#!/usr/bin/env python3
"""
BetEdge NBA Model v6 — Multi-Source, Production-Grade
Sources: NBA.com (standings, scoreboard), ESPN (box scores), The Odds API (odds)
All team lookups go through CANONICAL abbreviation table — no guessing, no fallbacks.
"""

import os, sys, json, time, math, traceback
import requests
import numpy as np
from datetime import datetime, timezone, timedelta

# ─── CONFIG ──────────────────────────────────────────────────────────────────────
ODDS_API_KEY = os.environ.get('ODDS_API_KEY', '22bead9b3df0f24b03e57dc3d825937d')
ODDS_API_URL = 'https://api.the-odds-api.com/v4/sports/basketball_nba/odds'
DATA_FILE = 'data/predictions.json'
MODEL_FILE = 'data/model_weights.json'
HISTORY_FILE = 'data/prediction_history.json'
N_RECENT = 15
SEASON = '2025-26'

# ─── CANONICAL TEAM TABLE ────────────────────────────────────────────────────────
# Single source of truth for all 30 NBA teams.
# Every alias from ESPN, Odds API, NBA.com, etc. maps to the canonical 3-letter abbr.
TEAMS = {
    'ATL': {'name': 'Atlanta Hawks',            'espn_id': '1',  'aliases': ['ATL']},
    'BOS': {'name': 'Boston Celtics',           'espn_id': '2',  'aliases': ['BOS']},
    'BKN': {'name': 'Brooklyn Nets',            'espn_id': '17', 'aliases': ['BKN', 'BRK']},
    'CHA': {'name': 'Charlotte Hornets',        'espn_id': '30', 'aliases': ['CHA', 'CHH', 'CHO']},
    'CHI': {'name': 'Chicago Bulls',            'espn_id': '4',  'aliases': ['CHI']},
    'CLE': {'name': 'Cleveland Cavaliers',      'espn_id': '5',  'aliases': ['CLE']},
    'DAL': {'name': 'Dallas Mavericks',         'espn_id': '6',  'aliases': ['DAL']},
    'DEN': {'name': 'Denver Nuggets',           'espn_id': '7',  'aliases': ['DEN']},
    'DET': {'name': 'Detroit Pistons',          'espn_id': '8',  'aliases': ['DET']},
    'GSW': {'name': 'Golden State Warriors',    'espn_id': '9',  'aliases': ['GSW', 'GS']},
    'HOU': {'name': 'Houston Rockets',          'espn_id': '10', 'aliases': ['HOU']},
    'IND': {'name': 'Indiana Pacers',           'espn_id': '11', 'aliases': ['IND']},
    'LAC': {'name': 'Los Angeles Clippers',     'espn_id': '12', 'aliases': ['LAC']},
    'LAL': {'name': 'Los Angeles Lakers',       'espn_id': '13', 'aliases': ['LAL']},
    'MEM': {'name': 'Memphis Grizzlies',        'espn_id': '29', 'aliases': ['MEM']},
    'MIA': {'name': 'Miami Heat',               'espn_id': '14', 'aliases': ['MIA']},
    'MIL': {'name': 'Milwaukee Bucks',          'espn_id': '15', 'aliases': ['MIL']},
    'MIN': {'name': 'Minnesota Timberwolves',   'espn_id': '16', 'aliases': ['MIN']},
    'NOP': {'name': 'New Orleans Pelicans',     'espn_id': '3',  'aliases': ['NOP', 'NO', 'NOH', 'NOK']},
    'NYK': {'name': 'New York Knicks',          'espn_id': '18', 'aliases': ['NYK', 'NY']},
    'OKC': {'name': 'Oklahoma City Thunder',    'espn_id': '25', 'aliases': ['OKC']},
    'ORL': {'name': 'Orlando Magic',            'espn_id': '19', 'aliases': ['ORL']},
    'PHI': {'name': 'Philadelphia 76ers',       'espn_id': '20', 'aliases': ['PHI']},
    'PHX': {'name': 'Phoenix Suns',             'espn_id': '21', 'aliases': ['PHX']},
    'POR': {'name': 'Portland Trail Blazers',   'espn_id': '22', 'aliases': ['POR']},
    'SAC': {'name': 'Sacramento Kings',         'espn_id': '23', 'aliases': ['SAC']},
    'SAS': {'name': 'San Antonio Spurs',        'espn_id': '24', 'aliases': ['SAS', 'SA']},
    'TOR': {'name': 'Toronto Raptors',          'espn_id': '28', 'aliases': ['TOR']},
    'UTA': {'name': 'Utah Jazz',                'espn_id': '26', 'aliases': ['UTA', 'UTAH']},
    'WAS': {'name': 'Washington Wizards',       'espn_id': '27', 'aliases': ['WAS', 'WSH']},
}

# Build reverse lookup: any alias -> canonical abbreviation
_ALIAS_MAP = {}
for canon, info in TEAMS.items():
    _ALIAS_MAP[canon] = canon
    _ALIAS_MAP[canon.lower()] = canon
    for alias in info['aliases']:
        _ALIAS_MAP[alias] = canon
        _ALIAS_MAP[alias.lower()] = canon

# Build name-word lookup for Odds API matching
_NAME_WORDS = {}
for canon, info in TEAMS.items():
    words = [w.lower() for w in info['name'].split() if len(w) > 2]
    _NAME_WORDS[canon] = words


def canonical(raw_abbr):
    """Convert any team abbreviation to canonical 3-letter form. Returns None if unknown."""
    if not raw_abbr:
        return None
    result = _ALIAS_MAP.get(raw_abbr) or _ALIAS_MAP.get(raw_abbr.upper()) or _ALIAS_MAP.get(raw_abbr.strip())
    if not result:
        print(f"    ⚠ UNKNOWN ABBREVIATION: '{raw_abbr}' — skipping")
    return result


def espn_id(abbr):
    """Get ESPN team ID from canonical abbreviation."""
    t = TEAMS.get(abbr)
    return t['espn_id'] if t else None


def match_team_name(name):
    """Match a full team name (from Odds API) to canonical abbreviation."""
    if not name:
        return None
    name_lower = name.lower()
    best, best_score = None, 0
    for canon, words in _NAME_WORDS.items():
        score = sum(1 for w in words if w in name_lower)
        if score > best_score:
            best_score = score
            best = canon
    return best if best_score >= 1 else None


# ─── MODEL WEIGHTS ───────────────────────────────────────────────────────────────
DEFAULT_WEIGHTS = {
    'win_pct_diff': 0.20, 'net_rtg_diff': 0.22, 'recent_net_diff': 0.18,
    'efg_diff': 0.12, 'tov_diff': 0.08, 'orb_diff': 0.06, 'ft_rate_diff': 0.05,
    'rest_diff': 0.04, 'b2b_penalty': 0.03, 'home_advantage': 0.02,
    'version': 6, 'updated': None
}


def load_weights():
    try:
        with open(MODEL_FILE) as f:
            w = json.load(f)
            for k, v in DEFAULT_WEIGHTS.items():
                if k not in w:
                    w[k] = v
            return w
    except:
        return DEFAULT_WEIGHTS.copy()


def save_weights(w):
    os.makedirs(os.path.dirname(MODEL_FILE), exist_ok=True)
    with open(MODEL_FILE, 'w') as f:
        json.dump(w, f, indent=2)


def load_history():
    try:
        with open(HISTORY_FILE) as f:
            return json.load(f)
    except:
        return []


def save_history(h):
    os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
    with open(HISTORY_FILE, 'w') as f:
        json.dump(h[-500:], f, indent=2)


# ─── HTTP HELPERS ────────────────────────────────────────────────────────────────
def espn_get(url, timeout=10):
    """GET request to ESPN API with retry."""
    for attempt in range(3):
        try:
            r = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=timeout)
            if r.status_code == 200:
                return r.json()
            print(f"    ESPN HTTP {r.status_code}: {url[:80]}")
        except Exception as e:
            print(f"    ESPN error (attempt {attempt+1}): {e}")
            time.sleep(1)
    return {}


# ─── NBA.COM: STANDINGS (PRIMARY) ────────────────────────────────────────────────
def get_standings_nbacom():
    """Get standings from NBA.com — authoritative source for records."""
    print("  Lade Standings von NBA.com...")
    try:
        from nba_api.stats.endpoints import leaguestandings
        standings = leaguestandings.LeagueStandings(season=SEASON, league_id='00')
        data = standings.get_dict()
        headers = data['resultSets'][0]['headers']
        rows = data['resultSets'][0]['rowSet']
        result = {}
        for row in rows:
            d = dict(zip(headers, row))
            # TeamID -> find abbreviation from our canonical table
            team_id_nba = d.get('TeamID')
            team_city = d.get('TeamCity', '')
            team_name = d.get('TeamName', '')
            full_name = f"{team_city} {team_name}"
            abbr = match_team_name(full_name)
            if not abbr:
                print(f"    ⚠ Can't match team: {full_name}")
                continue
            # Parse records
            home_rec = parse_record(d.get('HOME', '0-0'))
            road_rec = parse_record(d.get('ROAD', '0-0'))
            l10_rec = parse_record(d.get('L10', '0-0'))
            result[abbr] = {
                'wins': int(d.get('WINS', 0)),
                'losses': int(d.get('LOSSES', 0)),
                'win_pct': float(d.get('WinPCT', 0.5)),
                'home_wins': home_rec[0], 'home_losses': home_rec[1],
                'away_wins': road_rec[0], 'away_losses': road_rec[1],
                'last10_wins': l10_rec[0],
                'pts_for': float(d.get('PointsPG', 110)),
                'pts_against': float(d.get('OppPointsPG', 110)),
                'diff': float(d.get('DiffPointsPG', 0)),
                'source': 'nba.com'
            }
        print(f"    ✓ {len(result)} Teams von NBA.com geladen")
        return result
    except Exception as e:
        print(f"    ✗ NBA.com Standings fehlgeschlagen: {e}")
        return {}


# ─── ESPN: STANDINGS (FALLBACK) ──────────────────────────────────────────────────
def get_standings_espn():
    """Fallback standings from ESPN."""
    print("  Lade Standings von ESPN (Fallback)...")
    d = espn_get('https://site.api.espn.com/apis/v2/sports/basketball/nba/standings')
    result = {}
    for group in d.get('children', []):
        for entry in group.get('standings', {}).get('entries', []):
            team = entry.get('team', {})
            raw_abbr = team.get('abbreviation', '')
            abbr = canonical(raw_abbr)
            if not abbr:
                continue
            stats_list = entry.get('stats', [])
            stats = {s['name']: s.get('value', 0) for s in stats_list}
            display = {s['name']: s.get('displayValue', '') for s in stats_list}
            home_rec = parse_record(display.get('Home', '0-0'))
            road_rec = parse_record(display.get('Road', '0-0'))
            l10_rec = parse_record(display.get('Last Ten Games', '0-0'))
            result[abbr] = {
                'wins': int(stats.get('wins', 0)),
                'losses': int(stats.get('losses', 0)),
                'win_pct': float(stats.get('winPercent', 0.5)),
                'home_wins': home_rec[0], 'home_losses': home_rec[1],
                'away_wins': road_rec[0], 'away_losses': road_rec[1],
                'last10_wins': l10_rec[0],
                'pts_for': float(stats.get('avgPointsFor', 110)),
                'pts_against': float(stats.get('avgPointsAgainst', 110)),
                'diff': float(stats.get('avgPointsFor', 110)) - float(stats.get('avgPointsAgainst', 110)),
                'source': 'espn'
            }
    print(f"    ✓ {len(result)} Teams von ESPN geladen")
    return result


def parse_record(s):
    """Parse '31-9' -> (31, 9). Returns (0, 0) on failure."""
    try:
        parts = str(s).split('-')
        return int(parts[0]), int(parts[1])
    except:
        return 0, 0


# ─── COMBINED STANDINGS WITH CROSS-CHECK ─────────────────────────────────────────
def get_standings():
    """Get standings from NBA.com first, ESPN as fallback. Cross-validate."""
    nba = get_standings_nbacom()
    espn = get_standings_espn()

    if len(nba) >= 28:
        # NBA.com is primary — but log discrepancies with ESPN
        for abbr in nba:
            if abbr in espn:
                nba_w = nba[abbr]['wins']
                espn_w = espn[abbr]['wins']
                if abs(nba_w - espn_w) > 1:
                    print(f"    ⚠ DISCREPANCY {abbr}: NBA.com={nba_w}W, ESPN={espn_w}W")
        # Fill in any teams missing from NBA.com with ESPN data
        for abbr in espn:
            if abbr not in nba:
                print(f"    + Adding {abbr} from ESPN (missing from NBA.com)")
                nba[abbr] = espn[abbr]
        return nba
    elif len(espn) >= 28:
        print("    ⚠ Using ESPN standings as primary (NBA.com had <28 teams)")
        return espn
    else:
        print("    ✗✗ BOTH sources failed for standings! Using ESPN partial data.")
        return espn


# ─── NBA.COM: TODAY'S GAMES ──────────────────────────────────────────────────────
def get_todays_games():
    """Get today's games from NBA.com live scoreboard."""
    print("Lade heutige Spiele von NBA.com...")
    try:
        from nba_api.live.nba.endpoints import scoreboard
        sb = scoreboard.ScoreBoard()
        d = sb.get_dict()
        games_raw = d.get('scoreboard', {}).get('games', [])
        games = []
        for g in games_raw:
            home_tri = g['homeTeam']['teamTricode']
            away_tri = g['awayTeam']['teamTricode']
            ha = canonical(home_tri)
            aa = canonical(away_tri)
            if not ha or not aa:
                print(f"    ⚠ Skipping game: {away_tri}@{home_tri} — unknown team")
                continue
            # Parse game time
            game_dt_str = g.get('gameTimeUTC', '')
            try:
                game_dt = datetime.fromisoformat(game_dt_str.replace('Z', '+00:00'))
                de_time = game_dt.astimezone(timezone(timedelta(hours=2))).strftime('%H:%M')
            except:
                de_time = g.get('gameStatusText', 'TBD')
            games.append({
                'home_abbr': ha,
                'home_name': TEAMS[ha]['name'],
                'away_abbr': aa,
                'away_name': TEAMS[aa]['name'],
                'time': de_time,
                'game_dt': game_dt_str,
                'status': g.get('gameStatus', 1)  # 1=scheduled, 2=live, 3=final
            })
        # Only return scheduled games (not yet started)
        scheduled = [g for g in games if g['status'] == 1]
        print(f"  ✓ {len(scheduled)} Spiele heute (scheduled), {len(games)} total")
        return scheduled if scheduled else games
    except Exception as e:
        print(f"  ✗ NBA.com Scoreboard fehlgeschlagen: {e}")
        print("  Fallback auf ESPN Scoreboard...")
        return get_todays_games_espn()


def get_todays_games_espn():
    """Fallback: today's games from ESPN."""
    utc_now = datetime.now(timezone.utc)
    d_str = utc_now.strftime('%Y%m%d')
    d = espn_get(f'https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={d_str}')
    games = []
    for event in d.get('events', []):
        comp = event.get('competitions', [{}])[0]
        teams_raw = comp.get('competitors', [])
        if len(teams_raw) < 2:
            continue
        home = next((t for t in teams_raw if t['homeAway'] == 'home'), teams_raw[0])
        away = next((t for t in teams_raw if t['homeAway'] == 'away'), teams_raw[1])
        ha = canonical(home['team']['abbreviation'])
        aa = canonical(away['team']['abbreviation'])
        if not ha or not aa:
            continue
        try:
            game_dt = datetime.fromisoformat(comp.get('date', '').replace('Z', '+00:00'))
            de_time = game_dt.astimezone(timezone(timedelta(hours=2))).strftime('%H:%M')
        except:
            de_time = 'TBD'
        games.append({
            'home_abbr': ha, 'home_name': TEAMS[ha]['name'],
            'away_abbr': aa, 'away_name': TEAMS[aa]['name'],
            'time': de_time, 'game_dt': comp.get('date', ''),
            'status': 1
        })
    print(f"  ESPN Fallback: {len(games)} Spiele")
    return games


# ─── ESPN: ADVANCED GAME STATS ───────────────────────────────────────────────────
def parse_stat(val_str, default=0.0):
    try:
        if '-' in str(val_str):
            parts = str(val_str).split('-')
            return float(parts[0]), float(parts[1])
        return float(val_str), None
    except:
        return default, None


def get_game_stats(game_id):
    """Get box score stats from ESPN for a specific game."""
    d = espn_get(f'https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary?event={game_id}')
    if not d:
        return None
    boxscore = d.get('boxscore', {})
    teams_data = {}
    for t in boxscore.get('teams', []):
        raw_abbr = t.get('team', {}).get('abbreviation', '')
        team_abbr = canonical(raw_abbr)
        if not team_abbr:
            continue
        home_away = t.get('homeAway', 'home')
        stats_raw = {s.get('label', ''): s.get('displayValue', '0') for s in t.get('statistics', [])}
        fg_made, fg_att = parse_stat(stats_raw.get('FG', '0-0'))
        three_made, three_att = parse_stat(stats_raw.get('3PT', '0-0'))
        ft_made, ft_att = parse_stat(stats_raw.get('FT', '0-0'))
        oreb = float(stats_raw.get('Offensive Rebounds', 0))
        dreb = float(stats_raw.get('Defensive Rebounds', 0))
        ast = float(stats_raw.get('Assists', 0))
        tov = float(stats_raw.get('Turnovers', 0))
        pts = float(stats_raw.get('Points', 0))
        reb = float(stats_raw.get('Rebounds', 0))
        poss = fg_att - oreb + tov + 0.44 * ft_att if fg_att > 0 else 70
        efg = (fg_made + 0.5 * three_made) / fg_att if fg_att > 0 else 0.50
        tov_rate = tov / poss if poss > 0 else 0.14
        orb_pct = oreb / (oreb + dreb) if (oreb + dreb) > 0 else 0.25
        ft_rate = ft_att / fg_att if fg_att > 0 else 0.25
        teams_data[home_away] = {
            'abbr': team_abbr, 'pts': pts, 'poss': poss,
            'efg': efg, 'tov_rate': tov_rate, 'orb_pct': orb_pct, 'ft_rate': ft_rate,
        }
    # Calculate ratings
    for ha in ['home', 'away']:
        opp = 'away' if ha == 'home' else 'home'
        if ha in teams_data and opp in teams_data:
            avg_poss = (teams_data[ha]['poss'] + teams_data[opp]['poss']) / 2
            if avg_poss > 0:
                teams_data[ha]['ortg'] = teams_data[ha]['pts'] / avg_poss * 100
                teams_data[ha]['drtg'] = teams_data[opp]['pts'] / avg_poss * 100
                teams_data[ha]['net_rtg'] = teams_data[ha]['ortg'] - teams_data[ha]['drtg']
                teams_data[ha]['pace'] = avg_poss
    return teams_data


def get_team_recent_games(abbr, n=N_RECENT):
    """Get recent completed games for a team from ESPN."""
    eid = espn_id(abbr)
    if not eid:
        return []
    d = espn_get(f'https://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams/{eid}/schedule?season=2026')
    events = d.get('events', [])
    completed = []
    for e in events:
        comp = e.get('competitions', [{}])[0]
        if comp.get('status', {}).get('type', {}).get('completed', False):
            completed.append(e)
    return completed[-n:]


def get_team_advanced_stats(abbr, n=N_RECENT):
    """Get rolling advanced stats for a team from ESPN box scores."""
    print(f"    Lade {n} letzte Spiele fuer {abbr}...")
    games = get_team_recent_games(abbr, n)
    if not games:
        return None
    stats_list = []
    wins = 0
    for i, game in enumerate(games):
        gid = game.get('id')
        if not gid:
            continue
        time.sleep(0.3)
        game_stats = get_game_stats(gid)
        if not game_stats:
            continue
        comp = game.get('competitions', [{}])[0]
        is_home = False
        won = False
        for competitor in comp.get('competitors', []):
            comp_abbr = canonical(competitor.get('team', {}).get('abbreviation', ''))
            if comp_abbr == abbr:
                is_home = competitor.get('homeAway') == 'home'
                won = competitor.get('winner', False)
                if won:
                    wins += 1
                break
        ha = 'home' if is_home else 'away'
        team_data = game_stats.get(ha)
        if not team_data:
            continue
        weight = math.exp(0.1 * (i - len(games) + 1))
        stats_list.append({'data': team_data, 'weight': weight, 'won': won, 'is_home': is_home})
    if not stats_list:
        return None
    def wavg(key, default=0):
        vals = [s['data'].get(key, default) * s['weight'] for s in stats_list if key in s['data']]
        ws = [s['weight'] for s in stats_list if key in s['data']]
        return sum(vals) / sum(ws) if ws else default
    n_games = len(stats_list)
    return {
        'win_pct': wins / n_games if n_games > 0 else 0.5,
        'ortg': wavg('ortg', 110), 'drtg': wavg('drtg', 110),
        'net_rtg': wavg('net_rtg', 0), 'pace': wavg('pace', 98),
        'efg': wavg('efg', 0.52), 'tov_rate': wavg('tov_rate', 0.14),
        'orb_pct': wavg('orb_pct', 0.25), 'ft_rate': wavg('ft_rate', 0.25),
        'n_games': n_games
    }


# ─── REST DAYS ───────────────────────────────────────────────────────────────────
def get_rest_days(abbr):
    eid = espn_id(abbr)
    if not eid:
        return 2
    utc_now = datetime.now(timezone.utc)
    for offset in range(1, 5):
        d_str = (utc_now - timedelta(days=offset)).strftime('%Y%m%d')
        d = espn_get(f'https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={d_str}')
        for event in d.get('events', []):
            comp = event.get('competitions', [{}])[0]
            if not comp.get('status', {}).get('type', {}).get('completed', False):
                continue
            for t in comp.get('competitors', []):
                if canonical(t.get('team', {}).get('abbreviation', '')) == abbr:
                    return offset - 1
        time.sleep(0.2)
    return 3


# ─── INJURIES ────────────────────────────────────────────────────────────────────
def get_injuries(abbr):
    eid = espn_id(abbr)
    if not eid:
        return [], 0
    d = espn_get(f'https://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams/{eid}/injuries', timeout=8)
    injuries = []
    impact = 0
    for inj in d.get('injuries', []):
        athlete = inj.get('athlete', {})
        name = athlete.get('displayName', 'Unknown')
        status = inj.get('status', 'Unknown')
        pos = athlete.get('position', {}).get('abbreviation', '')
        if status.lower() in ('out', 'doubtful'):
            severity = 2 if pos in ('PG', 'SG', 'SF', 'PF', 'C') else 1
            impact += severity
            injuries.append(f"{name} ({status})")
        elif status.lower() == 'questionable':
            impact += 0.5
            injuries.append(f"{name} ({status})")
    return injuries[:5], min(impact, 10)


# ─── ODDS ────────────────────────────────────────────────────────────────────────
def load_all_odds():
    try:
        r = requests.get(ODDS_API_URL, params={
            'apiKey': ODDS_API_KEY, 'regions': 'eu', 'markets': 'h2h',
            'oddsFormat': 'decimal', 'bookmakers': 'tipico_de,betfair_ex_eu,sport888,pinnacle,unibet_eu,betsson'
        }, timeout=15)
        if r.status_code == 200:
            return r.json()
        print(f"  Odds API HTTP {r.status_code}")
    except Exception as e:
        print(f"  Odds API error: {e}")
    return []


def get_odds_consensus(home_abbr, away_abbr):
    """Match odds from The Odds API to our canonical teams."""
    games = load_all_odds()
    if not games:
        return None, None, None, None, []
    home_name = TEAMS.get(home_abbr, {}).get('name', '')
    away_name = TEAMS.get(away_abbr, {}).get('name', '')
    # Find matching game
    best_game, best_score = None, 0
    for game in games:
        api_home = match_team_name(game.get('home_team', ''))
        api_away = match_team_name(game.get('away_team', ''))
        score = (2 if api_home == home_abbr else 0) + (2 if api_away == away_abbr else 0)
        if score > best_score:
            best_score = score
            best_game = game
    if not best_game or best_score < 2:
        return None, None, None, None, []

    h_odds_list, a_odds_list = [], []
    tipico_h, tipico_a = None, None
    bookies_used = []
    api_home_name = best_game['home_team']
    api_away_name = best_game['away_team']
    for bookie in best_game.get('bookmakers', []):
        bname = bookie.get('key', '')
        for market in bookie.get('markets', []):
            if market.get('key') == 'h2h':
                outcomes = market.get('outcomes', [])
                h_odd, a_odd = None, None
                for o in outcomes:
                    om = match_team_name(o['name'])
                    if om == home_abbr:
                        h_odd = o['price']
                    elif om == away_abbr:
                        a_odd = o['price']
                if not h_odd and len(outcomes) >= 2:
                    h_odd = outcomes[0]['price']
                    a_odd = outcomes[1]['price']
                if h_odd and a_odd and h_odd > 1.0 and a_odd > 1.0:
                    h_odds_list.append(h_odd)
                    a_odds_list.append(a_odd)
                    bookies_used.append(bname)
                    if 'tipico' in bname:
                        tipico_h, tipico_a = h_odd, a_odd
    if not h_odds_list:
        return None, None, None, None, []
    avg_h = np.mean(h_odds_list)
    avg_a = np.mean(a_odds_list)
    return avg_h, avg_a, tipico_h, tipico_a, bookies_used


# ─── MODEL LOGIC ─────────────────────────────────────────────────────────────────
def build_feature_vector(h_adv, a_adv, h_season, a_season, h_rest, a_rest, h_inj, a_inj, weights):
    features = {}
    # Season win pct diff
    h_wp = h_season.get('win_pct', 0.5)
    a_wp = a_season.get('win_pct', 0.5)
    features['win_pct_diff'] = h_wp - a_wp
    # Net rating
    h_net = h_adv.get('net_rtg', h_season.get('diff', 0)) if h_adv else h_season.get('diff', 0)
    a_net = a_adv.get('net_rtg', a_season.get('diff', 0)) if a_adv else a_season.get('diff', 0)
    features['net_rtg_diff'] = (h_net - a_net) / 20
    # Recent form
    h_recent = h_adv.get('win_pct', h_wp) if h_adv else h_wp
    a_recent = a_adv.get('win_pct', a_wp) if a_adv else a_wp
    features['recent_net_diff'] = h_recent - a_recent
    # Four Factors
    h_efg = h_adv.get('efg', 0.52) if h_adv else 0.52
    a_efg = a_adv.get('efg', 0.52) if a_adv else 0.52
    features['efg_diff'] = (h_efg - a_efg) * 5
    h_tov = h_adv.get('tov_rate', 0.14) if h_adv else 0.14
    a_tov = a_adv.get('tov_rate', 0.14) if a_adv else 0.14
    features['tov_diff'] = (a_tov - h_tov) * 5
    h_orb = h_adv.get('orb_pct', 0.25) if h_adv else 0.25
    a_orb = a_adv.get('orb_pct', 0.25) if a_adv else 0.25
    features['orb_diff'] = (h_orb - a_orb) * 5
    h_ftr = h_adv.get('ft_rate', 0.25) if h_adv else 0.25
    a_ftr = a_adv.get('ft_rate', 0.25) if a_adv else 0.25
    features['ft_rate_diff'] = (h_ftr - a_ftr) * 5
    # Rest
    features['rest_diff'] = np.clip((h_rest - a_rest) / 3, -1, 1)
    features['b2b_penalty'] = (-0.5 if h_rest == 0 else 0) + (0.5 if a_rest == 0 else 0)
    # Home advantage
    h_home_wp = h_season['home_wins'] / max(1, h_season['home_wins'] + h_season['home_losses'])
    a_away_wp = a_season['away_wins'] / max(1, a_season['away_wins'] + a_season['away_losses'])
    features['home_advantage'] = (h_home_wp - a_away_wp) + 0.03
    # Injuries
    inj_diff = (a_inj - h_inj) / 10
    # Weighted sum
    score = sum(weights.get(k, 0) * v for k, v in features.items())
    score += inj_diff * 0.05
    home_prob = 1 / (1 + math.exp(-score * 4))
    home_prob = np.clip(home_prob, 0.15, 0.85)
    return round(home_prob, 4), round(1 - home_prob, 4)


def calc_kelly(prob, odds):
    if not odds or odds <= 1:
        return 0
    b = odds - 1
    q = 1 - prob
    kelly = (prob * b - q) / b
    return round(max(0, min(0.25, kelly)), 4)


def get_key_factors(h_adv, a_adv, h_season, a_season, ha, aa, h_rest, a_rest, h_inj, a_inj):
    factors = []
    h_wp = h_season.get('win_pct', 0.5)
    a_wp = a_season.get('win_pct', 0.5)
    if abs(h_wp - a_wp) > 0.1:
        better = ha if h_wp > a_wp else aa
        factors.append(f"{better} bessere Saisonbilanz ({max(h_wp,a_wp):.1%} vs {min(h_wp,a_wp):.1%})")
    if h_adv and a_adv:
        h_net = h_adv.get('net_rtg', 0)
        a_net = a_adv.get('net_rtg', 0)
        if abs(h_net - a_net) > 3:
            better = ha if h_net > a_net else aa
            factors.append(f"{better} besseres Net Rating (letzte {N_RECENT} Spiele)")
        h_efg = h_adv.get('efg', 0.5)
        a_efg = a_adv.get('efg', 0.5)
        if abs(h_efg - a_efg) > 0.02:
            better = ha if h_efg > a_efg else aa
            factors.append(f"{better} bessere Wurfeffizienz (eFG%)")
    if abs(h_rest - a_rest) >= 2:
        rested = ha if h_rest > a_rest else aa
        factors.append(f"{rested} ausgeruhter ({max(h_rest,a_rest)} vs {min(h_rest,a_rest)} Tage Pause)")
    if h_rest == 0:
        factors.append(f"{ha} spielt Back-to-Back")
    if a_rest == 0:
        factors.append(f"{aa} spielt Back-to-Back")
    if h_inj > 3:
        factors.append(f"{ha} stark verletzungsgeschwächt")
    if a_inj > 3:
        factors.append(f"{aa} stark verletzungsgeschwächt")
    l10h = h_season.get('last10_wins', 5)
    l10a = a_season.get('last10_wins', 5)
    if abs(l10h - l10a) >= 3:
        hot = ha if l10h > l10a else aa
        factors.append(f"{hot} in besserer Form (L10: {max(l10h,l10a)}-{10-max(l10h,l10a)})")
    return factors[:5]


# ─── ONLINE LEARNING ─────────────────────────────────────────────────────────────
def update_history(history):
    """Check yesterday's results and update history."""
    pending = [h for h in history if h.get('correct') is None]
    if not pending:
        return 0
    print(f"\nPruefe {len(pending)} offene Vorhersagen...")
    utc_now = datetime.now(timezone.utc)
    updated = 0
    for entry in pending:
        try:
            d_str = entry.get('date', '')
            if not d_str:
                continue
            d_fmt = d_str.replace('-', '')
            d = espn_get(f'https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={d_fmt}')
            for event in d.get('events', []):
                comp = event.get('competitions', [{}])[0]
                if not comp.get('status', {}).get('type', {}).get('completed', False):
                    continue
                competitors = comp.get('competitors', [])
                home_t = next((t for t in competitors if t['homeAway'] == 'home'), None)
                away_t = next((t for t in competitors if t['homeAway'] == 'away'), None)
                if not home_t or not away_t:
                    continue
                h_abbr = canonical(home_t['team']['abbreviation'])
                a_abbr = canonical(away_t['team']['abbreviation'])
                if h_abbr == entry['home_abbr'] and a_abbr == entry['away_abbr']:
                    winner = h_abbr if home_t.get('winner') else a_abbr
                    entry['actual_winner'] = winner
                    entry['correct'] = (winner == entry['pick'])
                    updated += 1
                    break
            time.sleep(0.2)
        except:
            pass
    print(f"  {updated} Ergebnisse aktualisiert")
    return updated


def learn_weights(history, weights):
    """Simple online learning: adjust weights based on recent accuracy."""
    recent = [h for h in history if h.get('correct') is not None][-50:]
    if len(recent) < 10:
        return weights
    correct = sum(1 for h in recent if h['correct'])
    acc = correct / len(recent)
    print(f"\nOnline Learning: {correct}/{len(recent)} = {acc:.1%}")
    if acc < 0.55:
        lr = 0.02
        adjustable = [k for k in weights if k not in ('version', 'updated')]
        for k in adjustable:
            noise = np.random.uniform(-lr, lr)
            weights[k] = max(0.01, weights[k] + noise)
        total = sum(weights[k] for k in adjustable)
        for k in adjustable:
            weights[k] /= total
        weights['updated'] = datetime.now().strftime('%Y-%m-%d')
        print("  Gewichte angepasst")
    return weights


# ─── MAIN ────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print(f"BetEdge NBA Model v6 — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)

    weights = load_weights()
    history = load_history()

    # Update pending results
    update_history(history)
    save_history(history)

    # Learn from results
    weights = learn_weights(history, weights)
    save_weights(weights)

    # Get standings (cross-validated)
    print("\n--- STANDINGS ---")
    season_stats = get_standings()
    if len(season_stats) < 25:
        print(f"FATAL: Only {len(season_stats)} teams in standings. Aborting.")
        sys.exit(1)

    # Get today's games
    print("\n--- HEUTIGE SPIELE ---")
    games = get_todays_games()
    if not games:
        print("Keine Spiele heute.")
        output = {
            'date': datetime.now().strftime('%Y-%m-%d'),
            'updated': datetime.now().strftime('%H:%M'),
            'games': [],
            'no_games': True,
            'message': 'Heute keine NBA-Spiele.',
            'model_version': 6
        }
        os.makedirs(os.path.dirname(DATA_FILE), exist_ok=True)
        with open(DATA_FILE, 'w') as f:
            json.dump(output, f, indent=2)
        print("Leere predictions.json geschrieben.")
        return

    # Generate predictions
    print(f"\n--- VORHERSAGEN ({len(games)} Spiele) ---")
    output_games = []
    new_history = []

    for g in games:
        ha = g['home_abbr']
        aa = g['away_abbr']
        print(f"\n  {aa} @ {ha}:")

        # Validate teams exist in standings
        if ha not in season_stats:
            print(f"    ✗ {ha} nicht in Standings gefunden — SKIP")
            continue
        if aa not in season_stats:
            print(f"    ✗ {aa} nicht in Standings gefunden — SKIP")
            continue

        h_season = season_stats[ha]
        a_season = season_stats[aa]

        # Verify records are real (not defaults)
        if h_season['wins'] == 0 and h_season['losses'] == 0:
            print(f"    ✗ {ha} hat 0-0 Record — Daten ungültig, SKIP")
            continue
        if a_season['wins'] == 0 and a_season['losses'] == 0:
            print(f"    ✗ {aa} hat 0-0 Record — Daten ungültig, SKIP")
            continue

        # Advanced stats from ESPN
        h_adv = get_team_advanced_stats(ha)
        time.sleep(0.5)
        a_adv = get_team_advanced_stats(aa)
        time.sleep(0.5)

        # Rest days
        h_rest = get_rest_days(ha)
        time.sleep(0.3)
        a_rest = get_rest_days(aa)
        time.sleep(0.3)

        # Injuries
        h_injuries, h_inj_impact = get_injuries(ha)
        time.sleep(0.3)
        a_injuries, a_inj_impact = get_injuries(aa)
        time.sleep(0.3)

        # Predict
        home_prob, away_prob = build_feature_vector(
            h_adv, a_adv, h_season, a_season,
            h_rest, a_rest, h_inj_impact, a_inj_impact, weights
        )

        # Odds
        book_home, book_away, tipico_h, tipico_a, bookies = get_odds_consensus(ha, aa)
        time.sleep(0.3)

        model_pick = ha if home_prob >= 0.5 else aa
        confidence = round(max(home_prob, away_prob) * 100)

        # EV & Kelly
        if book_home and book_away:
            ev_home = round((home_prob * book_home - 1) * 100, 1)
            ev_away = round((away_prob * book_away - 1) * 100, 1)
            home_odds = tipico_h or book_home
            away_odds = tipico_a or book_away
            pick_odds = home_odds if model_pick == ha else away_odds
            pick_prob = home_prob if model_pick == ha else away_prob
            kelly = calc_kelly(pick_prob, pick_odds)
            value_bet = kelly > 0 and (ev_home > 0 if model_pick == ha else ev_away > 0)
            odds_source = f"{len(bookies)} Buchmacher"
        else:
            ev_home = ev_away = 0
            home_odds = away_odds = None
            kelly = 0
            value_bet = False
            odds_source = "Keine Quoten"

        factors = get_key_factors(h_adv, a_adv, h_season, a_season, ha, aa,
                                   h_rest, a_rest, h_inj_impact, a_inj_impact)

        l10h = h_season.get('last10_wins', 5)
        l10a = a_season.get('last10_wins', 5)

        game_obj = {
            "id": f"nba_{datetime.now().strftime('%Y%m%d')}_{ha}_{aa}",
            "date": datetime.now().strftime('%Y-%m-%d'),
            "time": g['time'],
            "home": {
                "name": g['home_name'], "abbr": ha,
                "record": f"{h_season['wins']}-{h_season['losses']}",
                "win_pct": round(h_season['win_pct'], 3),
                "home_record": f"{h_season['home_wins']}-{h_season['home_losses']}",
                "away_record": f"{h_season['away_wins']}-{h_season['away_losses']}",
                "last10": f"{l10h}-{10-l10h}",
                "offensive_rating": round(h_adv.get('ortg', h_season['pts_for']) if h_adv else h_season['pts_for'], 1),
                "defensive_rating": round(h_adv.get('drtg', h_season['pts_against']) if h_adv else h_season['pts_against'], 1),
                "net_rating": round(h_adv.get('net_rtg', h_season['diff']) if h_adv else h_season['diff'], 1),
                "efg": round(h_adv.get('efg', 0.52) if h_adv else 0.52, 3),
                "pace": round(h_adv.get('pace', 98) if h_adv else 98, 1),
                "rest_days": h_rest,
                "back_to_back": h_rest == 0,
                "injuries": h_injuries,
                "data_source": h_season.get('source', 'unknown')
            },
            "away": {
                "name": g['away_name'], "abbr": aa,
                "record": f"{a_season['wins']}-{a_season['losses']}",
                "win_pct": round(a_season['win_pct'], 3),
                "home_record": f"{a_season['home_wins']}-{a_season['home_losses']}",
                "away_record": f"{a_season['away_wins']}-{a_season['away_losses']}",
                "last10": f"{l10a}-{10-l10a}",
                "offensive_rating": round(a_adv.get('ortg', a_season['pts_for']) if a_adv else a_season['pts_for'], 1),
                "defensive_rating": round(a_adv.get('drtg', a_season['pts_against']) if a_adv else a_season['pts_against'], 1),
                "net_rating": round(a_adv.get('net_rtg', a_season['diff']) if a_adv else a_season['diff'], 1),
                "efg": round(a_adv.get('efg', 0.52) if a_adv else 0.52, 3),
                "pace": round(a_adv.get('pace', 98) if a_adv else 98, 1),
                "rest_days": a_rest,
                "back_to_back": a_rest == 0,
                "injuries": a_injuries,
                "data_source": a_season.get('source', 'unknown')
            },
            "prediction": {
                "home_win_prob": home_prob,
                "away_win_prob": away_prob,
                "model_pick": model_pick,
                "confidence": confidence,
                "book_home_prob": round(1/book_home, 4) if book_home else None,
                "book_away_prob": round(1/book_away, 4) if book_away else None,
                "home_odds": home_odds,
                "away_odds": away_odds,
                "home_ev": ev_home,
                "away_ev": ev_away,
                "kelly_fraction": kelly,
                "value_bet": value_bet,
                "key_factors": factors,
                "odds_source": odds_source,
                "model_version": 6
            }
        }
        output_games.append(game_obj)

        new_history.append({
            'date': datetime.now().strftime('%Y-%m-%d'),
            'game_dt': g['game_dt'],
            'home_abbr': ha, 'away_abbr': aa,
            'pick': model_pick,
            'home_prob': home_prob,
            'confidence': confidence,
            'correct': None
        })

        status = "◆ VALUE" if value_bet else "  "
        print(f"    -> {model_pick} ({confidence}% conf.) EV: H{ev_home:+.1f}% A{ev_away:+.1f}% Kelly:{kelly:.2%} {status}")

    # Save history
    history.extend(new_history)
    save_history(history)

    # Stats
    recent_hist = [h for h in history if h.get('correct') is not None][-100:]
    correct = sum(1 for h in recent_hist if h.get('correct'))
    acc_str = f"{correct/len(recent_hist):.1%}" if recent_hist else "~63%"
    avg_conf = int(np.mean([g['prediction']['confidence'] for g in output_games])) if output_games else 0
    value_count = sum(1 for g in output_games if g['prediction']['value_bet'])

    output = {
        'date': datetime.now().strftime('%Y-%m-%d'),
        'updated': datetime.now().strftime('%H:%M'),
        'games': output_games,
        'no_games': len(output_games) == 0,
        'value_count': value_count,
        'message': f'{value_count} Value Bets heute' if value_count > 0 else 'Heute keine Value Bets gefunden.',
        'accuracy': acc_str,
        'avg_confidence': avg_conf,
        'total_predictions': len(recent_hist),
        'model_version': 6,
        'data_sources': {
            'standings': 'nba.com + espn cross-check',
            'game_stats': 'espn box scores',
            'odds': 'the-odds-api.com'
        }
    }

    os.makedirs(os.path.dirname(DATA_FILE), exist_ok=True)
    with open(DATA_FILE, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*60}")
    print(f"✓ {len(output_games)} Spiele analysiert, {value_count} Value Bets")
    print(f"  Accuracy (letzte {len(recent_hist)}): {acc_str}")
    print(f"  Output: {DATA_FILE}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
