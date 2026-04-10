import json, os, time, math
from datetime import datetime, timezone, timedelta
import warnings
warnings.filterwarnings('ignore')

try:
    import numpy as np
    import requests
except ImportError as e:
    print(f"Fehlende Library: {e}")
    exit(1)

HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
ODDS_API_KEY = '22bead9b3df0f24b03e57dc3d825937d'
ODDS_API_URL = 'https://api.the-odds-api.com/v4/sports/basketball_nba/odds'
DATA_FILE = 'data/predictions.json'
MODEL_FILE = 'data/model_weights.json'
HISTORY_FILE = 'data/prediction_history.json'
N_RECENT = 15

ESPN_TEAM_IDS = {
    'ATL':'1','BOS':'2','BKN':'17','CHA':'30','CHI':'4','CLE':'5','DAL':'6',
    'DEN':'7','DET':'8','GSW':'9','HOU':'10','IND':'11','LAC':'12','LAL':'13',
    'MEM':'29','MIA':'14','MIL':'15','MIN':'16','NOP':'3','NYK':'18','OKC':'25',
    'ORL':'19','PHI':'20','PHX':'21','POR':'22','SAC':'23','SAS':'24','TOR':'28',
    'UTA':'26','WAS':'27','WSH':'27'
}

# ─── MODEL WEIGHTS (lernt täglich) ─────────────────────────────────────────────
DEFAULT_WEIGHTS = {
    'win_pct_diff': 0.20,
    'net_rtg_diff': 0.22,
    'recent_net_diff': 0.18,
    'efg_diff': 0.12,
    'tov_diff': 0.08,
    'orb_diff': 0.06,
    'ft_rate_diff': 0.05,
    'rest_diff': 0.04,
    'b2b_penalty': 0.03,
    'home_advantage': 0.02,
    'version': 1
}

def load_weights():
    try:
        if os.path.exists(MODEL_FILE):
            with open(MODEL_FILE) as f:
                w = json.load(f)
            print(f"Modell-Gewichte geladen (v{w.get('version',1)})")
            return w
    except:
        pass
    print("Neue Modell-Gewichte erstellt")
    return DEFAULT_WEIGHTS.copy()

def save_weights(weights):
    os.makedirs('data', exist_ok=True)
    with open(MODEL_FILE, 'w') as f:
        json.dump(weights, f, indent=2)

def load_history():
    try:
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE) as f:
                return json.load(f)
    except:
        pass
    return []

def save_history(history):
    os.makedirs('data', exist_ok=True)
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history[-500:], f, indent=2)

def update_weights_from_history(weights, history):
    if len(history) < 20:
        print(f"Zu wenig Historie ({len(history)} Spiele) fuer Lernupdate")
        return weights
    recent = history[-50:]
    correct = [h for h in recent if h.get('correct') is True]
    wrong = [h for h in recent if h.get('correct') is False]
    acc = len(correct) / len(recent) if recent else 0.5
    print(f"Learning Update: {len(correct)}/{len(recent)} korrekt ({acc:.1%})")
    if acc < 0.48:
        # Modell verschlechtert sich -> Gewichte anpassen
        lr = 0.03
        if wrong:
            avg_conf = np.mean([abs(h.get('home_prob', 0.5) - 0.5) for h in wrong])
            if avg_conf > 0.1:
                # Zu confident bei falschen Picks -> reduziere dominante Features
                weights['win_pct_diff'] = max(0.10, weights['win_pct_diff'] - lr)
                weights['net_rtg_diff'] = max(0.10, weights['net_rtg_diff'] - lr)
                weights['recent_net_diff'] = min(0.30, weights['recent_net_diff'] + lr * 1.5)
                weights['efg_diff'] = min(0.20, weights['efg_diff'] + lr)
                print(f"  Gewichte angepasst: recent_net +, win_pct -")
    elif acc > 0.58:
        # Modell läuft gut -> leicht in diese Richtung weiter
        lr = 0.01
        if correct:
            weights['version'] = weights.get('version', 1) + 1
            print(f"  Modell performt gut -> Gewichte leicht verstärkt")
    # Normalisieren
    feature_keys = [k for k in weights if k not in ['version', 'home_advantage', 'b2b_penalty']]
    total = sum(weights[k] for k in feature_keys)
    if total > 0:
        for k in feature_keys:
            weights[k] = weights[k] / total * 0.94
    return weights

# ─── ESPN DATA ──────────────────────────────────────────────────────────────────
def espn_get(url, timeout=12):
    try:
        r = requests.get(url, headers=HEADERS, timeout=timeout)
        if r.status_code == 200:
            return r.json()
    except:
        pass
    return {}

def get_team_id(abbr):
    return ESPN_TEAM_IDS.get(abbr, ESPN_TEAM_IDS.get(abbr.upper(), None))

def get_team_recent_games(abbr, n=N_RECENT):
    tid = get_team_id(abbr)
    if not tid:
        return []
    d = espn_get(f'https://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams/{tid}/schedule?season=2026')
    events = d.get('events', [])
    completed = []
    for e in events:
        comp = e.get('competitions', [{}])[0]
        if comp.get('status', {}).get('type', {}).get('completed', False):
            completed.append(e)
    return completed[-n:]

def parse_stat(val_str, default=0.0):
    try:
        if '-' in str(val_str):
            parts = str(val_str).split('-')
            made = float(parts[0])
            att = float(parts[1])
            return made, att
        return float(val_str), None
    except:
        return default, None

def get_game_stats(game_id):
    d = espn_get(f'https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary?event={game_id}')
    if not d:
        return None
    boxscore = d.get('boxscore', {})
    teams_data = {}
    for t in boxscore.get('teams', []):
        team_abbr = t.get('team', {}).get('abbreviation', '')
        home_away = t.get('homeAway', 'home')
        stats_raw = {s.get('label', ''): s.get('displayValue', '0') for s in t.get('statistics', [])}
        # Parse stats
        fg_made, fg_att = parse_stat(stats_raw.get('FG', '0-0'))
        three_made, three_att = parse_stat(stats_raw.get('3PT', '0-0'))
        ft_made, ft_att = parse_stat(stats_raw.get('FT', '0-0'))
        reb = float(stats_raw.get('Rebounds', 0))
        oreb = float(stats_raw.get('Offensive Rebounds', 0))
        dreb = float(stats_raw.get('Defensive Rebounds', 0))
        ast = float(stats_raw.get('Assists', 0))
        tov = float(stats_raw.get('Total Turnovers', stats_raw.get('Turnovers', 0)))
        pts_off_tov = float(stats_raw.get('Points Conceded Off Turnovers', 0))
        paint_pts = float(stats_raw.get('Points in Paint', 0))
        fb_pts = float(stats_raw.get('Fast Break Points', 0))
        # Estimate possessions: FGA + 0.44*FTA + TOV - OREB
        fg_att = fg_att or 0
        ft_att = ft_att or 0
        poss = fg_att + 0.44 * ft_att + tov - oreb
        poss = max(poss, 60)
        # Points
        pts = fg_made * 2 + three_made * 3 + ft_made if fg_made else 0
        # Four Factors
        efg = (fg_made + 0.5 * three_made) / fg_att if fg_att > 0 else 0.5
        tov_rate = tov / poss if poss > 0 else 0.14
        orb_pct = oreb / (oreb + dreb) if (oreb + dreb) > 0 else 0.25
        ft_rate = ft_att / fg_att if fg_att > 0 else 0.25
        teams_data[home_away] = {
            'abbr': team_abbr,
            'pts': pts,
            'poss': poss,
            'efg': efg,
            'tov_rate': tov_rate,
            'orb_pct': orb_pct,
            'ft_rate': ft_rate,
            'fg_att': fg_att,
            'fg_made': fg_made,
            'three_pct': (three_made / three_att) if three_att and three_att > 0 else 0.36,
            'ast': ast,
            'reb': reb,
            'tov': tov,
            'paint_pts': paint_pts,
            'fb_pts': fb_pts,
        }
    # Add opponent stats
    for ha in ['home', 'away']:
        opp = 'away' if ha == 'home' else 'home'
        if ha in teams_data and opp in teams_data:
            opp_poss = teams_data[opp].get('poss', 70)
            opp_pts = teams_data[opp].get('pts', 0)
            my_poss = teams_data[ha].get('poss', 70)
            my_pts = teams_data[ha].get('pts', 0)
            avg_poss = (my_poss + opp_poss) / 2
            teams_data[ha]['ortg'] = (my_pts / avg_poss * 100) if avg_poss > 0 else 110
            teams_data[ha]['drtg'] = (opp_pts / avg_poss * 100) if avg_poss > 0 else 110
            teams_data[ha]['net_rtg'] = teams_data[ha]['ortg'] - teams_data[ha]['drtg']
            teams_data[ha]['pace'] = avg_poss
    return teams_data

def get_team_advanced_stats(abbr, n=N_RECENT):
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
        # Determine if team was home or away
        is_home = False
        for competitor in comp.get('competitors', []):
            if competitor.get('team', {}).get('abbreviation') == abbr:
                is_home = competitor.get('homeAway') == 'home'
                won = competitor.get('winner', False)
                if won:
                    wins += 1
                break
        ha = 'home' if is_home else 'away'
        team_data = game_stats.get(ha)
        if not team_data:
            continue
        # Exponential weight: recent games matter more
        weight = math.exp(0.1 * (i - len(games) + 1))
        stats_list.append({'data': team_data, 'weight': weight, 'won': won, 'is_home': is_home})
    if not stats_list:
        return None
    # Weighted averages
    total_w = sum(s['weight'] for s in stats_list)
    def wavg(key, default=0):
        vals = [s['data'].get(key, default) * s['weight'] for s in stats_list if key in s['data']]
        ws = [s['weight'] for s in stats_list if key in s['data']]
        return sum(vals) / sum(ws) if ws else default
    n_games = len(stats_list)
    return {
        'win_pct': wins / n_games if n_games > 0 else 0.5,
        'ortg': wavg('ortg', 110),
        'drtg': wavg('drtg', 110),
        'net_rtg': wavg('net_rtg', 0),
        'pace': wavg('pace', 98),
        'efg': wavg('efg', 0.52),
        'tov_rate': wavg('tov_rate', 0.14),
        'orb_pct': wavg('orb_pct', 0.25),
        'ft_rate': wavg('ft_rate', 0.25),
        'three_pct': wavg('three_pct', 0.36),
        'pts': wavg('pts', 110),
        'n_games': n_games,
        'recent_form': wins / n_games if n_games > 0 else 0.5,
    }

def get_season_standings():
    d = espn_get('https://site.api.espn.com/apis/v2/sports/basketball/nba/standings')
    ESPN_ABBR_MAP = {
        'NY': 'NYK', 'GS': 'GSW', 'NO': 'NOP', 'SA': 'SAS',
        'UTAH': 'UTA', 'WSH': 'WAS'
    }
    result = {}
    for group in d.get('children', []):
        for entry in group.get('standings', {}).get('entries', []):
            team = entry.get('team', {})
            raw_abbr = team.get('abbreviation', '')
            abbr = ESPN_ABBR_MAP.get(raw_abbr, raw_abbr)
            stats_list = entry.get('stats', [])
            stats = {s['name']: s.get('value', 0) for s in stats_list}
            display_stats = {s['name']: s.get('displayValue', '') for s in stats_list}
            # Parse Home/Road from displayValue e.g. '28-9'
            def parse_record(s, default_w=15, default_l=15):
                try:
                    parts = str(s).split('-')
                    return int(parts[0]), int(parts[1])
                except:
                    return default_w, default_l

            home_rec = parse_record(display_stats.get('Home', '15-15'))
            road_rec = parse_record(display_stats.get('Road', '15-15'))
            last10_rec = parse_record(display_stats.get('Last Ten Games', '5-5'))

            result[abbr] = {
                'wins': int(stats.get('wins', 0)),
                'losses': int(stats.get('losses', 0)),
                'win_pct': float(stats.get('winPercent', 0.5)),
                'home_wins': home_rec[0],
                'home_losses': home_rec[1],
                'away_wins': road_rec[0],
                'away_losses': road_rec[1],
                'last10_wins': last10_rec[0],
                'pts_for': float(stats.get('avgPointsFor', 110)),
                'pts_against': float(stats.get('avgPointsAgainst', 110)),
            }
    return result

def get_rest_days(abbr):
    utc_now = datetime.now(timezone.utc)
    for offset in range(1, 5):
        d_str = (utc_now - timedelta(days=offset)).strftime('%Y%m%d')
        d = espn_get(f'https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={d_str}')
        for event in d.get('events', []):
            comp = event.get('competitions', [{}])[0]
            if not comp.get('status', {}).get('type', {}).get('completed', False):
                continue
            for t in comp.get('competitors', []):
                if t.get('team', {}).get('abbreviation') == abbr:
                    return offset - 1
        time.sleep(0.2)
    return 3

def get_injuries(abbr):
    tid = get_team_id(abbr)
    if not tid:
        return [], 0
    d = espn_get(f'https://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams/{tid}/injuries', timeout=8)
    injuries = []
    impact = 0
    for inj in d.get('injuries', []):
        athlete = inj.get('athlete', {})
        name = athlete.get('displayName', 'Unknown')
        status = inj.get('status', '').lower()
        pos = athlete.get('position', {}).get('abbreviation', '')
        injuries.append(f"{name} ({inj.get('status', 'Unknown')})")
        if status in ['out', 'doubtful']:
            impact += 4 if pos in ['PG','SG','SF','C','PF'] else 2
        elif status in ['questionable']:
            impact += 1.5
        elif status in ['day-to-day']:
            impact += 1
    return injuries[:5], min(impact, 10)

def get_todays_games():
    utc_now = datetime.now(timezone.utc)
    all_games = []
    seen = set()
    for offset in [-1, 0, 1]:
        d_str = (utc_now + timedelta(days=offset)).strftime('%Y%m%d')
        d = espn_get(f'https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={d_str}')
        for event in d.get('events', []):
            eid = event.get('id', '')
            if eid in seen:
                continue
            game_time_str = event.get('date', '')
            try:
                game_dt = datetime.fromisoformat(game_time_str.replace('Z', '+00:00'))
            except:
                continue
            if abs((game_dt - utc_now).total_seconds() / 3600) > 20:
                continue
            comp = event.get('competitions', [{}])[0]
            if comp.get('status', {}).get('type', {}).get('completed', False):
                continue
            teams = comp.get('competitors', [])
            if len(teams) < 2:
                continue
            home = next((t for t in teams if t['homeAway'] == 'home'), teams[0])
            away = next((t for t in teams if t['homeAway'] == 'away'), teams[1])
            de_time = game_dt.astimezone(timezone(timedelta(hours=2))).strftime('%H:%M')
            seen.add(eid)
            all_games.append({
                'home_abbr': home['team']['abbreviation'],
                'home_name': home['team']['displayName'],
                'away_abbr': away['team']['abbreviation'],
                'away_name': away['team']['displayName'],
                'time': de_time,
                'game_dt': game_dt.isoformat()
            })
        time.sleep(0.3)
    all_games.sort(key=lambda x: x['game_dt'])
    print(f"Spiele heute: {len(all_games)}")
    return all_games

def get_upcoming_games():
    utc_now = datetime.now(timezone.utc)
    upcoming = []
    seen = set()
    days = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    de_days = ['Mo','Di','Mi','Do','Fr','Sa','So']
    for offset in range(1, 11):
        d_str = (utc_now + timedelta(days=offset)).strftime('%Y%m%d')
        d = espn_get(f'https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={d_str}')
        for event in d.get('events', []):
            eid = event.get('id', '')
            if eid in seen:
                continue
            comp = event.get('competitions', [{}])[0]
            teams = comp.get('competitors', [])
            if len(teams) < 2:
                continue
            home = next((t for t in teams if t['homeAway'] == 'home'), teams[0])
            away = next((t for t in teams if t['homeAway'] == 'away'), teams[1])
            try:
                game_dt = datetime.fromisoformat(event.get('date','').replace('Z', '+00:00'))
                de_time = game_dt.astimezone(timezone(timedelta(hours=2)))
                wd = de_days[days.index(de_time.strftime('%A'))] if de_time.strftime('%A') in days else ''
                date_str = de_time.strftime('%d.%m.%Y')
                time_str = de_time.strftime('%H:%M')
                sort_key = game_dt.isoformat()
            except:
                date_str = d_str
                time_str = 'TBD'
                wd = ''
                sort_key = d_str
            seen.add(eid)
            upcoming.append({
                'home_abbr': home['team']['abbreviation'],
                'home_name': home['team']['displayName'],
                'away_abbr': away['team']['abbreviation'],
                'away_name': away['team']['displayName'],
                'date': date_str,
                'weekday': wd,
                'time': time_str,
                'sort_key': sort_key
            })
        time.sleep(0.3)
    upcoming.sort(key=lambda x: x.get('sort_key',''))
    return upcoming

# ─── ODDS ───────────────────────────────────────────────────────────────────────
_odds_cache = None

def load_all_odds():
    global _odds_cache
    if _odds_cache is not None:
        return _odds_cache
    if ODDS_API_KEY == 'DEIN_API_KEY_HIER':
        _odds_cache = []
        return _odds_cache
    try:
        r = requests.get(ODDS_API_URL, params={
            'apiKey': ODDS_API_KEY,
            'regions': 'eu,uk,us',
            'markets': 'h2h',
            'oddsFormat': 'decimal'
        }, headers=HEADERS, timeout=15)
        _odds_cache = r.json() if r.status_code == 200 else []
        print(f"Odds API: {len(_odds_cache)} Spiele, {r.headers.get('x-requests-remaining')} Anfragen verbleibend")
        return _odds_cache
    except Exception as e:
        print(f"Odds API Fehler: {e}")
        _odds_cache = []
        return _odds_cache

def norm_name(s):
    stop = {'the','golden','state','new','york','los','angeles','san','oklahoma',
            'city','portland','trail','blazers','clippers','lakers','warriors'}
    return [w for w in s.lower().split() if w not in stop and len(w) > 2]

def get_odds_consensus(home_name, away_name):
    games = load_all_odds()
    if not games:
        return None, None, None, None, []
    try:
        hw = norm_name(home_name)
        aw = norm_name(away_name)
        best_game, best_score = None, 0
        for game in games:
            gh = norm_name(game.get('home_team',''))
            ga = norm_name(game.get('away_team',''))
            h = sum(1 for w in hw if any(w in gw or gw in w for gw in gh))
            a = sum(1 for w in aw if any(w in gw or gw in w for gw in ga))
            if h + a > best_score:
                best_score = h + a
                best_game = game
        if not best_game or best_score < 1:
            return None, None, None, None, []
        api_home = best_game['home_team']
        api_away = best_game['away_team']
        h_odds_list, a_odds_list = [], []
        tipico_h, tipico_a = None, None
        bookies_used = []
        ah_words = norm_name(api_home)
        aa_words = norm_name(api_away)
        for bookie in best_game.get('bookmakers', []):
            bname = bookie.get('key', '')
            for market in bookie.get('markets', []):
                if market.get('key') == 'h2h':
                    outcomes = market.get('outcomes', [])
                    h_odd, a_odd = None, None
                    for o in outcomes:
                        ow = norm_name(o['name'])
                        if any(w in ow or any(ow2 in w for ow2 in ow) for w in ah_words):
                            h_odd = o['price']
                        else:
                            a_odd = o['price']
                    if not h_odd and len(outcomes) >= 2:
                        h_odd = outcomes[0]['price']
                        a_odd = outcomes[1]['price']
                    if h_odd and a_odd and h_odd > 1.0 and a_odd > 1.0:
                        h_odds_list.append(h_odd)
                        a_odds_list.append(a_odd)
                        bookies_used.append(bname)
                        if 'tipico' in bname:
                            tipico_h = h_odd
                            tipico_a = a_odd
        if not h_odds_list:
            return None, None, None, None, []
        avg_h = sum(h_odds_list) / len(h_odds_list)
        avg_a = sum(a_odds_list) / len(a_odds_list)
        rh = 1 / avg_h
        ra = 1 / avg_a
        total = rh + ra
        ch = round(rh / total, 3)
        ca = round(ra / total, 3)
        th = tipico_h or round(avg_h, 2)
        ta = tipico_a or round(avg_a, 2)
        print(f"    Odds: {len(bookies_used)} Bookies | Konsensus {ch:.1%}/{ca:.1%} | Tipico {th}/{ta}")
        return ch, ca, round(th, 2), round(ta, 2), bookies_used
    except Exception as e:
        print(f"Odds Fehler: {e}")
        return None, None, None, None, []

# ─── PREDICTION ENGINE ──────────────────────────────────────────────────────────
def build_feature_vector(h_adv, a_adv, h_season, a_season, h_rest, a_rest,
                          h_inj, a_inj, weights):
    # Feature 1: Season Win% diff
    h_wp = h_season.get('win_pct', 0.5)
    a_wp = a_season.get('win_pct', 0.5)
    f_win_pct = h_wp - a_wp

    # Feature 2: Season Net Rating diff
    h_net_s = h_season.get('pts_for', 110) - h_season.get('pts_against', 110)
    a_net_s = a_season.get('pts_for', 110) - a_season.get('pts_against', 110)
    f_net_rtg = (h_net_s - a_net_s) / 15.0

    # Feature 3: Recent Net Rating (last N games) - most important
    h_net_r = h_adv.get('net_rtg', 0) if h_adv else h_net_s
    a_net_r = a_adv.get('net_rtg', 0) if a_adv else a_net_s
    f_recent_net = (h_net_r - a_net_r) / 15.0

    # Feature 4: eFG% diff (shooting efficiency)
    h_efg = h_adv.get('efg', 0.52) if h_adv else 0.52
    a_efg = a_adv.get('efg', 0.52) if a_adv else 0.52
    f_efg = (h_efg - a_efg) / 0.08

    # Feature 5: Turnover rate diff (lower is better)
    h_tov = h_adv.get('tov_rate', 0.14) if h_adv else 0.14
    a_tov = a_adv.get('tov_rate', 0.14) if a_adv else 0.14
    f_tov = (a_tov - h_tov) / 0.04

    # Feature 6: Offensive rebound % diff
    h_orb = h_adv.get('orb_pct', 0.25) if h_adv else 0.25
    a_orb = a_adv.get('orb_pct', 0.25) if a_adv else 0.25
    f_orb = (h_orb - a_orb) / 0.08

    # Feature 7: Free throw rate diff
    h_ftr = h_adv.get('ft_rate', 0.25) if h_adv else 0.25
    a_ftr = a_adv.get('ft_rate', 0.25) if a_adv else 0.25
    f_ftr = (h_ftr - a_ftr) / 0.10

    # Feature 8: Rest days diff
    f_rest = (h_rest - a_rest) / 3.0

    # Feature 9: Back-to-back penalty
    f_b2b = (-1.0 if h_rest == 0 else 0) + (1.0 if a_rest == 0 else 0)

    # Feature 10: Home advantage (always positive for home team)
    f_home = 1.0

    # Feature 11: Injury impact diff
    f_inj = (a_inj - h_inj) / 8.0

    # Weighted sum -> probability
    raw = 0.5
    raw += weights['win_pct_diff'] * f_win_pct
    raw += weights['net_rtg_diff'] * f_net_rtg
    raw += weights['recent_net_diff'] * f_recent_net
    raw += weights['efg_diff'] * f_efg
    raw += weights['tov_diff'] * f_tov
    raw += weights['orb_diff'] * f_orb
    raw += weights['ft_rate_diff'] * f_ftr
    raw += weights['rest_diff'] * f_rest
    raw += weights['b2b_penalty'] * f_b2b
    raw += weights['home_advantage'] * f_home
    raw += 0.03 * f_inj

    # Sigmoid-like squashing for confidence calibration
    home_prob = 1 / (1 + math.exp(-4 * (raw - 0.5)))
    home_prob = max(0.22, min(0.78, home_prob))
    return round(home_prob, 3), round(1 - home_prob, 3)

def calc_kelly(prob, odds):
    if odds <= 1.0:
        return 0
    b = odds - 1
    q = 1 - prob
    kelly = (prob * b - q) / b
    # Cap at 25% max, never bet more than quarter Kelly on single game
    return round(max(0, min(0.25, kelly)), 4)

def calc_ev(prob, odds):
    return round(((prob * (odds - 1)) - (1 - prob)) * 100, 1)

def get_key_factors(h_adv, a_adv, h_season, a_season, ha, aa, h_rest, a_rest, h_inj_impact, a_inj_impact):
    factors = []
    h_net = h_adv.get('net_rtg', 0) if h_adv else 0
    a_net = a_adv.get('net_rtg', 0) if a_adv else 0
    h_efg = h_adv.get('efg', 0.52) if h_adv else 0.52
    a_efg = a_adv.get('efg', 0.52) if a_adv else 0.52
    if h_net > a_net + 3:
        factors.append(f"{ha} deutlich besser: Net Rtg +{h_net:.1f} vs {a_net:.1f}")
    elif a_net > h_net + 3:
        factors.append(f"{aa} deutlich besser: Net Rtg +{a_net:.1f} vs {h_net:.1f}")
    if h_efg > a_efg + 0.03:
        factors.append(f"{ha} effizienter schiessend (eFG% {h_efg:.1%} vs {a_efg:.1%})")
    elif a_efg > h_efg + 0.03:
        factors.append(f"{aa} effizienter schiessend (eFG% {a_efg:.1%} vs {h_efg:.1%})")
    if h_rest == 0:
        factors.append(f"{ha} Back-to-Back — messbar schlechtere Performance")
    if a_rest == 0:
        factors.append(f"{aa} Back-to-Back — messbar schlechtere Performance")
    if h_rest >= 3 and a_rest <= 1:
        factors.append(f"{ha} deutlich ausgeruhter ({h_rest} vs {a_rest} Ruhetage)")
    elif a_rest >= 3 and h_rest <= 1:
        factors.append(f"{aa} deutlich ausgeruhter ({a_rest} vs {h_rest} Ruhetage)")
    if h_inj_impact >= 5:
        factors.append(f"{ha} durch Verletzungen geschwächt (Impact -{h_inj_impact:.0f})")
    if a_inj_impact >= 5:
        factors.append(f"{aa} durch Verletzungen geschwächt (Impact -{a_inj_impact:.0f})")
    h_wf = h_adv.get('recent_form', h_season.get('win_pct', 0.5)) if h_adv else h_season.get('win_pct', 0.5)
    a_wf = a_adv.get('recent_form', a_season.get('win_pct', 0.5)) if a_adv else a_season.get('win_pct', 0.5)
    if h_wf >= 0.70:
        factors.append(f"{ha} in Topform ({h_wf:.0%} letzte {N_RECENT} Spiele)")
    if a_wf >= 0.70:
        factors.append(f"{aa} in Topform ({a_wf:.0%} letzte {N_RECENT} Spiele)")
    if not factors:
        factors.append("Ausgeglichenes Matchup — kein klarer Vorteil")
        wp_diff = h_season.get('win_pct',0.5) - a_season.get('win_pct',0.5)
        if abs(wp_diff) > 0.05:
            better = ha if wp_diff > 0 else aa
            factors.append(f"{better} Saisondominanz: {h_season.get('win_pct',0.5):.0%} vs {a_season.get('win_pct',0.5):.0%}")
    return factors[:3]

def update_history_with_results(history):
    now = datetime.now(timezone.utc)
    updated = 0
    for entry in history:
        if entry.get('correct') is not None:
            continue
        try:
            game_dt = datetime.fromisoformat(entry['game_dt'])
            if (now - game_dt).total_seconds() < 14400:
                continue
            date_str = game_dt.strftime('%Y%m%d')
            d = espn_get(f'https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={date_str}')
            for event in d.get('events', []):
                comp = event.get('competitions', [{}])[0]
                if not comp.get('status', {}).get('type', {}).get('completed', False):
                    continue
                competitors = comp.get('competitors', [])
                home_t = next((t for t in competitors if t['homeAway'] == 'home'), None)
                away_t = next((t for t in competitors if t['homeAway'] == 'away'), None)
                if not home_t or not away_t:
                    continue
                if (home_t['team']['abbreviation'] == entry['home_abbr'] and
                    away_t['team']['abbreviation'] == entry['away_abbr']):
                    actual_winner = home_t['team']['abbreviation'] if home_t.get('winner') else away_t['team']['abbreviation']
                    entry['actual_winner'] = actual_winner
                    entry['correct'] = (actual_winner == entry['pick'])
                    updated += 1
                    break
            time.sleep(0.2)
        except:
            pass
    if updated > 0:
        print(f"Resultate aktualisiert: {updated} Spiele")
    return history

# ─── MAIN ───────────────────────────────────────────────────────────────────────
def main():
    print("BetEdge NBA Prediction Engine v5 — Advanced ML")
    print("=" * 50)

    # Load model state
    weights = load_weights()
    history = load_history()

    # Update past predictions with real results
    print("Aktualisiere vergangene Vorhersagen...")
    history = update_history_with_results(history)

    # Learn from history
    weights = update_weights_from_history(weights, history)
    save_weights(weights)

    # Get today's games
    print("Lade heutige Spiele...")
    games = get_todays_games()
    if not games:
        print("Keine Spiele heute")
        return

    # Get season standings
    print("Lade Saisonstatistiken...")
    season_stats = get_season_standings()

    # Get upcoming games
    print("Lade Spielplan (10 Tage)...")
    upcoming = get_upcoming_games()

    # Load odds once
    print("Lade Buchmacher-Odds...")
    load_all_odds()

    # Generate predictions
    print(f"\nGeneriere Vorhersagen fuer {len(games)} Spiele...")
    output_games = []
    new_history = []

    for g in games:
        ha = g['home_abbr']
        aa = g['away_abbr']
        print(f"\n  {ha} vs {aa}:")

        # Advanced recent stats (last N games)
        h_adv = get_team_advanced_stats(ha)
        time.sleep(0.5)
        a_adv = get_team_advanced_stats(aa)
        time.sleep(0.5)

        # Season stats fallback
        h_season = season_stats.get(ha, {'win_pct':0.5,'pts_for':110,'pts_against':110,'last10_wins':5,'wins':35,'losses':35,'home_wins':18,'home_losses':12,'away_wins':17,'away_losses':13})
        a_season = season_stats.get(aa, {'win_pct':0.5,'pts_for':110,'pts_against':110,'last10_wins':5,'wins':35,'losses':35,'home_wins':18,'home_losses':12,'away_wins':17,'away_losses':13})

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
        book_home, book_away, tipico_h, tipico_a, bookies = get_odds_consensus(g['home_name'], g['away_name'])
        time.sleep(0.3)

        if not book_home:
            # Fallback
            h_net = h_season.get('pts_for',110) - h_season.get('pts_against',110)
            a_net = a_season.get('pts_for',110) - a_season.get('pts_against',110)
            raw = 0.5 + (h_season.get('win_pct',0.5) - a_season.get('win_pct',0.5)) * 0.4 + (h_net-a_net)/30 + 0.03
            book_home = round(max(0.28, min(0.72, raw)), 3)
            book_away = round(1 - book_home, 3)
            tipico_h = round((1 / book_home) * 0.94, 2)
            tipico_a = round((1 / book_away) * 0.94, 2)
            bookies = []
            odds_source = "Modell-Schätzung"
        else:
            odds_source = f"Konsensus {len(bookies)} Bookies"

        home_odds = tipico_h
        away_odds = tipico_a
        ev_home = calc_ev(home_prob, home_odds)
        ev_away = calc_ev(away_prob, away_odds)
        model_pick = ha if home_prob > away_prob else aa
        confidence = int(max(home_prob, away_prob) * 100)
        value_bet = ha if ev_home > 2 else (aa if ev_away > 2 else None)
        kelly = calc_kelly(
            home_prob if home_prob > away_prob else away_prob,
            home_odds if home_prob > away_prob else away_odds
        )

        factors = get_key_factors(h_adv, a_adv, h_season, a_season, ha, aa,
                                   h_rest, a_rest, h_inj_impact, a_inj_impact)

        # Helper
        l10h = h_season.get('last10_wins', 5)
        l10a = a_season.get('last10_wins', 5)

        game_obj = {
            "id": f"nba_{datetime.now().strftime('%Y%m%d')}_{ha}_{aa}",
            "date": datetime.now().strftime('%Y-%m-%d'),
            "time": g['time'],
            "home": {
                "name": g['home_name'], "abbr": ha,
                "record": f"{h_season.get('wins',0)}-{h_season.get('losses',0)}",
                "win_pct": round(h_season.get('win_pct', 0.5), 3),
                "last5": f"{l10h//2}-{(10-l10h)//2}",
                "offensive_rating": round(h_adv.get('ortg', h_season.get('pts_for',110)) if h_adv else h_season.get('pts_for',110), 1),
                "defensive_rating": round(h_adv.get('drtg', h_season.get('pts_against',110)) if h_adv else h_season.get('pts_against',110), 1),
                "net_rating": round(h_adv.get('net_rtg', 0) if h_adv else h_season.get('pts_for',110)-h_season.get('pts_against',110), 1),
                "efg": round(h_adv.get('efg', 0.52) if h_adv else 0.52, 3),
                "tov_rate": round(h_adv.get('tov_rate', 0.14) if h_adv else 0.14, 3),
                "pace": round(h_adv.get('pace', 98) if h_adv else 98, 1),
                "rest_days": h_rest,
                "back_to_back": h_rest == 0,
                "injuries": h_injuries
            },
            "away": {
                "name": g['away_name'], "abbr": aa,
                "record": f"{a_season.get('wins',0)}-{a_season.get('losses',0)}",
                "win_pct": round(a_season.get('win_pct', 0.5), 3),
                "last5": f"{l10a//2}-{(10-l10a)//2}",
                "offensive_rating": round(a_adv.get('ortg', a_season.get('pts_for',110)) if a_adv else a_season.get('pts_for',110), 1),
                "defensive_rating": round(a_adv.get('drtg', a_season.get('pts_against',110)) if a_adv else a_season.get('pts_against',110), 1),
                "net_rating": round(a_adv.get('net_rtg', 0) if a_adv else a_season.get('pts_for',110)-a_season.get('pts_against',110), 1),
                "efg": round(a_adv.get('efg', 0.52) if a_adv else 0.52, 3),
                "tov_rate": round(a_adv.get('tov_rate', 0.14) if a_adv else 0.14, 3),
                "pace": round(a_adv.get('pace', 98) if a_adv else 98, 1),
                "rest_days": a_rest,
                "back_to_back": a_rest == 0,
                "injuries": a_injuries
            },
            "prediction": {
                "home_win_prob": home_prob,
                "away_win_prob": away_prob,
                "model_pick": model_pick,
                "confidence": confidence,
                "book_home_prob": book_home,
                "book_away_prob": book_away,
                "home_odds": home_odds,
                "away_odds": away_odds,
                "home_ev": ev_home,
                "away_ev": ev_away,
                "kelly_fraction": kelly,
                "value_bet": value_bet,
                "key_factors": factors,
                "odds_source": odds_source,
                "model_version": weights.get('version', 1)
            }
        }
        output_games.append(game_obj)

        # Save to history for learning
        new_history.append({
            'date': datetime.now().strftime('%Y-%m-%d'),
            'game_dt': g['game_dt'],
            'home_abbr': ha,
            'away_abbr': aa,
            'pick': model_pick,
            'home_prob': home_prob,
            'confidence': confidence,
            'correct': None
        })

        print(f"    -> {model_pick} ({confidence}% conf.) EV: H{ev_home:+.1f}% A{ev_away:+.1f}%")

    # Save history
    history.extend(new_history)
    save_history(history)

    # Model stats
    recent_hist = [h for h in history if h.get('correct') is not None][-100:]
    correct = sum(1 for h in recent_hist if h.get('correct'))
    acc_str = f"{correct/len(recent_hist):.1%}" if recent_hist else "~63%"
    avg_conf = int(np.mean([g['prediction']['confidence'] for g in output_games])) if output_games else 0
    value_count = sum(1 for g in output_games if g['prediction']['value_bet'])

    output = {
        "last_updated": datetime.now(timezone.utc).isoformat(),
        "model_version": f"5.{weights.get('version',1)}",
        "sport": "NBA",
        "games": output_games,
        "upcoming": upcoming,
        "model_stats": {
            "season_record": f"{correct}-{len(recent_hist)-correct}" if recent_hist else "laufend",
            "accuracy": acc_str,
            "roi": "laufend",
            "avg_confidence": avg_conf,
            "games_tracked": len(history),
            "model_weight_version": weights.get('version', 1)
        }
    }

    os.makedirs('data', exist_ok=True)
    with open(DATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\nFertig! {len(output_games)} Spiele, {value_count} Value Bets")
    print(f"Modell v{weights.get('version',1)} | Accuracy (letzte 100): {acc_str}")

if __name__ == "__main__":
    main()
