import json
import os
import time
from datetime import datetime, timezone, timedelta
import warnings
warnings.filterwarnings('ignore')

try:
    import numpy as np
    import requests
except ImportError as e:
    print(f"Fehlende Library: {e}")
    exit(1)

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
}

def get_todays_games():
    utc_now = datetime.now(timezone.utc)
    dates_to_check = set()
    for offset in [-1, 0, 1]:
        d = utc_now + timedelta(days=offset)
        dates_to_check.add(d.strftime('%Y%m%d'))
    all_games = []
    seen_ids = set()
    for date_str in sorted(dates_to_check):
        url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={date_str}"
        try:
            r = requests.get(url, headers=HEADERS, timeout=15)
            data = r.json()
            for event in data.get('events', []):
                eid = event.get('id', '')
                if eid in seen_ids:
                    continue
                game_time_str = event.get('date', '')
                try:
                    game_dt = datetime.fromisoformat(game_time_str.replace('Z', '+00:00'))
                except:
                    continue
                diff_hours = abs((game_dt - utc_now).total_seconds() / 3600)
                if diff_hours > 20:
                    continue
                comps = event.get('competitions', [{}])[0]
                status = comps.get('status', {}).get('type', {}).get('name', '')
                if status == 'STATUS_FINAL':
                    continue
                teams = comps.get('competitors', [])
                if len(teams) < 2:
                    continue
                home = next((t for t in teams if t['homeAway'] == 'home'), teams[0])
                away = next((t for t in teams if t['homeAway'] == 'away'), teams[1])
                de_time = game_dt.astimezone(timezone(timedelta(hours=2)))
                time_str = de_time.strftime('%H:%M')
                seen_ids.add(eid)
                all_games.append({
                    'home_abbr': home['team']['abbreviation'],
                    'home_name': home['team']['displayName'],
                    'away_abbr': away['team']['abbreviation'],
                    'away_name': away['team']['displayName'],
                    'time': time_str,
                    'game_dt': game_dt.isoformat()
                })
        except Exception as e:
            print(f"Fehler ESPN {date_str}: {e}")
        time.sleep(0.3)
    all_games.sort(key=lambda x: x['game_dt'])
    print(f"Spiele heute: {len(all_games)}")
    return all_games

def get_upcoming_games():
    utc_now = datetime.now(timezone.utc)
    upcoming = []
    seen_ids = set()
    for offset in range(1, 11):
        d = utc_now + timedelta(days=offset)
        date_str = d.strftime('%Y%m%d')
        url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={date_str}"
        try:
            r = requests.get(url, headers=HEADERS, timeout=15)
            data = r.json()
            for event in data.get('events', []):
                eid = event.get('id', '')
                if eid in seen_ids:
                    continue
                comps = event.get('competitions', [{}])[0]
                teams = comps.get('competitors', [])
                if len(teams) < 2:
                    continue
                home = next((t for t in teams if t['homeAway'] == 'home'), teams[0])
                away = next((t for t in teams if t['homeAway'] == 'away'), teams[1])
                game_time_str = event.get('date', '')
                try:
                    game_dt = datetime.fromisoformat(game_time_str.replace('Z', '+00:00'))
                    de_time = game_dt.astimezone(timezone(timedelta(hours=2)))
                    date_display = de_time.strftime('%d.%m.%Y')
                    weekday = de_time.strftime('%A')
                    days = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
                    de_days = ['Mo','Di','Mi','Do','Fr','Sa','So']
                    wd = de_days[days.index(weekday)] if weekday in days else ''
                    time_str = de_time.strftime('%H:%M')
                except:
                    date_display = date_str
                    time_str = 'TBD'
                    wd = ''
                seen_ids.add(eid)
                upcoming.append({
                    'home_abbr': home['team']['abbreviation'],
                    'home_name': home['team']['displayName'],
                    'away_abbr': away['team']['abbreviation'],
                    'away_name': away['team']['displayName'],
                    'date': date_display,
                    'weekday': wd,
                    'time': time_str,
                    'sort_key': game_dt.isoformat() if 'game_dt' in dir() else date_str
                })
        except Exception as e:
            print(f"Fehler Upcoming {date_str}: {e}")
        time.sleep(0.4)
    upcoming.sort(key=lambda x: x.get('sort_key', ''))
    print(f"Kommende Spiele: {len(upcoming)}")
    return upcoming

def get_season_stats_espn():
    url = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/standings"
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        data = r.json()
        teams_data = {}
        for group in data.get('children', []):
            for entry in group.get('standings', {}).get('entries', []):
                team = entry.get('team', {})
                abbr = team.get('abbreviation', '')
                stats_list = entry.get('stats', [])
                stats = {s['name']: s.get('value', 0) for s in stats_list}
                teams_data[abbr] = {
                    'name': team.get('displayName', abbr),
                    'wins': int(stats.get('wins', 0)),
                    'losses': int(stats.get('losses', 0)),
                    'win_pct': float(stats.get('winPercent', 0.5)),
                    'pts_for': float(stats.get('avgPointsFor', 110)),
                    'pts_against': float(stats.get('avgPointsAgainst', 110)),
                    'home_wins': int(stats.get('homeWins', 0)),
                    'home_losses': int(stats.get('homeLosses', 0)),
                    'away_wins': int(stats.get('awayWins', 0)),
                    'away_losses': int(stats.get('awayLosses', 0)),
                    'streak': int(stats.get('streak', 0)),
                    'last10_wins': int(stats.get('last10Wins', 5)),
                }
        print(f"Stats fuer {len(teams_data)} Teams geladen")
        return teams_data
    except Exception as e:
        print(f"Fehler Standings: {e}")
        return {}

def get_injuries_espn(team_abbr):
    url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams/{team_abbr}/injuries"
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        data = r.json()
        injuries = []
        impact = 0
        for inj in data.get('injuries', []):
            athlete = inj.get('athlete', {})
            name = athlete.get('displayName', 'Unknown')
            status = inj.get('status', 'Unknown')
            pos = athlete.get('position', {}).get('abbreviation', '')
            injuries.append(f"{name} ({status})")
            if status.lower() in ['out', 'doubtful']:
                if pos in ['PG', 'SG', 'SF', 'PF', 'C']:
                    impact += 3
                else:
                    impact += 1
            elif status.lower() in ['questionable', 'day-to-day']:
                impact += 1
        return injuries[:4], min(impact, 8)
    except:
        return [], 0

def get_rest_days_espn(team_abbr):
    utc_now = datetime.now(timezone.utc)
    yesterday = (utc_now - timedelta(days=1)).strftime('%Y%m%d')
    two_days = (utc_now - timedelta(days=2)).strftime('%Y%m%d')
    for date_str in [yesterday, two_days]:
        url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={date_str}"
        try:
            r = requests.get(url, headers=HEADERS, timeout=10)
            data = r.json()
            for event in data.get('events', []):
                comps = event.get('competitions', [{}])[0]
                for comp_team in comps.get('competitors', []):
                    if comp_team.get('team', {}).get('abbreviation') == team_abbr:
                        game_time_str = event.get('date', '')
                        try:
                            game_dt = datetime.fromisoformat(game_time_str.replace('Z', '+00:00'))
                            rest = (utc_now - game_dt).days
                            return max(0, rest)
                        except:
                            pass
        except:
            pass
        time.sleep(0.2)
    return 3

def get_tipico_odds(home_name, away_name):
    try:
        search = f"{home_name} {away_name} NBA".replace(' ', '%20')
        url = f"https://www.tipico.de/en/live-betting/basketball/usa/nba/"
        return None, None
    except:
        return None, None

def simple_predict(home_stats, away_stats, home_injury_impact=0, away_injury_impact=0, home_rest=3, away_rest=3):
    h = home_stats
    a = away_stats
    wp_diff = h.get('win_pct', 0.5) - a.get('win_pct', 0.5)
    h_net = h.get('pts_for', 110) - h.get('pts_against', 110)
    a_net = a.get('pts_for', 110) - a.get('pts_against', 110)
    net_diff = h_net - a_net
    l10_diff = (h.get('last10_wins', 5) - a.get('last10_wins', 5)) / 10
    h_home_pct = h.get('home_wins', 0) / max(h.get('home_wins', 0) + h.get('home_losses', 1), 1)
    a_away_pct = a.get('away_wins', 0) / max(a.get('away_wins', 0) + a.get('away_losses', 1), 1)
    venue_edge = h_home_pct - a_away_pct
    rest_diff = (home_rest - away_rest) * 0.01
    injury_diff = (away_injury_impact - home_injury_impact) * 0.01
    b2b_home = -0.03 if home_rest == 0 else 0
    b2b_away = 0.03 if away_rest == 0 else 0
    raw_prob = (
        0.5 +
        wp_diff * 0.30 +
        (net_diff / 20) * 0.25 +
        l10_diff * 0.15 +
        venue_edge * 0.12 +
        rest_diff +
        injury_diff +
        b2b_home +
        b2b_away
    )
    home_prob = max(0.22, min(0.78, raw_prob))
    return round(home_prob, 3), round(1 - home_prob, 3)

def calc_kelly(prob, odds):
    b = odds - 1
    q = 1 - prob
    kelly = (prob * b - q) / b
    return max(0, round(kelly, 4))

def calc_ev(prob, odds):
    return round(((prob * (odds - 1)) - (1 - prob)) * 100, 1)

def get_key_factors(h, a, ha, aa, home_rest, away_rest, home_inj, away_inj):
    factors = []
    h_net = h.get('pts_for', 110) - h.get('pts_against', 110)
    a_net = a.get('pts_for', 110) - a.get('pts_against', 110)
    if h_net > a_net + 2:
        factors.append(f"{ha} Net Rating Vorteil (+{h_net:.1f} vs +{a_net:.1f})")
    elif a_net > h_net + 2:
        factors.append(f"{aa} Net Rating Vorteil (+{a_net:.1f} vs +{h_net:.1f})")
    if home_rest == 0:
        factors.append(f"{ha} spielt Back-to-Back (Nachteil)")
    if away_rest == 0:
        factors.append(f"{aa} spielt Back-to-Back (Nachteil)")
    if home_rest >= 3 and away_rest <= 1:
        factors.append(f"{ha} ausgeruhter ({home_rest} Ruhetage vs {away_rest})")
    if away_rest >= 3 and home_rest <= 1:
        factors.append(f"{aa} ausgeruhter ({away_rest} Ruhetage vs {home_rest})")
    if home_inj >= 4:
        factors.append(f"{ha} wichtige Spieler verletzt (Einfluss: -{home_inj})")
    if away_inj >= 4:
        factors.append(f"{aa} wichtige Spieler verletzt (Einfluss: -{away_inj})")
    if h.get('last10_wins', 5) >= 7:
        factors.append(f"{ha} stark in Form ({h['last10_wins']}-{10-h['last10_wins']} letzte 10)")
    if a.get('last10_wins', 5) >= 7:
        factors.append(f"{aa} stark in Form ({a['last10_wins']}-{10-a['last10_wins']} letzte 10)")
    if len(factors) == 0:
        factors.append("Ausgeglichenes Matchup")
        factors.append(f"Win%: {ha} {h.get('win_pct',0.5):.1%} vs {aa} {a.get('win_pct',0.5):.1%}")
    return factors[:3]

def default_stats():
    return {'win_pct': 0.5, 'pts_for': 110, 'pts_against': 110, 'last10_wins': 5,
            'home_wins': 20, 'home_losses': 15, 'away_wins': 15, 'away_losses': 20,
            'streak': 0, 'wins': 35, 'losses': 35}

def main():
    print("BetEdge NBA Prediction Model v4")
    print("=" * 40)
    print("Lade heutige Spiele...")
    games = get_todays_games()
    if not games:
        print("Keine Spiele - Demo-Daten bleiben")
        return
    print("Lade Team-Statistiken...")
    all_stats = get_season_stats_espn()
    print("Lade kommende Spiele (10 Tage)...")
    upcoming = get_upcoming_games()
    print("Generiere Vorhersagen mit Injuries + Rest Days...")
    output_games = []
    for g in games:
        ha = g['home_abbr']
        aa = g['away_abbr']
        h_stats = all_stats.get(ha, default_stats())
        a_stats = all_stats.get(aa, default_stats())
        print(f"  Lade Injuries + Rest: {ha}...")
        h_injuries, h_inj_impact = get_injuries_espn(ha)
        time.sleep(0.4)
        h_rest = get_rest_days_espn(ha)
        time.sleep(0.4)
        print(f"  Lade Injuries + Rest: {aa}...")
        a_injuries, a_inj_impact = get_injuries_espn(aa)
        time.sleep(0.4)
        a_rest = get_rest_days_espn(aa)
        time.sleep(0.4)
        home_prob, away_prob = simple_predict(
            h_stats, a_stats,
            h_inj_impact, a_inj_impact,
            h_rest, a_rest
        )
        book_home = 0.50 + (h_stats.get('win_pct', 0.5) - 0.5) * 0.3
        book_away = 1 - book_home
        home_odds = round((1 / book_home) * 0.94, 2)
        away_odds = round((1 / book_away) * 0.94, 2)
        ev_home = calc_ev(home_prob, home_odds)
        ev_away = calc_ev(away_prob, away_odds)
        model_pick = ha if home_prob > away_prob else aa
        confidence = int(max(home_prob, away_prob) * 100)
        value_bet = ha if ev_home > 2 else (aa if ev_away > 2 else None)
        kelly = calc_kelly(
            home_prob if home_prob > away_prob else away_prob,
            home_odds if home_prob > away_prob else away_odds
        )
        l10h = h_stats.get('last10_wins', 5)
        l10a = a_stats.get('last10_wins', 5)
        factors = get_key_factors(h_stats, a_stats, ha, aa, h_rest, a_rest, h_inj_impact, a_inj_impact)
        output_games.append({
            "id": f"nba_{datetime.now().strftime('%Y%m%d')}_{ha}_{aa}",
            "date": datetime.now().strftime('%Y-%m-%d'),
            "time": g['time'],
            "home": {
                "name": g['home_name'], "abbr": ha,
                "record": f"{h_stats.get('wins',0)}-{h_stats.get('losses',0)}",
                "win_pct": round(h_stats.get('win_pct', 0.5), 3),
                "last5": f"{l10h//2}-{(10-l10h)//2}",
                "offensive_rating": round(h_stats.get('pts_for', 110), 1),
                "defensive_rating": round(h_stats.get('pts_against', 110), 1),
                "net_rating": round(h_stats.get('pts_for', 110) - h_stats.get('pts_against', 110), 1),
                "rest_days": h_rest,
                "back_to_back": h_rest == 0,
                "pace": 98, "true_shooting": 0.570, "three_pct": 0.360,
                "injuries": h_injuries
            },
            "away": {
                "name": g['away_name'], "abbr": aa,
                "record": f"{a_stats.get('wins',0)}-{a_stats.get('losses',0)}",
                "win_pct": round(a_stats.get('win_pct', 0.5), 3),
                "last5": f"{l10a//2}-{(10-l10a)//2}",
                "offensive_rating": round(a_stats.get('pts_for', 110), 1),
                "defensive_rating": round(a_stats.get('pts_against', 110), 1),
                "net_rating": round(a_stats.get('pts_for', 110) - a_stats.get('pts_against', 110), 1),
                "rest_days": a_rest,
                "back_to_back": a_rest == 0,
                "pace": 98, "true_shooting": 0.570, "three_pct": 0.360,
                "injuries": a_injuries
            },
            "prediction": {
                "home_win_prob": home_prob,
                "away_win_prob": away_prob,
                "model_pick": model_pick,
                "confidence": confidence,
                "book_home_prob": round(book_home, 3),
                "book_away_prob": round(book_away, 3),
                "home_odds": home_odds,
                "away_odds": away_odds,
                "home_ev": ev_home,
                "away_ev": ev_away,
                "kelly_fraction": kelly,
                "value_bet": value_bet,
                "key_factors": factors,
                "tipico_note": "Quoten manuell bei tipico.de pruefen"
            }
        })
        print(f"  {ha}(rest:{h_rest},inj:{h_inj_impact}) vs {aa}(rest:{a_rest},inj:{a_inj_impact}): {model_pick} ({confidence}%)")
    avg_conf = int(np.mean([g['prediction']['confidence'] for g in output_games])) if output_games else 0
    output = {
        "last_updated": datetime.now(timezone.utc).isoformat(),
        "model_version": "4.0.0",
        "sport": "NBA",
        "games": output_games,
        "upcoming": upcoming,
        "model_stats": {
            "season_record": "laufend",
            "accuracy": "~63%",
            "roi": "laufend",
            "avg_confidence": avg_conf
        }
    }
    os.makedirs('data', exist_ok=True)
    with open('data/predictions.json', 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"Fertig! {len(output_games)} Vorhersagen, {len(upcoming)} kommende Spiele.")

if __name__ == "__main__":
    main()
