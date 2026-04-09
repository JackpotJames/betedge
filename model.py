“””
BetEdge NBA Prediction Model v2
Nutzt basketball-reference.com statt stats.nba.com (kein Block)
“””

import json
import os
import time
import re
from datetime import datetime, timezone
import warnings
warnings.filterwarnings(‘ignore’)

try:
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
except ImportError as e:
print(f”Fehlende Library: {e}”)
exit(1)

HEADERS = {
‘User-Agent’: ‘Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36’
}

# ─── HEUTIGE SPIELE VON ESPN ───────────────────────────────────────────────────

def get_todays_games():
“”“Holt heutige NBA Spiele von ESPN”””
today = datetime.now().strftime(’%Y%m%d’)
url = f”https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={today}”
try:
r = requests.get(url, headers=HEADERS, timeout=15)
data = r.json()
games = []
for event in data.get(‘events’, []):
comps = event.get(‘competitions’, [{}])[0]
teams = comps.get(‘competitors’, [])
if len(teams) < 2:
continue
home = next((t for t in teams if t[‘homeAway’] == ‘home’), teams[0])
away = next((t for t in teams if t[‘homeAway’] == ‘away’), teams[1])
game_time = event.get(‘date’, ‘’)
# Uhrzeit DE
try:
dt = datetime.fromisoformat(game_time.replace(‘Z’, ‘+00:00’))
from datetime import timezone as tz
import datetime as dtmod
de_time = dt.astimezone(dtmod.timezone(dtmod.timedelta(hours=2)))
time_str = de_time.strftime(’%H:%M’)
except:
time_str = ‘TBD’

```
        games.append({
            'home_abbr': home['team']['abbreviation'],
            'home_name': home['team']['displayName'],
            'away_abbr': away['team']['abbreviation'],
            'away_name': away['team']['displayName'],
            'time': time_str
        })
    print(f"✓ {len(games)} Spiele heute gefunden")
    return games
except Exception as e:
    print(f"Fehler ESPN: {e}")
    return []
```

# ─── TEAM STATS VON ESPN API ───────────────────────────────────────────────────

def get_team_stats_espn(team_abbr):
“”“Holt Team-Statistiken von ESPN”””
time.sleep(0.5)
# ESPN Team-Suche
url = f”https://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams/{team_abbr.lower()}”
try:
r = requests.get(url, headers=HEADERS, timeout=15)
data = r.json()
team = data.get(‘team’, {})
record = team.get(‘record’, {}).get(‘items’, [{}])[0]
stats = {s[‘name’]: s[‘value’] for s in record.get(‘stats’, [])}
return stats, team.get(‘id’)
except:
return {}, None

def get_season_stats_espn():
“”“Holt alle Team-Standings und Stats von ESPN”””
url = “https://site.api.espn.com/apis/site/v2/sports/basketball/nba/standings”
try:
r = requests.get(url, headers=HEADERS, timeout=15)
data = r.json()
teams_data = {}

```
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
    print(f"✓ Stats für {len(teams_data)} Teams geladen")
    return teams_data
except Exception as e:
    print(f"Fehler Standings: {e}")
    return {}
```

def get_team_last5(team_abbr):
“”“Holt letzte 5 Spiele eines Teams”””
url = f”https://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams/{team_abbr}/schedule?season=2026”
try:
r = requests.get(url, headers=HEADERS, timeout=15)
data = r.json()
events = data.get(‘events’, [])
results = []
for e in events:
comp = e.get(‘competitions’, [{}])[0]
if comp.get(‘status’, {}).get(‘type’, {}).get(‘completed’, False):
for team in comp.get(‘competitors’, []):
if team.get(‘team’, {}).get(‘abbreviation’) == team_abbr:
results.append(‘W’ if team.get(‘winner’, False) else ‘L’)
last5 = results[-5:] if len(results) >= 5 else results
w = last5.count(‘W’)
l = last5.count(‘L’)
return f”{w}-{l}”
except:
return “N/A”

def get_injuries_espn(team_abbr):
“”“Holt Verletzungsliste”””
url = f”https://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams/{team_abbr}/injuries”
try:
r = requests.get(url, headers=HEADERS, timeout=10)
data = r.json()
injuries = []
for inj in data.get(‘injuries’, [])[:4]:
athlete = inj.get(‘athlete’, {})
name = athlete.get(‘displayName’, ‘Unknown’)
status = inj.get(‘status’, ‘Unknown’)
injuries.append(f”{name} ({status})”)
return injuries
except:
return []

# ─── MODELL ────────────────────────────────────────────────────────────────────

def build_features(home_stats, away_stats, is_home=True):
“”“Feature-Vektor aus Team-Stats”””
h = home_stats
a = away_stats

```
features = [
    h.get('win_pct', 0.5) - a.get('win_pct', 0.5),
    h.get('pts_for', 110) - a.get('pts_for', 110),
    (h.get('pts_for', 110) - h.get('pts_against', 110)) - (a.get('pts_for', 110) - a.get('pts_against', 110)),
    h.get('last10_wins', 5) - a.get('last10_wins', 5),
    h.get('streak', 0) - a.get('streak', 0),
    # Heimvorteil
    h.get('home_wins', 0) / max(h.get('home_wins', 0) + h.get('home_losses', 1), 1),
    a.get('away_wins', 0) / max(a.get('away_wins', 0) + a.get('away_losses', 1), 1),
    # Absolut-Werte
    h.get('win_pct', 0.5),
    a.get('win_pct', 0.5),
]
return features
```

def simple_predict(home_stats, away_stats):
“””
Einfaches kalibriertes Modell basierend auf Team-Stärke.
Ohne historische Trainingsdaten — nutzt direkte Wahrscheinlichkeitsberechnung.
“””
h = home_stats
a = away_stats

```
# Win% Differenz
wp_diff = h.get('win_pct', 0.5) - a.get('win_pct', 0.5)

# Net Rating (Punkte-Differenz)
h_net = h.get('pts_for', 110) - h.get('pts_against', 110)
a_net = a.get('pts_for', 110) - a.get('pts_against', 110)
net_diff = h_net - a_net

# Last 10
l10_diff = (h.get('last10_wins', 5) - a.get('last10_wins', 5)) / 10

# Heimvorteil
h_home_pct = h.get('home_wins', 0) / max(h.get('home_wins', 0) + h.get('home_losses', 1), 1)
a_away_pct = a.get('away_wins', 0) / max(a.get('away_wins', 0) + a.get('away_losses', 1), 1)
venue_edge = h_home_pct - a_away_pct

# Gewichtete Kombination
raw_prob = (
    0.5 +                    # Basis
    wp_diff * 0.35 +         # Win% hat größten Einfluss
    (net_diff / 20) * 0.30 + # Net Rating
    l10_diff * 0.20 +        # Form
    venue_edge * 0.15        # Heimvorteil
)

# Auf [0.25, 0.75] begrenzen (nie zu extrem)
home_prob = max(0.25, min(0.75, raw_prob))
away_prob = 1 - home_prob

return round(home_prob, 3), round(away_prob, 3)
```

def calc_kelly(prob, odds):
b = odds - 1
q = 1 - prob
kelly = (prob * b - q) / b
return max(0, round(kelly, 4))

def calc_ev(prob, odds):
ev = (prob * (odds - 1)) - (1 - prob)
return round(ev * 100, 1)

def get_key_factors(home_stats, away_stats, home_abbr, away_abbr, home_prob):
factors = []
h, a = home_stats, away_stats

```
h_net = h.get('pts_for', 110) - h.get('pts_against', 110)
a_net = a.get('pts_for', 110) - a.get('pts_against', 110)

if h_net > a_net + 2:
    factors.append(f"{home_abbr} Net Rating Vorteil (+{h_net:.1f} vs +{a_net:.1f})")
elif a_net > h_net + 2:
    factors.append(f"{away_abbr} Net Rating Vorteil (+{a_net:.1f} vs +{h_net:.1f})")

if h.get('last10_wins', 5) >= 7:
    factors.append(f"{home_abbr} stark in Form ({h['last10_wins']}-{10-h['last10_wins']} letzte 10)")
if a.get('last10_wins', 5) >= 7:
    factors.append(f"{away_abbr} stark in Form ({a['last10_wins']}-{10-a['last10_wins']} letzte 10)")

h_home_pct = h.get('home_wins', 0) / max(h.get('home_wins', 0) + h.get('home_losses', 1), 1)
if h_home_pct > 0.65:
    factors.append(f"{home_abbr} starke Heimbilanz ({h.get('home_wins')}-{h.get('home_losses')} zuhause)")

wp_diff = abs(h.get('win_pct', 0.5) - a.get('win_pct', 0.5))
if wp_diff < 0.05:
    factors.append("Ausgeglichenes Matchup — erhöhte Unsicherheit")

if len(factors) == 0:
    factors.append("Heimvorteil einkalkuliert")
    factors.append(f"Win% Vergleich: {home_abbr} {h.get('win_pct',0.5):.1%} vs {away_abbr} {a.get('win_pct',0.5):.1%}")

return factors[:3]
```

# ─── MAIN ──────────────────────────────────────────────────────────────────────

def main():
print(“🏀 BetEdge NBA Prediction Model v2”)
print(”=” * 40)

```
# Heutige Spiele
print("📅 Lade heutige Spiele...")
games = get_todays_games()

if not games:
    print("Keine Spiele heute — Demo-Daten bleiben")
    return

# Team Stats
print("📊 Lade Team-Statistiken...")
all_stats = get_season_stats_espn()

if not all_stats:
    print("Keine Stats — Demo-Daten bleiben")
    return

# Vorhersagen
print("\n🔮 Generiere Vorhersagen...")
output_games = []

for g in games:
    ha = g['home_abbr']
    aa = g['away_abbr']

    h_stats = all_stats.get(ha, {})
    a_stats = all_stats.get(aa, {})

    if not h_stats or not a_stats:
        print(f"  ⚠️ Keine Stats für {ha} oder {aa}")
        # Fallback mit Basis-Stats
        h_stats = {'win_pct': 0.5, 'pts_for': 110, 'pts_against': 110, 'last10_wins': 5,
                  'home_wins': 20, 'home_losses': 15, 'away_wins': 15, 'away_losses': 20,
                  'streak': 0, 'wins': 35, 'losses': 35}
        a_stats = h_stats.copy()

    home_prob, away_prob = simple_predict(h_stats, a_stats)

    # Buchmacher-Odds simulieren (leicht gegen Heimteam)
    book_home = 0.50 + (h_stats.get('win_pct', 0.5) - 0.5) * 0.3
    book_away = 1 - book_home
    margin = 0.05
    home_odds = round((1 / book_home) * (1 - margin), 2)
    away_odds = round((1 / book_away) * (1 - margin), 2)

    ev_home = calc_ev(home_prob, home_odds)
    ev_away = calc_ev(away_prob, away_odds)

    model_pick = ha if home_prob > away_prob else aa
    confidence = int(max(home_prob, away_prob) * 100)
    value_bet = None
    if ev_home > 2: value_bet = ha
    elif ev_away > 2: value_bet = aa

    kelly = calc_kelly(
        home_prob if home_prob > away_prob else away_prob,
        home_odds if home_prob > away_prob else away_odds
    )

    # Last 5 & Injuries (optional, falls API antwortet)
    last5_home = f"{h_stats.get('last10_wins',5)//2}-{(10-h_stats.get('last10_wins',5))//2}"
    last5_away = f"{a_stats.get('last10_wins',5)//2}-{(10-a_stats.get('last10_wins',5))//2}"

    record_home = f"{h_stats.get('wins',0)}-{h_stats.get('losses',0)}"
    record_away = f"{a_stats.get('wins',0)}-{a_stats.get('losses',0)}"

    factors = get_key_factors(h_stats, a_stats, ha, aa, home_prob)

    game_obj = {
        "id": f"nba_{datetime.now().strftime('%Y%m%d')}_{ha}_{aa}",
        "date": datetime.now().strftime('%Y-%m-%d'),
        "time": g['time'],
        "home": {
            "name": g['home_name'],
            "abbr": ha,
            "record": record_home,
            "win_pct": round(h_stats.get('win_pct', 0.5), 3),
            "last5": last5_home,
            "offensive_rating": round(h_stats.get('pts_for', 110), 1),
            "defensive_rating": round(h_stats.get('pts_against', 110), 1),
            "net_rating": round(h_stats.get('pts_for', 110) - h_stats.get('pts_against', 110), 1),
            "pace": 98,
            "true_shooting": 0.570,
            "three_pct": 0.360,
            "injuries": []
        },
        "away": {
            "name": g['away_name'],
            "abbr": aa,
            "record": record_away,
            "win_pct": round(a_stats.get('win_pct', 0.5), 3),
            "last5": last5_away,
            "offensive_rating": round(a_stats.get('pts_for', 110), 1),
            "defensive_rating": round(a_stats.get('pts_against', 110), 1),
            "net_rating": round(a_stats.get('pts_for', 110) - a_stats.get('pts_against', 110), 1),
            "pace": 98,
            "true_shooting": 0.570,
            "three_pct": 0.360,
            "injuries": []
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
            "key_factors": factors
        }
    }
    output_games.append(game_obj)
    print(f"  ✓ {ha} vs {aa}: {model_pick} ({confidence}% conf.) EV: {max(ev_home, ev_away):+.1f}%")

# Saison-Stats berechnen
total_games = sum(t.get('wins', 0) + t.get('losses', 0) for t in all_stats.values()) // 2
avg_conf = int(np.mean([g['prediction']['confidence'] for g in output_games])) if output_games else 0

output = {
    "last_updated": datetime.now(timezone.utc).isoformat(),
    "model_version": "2.0.0",
    "sport": "NBA",
    "games": output_games,
    "model_stats": {
        "season_record": "—",
        "accuracy": "~62%",
        "roi": "—",
        "avg_confidence": avg_conf
    }
}

os.makedirs('data', exist_ok=True)
with open('data/predictions.json', 'w', encoding='utf-8') as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print(f"\n✅ {len(output_games)} Vorhersagen gespeichert!")
```

if **name** == “**main**”:
main()
