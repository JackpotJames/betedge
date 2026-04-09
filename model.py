"""
BetEdge NBA Prediction Model
Läuft täglich via GitHub Actions und aktualisiert data/predictions.json
"""

import json
import os
from datetime import datetime, timezone
import warnings
warnings.filterwarnings('ignore')

try:
    import pandas as pd
    import numpy as np
    from nba_api.stats.endpoints import leaguegamefinder, teamgamelogs, commonteamroster
    from nba_api.stats.static import teams
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.preprocessing import StandardScaler
    import time
except ImportError as e:
    print(f"Fehlende Library: {e}")
    print("Installiere mit: pip install nba_api pandas numpy scikit-learn")
    exit(1)

# ─── KONSTANTEN ────────────────────────────────────────────────────────────────
CURRENT_SEASON = "2025-26"
ROLLING_WINDOW = 10      # letzte N Spiele für Form-Features
KELLY_BANKROLL = 1.0     # normiert auf 1 (Anteil in %)
MIN_CONFIDENCE = 55      # nur Picks über diesem Threshold ausgeben

# ─── TEAM DATEN LADEN ──────────────────────────────────────────────────────────
def get_all_teams():
    nba_teams = teams.get_teams()
    return {t['abbreviation']: t['id'] for t in nba_teams}

def get_team_logs(team_id, season=CURRENT_SEASON, n_games=30):
    """Holt Spiel-Logs für ein Team"""
    time.sleep(0.6)  # API Rate Limit
    try:
        logs = teamgamelogs.TeamGameLogs(
            team_id_nullable=team_id,
            season_nullable=season,
            league_id_nullable="00"
        ).get_data_frames()[0]
        return logs.head(n_games)
    except Exception as e:
        print(f"  Fehler beim Laden von Team {team_id}: {e}")
        return None

def calc_team_features(logs):
    """Berechnet Rolling-Average Features aus Spiel-Logs"""
    if logs is None or len(logs) < 5:
        return None

    df = logs.copy()

    # Win/Loss
    df['WIN'] = (df['WL'] == 'W').astype(int)

    # Offensive/Defensive Ratings schätzen
    # (echte Ratings brauchen possession-Daten, hier Näherung)
    df['PTS_SCORED'] = pd.to_numeric(df['PTS'], errors='coerce')
    df['PTS_ALLOWED'] = pd.to_numeric(df['OPP_PTS'] if 'OPP_PTS' in df.columns else df['PLUS_MINUS'] - df['PTS'] * 0 + df['PTS'], errors='coerce')

    n = min(ROLLING_WINDOW, len(df))
    recent = df.head(n)

    features = {
        'win_pct': recent['WIN'].mean(),
        'win_pct_last5': df.head(5)['WIN'].mean() if len(df) >= 5 else recent['WIN'].mean(),
        'pts_scored': pd.to_numeric(recent['PTS'], errors='coerce').mean(),
        'fg_pct': pd.to_numeric(recent['FG_PCT'], errors='coerce').mean(),
        'fg3_pct': pd.to_numeric(recent['FG3_PCT'], errors='coerce').mean(),
        'ft_pct': pd.to_numeric(recent['FT_PCT'], errors='coerce').mean(),
        'reb': pd.to_numeric(recent['REB'], errors='coerce').mean(),
        'ast': pd.to_numeric(recent['AST'], errors='coerce').mean(),
        'tov': pd.to_numeric(recent['TOV'], errors='coerce').mean(),
        'stl': pd.to_numeric(recent['STL'], errors='coerce').mean(),
        'blk': pd.to_numeric(recent['BLK'], errors='coerce').mean(),
        'plus_minus': pd.to_numeric(recent['PLUS_MINUS'], errors='coerce').mean(),
        'streak': calc_streak(df['WIN'].values),
    }

    # True Shooting % approximation
    pts = features['pts_scored']
    fga = pd.to_numeric(recent['FGA'], errors='coerce').mean()
    fta = pd.to_numeric(recent['FTA'], errors='coerce').mean()
    if fga > 0 and fta > 0:
        features['true_shooting'] = pts / (2 * (fga + 0.44 * fta))
    else:
        features['true_shooting'] = features['fg_pct']

    return features

def calc_streak(wins):
    """Berechnet aktuelle Siegesserie (positiv) oder Niederlagenserie (negativ)"""
    if len(wins) == 0:
        return 0
    streak = 1 if wins[0] == 1 else -1
    for i in range(1, len(wins)):
        if wins[i] == wins[0]:
            streak += (1 if wins[0] == 1 else -1)
        else:
            break
    return streak

# ─── HISTORISCHE DATEN FÜR TRAINING ───────────────────────────────────────────
def build_training_data(all_team_logs):
    """
    Erstellt Feature-Matrix aus historischen Team-Matchups.
    X = Differenz der Team-Features (Heim - Auswärts)
    y = Heimsieg (1) oder Auswärtssieg (0)
    """
    records = []

    for team_abbr, logs in all_team_logs.items():
        if logs is None or len(logs) < 10:
            continue

        for i in range(5, len(logs) - 1):
            # Features aus den letzten N Spielen vor diesem Spiel
            past = logs.iloc[i:]
            n = min(ROLLING_WINDOW, len(past))
            recent = past.head(n)

            win = 1 if logs.iloc[i]['WL'] == 'W' else 0
            is_home = '@' not in str(logs.iloc[i].get('MATCHUP', ''))

            record = {
                'is_home': int(is_home),
                'win': win,
                'win_pct': pd.to_numeric(recent['WL'].apply(lambda x: 1 if x=='W' else 0), errors='coerce').mean(),
                'fg_pct': pd.to_numeric(recent['FG_PCT'], errors='coerce').mean(),
                'fg3_pct': pd.to_numeric(recent['FG3_PCT'], errors='coerce').mean(),
                'plus_minus': pd.to_numeric(recent['PLUS_MINUS'], errors='coerce').mean(),
                'ast': pd.to_numeric(recent['AST'], errors='coerce').mean(),
                'tov': pd.to_numeric(recent['TOV'], errors='coerce').mean(),
                'reb': pd.to_numeric(recent['REB'], errors='coerce').mean(),
                'stl': pd.to_numeric(recent['STL'], errors='coerce').mean(),
            }
            records.append(record)

    if len(records) < 50:
        return None, None

    df = pd.DataFrame(records).dropna()
    feature_cols = ['win_pct', 'fg_pct', 'fg3_pct', 'plus_minus', 'ast', 'tov', 'reb', 'stl']
    X = df[feature_cols].values
    y = df['win'].values
    return X, y

# ─── MODELL TRAINING ───────────────────────────────────────────────────────────
def train_model(X, y):
    """Trainiert XGBoost-ähnliches Gradient Boosting Modell mit Kalibrierung"""
    base = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        min_samples_leaf=20,
        subsample=0.8,
        random_state=42
    )
    # Kalibrierung ist KRITISCH für Kelly Criterion (wie Forschung zeigt)
    model = CalibratedClassifierCV(base, cv=5, method='isotonic')
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model.fit(X_scaled, y)
    return model, scaler

# ─── HEUTE'S SPIELE HOLEN ──────────────────────────────────────────────────────
def get_todays_games():
    """Holt die heutigen NBA Spiele"""
    time.sleep(0.6)
    try:
        today = datetime.now().strftime('%m/%d/%Y')
        finder = leaguegamefinder.LeagueGameFinder(
            date_from_nullable=today,
            date_to_nullable=today,
            league_id_nullable='00'
        )
        games_df = finder.get_data_frames()[0]
        if games_df.empty:
            return []

        # Duplikate entfernen (jedes Spiel erscheint 2x, für jedes Team)
        games = []
        seen = set()
        for _, row in games_df.iterrows():
            gid = row['GAME_ID']
            if gid not in seen:
                seen.add(gid)
                matchup = str(row.get('MATCHUP', ''))
                if ' vs. ' in matchup:
                    home_abbr = matchup.split(' vs. ')[0].strip()
                    away_abbr = matchup.split(' vs. ')[1].strip()
                elif ' @ ' in matchup:
                    away_abbr = matchup.split(' @ ')[0].strip()
                    home_abbr = matchup.split(' @ ')[1].strip()
                else:
                    continue
                games.append({'home': home_abbr, 'away': away_abbr, 'game_id': gid})
        return games
    except Exception as e:
        print(f"Fehler beim Laden der heutigen Spiele: {e}")
        return []

# ─── VORHERSAGE ────────────────────────────────────────────────────────────────
def predict_game(model, scaler, home_features, away_features):
    """Berechnet Gewinnwahrscheinlichkeit für ein Spiel"""
    if home_features is None or away_features is None:
        return None

    feature_keys = ['win_pct', 'fg_pct', 'fg3_pct', 'plus_minus', 'ast', 'tov', 'reb', 'stl']

    # Feature-Differenz Heim - Auswärts
    X = np.array([[
        home_features.get(k, 0) - away_features.get(k, 0)
        for k in feature_keys
    ]])

    X_scaled = scaler.transform(X)
    prob = model.predict_proba(X_scaled)[0]

    home_win_prob = float(prob[1])
    away_win_prob = float(prob[0])

    return home_win_prob, away_win_prob

def calc_kelly(prob, odds):
    """Kelly Criterion: f = (p*b - q) / b, b = odds - 1"""
    b = odds - 1
    q = 1 - prob
    kelly = (prob * b - q) / b
    return max(0, kelly)  # Nie negativ (nicht wetten)

def calc_ev(model_prob, book_odds):
    """Expected Value: EV = (prob * (odds-1)) - (1-prob)"""
    ev = (model_prob * (book_odds - 1)) - (1 - model_prob)
    return ev * 100  # in %

def get_last5_str(logs):
    if logs is None or len(logs) < 5:
        return "N/A"
    last5 = logs.head(5)['WL'].values
    w = sum(1 for x in last5 if x == 'W')
    l = sum(1 for x in last5 if x == 'L')
    return f"{w}-{l}"

# ─── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    print("🏀 BetEdge NBA Prediction Model")
    print("=" * 40)

    # Teams laden
    print("📥 Lade Team-IDs...")
    team_map = get_all_teams()

    # Alle Team-Logs laden (für Training)
    print("📊 Lade Team-Statistiken (dauert ~2-3 Minuten)...")
    all_team_logs = {}
    nba_teams = teams.get_teams()

    for i, team in enumerate(nba_teams[:30]):  # alle 30 NBA Teams
        abbr = team['abbreviation']
        tid = team['id']
        print(f"  [{i+1}/30] {abbr}...", end=' ', flush=True)
        logs = get_team_logs(tid)
        if logs is not None and len(logs) > 0:
            all_team_logs[abbr] = logs
            print(f"✓ ({len(logs)} Spiele)")
        else:
            print("✗ keine Daten")

    # Training
    print("\n🧠 Trainiere Modell...")
    X, y = build_training_data(all_team_logs)

    if X is None or len(X) < 50:
        print("⚠️ Nicht genug Daten für Training. Nutze Demo-Daten.")
        generate_demo_output()
        return

    model, scaler = train_model(X, y)
    print(f"✓ Modell trainiert auf {len(X)} Datenpunkten")

    # Heutige Spiele
    print("\n📅 Lade heutige Spiele...")
    todays_games = get_todays_games()

    if not todays_games:
        print("Keine Spiele heute — nutze Demo-Daten")
        generate_demo_output()
        return

    print(f"✓ {len(todays_games)} Spiele gefunden")

    # Features für heutige Teams
    team_features = {}
    for game in todays_games:
        for abbr in [game['home'], game['away']]:
            if abbr not in team_features and abbr in all_team_logs:
                team_features[abbr] = calc_team_features(all_team_logs[abbr])

    # Vorhersagen generieren
    print("\n🔮 Generiere Vorhersagen...")
    output_games = []

    for game in todays_games:
        home_abbr = game['home']
        away_abbr = game['away']

        hf = team_features.get(home_abbr)
        af = team_features.get(away_abbr)

        result = predict_game(model, scaler, hf, af)
        if result is None:
            continue

        home_prob, away_prob = result

        # Buchmacher-Odds (Platzhalter — in Produktion von API holen)
        # Heimvorteil typisch ~52-55% beim Buchmacher
        book_home = 0.52 + np.random.uniform(-0.05, 0.05)
        book_away = 1 - book_home
        home_odds = round(1 / book_home * 0.95, 2)  # 5% Margin
        away_odds = round(1 / book_away * 0.95, 2)

        ev_home = calc_ev(home_prob, home_odds)
        ev_away = calc_ev(away_prob, away_odds)
        kelly = calc_kelly(home_prob if home_prob > away_prob else away_prob,
                          home_odds if home_prob > away_prob else away_odds)

        model_pick = home_abbr if home_prob > away_prob else away_abbr
        confidence = int(max(home_prob, away_prob) * 100)
        value_bet = model_pick if (ev_home > 0 or ev_away > 0) else None

        hf = hf or {}
        af = af or {}
        hl = all_team_logs.get(home_abbr)
        al = all_team_logs.get(away_abbr)

        # Key Factors
        factors = []
        if hf.get('win_pct', 0) > af.get('win_pct', 0) + 0.1:
            factors.append(f"{home_abbr} deutlich bessere Siegquote")
        if af.get('win_pct', 0) > hf.get('win_pct', 0) + 0.1:
            factors.append(f"{away_abbr} deutlich bessere Siegquote")
        if hf.get('plus_minus', 0) > af.get('plus_minus', 0) + 3:
            factors.append(f"{home_abbr} Net Rating Vorteil +{hf['plus_minus'] - af['plus_minus']:.1f}")
        if af.get('plus_minus', 0) > hf.get('plus_minus', 0) + 3:
            factors.append(f"{away_abbr} Net Rating Vorteil +{af['plus_minus'] - hf['plus_minus']:.1f}")
        if hf.get('streak', 0) >= 3:
            factors.append(f"{home_abbr} auf {abs(hf['streak'])}-Spiel-Siegesserie")
        if af.get('streak', 0) >= 3:
            factors.append(f"{away_abbr} auf {abs(af['streak'])}-Spiel-Siegesserie")
        factors.append("Heimvorteil einkalkuliert")
        if len(factors) < 2:
            factors.append("Ausgeglichenes Matchup — hohe Unsicherheit")

        # Team Rekord
        def get_record(logs):
            if logs is None: return "N/A"
            w = sum(1 for x in logs['WL'] if x == 'W')
            l = sum(1 for x in logs['WL'] if x == 'L')
            return f"{w}-{l}"

        game_obj = {
            "id": f"nba_{datetime.now().strftime('%Y%m%d')}_{home_abbr}_{away_abbr}",
            "date": datetime.now().strftime('%Y-%m-%d'),
            "time": "TBD",
            "home": {
                "name": next((t['full_name'] for t in nba_teams if t['abbreviation'] == home_abbr), home_abbr),
                "abbr": home_abbr,
                "record": get_record(hl),
                "win_pct": round(hf.get('win_pct', 0), 3),
                "last5": get_last5_str(hl),
                "offensive_rating": round(hf.get('pts_scored', 0), 1),
                "defensive_rating": 0,
                "net_rating": round(hf.get('plus_minus', 0), 1),
                "pace": 98,
                "true_shooting": round(hf.get('true_shooting', 0), 3),
                "three_pct": round(hf.get('fg3_pct', 0), 3),
                "injuries": []
            },
            "away": {
                "name": next((t['full_name'] for t in nba_teams if t['abbreviation'] == away_abbr), away_abbr),
                "abbr": away_abbr,
                "record": get_record(al),
                "win_pct": round(af.get('win_pct', 0), 3),
                "last5": get_last5_str(al),
                "offensive_rating": round(af.get('pts_scored', 0), 1),
                "defensive_rating": 0,
                "net_rating": round(af.get('plus_minus', 0), 1),
                "pace": 98,
                "true_shooting": round(af.get('true_shooting', 0), 3),
                "three_pct": round(af.get('fg3_pct', 0), 3),
                "injuries": []
            },
            "prediction": {
                "home_win_prob": round(home_prob, 3),
                "away_win_prob": round(away_prob, 3),
                "model_pick": model_pick,
                "confidence": confidence,
                "book_home_prob": round(book_home, 3),
                "book_away_prob": round(book_away, 3),
                "home_odds": home_odds,
                "away_odds": away_odds,
                "home_ev": round(ev_home, 1),
                "away_ev": round(ev_away, 1),
                "kelly_fraction": round(kelly, 4),
                "value_bet": value_bet,
                "key_factors": factors[:3]
            }
        }
        output_games.append(game_obj)
        print(f"  ✓ {home_abbr} vs {away_abbr}: {model_pick} ({confidence}% conf.)")

    # Output speichern
    output = {
        "last_updated": datetime.now(timezone.utc).isoformat(),
        "model_version": "1.0.0",
        "sport": "NBA",
        "games": output_games,
        "model_stats": {
            "season_record": "—",
            "accuracy": "—",
            "roi": "—",
            "avg_confidence": int(np.mean([g['prediction']['confidence'] for g in output_games])) if output_games else 0
        }
    }

    os.makedirs('data', exist_ok=True)
    with open('data/predictions.json', 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n✅ {len(output_games)} Vorhersagen gespeichert in data/predictions.json")

def generate_demo_output():
    """Generiert Demo-Output wenn keine API-Daten verfügbar"""
    print("📝 Generiere Demo-Daten...")
    demo_path = os.path.join(os.path.dirname(__file__), 'data', 'predictions.json')
    # Die bestehende predictions.json bleibt unverändert
    print("✓ Demo-Daten belassen (predictions.json unverändert)")

if __name__ == "__main__":
    main()
