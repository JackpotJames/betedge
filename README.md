# BetEdge 🏀

AI-gestützte Sportwetten-Analyse mit Kelly Criterion.

## Setup (5 Minuten)

### 1. Repository erstellen
- Gehe auf github.com → "New repository"
- Name: `betedge` (oder was du willst)
- Public ✓
- Create repository

### 2. Dateien hochladen
Lade alle diese Dateien hoch:
- `index.html`
- `model.py`
- `data/predictions.json`
- `.github/workflows/predict.yml`

### 3. GitHub Pages aktivieren
- Settings → Pages
- Source: "Deploy from branch"
- Branch: main / root
- Save

Deine Seite ist dann erreichbar unter:
`https://DEIN-USERNAME.github.io/betedge`

### 4. Automatische Updates
Das Modell läuft täglich automatisch um 14:00 Uhr (DE).
Du kannst es auch manuell starten: Actions → "NBA Daily Predictions" → "Run workflow"

## Features
- 🧠 ML-Modell (Gradient Boosting + Kalibrierung)
- 📐 Kelly Criterion für optimale Einsatzgröße
- 💰 Expected Value Berechnung
- ⚠️ Injury Reports
- 📊 Team-Statistiken Vergleich
- 🔄 Täglich automatisch aktualisiert
