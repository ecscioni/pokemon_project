# PokéBattle Predictor — “Which Pokémon would win?”

An AI-powered tool that predicts the likely winner between two Pokémon using their **base stats** and a small **machine learning** model. Includes a simple command-line interface (CLI) and optional figures for your report.

---

## Features
- Predicts the winner for any two Pokémon (by **name** or **Pokédex id**)
- Shows a probability-style **confidence**
- Computes a simple **“worth-it” score** (mean of base stats) for any Pokémon
- Generates basic **figures** (histograms + correlation heatmap)

---

## Project Structure
```
pokemon_project/
├─ build_model.py                # Train model from combats.csv
├─ predict_cli.py                # CLI: predict winners / compute scores
├─ analysis.py                   # Create figures (PNG) for the report
├─ pokemon_winner_model.pkl      # Trained model (created by build_model.py)
├─ pokemon.csv                   # Pokédex stats (types + base stats)
├─ combats.csv                   # Head-to-head battles with winners
├─ figures/                      # Images saved by analysis.py
│   ├─ hp_distribution.png
│   ├─ attack_distribution.png
│   ├─ defense_distribution.png
│   ├─ sp_atk_distribution.png
│   ├─ sp_def_distribution.png
│   ├─ speed_distribution.png
│   └─ base_stats_correlation.png
└─ report.md                     # Method, results, findings, limits, next steps
```

---

## Setup

### Windows (PowerShell or VS Code Terminal)
```powershell
# inside pokemon_project/
py -m venv .venv
.venv\Scripts\Activate

py -m pip install --upgrade pip
pip install pandas scikit-learn matplotlib numpy joblib
# Optional (prettier heatmap): pip install seaborn
```

### macOS / Linux
```bash
python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
pip install pandas scikit-learn matplotlib numpy joblib
# Optional: pip install seaborn
```

---

## Quick Start

### 1) Train (or re-train) the model
```bash
python build_model.py
```
What it does:
- Loads `pokemon.csv` and `combats.csv`
- Builds features = **differences** of base stats between the two Pokémon
- Trains a `RandomForestClassifier(n_estimators=200, random_state=42)` on a stratified 80/20 split
- Prints accuracy + classification report
- Saves `pokemon_winner_model.pkl`

Typical result: ~**0.95 accuracy** on the held-out set (example run).

### 2) Predict a battle (CLI)
By **name**:
```bash
python predict_cli.py --pokemon1 Pikachu --pokemon2 Bulbasaur
```
By **id**:
```bash
python predict_cli.py --pokemon1 25 --pokemon2 1
```
Output shows the predicted **winner** and a **confidence** value.

### 3) “Worth-it” score for a single Pokémon
```bash
python predict_cli.py --score Charizard
# or
python predict_cli.py --score 6
```
Score = mean of `{hp, attack, defense, sp_atk, sp_def, speed}`.

### 4) Generate figures for your report
```bash
python analysis.py
```
Images are saved in `figures/`.

> If `analysis.py` asks for seaborn and you don’t want to install it, you can switch to the matplotlib-only version (or just run `pip install seaborn`).

---

## How It Works (short)

**Inputs**  
- `pokemon.csv`: columns include `#` (id), `Name`, `Type 1`, `Type 2`, `HP`, `Attack`, `Defense`, `Sp. Atk`, `Sp. Def`, `Speed`, `Generation`, `Legendary`
- `combats.csv`: columns `First_pokemon`, `Second_pokemon`, `Winner` (as id)

**Features**  
For a pair (A, B), features are stat differences:
```
[hp_A - hp_B,
 attack_A - attack_B,
 defense_A - defense_B,
 sp_atk_A - sp_atk_B,
 sp_def_A - sp_def_B,
 speed_A - speed_B]
```

**Model**  
Random Forest (200 trees).  
Metric: accuracy on a held-out 20% test split (stratified).

**CLI**  
Looks up two Pokémon (name or id), builds the feature vector, loads the trained model, prints predicted winner + confidence.

---

## Assumptions & Limitations
- Uses **base stats only** (no moves, abilities, items, weather, or detailed type matchups)
- Depends on coverage/quality of `combats.csv`
- Confidence is model probability, not a guaranteed win rate in real gameplay

---

## Ideas to Improve
- Encode **type matchups** or add type-effectiveness multipliers as features
- Add a **logistic regression** baseline and compare
- Use **cross-validation** and **hyperparameter search** (`RandomizedSearchCV`)
- Add model **explainability** (permutation importance or SHAP)

---

## Requirements
```
Python 3.9–3.12
pandas
numpy
scikit-learn
matplotlib
joblib
seaborn   # optional
```
You can also create a `requirements.txt` with the list above and install via:
```bash
pip install -r requirements.txt
```

---

## Troubleshooting
- `ModuleNotFoundError` → `pip install <module>`
- `python not found` (Windows) → install Python from python.org and reopen terminal; try `py` instead of `python`
- Figures missing → run `python analysis.py`
- Name not found → use exact spelling from `pokemon.csv` **or** pass the Pokédex id


