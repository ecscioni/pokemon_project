# pokemon_project

PokéBattle Predictor — “Which Pokémon would win?”

An AI-powered analysis tool that predicts the winner between two Pokémon using their base stats and a machine-learning model. Includes a simple command-line interface (CLI) and optional visuals for your report.

What this does

Predicts the likely winner between two Pokémon (by name or Pokédex id).

Gives a probability-style confidence.

Provides a simple “worth-it” score for any Pokémon (average of base stats).

Generates figures (histograms + correlation heatmap) for documentation.

Project structure
pokemon_project/
├─ build_model.py                # Train model from combats.csv
├─ predict_cli.py                # CLI: predict winners / compute scores
├─ analysis.py                   # Create plots for the report
├─ pokemon_winner_model.pkl      # Trained model (created by build_model.py)
├─ pokemon.csv                   # Pokédex stats (hp/atk/def/sp_atk/sp_def/speed, etc.)
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

Setup
Windows (PowerShell or VS Code Terminal)
# inside pokemon_project/
py -m venv .venv
.venv\Scripts\Activate

py -m pip install --upgrade pip
pip install pandas scikit-learn matplotlib numpy joblib
# Optional (pretty plots): pip install seaborn

macOS / Linux
python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
pip install pandas scikit-learn matplotlib numpy joblib
# Optional: pip install seaborn

Quick start (most common commands)
1) Train (or re-train) the model
python build_model.py


Loads pokemon.csv and combats.csv

Builds features = differences of base stats between two Pokémon

Trains a RandomForest (200 trees) with a stratified 80/20 split

Prints accuracy and classification report

Saves pokemon_winner_model.pkl

Typical result (example run): ~0.95 accuracy on the test split.

2) Predict a battle (CLI)

By name:

python predict_cli.py --pokemon1 Pikachu --pokemon2 Bulbasaur


By id:

python predict_cli.py --pokemon1 25 --pokemon2 1


Output shows the predicted winner and a confidence value.

3) “Worth-it” score for a single Pokémon
python predict_cli.py --score Charizard
# or
python predict_cli.py --score 6


Score = average of {hp, attack, defense, sp_atk, sp_def, speed}.

4) Generate figures for your report
python analysis.py


Images are saved in figures/.

If analysis.py asks for seaborn and you don’t want to install it, use the provided matplotlib-only version (or install seaborn with pip install seaborn).

How it works (short version)

Inputs
pokemon.csv with base stats + metadata; combats.csv with head-to-head battles and the winner.

Features
For a pair (A, B), compute stat differences:
[hp_A - hp_B, attack_A - attack_B, defense_A - defense_B, sp_atk_A - sp_atk_B, sp_def_A - sp_def_B, speed_A - speed_B].

Model
RandomForestClassifier(n_estimators=200, random_state=42).
Metric: accuracy on held-out 20% test split (stratified).

CLI
Looks up the two Pokémon (name or id), builds the feature vector, loads the trained model, prints the predicted winner and confidence.

Assumptions & limitations

Uses base stats only (no moves, abilities, items, weather, type matchups, status effects).

Dependent on the quality/coverage of combats.csv.

Duplicate or unusual entries in combats.csv can bias training.

Confidence is model probability; it’s not guaranteed win chance in real games.

Ideas to improve (if you have time)

Add type matchup features (e.g., one-hot encode types or use type effectiveness multipliers).

Try a logistic regression baseline and compare.

Add cross-validation and hyperparameter search (e.g., RandomizedSearchCV).

Model explainability: permutation importance or SHAP for which stats matter most.

Requirements

Python 3.9–3.12 recommended

Packages:

pandas
numpy
scikit-learn
matplotlib
joblib
seaborn   # optional


You can also create a requirements.txt with the list above and install via:

pip install -r requirements.txt

Typical troubleshooting

ModuleNotFoundError → pip install <module>

python not found (Windows) → install Python from python.org, reopen terminal; try py instead of python

Git/VS Code line endings on Windows → optional: git config --global core.autocrlf true