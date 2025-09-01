"""
Script to train a simple machine‑learning model that predicts the winner of a
Pokémon battle.  The model is trained on the provided `combats.csv` file,
which lists pairs of Pokémon and the winning combatant.  For each battle we
construct a feature vector consisting of the differences between the two
Pokémon's base statistics (HP, Attack, Defense, Special Attack, Special
Defense and Speed).  A RandomForestClassifier is fitted on these feature
difference vectors to learn which Pokémon tends to win given superior stats.

Running this script will train the model and save it to disk as
`pokemon_winner_model.pkl` so that it can be reused by the prediction
interface (`predict_cli.py`).

Usage:
    python build_model.py
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier


def load_data(pokemon_path: str, combats_path: str):
    """Load and prepare the Pokémon and combats datasets.

    Parameters
    ----------
    pokemon_path: str
        Path to the CSV containing Pokémon attributes.
    combats_path: str
        Path to the CSV containing combat outcomes.

    Returns
    -------
    tuple
        A tuple containing the processed Pokémon DataFrame, the feature matrix
        X and the target vector y.
    """
    # Load Pokémon data
    pokemon = pd.read_csv(pokemon_path)

    # Standardise column names for easier access
    pokemon = pokemon.rename(
        columns={
            '#': 'id',
            'Type 1': 'type1',
            'Type 2': 'type2',
            'HP': 'hp',
            'Attack': 'attack',
            'Defense': 'defense',
            'Sp. Atk': 'sp_atk',
            'Sp. Def': 'sp_def',
            'Speed': 'speed',
            'Generation': 'generation',
            'Legendary': 'legendary',
        }
    )
    # Use the Pokémon ID as the index for fast lookups
    pokemon.set_index('id', inplace=True)

    # Load combats data
    combats = pd.read_csv(combats_path)

    # Define the base stats to use when comparing Pokémon
    stat_columns = ['hp', 'attack', 'defense', 'sp_atk', 'sp_def', 'speed']

    features = []
    labels = []
    # Iterate through each combat record, compute stat differences and label
    for row in combats.itertuples(index=False):
        # Extract the two Pokémon participating in the battle
        first_id = row.First_pokemon
        second_id = row.Second_pokemon
        winner_id = row.Winner

        # Some Pokémon in the combats file might not exist in the Pokémon
        # dataset due to missing entries; skip those battles
        if first_id not in pokemon.index or second_id not in pokemon.index:
            continue

        p1 = pokemon.loc[first_id]
        p2 = pokemon.loc[second_id]

        # Compute the difference in base stats (first minus second)
        diff = p1[stat_columns].values - p2[stat_columns].values
        features.append(diff)

        # Label is 1 if the first Pokémon won, otherwise 0
        labels.append(1 if winner_id == first_id else 0)

    X = np.array(features)
    y = np.array(labels)
    return pokemon, X, y


def train_model(X: np.ndarray, y: np.ndarray) -> RandomForestClassifier:
    """Train a RandomForestClassifier on the provided data.

    Parameters
    ----------
    X: np.ndarray
        Feature matrix consisting of stat differences.
    y: np.ndarray
        Target vector indicating whether the first Pokémon won the battle.

    Returns
    -------
    RandomForestClassifier
        A trained random forest model.
    """
    # Use a modest number of trees to balance speed and performance
    clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    clf.fit(X, y)
    return clf


def main():
    # Paths relative to the project directory
    pokemon_path = 'pokemon.csv'
    combats_path = 'combats.csv'

    pokemon_df, X, y = load_data(pokemon_path, combats_path)
    model = train_model(X, y)

    # Persist the model and the feature list for later use
    with open('pokemon_winner_model.pkl', 'wb') as f:
        pickle.dump({'model': model, 'features': ['hp', 'attack', 'defense', 'sp_atk', 'sp_def', 'speed']}, f)

    print('Model training complete. Saved to pokemon_winner_model.pkl.')


if __name__ == '__main__':
    main()