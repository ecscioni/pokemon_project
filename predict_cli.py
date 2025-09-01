"""
Interactive command line interface for predicting the outcome of a
Pokémon battle using a trained machine‑learning model.  The tool
prompts the user for the names (or Pokédex numbers) of two Pokémon,
looks up their stats in `pokemon.csv`, constructs the feature vector
as defined in `build_model.py` and uses the pre‑trained model to
determine which Pokémon is more likely to win.  It also supports
obtaining a simple "worth it" score for an individual Pokémon based
on its base statistics.

Usage examples::

    # Predict the winner between Pikachu and Bulbasaur
    python predict_cli.py --pokemon1 Pikachu --pokemon2 Bulbasaur

    # Score a single Pokémon
    python predict_cli.py --score Charizard

The model file `pokemon_winner_model.pkl` must be present in the
same directory as this script.  If not found, run `build_model.py`
first to generate it.
"""

import argparse
import pandas as pd
import pickle
import sys


def load_pokemon_data(path: str) -> pd.DataFrame:
    """Load and preprocess the Pokémon dataset.

    Returns a DataFrame indexed by Pokédex number with simplified
    column names for consistent access.
    """
    df = pd.read_csv(path)
    df = df.rename(
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
    df.set_index('id', inplace=True)
    return df


def build_name_lookup(df: pd.DataFrame) -> dict:
    """Construct a dictionary mapping Pokémon names to their Pokédex numbers.

    The lookup is case‑insensitive and handles names with spaces or
    hyphens.  Duplicate names (e.g. Mega evolutions) are mapped to
    their first occurrence.
    """
    lookup = {}
    for pid, row in df.reset_index()[['id', 'Name']].itertuples(index=False):
        name_key = row.strip().lower()
        # Avoid overwriting entries for duplicate names
        lookup.setdefault(name_key, pid)
    return lookup


def predict_winner(p1_name: str, p2_name: str, pokemon_df: pd.DataFrame, model_data: dict) -> str:
    """Predict which Pokémon will win based on their names.

    Parameters
    ----------
    p1_name, p2_name: str
        Names or Pokédex numbers of the two Pokémon to compare.
    pokemon_df: pd.DataFrame
        DataFrame of Pokémon stats indexed by Pokédex number.
    model_data: dict
        Dictionary containing the trained model and the feature list.

    Returns
    -------
    str
        The name of the predicted winner.
    """
    # Build name lookup table for convenience
    name_to_id = build_name_lookup(pokemon_df)

    # Determine IDs (accept numeric IDs directly)
    def resolve(identifier: str) -> int:
        # Try numeric conversion first
        if identifier.isdigit():
            return int(identifier)
        key = identifier.strip().lower()
        if key not in name_to_id:
            raise ValueError(f'Unknown Pokémon: {identifier}')
        return name_to_id[key]

    id1 = resolve(p1_name)
    id2 = resolve(p2_name)

    if id1 not in pokemon_df.index or id2 not in pokemon_df.index:
        raise ValueError('One of the provided Pokédex numbers does not exist in the dataset.')

    # Extract features
    features = model_data['features']
    p1_stats = pokemon_df.loc[id1][features]
    p2_stats = pokemon_df.loc[id2][features]
    diff = (p1_stats - p2_stats).values.reshape(1, -1)

    # Make prediction (1 => first Pokémon wins, 0 => second wins)
    model = model_data['model']
    pred = model.predict(diff)[0]
    winner_id = id1 if pred == 1 else id2

    return pokemon_df.loc[winner_id]['Name']


def score_pokemon(name_or_id: str, pokemon_df: pd.DataFrame) -> float:
    """Compute a simple worth‑it score for a Pokémon.

    The score is defined as the mean of its base statistics (HP, attack,
    defence, special attack, special defence and speed).  Higher scores
    indicate better overall stats.  This function accepts a name or
    Pokédex number and returns the computed score.
    """
    name_lookup = build_name_lookup(pokemon_df)

    def resolve(identifier: str) -> int:
        if identifier.isdigit():
            return int(identifier)
        key = identifier.strip().lower()
        if key not in name_lookup:
            raise ValueError(f'Unknown Pokémon: {identifier}')
        return name_lookup[key]

    pid = resolve(name_or_id)
    if pid not in pokemon_df.index:
        raise ValueError('Provided Pokédex number does not exist in the dataset.')
    stats = pokemon_df.loc[pid][['hp', 'attack', 'defense', 'sp_atk', 'sp_def', 'speed']]
    return float(stats.mean())


def main():
    parser = argparse.ArgumentParser(description='Predict Pokémon battle outcomes or compute worth‑it scores.')
    parser.add_argument('--pokemon1', help='Name or Pokédex number of the first Pokémon')
    parser.add_argument('--pokemon2', help='Name or Pokédex number of the second Pokémon')
    parser.add_argument('--score', help='Name or Pokédex number of a single Pokémon to score')

    args = parser.parse_args()

    # Load model and data
    try:
        with open('pokemon_winner_model.pkl', 'rb') as f:
            model_data = pickle.load(f)
    except FileNotFoundError:
        print('Error: model file not found. Please run build_model.py first.', file=sys.stderr)
        sys.exit(1)

    pokemon_df = load_pokemon_data('pokemon.csv')

    # Determine action
    if args.score:
        score = score_pokemon(args.score, pokemon_df)
        print(f'Worth‑it score for {args.score}: {score:.2f}')
        return

    if not (args.pokemon1 and args.pokemon2):
        print('Error: you must specify two Pokémon names or numbers using --pokemon1 and --pokemon2, or specify --score for a single Pokémon.')
        sys.exit(1)

    try:
        winner = predict_winner(args.pokemon1, args.pokemon2, pokemon_df, model_data)
    except ValueError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)

    print(f'{args.pokemon1} vs {args.pokemon2} -> Predicted winner: {winner}')


if __name__ == '__main__':
    main()