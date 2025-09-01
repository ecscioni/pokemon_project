"""
Generate simple visualisations and summaries for the Pokémon dataset.  This
script illustrates the distribution of base stats and the relationships
between them.  The resulting figures are saved as PNG files in the
`figures` subdirectory.  Running this script is optional for the
project deliverable, but it produces useful graphics that can be
included in reports or presentations.

Usage::

    python analysis.py
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def main():
    pokemon = pd.read_csv('pokemon.csv')
    pokemon = pokemon.rename(
        columns={
            'HP': 'hp',
            'Attack': 'attack',
            'Defense': 'defense',
            'Sp. Atk': 'sp_atk',
            'Sp. Def': 'sp_def',
            'Speed': 'speed',
        }
    )

    # Create output directory
    os.makedirs('figures', exist_ok=True)

    # Histogram of each base stat
    stat_cols = ['hp', 'attack', 'defense', 'sp_atk', 'sp_def', 'speed']
    for col in stat_cols:
        plt.figure(figsize=(6, 4))
        sns.histplot(pokemon[col], bins=30, kde=True, color='skyblue')
        plt.title(f'Distribution of {col.capitalize()}')
        plt.xlabel(col.capitalize())
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig(f'figures/{col}_distribution.png')
        plt.close()

    # Pairwise correlation heatmap of base stats
    corr = pokemon[stat_cols].corr()
    plt.figure(figsize=(7, 5))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', square=True)
    plt.title('Correlation between base stats')
    plt.tight_layout()
    plt.savefig('figures/base_stats_correlation.png')
    plt.close()

    print('Analysis figures saved to figures/*.png')


if __name__ == '__main__':
    main()