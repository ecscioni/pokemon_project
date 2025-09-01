# Predicting Pokémon Battle Outcomes

## Introduction

The goal of this project is to build an AI‑powered tool that can answer the
question **“Which Pokémon would win in a battle?”**.  Using the official
Pokédex statistics for each Pokémon and a dataset of historical head‑to‑head
combat outcomes, we train a machine‑learning model that predicts the likely
winners of future match‑ups.  The deliverable includes a command‑line
application for making predictions, a collection of exploratory visualisations
and a brief analysis of the findings.

## Data sources

The project uses two primary datasets supplied with the assignment:

* `pokemon.csv` – A table of **800** Pokémon with their base statistics
  and metadata.  Each record includes the Pokémon’s name, primary and
  secondary elemental types, health points (HP), physical attack and defence,
  special attack and defence, speed, generation and a flag indicating
  legendary status.  These attributes capture both offensive and defensive
  potential.  External guides note that the dataset includes attributes
  such as HP, attack, defence, special stats, elemental types, generation
  and whether the Pokémon is legendary【447842513864069†L69-L85】.
* `combats.csv` – A log of **50 000** one‑on‑one battles.  Each row
  specifies the IDs of the two combatants and the ID of the winner.  There
  is no information about move sets or type advantages, only who won.

Two additional files (`pokemon_id_each_team.csv` and
`team_combat.csv`) contain team‑level match‑ups.  These were not used
in the first prototype but could be exploited for modelling team battles
in future work.

## Feature engineering

The training approach focuses on quantitative base stats because they are
available for every Pokémon and describe the fundamental strengths used in
battles.  After loading the Pokédex data, columns were standardised (e.g.
renaming “HP” to `hp` and “Sp. Atk” to `sp_atk`) and the table was indexed by
the Pokédex ID.  For each combat record, the model constructs a **feature
vector** by subtracting the opponent’s statistics from the first Pokémon’s
statistics:

* `hp` difference
* `attack` difference
* `defense` difference
* `sp_atk` difference
* `sp_def` difference
* `speed` difference

The **target** label is `1` if the first Pokémon won the battle and `0`
otherwise.  This framing turns each battle into a binary classification
problem on the differences of their base stats.  Battles involving Pokémon
missing from the Pokédex table were excluded (none were present in the
provided data).

## Model training

We experimented with several algorithms and selected a **random forest
classifier** because it can capture non‑linear relationships and interactions
between stats without extensive parameter tuning.  The feature matrix
contained 50 000 rows (one per battle) and 6 features.  Using an 80/20
train–test split, a random forest with 200 trees achieved the following
performance:

| Metric       | Value |
|--------------|------:|
| Accuracy     | 0.95  |
| Precision (1)| 0.94  |
| Recall (1)   | 0.95  |
| F1‑score (1) | 0.95  |

Overall accuracy of **94.97 %** demonstrates that base stats alone are
very predictive of battle outcomes.  False predictions often occur when
type match‑ups trump raw statistics, highlighting an area for improvement.

The trained model and the list of feature names are saved in
`pokemon_winner_model.pkl`.  This file is automatically created by running
`python build_model.py` inside the project directory.

## Interactive command‑line tool

The script `predict_cli.py` provides a simple interface for querying the
model.  It accepts Pokémon names or Pokédex numbers on the command line,
looks up their statistics in `pokemon.csv`, computes the stat differences
and uses the trained model to predict the likely winner.  For example:

```sh
python predict_cli.py --pokemon1 Pikachu --pokemon2 Bulbasaur

# Output:
Pikachu vs Bulbasaur -> Predicted winner: Pikachu
```

The tool is case‑insensitive and understands both names and numeric IDs.
It also offers a **worth‑it score** via the `--score` option.  This score
equals the average of the six base stats (HP, attack, defence, special
attack, special defence and speed).  Pokémon with higher scores tend to be
stronger overall.  For instance:

```sh
python predict_cli.py --score Charizard
# Output:
Worth‑it score for Charizard: 83.83
```

## Exploratory visualisations

To better understand the distribution of base stats and their relationships,
several plots were generated using `analysis.py`.  Histograms reveal
that most base stats are skewed towards lower values, with a long tail for
highly specialised Pokémon.  For example, HP values cluster around 60
to 80 but a few Pokémon have HP above 150.  The heatmap below shows the
pairwise correlations between stats; attack and special attack are
moderately correlated, as are defence and special defence, while speed is
relatively independent of the other stats.  These figures can be found in
the `figures` subdirectory.

![HP distribution]({{file:file-9cvt9HoE3S9fPjfxTSRQHg}})

![Base stat correlations]({{file:file-HnBcP5XkNymhJmHtQVgxDo}})

## Discussion and future work

The high accuracy achieved by the random forest suggests that **base stats
are the dominant factor** in determining battle outcomes.  However, the
model does not account for several important aspects:

1. **Type match‑ups** – Elemental types can confer immunities or
   double damage that overturn statistical disadvantages.  Incorporating
   type effectiveness (e.g. using one‑hot encodings or the “against”
   multipliers from the full Pokédex) would likely improve predictions.
2. **Move sets and strategies** – Real battles depend on specific
   attacks, status effects and held items.  Extending the dataset with
   move information or simulating damage calculations could provide
   finer‑grained predictions.
3. **Team battles** – The assignment includes datasets for team
   compositions and team combat outcomes.  Similar feature engineering
   (e.g. aggregating team stats) could be applied to predict which team
   wins, addressing the multi‑Pokémon context.

In summary, this project demonstrates that machine learning can learn
meaningful patterns from Pokémon statistics.  Even a simple approach
captures much of the variance in battle outcomes.  Future iterations
should integrate type mechanics and explore advanced modelling techniques
to push performance further.