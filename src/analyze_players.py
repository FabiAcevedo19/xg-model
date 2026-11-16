"""Which players scored more (or fewer) goals than their chances were worth?

Run from the project folder (after train.py):
    python src/analyze_players.py

Two things keep this honest:

* Out-of-sample xG. Every shot's xG comes from a model that was NOT trained
  on that shot's match (5-fold cross-validation grouped by match). Using the
  final model would be circular: it has already seen whether each shot scored.
* Luck. Even a perfectly average finisher will beat their xG in some seasons.
  If a player's shots have probabilities p1, p2, ..., the number of goals has
  standard deviation sqrt(sum of p * (1 - p)). The z-score says how many of
  those standard deviations the player is above or below expectation. Beyond
  about 2 is unlikely to be luck alone.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from features import build_features
from load_data import load_shots
from train import boosted_trees, cross_validate

ROOT = Path(__file__).resolve().parent.parent
MIN_SHOTS = 50


def player_table(shots: pd.DataFrame, min_shots: int) -> pd.DataFrame:
    """One row per player: shots, goals, xG, the difference, and a z-score."""
    shots = shots.assign(variance=shots["xg"] * (1 - shots["xg"]))
    table = shots.groupby("player").agg(
        team=("team", lambda t: t.mode().iloc[0]),
        shots=("is_goal", "size"),
        goals=("is_goal", "sum"),
        xg=("xg", "sum"),
        statsbomb_xg=("statsbomb_xg", "sum"),
        sd=("variance", lambda v: np.sqrt(v.sum())),
    )
    table = table[table["shots"] >= min_shots]
    table["goals_minus_xg"] = table["goals"] - table["xg"]
    table["z_score"] = table["goals_minus_xg"] / table["sd"]
    return table.drop(columns="sd").sort_values("goals_minus_xg", ascending=False)


def main():
    metrics = json.loads((ROOT / "results" / "metrics.json").read_text())
    features, params = metrics["final_features"], metrics["chosen_params"]

    shots = load_shots()
    shots = shots[shots["shot_type"] != "Penalty"].reset_index(drop=True)
    print(f"Predicting {len(shots)} non-penalty shots out-of-sample...", flush=True)
    X = build_features(shots)[features]
    shots["xg"] = cross_validate(boosted_trees(features, params), X,
                                 shots["is_goal"], shots["match_id"])

    clubs = shots[shots["competition"].str.contains("2015/16")]
    table = player_table(clubs, MIN_SHOTS)

    pd.set_option("display.width", 200)
    columns = ["team", "shots", "goals", "xg", "goals_minus_xg", "z_score"]
    print(f"\n2015/16 league seasons, players with at least {MIN_SHOTS} "
          f"non-penalty shots: {len(table)}")
    print("\nBiggest over-performers\n", table[columns].head(10).round(2).to_string())
    print("\nBiggest under-performers\n", table[columns].tail(10).round(2).to_string())

    beyond_2 = (table["z_score"].abs() > 2).sum()
    print(f"\nPlayers more than 2 standard deviations from their xG: {beyond_2} of "
          f"{len(table)} ({beyond_2 / len(table):.0%}). Luck alone predicts about 5%.")
    agreement = np.corrcoef(table["goals_minus_xg"],
                            table["goals"] - table["statsbomb_xg"])[0, 1]
    print(f"Agreement with the same ranking from StatsBomb's xG: r = {agreement:.2f}")

    (ROOT / "results").mkdir(exist_ok=True)
    table.round(3).to_csv(ROOT / "results" / "player_finishing_2015_16.csv")

    # ---- Figure: the ten biggest over- and under-performers ----------------
    extremes = pd.concat([table.head(10), table.tail(10)]).iloc[::-1]
    colors = ["#2a9d8f" if v > 0 else "#e63946" for v in extremes["goals_minus_xg"]]
    fig, ax = plt.subplots(figsize=(9, 7.5))
    ax.barh(extremes.index, extremes["goals_minus_xg"], color=colors)
    for name, row in extremes.iterrows():
        value = row["goals_minus_xg"]
        ax.text(value + (0.15 if value > 0 else -0.15), name,
                f"{int(row['goals'])} goals from {row['xg']:.1f} xG",
                va="center", ha="left" if value > 0 else "right", fontsize=8)
    ax.axvline(0, color="black", linewidth=0.8)
    limit = extremes["goals_minus_xg"].abs().max() + 6
    ax.set_xlim(-limit, limit)
    ax.set_xlabel("Non-penalty goals minus expected goals")
    ax.set_title("Finishing above and below expectation\n"
                 "Premier League, La Liga, Serie A and Ligue 1, 2015/16")
    ax.grid(axis="x", alpha=0.3)
    fig.savefig(ROOT / "figures" / "finishing_2015_16.png", dpi=150, bbox_inches="tight")
    print("\nSaved figures/finishing_2015_16.png and results/player_finishing_2015_16.csv")


if __name__ == "__main__":
    main()
