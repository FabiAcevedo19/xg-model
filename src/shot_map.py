"""Draw every shot of the tournament on a pitch.

Goals are filled circles, misses are hollow. Bigger circles are better
chances according to StatsBomb's own xG model.

Run from the project folder:
    python src/shot_map.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
from mplsoccer import VerticalPitch

from load_data import load_shots

FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"


def draw_shot_map(shots, title, save_as):
    # StatsBomb's pitch is 120 x 80. The goal being attacked is at x = 120.
    pitch = VerticalPitch(pitch_type="statsbomb", half=True,
                          pitch_color="#1b4332", line_color="white")
    fig, ax = pitch.draw(figsize=(9, 7))
    fig.set_facecolor("#1b4332")

    goals = shots[shots["is_goal"] == 1]
    misses = shots[shots["is_goal"] == 0]

    pitch.scatter(misses["x"], misses["y"], s=misses["statsbomb_xg"] * 600 + 20,
                  facecolors="none", edgecolors="white", alpha=0.45,
                  ax=ax, label=f"No goal ({len(misses)})")
    pitch.scatter(goals["x"], goals["y"], s=goals["statsbomb_xg"] * 600 + 20,
                  color="#ffd166", edgecolors="black", alpha=0.9,
                  ax=ax, label=f"Goal ({len(goals)})")

    ax.set_title(title, color="white", fontsize=16, pad=10)
    ax.legend(loc="lower center", ncol=2, frameon=False, labelcolor="white")

    FIGURES_DIR.mkdir(exist_ok=True)
    path = FIGURES_DIR / save_as
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved {path}")
    return fig


if __name__ == "__main__":
    shots = load_shots()

    # Penalties are always taken from the same spot, so leave them out.
    open_play = shots[shots["shot_type"] != "Penalty"]

    # All tournaments on one map would be too crowded, so draw just one.
    world_cup = open_play[open_play["competition"] == "World Cup 2022"]

    draw_shot_map(world_cup, "Every non-penalty shot, World Cup 2022",
                  "shot_map_world_cup_2022.png")
    plt.show()
