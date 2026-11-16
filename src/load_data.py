"""Download shot data from StatsBomb's free open-data release.

StatsBomb records every "event" in a match (passes, tackles, shots...).
We only keep the shots, one row per shot, and save them to data/shots.csv
so we don't have to download them again.

Run from the project folder:
    python src/load_data.py
"""

import json
import warnings
from pathlib import Path

import pandas as pd
from statsbombpy import sb

# statsbombpy warns that we're using the free data without a login. That's fine.
warnings.filterwarnings("ignore", message="credentials were not supplied")

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SHOTS_FILE = DATA_DIR / "shots.csv"

# (competition_id, season_id) pairs. Find more with sb.competitions().
#
# StatsBomb's free data only has two recent World Cups in full (2018, 2022).
# 2010, 2014 and 2026 are not in the free release. To get more shots we add
# the other big men's international tournaments from 2020 to 2024.
# Delete a line here (and delete data/shots.csv) to leave a tournament out.
COMPETITIONS = {
    "World Cup 2022": (43, 106),
    "World Cup 2018": (43, 3),
    "Euro 2024": (55, 282),
    "Euro 2020": (55, 43),
    "Copa America 2024": (223, 282),
    "African Cup of Nations 2023": (1267, 107),
}

# The columns we keep, renamed to something shorter.
COLUMNS = {
    "match_id": "match_id",
    "period": "period",  # 1-2 normal time, 3-4 extra time, 5 penalty shootout
    "minute": "minute",
    "team": "team",
    "player": "player",
    "shot_outcome": "outcome",
    "shot_body_part": "body_part",
    "shot_type": "shot_type",  # Open Play, Penalty, Free Kick, Corner
    "shot_technique": "technique",
    "under_pressure": "under_pressure",
    "shot_first_time": "first_time",
    "shot_statsbomb_xg": "statsbomb_xg",  # StatsBomb's own model: our benchmark
    "play_pattern": "play_pattern",  # Regular Play, From Counter, From Corner...
    # The "freeze frame": where every nearby player stood when the shot was hit.
    "shot_freeze_frame": "freeze_frame",
}


def _freeze_frame_to_text(frame) -> str:
    """Turn a freeze frame into compact JSON text so it fits in a CSV cell.

    We keep only what we need: [x, y, is_teammate, is_goalkeeper] per player.
    """
    if not isinstance(frame, list):
        return "[]"
    return json.dumps([
        [p["location"][0], p["location"][1], int(p["teammate"]),
         int(p["position"]["name"] == "Goalkeeper")]
        for p in frame
    ])


def download_shots() -> pd.DataFrame:
    """Download every shot from the competitions listed above."""
    all_shots = []
    for name, (competition_id, season_id) in COMPETITIONS.items():
        matches = sb.matches(competition_id=competition_id, season_id=season_id)
        print(f"{name}: {len(matches)} matches")

        for i, match_id in enumerate(matches["match_id"], start=1):
            events = sb.events(match_id=match_id)
            shots = events[events["type"] == "Shot"].copy()

            # "location" is a list [x, y]. Split it into two number columns.
            shots["x"] = shots["location"].str[0]
            shots["y"] = shots["location"].str[1]

            # Some columns only exist if the thing happened in that match.
            for col in COLUMNS:
                if col not in shots.columns:
                    shots[col] = None

            shots = shots[list(COLUMNS) + ["x", "y"]].rename(columns=COLUMNS)
            shots["freeze_frame"] = shots["freeze_frame"].map(_freeze_frame_to_text)
            shots["competition"] = name
            all_shots.append(shots)
            print(f"  match {i}/{len(matches)}: {len(shots)} shots", end="\r")
        print()

    df = pd.concat(all_shots, ignore_index=True)

    # The thing we want to predict: 1 if the shot was a goal, 0 if not.
    df["is_goal"] = (df["outcome"] == "Goal").astype(int)

    # Missing means "no", so fill blanks with False.
    for col in ["under_pressure", "first_time"]:
        df[col] = df[col].fillna(False).astype(bool)

    return df


def load_shots() -> pd.DataFrame:
    """Return the shots table, downloading it the first time only."""
    if SHOTS_FILE.exists():
        return pd.read_csv(SHOTS_FILE)

    df = download_shots()
    DATA_DIR.mkdir(exist_ok=True)
    df.to_csv(SHOTS_FILE, index=False)
    print(f"Saved {len(df)} shots to {SHOTS_FILE}")
    return df


if __name__ == "__main__":
    shots = load_shots()
    print(shots.drop(columns="freeze_frame").head())
    print(f"\n{len(shots)} shots, {shots['is_goal'].sum()} goals "
          f"({shots['is_goal'].mean():.1%} of shots score)")
    print("\nShots per tournament:")
    print(shots.groupby("competition")["is_goal"].agg(shots="count", goals="sum"))
