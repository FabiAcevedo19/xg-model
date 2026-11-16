"""Download shot data from StatsBomb's free open-data release.

StatsBomb records every "event" in a match (passes, tackles, shots...).
We only keep the shots, one row per shot, and save them to data/shots.csv
so we don't have to download them again.

Run from the project folder:
    python src/load_data.py
"""

import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

# The data is a public GitHub repository of JSON files:
#   matches/<competition>/<season>.json   the list of matches
#   events/<match_id>.json                everything that happened in a match
BASE_URL = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SHOTS_FILE = DATA_DIR / "shots.csv"

# (competition_id, season_id) pairs. The full list is in competitions.json.
#
# We only use competitions where StatsBomb released EVERY match. Their free
# data also has hundreds of Barcelona-only matches (the "Messi data"), but a
# model trained on those would mostly learn how Messi finishes.
#
# International tournaments. 2010, 2014 and 2026 World Cups are not free.
# Club football: complete 2015/16 seasons of four big European leagues.
# Delete a line here (and delete data/shots.csv) to leave a competition out.
COMPETITIONS = {
    "World Cup 2022": (43, 106),
    "World Cup 2018": (43, 3),
    "Euro 2024": (55, 282),
    "Euro 2020": (55, 43),
    "Copa America 2024": (223, 282),
    "African Cup of Nations 2023": (1267, 107),
    "Premier League 2015/16": (2, 27),
    "La Liga 2015/16": (11, 27),
    "Serie A 2015/16": (12, 27),
    "Ligue 1 2015/16": (7, 27),
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


def _get_json(path: str):
    """Fetch one JSON file from the open-data repository, retrying on errors."""
    for attempt in range(4):
        try:
            response = requests.get(f"{BASE_URL}/{path}", timeout=60)
            response.raise_for_status()
            return response.json()
        except requests.RequestException:
            if attempt == 3:
                raise
            time.sleep(2 * (attempt + 1))


def _name(event_part, key):
    """StatsBomb stores labels as {"id": 87, "name": "Open Play"}. Get the name."""
    return (event_part.get(key) or {}).get("name")


def _shots_from_match(match_id: int) -> list[dict]:
    """Download one match (about 3,500 events) and keep just its shots."""
    rows = []
    for event in _get_json(f"events/{match_id}.json"):
        if event["type"]["name"] != "Shot":
            continue
        shot = event["shot"]
        rows.append({
            "match_id": match_id,
            "period": event["period"],  # 1-2 normal time, 3-4 extra time, 5 shootout
            "minute": event["minute"],
            "team": _name(event, "team"),
            "player": _name(event, "player"),
            "outcome": _name(shot, "outcome"),
            "body_part": _name(shot, "body_part"),
            "shot_type": _name(shot, "type"),  # Open Play, Penalty, Free Kick, Corner
            "technique": _name(shot, "technique"),
            "under_pressure": bool(event.get("under_pressure", False)),
            "first_time": bool(shot.get("first_time", False)),
            "statsbomb_xg": shot["statsbomb_xg"],  # StatsBomb's own model: our benchmark
            "play_pattern": _name(event, "play_pattern"),
            # Where every nearby player stood when the shot was hit.
            "freeze_frame": _freeze_frame_to_text(shot.get("freeze_frame")),
            "x": event["location"][0],
            "y": event["location"][1],
        })
    return rows


def download_shots() -> pd.DataFrame:
    """Download every shot from the competitions listed above."""
    all_shots = []
    for name, (competition_id, season_id) in COMPETITIONS.items():
        matches = _get_json(f"matches/{competition_id}/{season_id}.json")
        match_ids = [m["match_id"] for m in matches]
        # Download 16 matches at a time instead of one by one.
        with ThreadPoolExecutor(max_workers=16) as pool:
            per_match = list(pool.map(_shots_from_match, match_ids))
        shots = pd.DataFrame([row for rows in per_match for row in rows])
        shots["competition"] = name
        all_shots.append(shots)
        print(f"{name}: {len(match_ids)} matches, {len(shots)} shots", flush=True)

    df = pd.concat(all_shots, ignore_index=True)

    # The thing we want to predict: 1 if the shot was a goal, 0 if not.
    df["is_goal"] = (df["outcome"] == "Goal").astype(int)
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
    print(f"\n{len(shots)} shots, {shots['is_goal'].sum()} goals "
          f"({shots['is_goal'].mean():.1%} of shots score)")
    print("\nShots per competition:")
    print(shots.groupby("competition")["is_goal"].agg(shots="count", goals="sum"))
    has_frame = (shots["freeze_frame"] != "[]").mean()
    print(f"\nShots with player positions: {has_frame:.1%}")
