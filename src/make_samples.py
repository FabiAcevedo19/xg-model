"""Save the shots of the 2022 World Cup final for the app to replay.

The full dataset is too big to commit, so the app ships with this one match.

Run from the project folder (after load_data.py):
    python src/make_samples.py
"""

from pathlib import Path

from load_data import load_shots

OUT_FILE = Path(__file__).resolve().parent.parent / "app_data" / "world_cup_final_2022.csv"

COLUMNS = ["minute", "period", "team", "player", "outcome", "body_part",
           "shot_type", "technique", "under_pressure", "first_time",
           "play_pattern", "statsbomb_xg", "x", "y", "freeze_frame"]

if __name__ == "__main__":
    shots = load_shots()
    world_cup = shots[shots["competition"] == "World Cup 2022"]

    # The final is the only match where both Argentina and France took shots.
    teams_per_match = world_cup.groupby("match_id")["team"].agg(set)
    final_id = teams_per_match[teams_per_match == {"Argentina", "France"}].index[0]

    final = world_cup[(world_cup["match_id"] == final_id)
                      & (world_cup["shot_type"] != "Penalty")]
    final = final.sort_values(["period", "minute"])[COLUMNS]

    OUT_FILE.parent.mkdir(exist_ok=True)
    final.to_csv(OUT_FILE, index=False)
    print(f"Saved {len(final)} shots to {OUT_FILE}")
    print(final[["minute", "team", "player", "outcome", "statsbomb_xg"]].to_string(index=False))
