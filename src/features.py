"""Feature engineering: turn raw shot data into numbers a model can learn from.

A model can't do much with raw (x, y) coordinates. What matters physically is:

  1. Shot geometry      how far is the goal, and how wide does it look?
  2. Defensive geometry who is standing between the ball and the goal?
  3. Goalkeeper         where is the keeper relative to the shot line?

Everything here is plain geometry on a 120 x 80 pitch (StatsBomb units, roughly
yards). The goal being attacked is centred at (120, 40) and is 8 wide.
"""

import json

import numpy as np
import pandas as pd

GOAL_X = 120.0
GOAL_Y = 40.0
GOAL_WIDTH = 8.0
GOAL_CENTRE = np.array([GOAL_X, GOAL_Y])
LEFT_POST = np.array([GOAL_X, GOAL_Y - GOAL_WIDTH / 2])
RIGHT_POST = np.array([GOAL_X, GOAL_Y + GOAL_WIDTH / 2])

# How much of a shot's path a player's body blocks, as a radius around them.
# A keeper covers more than an outfield player because they can use their arms.
DEFENDER_RADIUS = 0.5
KEEPER_RADIUS = 1.0
N_RAYS = 100

BASIC_FEATURES = [
    "distance", "angle_deg", "is_header", "is_free_kick", "is_volley",
    "first_time", "under_pressure", "from_counter",
]
FREEZE_FRAME_FEATURES = [
    "open_goal_fraction", "free_angle_deg", "defenders_in_cone",
    "nearest_defender_dist", "gk_dist_from_goal", "gk_off_line",
    "gk_dist_to_shot",
]
ALL_FEATURES = BASIC_FEATURES + FREEZE_FRAME_FEATURES

# Physics we know before seeing any data: +1 means "more of this can only
# raise the goal probability", -1 means it can only lower it. The XGBoost
# model is forced to respect these, which stops it learning noise.
MONOTONE = {
    "distance": -1,
    "angle_deg": +1,
    "open_goal_fraction": +1,
    "free_angle_deg": +1,
    "defenders_in_cone": -1,
}


# --------------------------------------------------------------------------
# 1. Shot geometry
# --------------------------------------------------------------------------

def shot_distance(x, y):
    """Straight-line distance from the shot to the centre of the goal."""
    return np.hypot(GOAL_X - x, GOAL_Y - y)


def shot_angle(x, y):
    """Angle (radians) between the two goalposts as seen from the shot.

    This is how "wide" the goal looks to the shooter. From the penalty spot
    it is about 37 degrees; from a tight angle near the byline it is close
    to zero even though the goal is near.

    Derivation: with a = vector to the left post and b = vector to the right
    post, tan(angle) = |a x b| / (a . b). Working that out gives the formula.
    """
    dx = GOAL_X - np.asarray(x, dtype=float)
    dy = np.asarray(y, dtype=float) - GOAL_Y
    angle = np.arctan2(GOAL_WIDTH * dx, dx**2 + dy**2 - (GOAL_WIDTH / 2) ** 2)
    return np.where(angle < 0, angle + np.pi, angle)


# --------------------------------------------------------------------------
# 2. Defensive geometry
# --------------------------------------------------------------------------

def _cross(o, a, b):
    """2D cross product of (a - o) and (b - o). Its sign says which side b is on."""
    return (a[..., 0] - o[..., 0]) * (b[..., 1] - o[..., 1]) - \
           (a[..., 1] - o[..., 1]) * (b[..., 0] - o[..., 0])


def in_shot_cone(points, shot):
    """True for each point inside the triangle shot / left post / right post."""
    points = np.atleast_2d(points)
    d1 = _cross(shot, LEFT_POST, points)
    d2 = _cross(LEFT_POST, RIGHT_POST, points)
    d3 = _cross(RIGHT_POST, shot, points)
    has_neg = (d1 < 0) | (d2 < 0) | (d3 < 0)
    has_pos = (d1 > 0) | (d2 > 0) | (d3 > 0)
    return ~(has_neg & has_pos)


def _distance_to_segments(points, start, ends):
    """Distance from each point to each line segment start -> end.

    points: (k, 2) player positions.  ends: (n, 2) segment end points.
    Returns a (k, n) array.
    """
    d = ends - start                                   # (n, 2) segment vectors
    length_sq = np.maximum((d**2).sum(axis=1), 1e-9)   # (n,)
    rel = points - start                               # (k, 2)
    # t = how far along the segment the closest point is, clipped to [0, 1]
    t = np.clip(rel @ d.T / length_sq, 0.0, 1.0)       # (k, n)
    closest = start + t[..., None] * d                 # (k, n, 2)
    return np.linalg.norm(points[:, None, :] - closest, axis=2)


def open_goal_fraction(shot, blockers, radii, n_rays=N_RAYS):
    """Fraction of the goal mouth the shooter has a clear line to (0 to 1).

    Ray casting: draw n_rays straight lines from the ball to evenly spaced
    points between the posts. A ray is blocked if any opposing player's body
    (a circle of the given radius) touches it. 1.0 means nothing in the way.
    """
    blockers = np.asarray(blockers, dtype=float).reshape(-1, 2)
    if len(blockers) == 0:
        return 1.0
    targets = np.column_stack([
        np.full(n_rays, GOAL_X),
        np.linspace(LEFT_POST[1], RIGHT_POST[1], n_rays),
    ])
    dist = _distance_to_segments(blockers, np.asarray(shot, dtype=float), targets)
    blocked = (dist < np.asarray(radii, dtype=float)[:, None]).any(axis=0)
    return 1.0 - blocked.mean()


def freeze_frame_features(x, y, frame_text):
    """All defender and goalkeeper features for one shot.

    frame_text is JSON: a list of [x, y, is_teammate, is_goalkeeper].
    """
    shot = np.array([x, y], dtype=float)
    players = np.array(json.loads(frame_text), dtype=float).reshape(-1, 4)
    opponents = players[players[:, 2] == 0]
    defenders = opponents[opponents[:, 3] == 0][:, :2]
    keepers = opponents[opponents[:, 3] == 1][:, :2]

    blockers = np.vstack([defenders, keepers])
    radii = np.r_[np.full(len(defenders), DEFENDER_RADIUS),
                  np.full(len(keepers), KEEPER_RADIUS)]
    open_fraction = open_goal_fraction(shot, blockers, radii)

    out = {
        "open_goal_fraction": open_fraction,
        "free_angle_deg": float(np.degrees(shot_angle(x, y))) * open_fraction,
        "defenders_in_cone": int(in_shot_cone(defenders, shot).sum()) if len(defenders) else 0,
        "nearest_defender_dist": np.nan,
        "gk_dist_from_goal": np.nan,
        "gk_off_line": np.nan,
        "gk_dist_to_shot": np.nan,
    }
    if len(defenders):
        out["nearest_defender_dist"] = np.linalg.norm(defenders - shot, axis=1).min()
    if len(keepers):
        gk = keepers[0]
        out["gk_dist_from_goal"] = np.linalg.norm(gk - GOAL_CENTRE)
        out["gk_dist_to_shot"] = np.linalg.norm(gk - shot)
        # How far the keeper is from the line ball -> centre of goal.
        # 0 means perfectly positioned; large means caught out of position.
        out["gk_off_line"] = _distance_to_segments(
            gk[None, :], shot, GOAL_CENTRE[None, :])[0, 0]
    return out


# --------------------------------------------------------------------------
# Put it all together
# --------------------------------------------------------------------------

def build_features(shots: pd.DataFrame) -> pd.DataFrame:
    """Return one row of model features per shot."""
    X = pd.DataFrame(index=shots.index)

    X["distance"] = shot_distance(shots["x"], shots["y"])
    X["angle_deg"] = np.degrees(shot_angle(shots["x"], shots["y"]))

    X["is_header"] = (shots["body_part"] == "Head").astype(int)
    X["is_free_kick"] = (shots["shot_type"] == "Free Kick").astype(int)
    X["is_volley"] = shots["technique"].isin(["Volley", "Half Volley"]).astype(int)
    X["first_time"] = shots["first_time"].astype(int)
    X["under_pressure"] = shots["under_pressure"].astype(int)
    X["from_counter"] = (shots["play_pattern"] == "From Counter").astype(int)

    frames = pd.DataFrame(
        [freeze_frame_features(x, y, f)
         for x, y, f in zip(shots["x"], shots["y"], shots["freeze_frame"])],
        index=shots.index,
    )
    return pd.concat([X, frames], axis=1)[ALL_FEATURES]


if __name__ == "__main__":
    from load_data import load_shots

    shots = load_shots()
    X = build_features(shots)
    pd.set_option("display.width", 200)
    print(X.describe().T[["mean", "min", "max"]].round(2))
