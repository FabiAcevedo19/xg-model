"""Checks that the geometry is right, using cases we can work out by hand.

Run from the project folder:
    python -m pytest
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from features import (freeze_frame_features, in_shot_cone, open_goal_fraction,
                      shot_angle, shot_distance)

PENALTY_SPOT = (108.0, 40.0)  # 12 yards from the goal line, dead centre


def test_distance_from_penalty_spot():
    assert shot_distance(*PENALTY_SPOT) == pytest.approx(12.0)


def test_angle_from_penalty_spot():
    # Two right triangles with sides 4 and 12: angle = 2 * atan(4 / 12)
    assert np.degrees(shot_angle(*PENALTY_SPOT)) == pytest.approx(
        np.degrees(2 * np.arctan(4 / 12)))


def test_angle_shrinks_from_the_side():
    central = shot_angle(108, 40)
    wide = shot_angle(108, 15)
    assert wide < central


def test_angle_on_the_byline_is_zero():
    assert float(shot_angle(120, 10)) == pytest.approx(0.0, abs=1e-9)


def test_open_goal_with_nobody_there():
    assert open_goal_fraction(PENALTY_SPOT, [], []) == 1.0


def test_defender_right_in_front_blocks_everything():
    # One yard in front of the ball, a 0.5 radius body covers the whole goal.
    assert open_goal_fraction(PENALTY_SPOT, [[109.0, 40.0]], [0.5]) == 0.0


def test_defender_behind_the_shooter_blocks_nothing():
    assert open_goal_fraction(PENALTY_SPOT, [[100.0, 40.0]], [0.5]) == 1.0


def test_keeper_on_the_line_blocks_part_of_the_goal():
    # A radius of 1 on an 8 wide goal blocks about a quarter of it.
    fraction = open_goal_fraction(PENALTY_SPOT, [[119.5, 40.0]], [1.0])
    assert 0.65 < fraction < 0.85


def test_cone_membership():
    shot = np.array(PENALTY_SPOT)
    points = np.array([[114.0, 40.0],   # between ball and goal
                       [114.0, 20.0],   # far out wide
                       [100.0, 40.0]])  # behind the shooter
    assert in_shot_cone(points, shot).tolist() == [True, False, False]


def test_freeze_frame_features():
    frame = json.dumps([
        [114.0, 40.0, 0, 0],   # defender in the cone
        [114.0, 20.0, 0, 0],   # defender out wide
        [119.0, 41.0, 0, 1],   # goalkeeper, slightly off centre
        [110.0, 44.0, 1, 0],   # teammate (ignored)
    ])
    f = freeze_frame_features(*PENALTY_SPOT, frame)
    assert f["defenders_in_cone"] == 1
    assert f["nearest_defender_dist"] == pytest.approx(6.0)
    assert f["gk_dist_to_shot"] == pytest.approx(np.hypot(11, 1))
    assert 0 < f["gk_off_line"] < 1.01
    assert 0 <= f["open_goal_fraction"] < 1


def test_empty_freeze_frame():
    f = freeze_frame_features(*PENALTY_SPOT, "[]")
    assert f["open_goal_fraction"] == 1.0
    assert f["defenders_in_cone"] == 0
    assert np.isnan(f["gk_dist_from_goal"])
