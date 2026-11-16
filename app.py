"""Interactive xG explorer.

Place the ball, the goalkeeper and defenders on the pitch and see the goal
probability update, along with the reasons behind it.

Run from the project folder:
    streamlit run app.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import xgboost as xgb

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from features import (DEFENDER_RADIUS, KEEPER_RADIUS, _goal_targets,  # noqa: E402
                      build_features, rays_blocked)

st.set_page_config(page_title="xG Explorer", page_icon="⚽", layout="wide")

GRASS, LINE = "#1b4332", "#f1faee"
BALL, KEEPER, DEFENDER, TEAMMATE = "#ffd166", "#f77f00", "#e63946", "#8ecae6"
N_DRAWN_RAYS = 30

FEATURE_LABELS = {
    "distance": "Distance to goal",
    "angle_deg": "Goal angle",
    "is_header": "Header",
    "is_free_kick": "Free kick",
    "is_volley": "Volley",
    "first_time": "First-time shot",
    "under_pressure": "Under pressure",
    "from_counter": "Counter-attack",
    "open_goal_fraction": "Open share of the goal",
    "free_angle_deg": "Unblocked goal angle",
    "defenders_in_cone": "Defenders in the way",
    "nearest_defender_dist": "Nearest defender",
    "gk_dist_from_goal": "Keeper's distance off the line",
    "gk_off_line": "Keeper out of position",
    "gk_dist_to_shot": "Keeper's distance to the ball",
    "unsaveable_fraction": "Unsaveable share of the goal",
    "scoring_window_deg": "Scoring window",
}

# name -> ball, keeper, defenders, and whether it is a header
PRESETS = {
    "One-on-one from the penalty spot": dict(
        ball=(108, 40), keeper=(118, 40), defenders=[]),
    "Crowded penalty box": dict(
        ball=(104, 38), keeper=(118, 40),
        defenders=[(107, 37), (108, 40), (110, 43), (112, 36), (106, 45)]),
    "Tight angle": dict(
        ball=(114, 57), keeper=(119, 43), defenders=[(116, 52)]),
    "Long-range effort": dict(
        ball=(90, 40), keeper=(118, 40), defenders=[(98, 38), (100, 43), (104, 40)]),
    "Keeper caught out of position": dict(
        ball=(110, 44), keeper=(112, 31), defenders=[(114, 47)]),
    "Header from a cross": dict(
        ball=(113, 43), keeper=(119, 40), defenders=[(114, 41), (113, 46)], header=True),
}
FLAG_KEYS = ["is_header", "is_volley", "first_time", "under_pressure",
             "is_free_kick", "from_counter"]


@st.cache_resource
def load_model():
    model = xgb.XGBClassifier()
    model.load_model(ROOT / "models" / "xg_model.json")
    features = json.loads((ROOT / "results" / "metrics.json").read_text())["final_features"]
    return model, features


@st.cache_data
def load_final_shots():
    shots = pd.read_csv(ROOT / "app_data" / "world_cup_final_2022.csv")
    shots["label"] = (shots["minute"].astype(str) + "' " + shots["player"]
                      + " (" + shots["team"] + "): " + shots["outcome"])
    return shots


# --------------------------------------------------------------------------
# The scene (ball, keeper, defenders) lives in st.session_state
# --------------------------------------------------------------------------

def set_scene(ball, keeper, defenders, teammates=(), real=None, **flags):
    s = st.session_state
    s.ball = tuple(ball)
    s.keeper = tuple(keeper) if keeper is not None else None
    s.defenders = [tuple(d) for d in defenders]
    s.teammates = [tuple(t) for t in teammates]
    s.real = real
    for key in FLAG_KEYS:
        s[key] = bool(flags.get(key, False))


def load_preset():
    preset = dict(PRESETS[st.session_state.preset_choice])
    set_scene(is_header=preset.pop("header", False), **preset)


def load_real_shot():
    shots = load_final_shots()
    shot = shots[shots["label"] == st.session_state.real_choice].iloc[0]
    players = json.loads(shot["freeze_frame"])  # [x, y, is_teammate, is_goalkeeper]
    keepers = [(p[0], p[1]) for p in players if not p[2] and p[3]]
    set_scene(
        ball=(shot["x"], shot["y"]),
        keeper=keepers[0] if keepers else None,
        defenders=[(p[0], p[1]) for p in players if not p[2] and not p[3]],
        teammates=[(p[0], p[1]) for p in players if p[2]],
        real=dict(label=shot["label"], outcome=shot["outcome"],
                  statsbomb_xg=float(shot["statsbomb_xg"])),
        is_header=shot["body_part"] == "Head",
        is_volley=shot["technique"] in ("Volley", "Half Volley"),
        first_time=bool(shot["first_time"]),
        under_pressure=bool(shot["under_pressure"]),
        is_free_kick=shot["shot_type"] == "Free Kick",
        from_counter=shot["play_pattern"] == "From Counter",
    )


if "ball" not in st.session_state:
    set_scene(**PRESETS["One-on-one from the penalty spot"])
    st.session_state.clicks = 0


# --------------------------------------------------------------------------
# Model prediction for the current scene
# --------------------------------------------------------------------------

def predict(model, features):
    s = st.session_state
    frame = [[x, y, 0, 0] for x, y in s.defenders]
    if s.keeper:
        frame.append([s.keeper[0], s.keeper[1], 0, 1])
    shot = pd.DataFrame([{
        "x": s.ball[0], "y": s.ball[1],
        "body_part": "Head" if s.is_header else "Right Foot",
        "shot_type": "Free Kick" if s.is_free_kick else "Open Play",
        "technique": "Volley" if s.is_volley else "Normal",
        "first_time": s.first_time,
        "under_pressure": s.under_pressure,
        "play_pattern": "From Counter" if s.from_counter else "Regular Play",
        "freeze_frame": json.dumps(frame),
    }])
    X = build_features(shot)[features]
    xg = float(model.predict_proba(X)[0, 1])
    # SHAP values: how much each feature pushed this prediction up or down.
    contributions = model.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)[0]
    return xg, X.iloc[0], pd.Series(contributions[:-1], index=features)


# --------------------------------------------------------------------------
# Drawing
# --------------------------------------------------------------------------

def pitch_figure():
    s = st.session_state
    fig = go.Figure()
    line = dict(color=LINE, width=2)

    # The pitch is drawn rotated so the goal is at the top of the screen:
    # the data's y (across the pitch) is the chart's horizontal axis, and the
    # data's x (towards goal) is the chart's vertical axis.
    for x0, y0, x1, y1 in [(60, 0, 120, 80),      # half of the pitch
                           (102, 18, 120, 62),    # penalty area
                           (114, 30, 120, 50),    # six-yard box
                           (120, 36, 122, 44)]:   # goal
        fig.add_shape(type="rect", x0=y0, y0=x0, x1=y1, y1=x1, line=line, layer="below")
    theta = np.linspace(np.arccos(-0.6), 2 * np.pi - np.arccos(-0.6), 40)
    fig.add_trace(go.Scatter(x=40 + 10 * np.sin(theta), y=108 + 10 * np.cos(theta),
                             mode="lines", line=line, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=[40], y=[108], mode="markers", hoverinfo="skip",
                             marker=dict(color=LINE, size=5)))

    # Rays from the ball to the goal: bright if clear, faint if blocked.
    blockers = list(s.defenders) + ([s.keeper] if s.keeper else [])
    radii = [DEFENDER_RADIUS] * len(s.defenders) + ([KEEPER_RADIUS] if s.keeper else [])
    blocked = rays_blocked(s.ball, blockers, radii, N_DRAWN_RAYS)
    targets = _goal_targets(N_DRAWN_RAYS)
    for is_blocked, color in [(True, "rgba(255,255,255,0.10)"), (False, "rgba(255,209,102,0.55)")]:
        xs, ys = [], []
        for target in targets[blocked == is_blocked]:
            xs += [s.ball[1], target[1], None]
            ys += [s.ball[0], target[0], None]
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", hoverinfo="skip",
                                 line=dict(color=color, width=1)))

    # An invisible grid of points covering the pitch. Clicking one tells us
    # where on the pitch the click landed.
    gx, gy = np.meshgrid(np.arange(61, 120.5, 1.0), np.arange(1, 80, 1.0))
    fig.add_trace(go.Scatter(x=gy.ravel(), y=gx.ravel(), mode="markers",
                             marker=dict(size=9, color="rgba(0,0,0,0)"),
                             hoverinfo="none", name="grid"))

    def players(points, color, name, size=17, symbol="circle"):
        if points:
            fig.add_trace(go.Scatter(
                x=[p[1] for p in points], y=[p[0] for p in points], mode="markers",
                name=name, hoverinfo="name",
                marker=dict(size=size, color=color, symbol=symbol,
                            line=dict(color="black", width=1.5))))

    players(s.teammates, TEAMMATE, "Teammate", size=13)
    players(s.defenders, DEFENDER, "Defender")
    players([s.keeper] if s.keeper else [], KEEPER, "Goalkeeper", size=19, symbol="square")
    players([s.ball], BALL, "Ball", size=14)

    # Keep every marker fully visible after a click selects one point.
    fig.update_traces(selector=dict(mode="markers"),
                      unselected=dict(marker=dict(opacity=1)))
    fig.update_layout(
        height=480, margin=dict(l=5, r=5, t=5, b=5), showlegend=False,
        plot_bgcolor=GRASS, paper_bgcolor=GRASS, dragmode=False, clickmode="event+select",
        xaxis=dict(range=[-0.5, 80.5], visible=False, fixedrange=True, constrain="domain"),
        yaxis=dict(range=[59.5, 123], visible=False, fixedrange=True,
                   scaleanchor="x", scaleratio=1, constrain="domain"),
    )
    return fig


def explanation_figure(contributions):
    top = contributions[contributions.abs().sort_values().index[-7:]]
    fig = go.Figure(go.Bar(
        x=top.values, y=[FEATURE_LABELS.get(f, f) for f in top.index], orientation="h",
        marker_color=[DEFENDER if v < 0 else "#2a9d8f" for v in top.values],
        hovertemplate="%{y}: %{x:+.2f}<extra></extra>"))
    fig.update_layout(height=290, margin=dict(l=5, r=10, t=5, b=30),
                      xaxis_title="Lowers xG  ←      →  Raises xG",
                      xaxis=dict(zeroline=True, zerolinecolor="grey", fixedrange=True),
                      yaxis=dict(fixedrange=True))
    return fig


def handle_click(mode, x, y):
    s = st.session_state
    s.real = None       # the scene no longer matches the real shot
    s.teammates = []
    if mode == "Move the ball":
        s.ball = (x, y)
    elif mode == "Move the keeper":
        s.keeper = (x, y)
    elif mode == "Add a defender":
        s.defenders.append((x, y))
    elif mode == "Remove a defender" and s.defenders:
        nearest = min(s.defenders, key=lambda d: (d[0] - x) ** 2 + (d[1] - y) ** 2)
        s.defenders.remove(nearest)


# --------------------------------------------------------------------------
# Page
# --------------------------------------------------------------------------

model, features = load_model()

st.title("⚽ xG Explorer")
st.caption("How likely is this shot to score? Set up a chance on the pitch and an "
           "XGBoost model trained on 45,000 professional shots gives its answer. "
           "The goal is at the top.")

with st.sidebar:
    st.header("Start from")
    st.selectbox("A preset situation", list(PRESETS), key="preset_choice",
                 on_change=load_preset)
    st.selectbox("A real shot from the 2022 World Cup final",
                 load_final_shots()["label"], key="real_choice", index=None,
                 placeholder="Choose a shot", on_change=load_real_shot)

    st.header("The shot")
    st.toggle("Header", key="is_header")
    st.toggle("Volley", key="is_volley")
    st.toggle("First-time shot", key="first_time")
    st.toggle("Under pressure", key="under_pressure")
    st.toggle("Free kick", key="is_free_kick")
    st.toggle("Counter-attack", key="from_counter")

    st.divider()
    st.caption("Data: StatsBomb Open Data. "
               "[Code on GitHub](https://github.com/FabiAcevedo19/xg-model)")

xg, values, contributions = predict(model, features)
pitch_col, info_col = st.columns([3, 2], gap="large")

with pitch_col:
    mode = st.radio("Click on the pitch to", ["Move the ball", "Move the keeper",
                                              "Add a defender", "Remove a defender"],
                    horizontal=True)
    event = st.plotly_chart(pitch_figure(), on_select="rerun", selection_mode="points",
                            key=f"pitch_{st.session_state.clicks}",
                            config={"displayModeBar": False})
    points = event.selection.points if event and event.selection else []
    if points:
        # The chart is rotated, so its vertical axis is the pitch's x.
        handle_click(mode, float(points[0]["y"]), float(points[0]["x"]))
        st.session_state.clicks += 1  # a new chart key clears the selection
        st.rerun()
    st.caption("🟡 ball   🟧 goalkeeper   🔴 defenders   🔵 teammates (shown for real shots)   "
               "Yellow lines are clear paths to goal; faint lines are blocked.")

with info_col:
    st.metric("Expected goals (xG)", f"{xg:.2f}")
    st.progress(min(xg, 1.0))
    st.write(f"A shot like this scores about **1 time in {max(1, round(1 / xg))}**.")

    real = st.session_state.real
    if real:
        a, b = st.columns(2)
        a.metric("StatsBomb's xG", f"{real['statsbomb_xg']:.2f}")
        b.metric("What happened", real["outcome"])

    a, b, c = st.columns(3)
    a.metric("Distance", f"{values['distance']:.0f} yd")
    b.metric("Goal angle", f"{values['angle_deg']:.0f}°")
    c.metric("Goal open", f"{values['open_goal_fraction']:.0%}")

    st.subheader("Why this number?")
    st.plotly_chart(explanation_figure(contributions), config={"displayModeBar": False})
    st.caption("Each bar is one input's push on this prediction (SHAP values, in "
               "log-odds). Green raises the chance of a goal, red lowers it.")
