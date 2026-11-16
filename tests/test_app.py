"""Smoke tests for the Streamlit app: it starts, and the main controls work.

These need the full requirements (streamlit, xgboost, plotly), so they are
skipped automatically where those are not installed.
"""

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
pytest.importorskip("xgboost")
pytest.importorskip("plotly")

from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def xg(app):
    return float(app.metric[0].value)


def test_app_starts_with_a_sensible_prediction():
    app = AppTest.from_file(APP, default_timeout=60).run()
    assert not app.exception
    assert 0 < xg(app) < 1


def test_every_preset_loads():
    app = AppTest.from_file(APP, default_timeout=60).run()
    for name in app.selectbox(key="preset_choice").options:
        app.selectbox(key="preset_choice").select(name).run()
        assert not app.exception
        assert 0 < xg(app) < 1


def test_long_range_is_harder_than_a_one_on_one():
    app = AppTest.from_file(APP, default_timeout=60).run()
    close = xg(app)
    app.selectbox(key="preset_choice").select("Long-range effort").run()
    assert xg(app) < close


def test_header_lowers_xg():
    app = AppTest.from_file(APP, default_timeout=60).run()
    foot = xg(app)
    app.toggle(key="is_header").set_value(True).run()
    assert xg(app) < foot


def test_every_real_shot_loads():
    app = AppTest.from_file(APP, default_timeout=60).run()
    for label in app.selectbox(key="real_choice").options:
        app.selectbox(key="real_choice").select(label).run()
        assert not app.exception
        assert 0 <= xg(app) < 1  # a very poor chance is displayed as 0.00
        # Real shots also show StatsBomb's xG and the actual outcome.
        assert "StatsBomb's xG" in [m.label for m in app.metric]
