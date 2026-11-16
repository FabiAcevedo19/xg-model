"""Train several xG models, compare them fairly, and save the best one.

Run from the project folder:
    python src/train.py

How the evaluation is kept honest
---------------------------------
1. Held-out test set. Every shot from the 2022 World Cup is locked away.
   No model is trained on it and no setting is tuned on it. It is used
   once, at the end, to report the final score.
2. Cross-validation on the rest (the "development" data), to compare models
   and choose XGBoost's settings. The data is split into 5 parts; a model
   trains on 4 and predicts the 5th, so every prediction is for a shot the
   model has not seen.
3. Grouped by match. All shots from one match stay in the same part, because
   shots from the same match are related (same teams, same keeper).
4. Benchmark. StatsBomb's commercial xG is scored on exactly the same shots.
5. Confidence intervals. The test score is recomputed on 2,000 random
   re-samplings of the test matches (a "bootstrap") to show how much of the
   gap to StatsBomb could be luck.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # save figures to files without opening windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.calibration import calibration_curve
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import (ALL_FEATURES, BASIC_FEATURES, FREEZE_FRAME_FEATURES,
                      MONOTONE, build_features)
from load_data import load_shots

ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = ROOT / "figures"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

TEST_COMPETITION = "World Cup 2022"
STATIC_FEATURES = BASIC_FEATURES + FREEZE_FRAME_FEATURES  # no keeper kinematics

# XGBoost settings to try. More trees and deeper trees can learn finer
# patterns but can also memorise noise; cross-validation picks the winner.
PARAM_GRID = [
    dict(n_estimators=300, learning_rate=0.03, max_depth=3, min_child_weight=10),
    dict(n_estimators=600, learning_rate=0.03, max_depth=3, min_child_weight=10),
    dict(n_estimators=600, learning_rate=0.03, max_depth=4, min_child_weight=10),
    dict(n_estimators=1000, learning_rate=0.02, max_depth=4, min_child_weight=20),
    dict(n_estimators=800, learning_rate=0.03, max_depth=5, min_child_weight=20),
    dict(n_estimators=1500, learning_rate=0.02, max_depth=5, min_child_weight=30),
]


def logistic():
    """Logistic regression: a straight-line model squashed into 0..1."""
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(max_iter=1000))


def boosted_trees(features, params):
    """XGBoost: many small decision trees, each fixing the last one's errors."""
    return xgb.XGBClassifier(
        subsample=0.8, colsample_bytree=0.8,
        monotone_constraints=tuple(MONOTONE.get(f, 0) for f in features),
        eval_metric="logloss", random_state=0, n_jobs=4, **params,
    )


def score(y, p):
    return {
        "log_loss": log_loss(y, p),        # punishes confident wrong answers. Lower is better.
        "brier": brier_score_loss(y, p),   # mean squared error of the probability. Lower is better.
        "roc_auc": roc_auc_score(y, p),    # can it rank goals above misses? Higher is better.
    }


def cross_validate(model, X, y, matches):
    return cross_val_predict(model, X, y, groups=matches, cv=GroupKFold(n_splits=5),
                             method="predict_proba")[:, 1]


def bootstrap_log_loss(y, p_ours, p_ref, matches, n_boot=2000, seed=0):
    """95% intervals for our log-loss, the reference's, and the difference.

    Whole matches are re-sampled (not single shots), for the same reason the
    cross-validation is grouped by match.
    """
    y = np.asarray(y)

    def per_shot_loss(p):
        p = np.clip(np.asarray(p, dtype=float), 1e-7, 1 - 1e-7)
        return -(y * np.log(p) + (1 - y) * np.log(1 - p))

    frame = pd.DataFrame({"m": np.asarray(matches), "ours": per_shot_loss(p_ours),
                          "ref": per_shot_loss(p_ref), "n": 1})
    per_match = frame.groupby("m").sum().to_numpy()   # columns: ours, ref, n
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(per_match), size=(n_boot, len(per_match)))
    sums = per_match[picks].sum(axis=1)               # (n_boot, 3)
    ours, ref = sums[:, 0] / sums[:, 2], sums[:, 1] / sums[:, 2]

    def interval(v):
        return [round(float(np.percentile(v, 2.5)), 4),
                round(float(np.percentile(v, 97.5)), 4)]

    return {"ours": interval(ours), "statsbomb": interval(ref),
            "difference": interval(ours - ref),
            "share_of_resamples_we_win": round(float((ours < ref).mean()), 3)}


def main():
    shots = load_shots()
    # Penalties are a fixed situation (about 76% score), so a model adds nothing.
    shots = shots[shots["shot_type"] != "Penalty"].reset_index(drop=True)

    print(f"Building features for {len(shots)} non-penalty shots...", flush=True)
    X = build_features(shots)
    y = shots["is_goal"]

    is_test = shots["competition"] == TEST_COMPETITION
    X_dev, y_dev, shots_dev = X[~is_test], y[~is_test], shots[~is_test]
    X_test, y_test, shots_test = X[is_test], y[is_test], shots[is_test]
    matches_dev = shots_dev["match_id"]
    print(f"Development: {len(X_dev)} shots, {matches_dev.nunique()} matches. "
          f"Held-out test ({TEST_COMPETITION}): {len(X_test)} shots.", flush=True)

    # ---- Step 1: choose XGBoost's settings by cross-validation ------------
    print("\nTuning XGBoost on the development data:")
    tuning = []
    for params in PARAM_GRID:
        p = cross_validate(boosted_trees(ALL_FEATURES, params),
                           X_dev[ALL_FEATURES], y_dev, matches_dev)
        tuning.append({**params, "cv_log_loss": round(log_loss(y_dev, p), 5)})
        print(f"  {params} -> {tuning[-1]['cv_log_loss']}", flush=True)
    best_params = min(zip(tuning, PARAM_GRID), key=lambda t: t[0]["cv_log_loss"])[1]
    print(f"Chosen: {best_params}")

    # ---- Step 2: compare models -------------------------------------------
    static = "XGBoost (+ defender and keeper positions)"
    kinematic = "XGBoost (+ goalkeeper kinematics)"
    models = {
        "Baseline (always the average)": (DummyClassifier(strategy="prior"), BASIC_FEATURES),
        "Logistic regression (distance + angle)": (logistic(), ["distance", "angle_deg"]),
        "Logistic regression (all features)": (logistic(), ALL_FEATURES),
        "XGBoost (no player positions)": (boosted_trees(BASIC_FEATURES, best_params), BASIC_FEATURES),
        static: (boosted_trees(STATIC_FEATURES, best_params), STATIC_FEATURES),
        kinematic: (boosted_trees(ALL_FEATURES, best_params), ALL_FEATURES),
    }
    benchmark = "StatsBomb xG (benchmark)"

    cv_pred, cv_results, test_pred, test_results = {}, {}, {}, {}
    for name, (model, features) in models.items():
        cv_pred[name] = cross_validate(model, X_dev[features], y_dev, matches_dev)
        cv_results[name] = score(y_dev, cv_pred[name])
        # Train on all development data, then predict the untouched test set.
        test_pred[name] = model.fit(X_dev[features], y_dev).predict_proba(X_test[features])[:, 1]
        test_results[name] = score(y_test, test_pred[name])

    cv_pred[benchmark] = shots_dev["statsbomb_xg"].to_numpy()
    test_pred[benchmark] = shots_test["statsbomb_xg"].to_numpy()
    cv_results[benchmark] = score(y_dev, cv_pred[benchmark])
    test_results[benchmark] = score(y_test, test_pred[benchmark])

    cv_table = pd.DataFrame(cv_results).T.round(4)
    test_table = pd.DataFrame(test_results).T.round(4)
    print("\nCross-validation on development data\n")
    print(cv_table.to_string())
    print(f"\nHeld-out test: {TEST_COMPETITION}\n")
    print(test_table.to_string())

    # The final model is whichever feature set did better in cross-validation.
    # The test set plays no part in this choice. If the extra kinematic
    # features do not earn their place, the simpler model wins.
    full = min([static, kinematic], key=lambda name: cv_results[name]["log_loss"])
    final_features = models[full][1]
    print(f"\nFinal model (chosen by cross-validation): {full}")

    ci = bootstrap_log_loss(y_test, test_pred[full], test_pred[benchmark],
                            shots_test["match_id"])
    corr = float(np.corrcoef(test_pred[full], test_pred[benchmark])[0, 1])
    print(f"\n95% intervals for test log-loss: ours {ci['ours']}, "
          f"StatsBomb {ci['statsbomb']}, difference {ci['difference']}")
    print(f"Correlation with StatsBomb xG on the test set: {corr:.3f}")
    print(f"Test goals: {int(y_test.sum())} | our xG: {test_pred[full].sum():.1f} "
          f"| StatsBomb xG: {test_pred[benchmark].sum():.1f}")

    FIGURES_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    # ---- Figure 1: calibration. When we say 30%, do 30% go in? -------------
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.plot([0, 0.7], [0, 0.7], "--", color="grey", label="Perfect")
    for name, color in [("Logistic regression (distance + angle)", "#8d99ae"),
                        (full, "#e63946"), (benchmark, "#1d3557")]:
        observed, predicted = calibration_curve(y_dev, cv_pred[name], n_bins=15,
                                                strategy="quantile")
        ax.plot(predicted, observed, "o-", color=color, label=name, markersize=4)
    ax.set_xlabel("Predicted goal probability")
    ax.set_ylabel("Share of shots that actually scored")
    ax.set_title("Calibration (cross-validated, development data)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.savefig(FIGURES_DIR / "calibration.png", dpi=150, bbox_inches="tight")

    # ---- Train the final model on every shot and save it -------------------
    final = boosted_trees(final_features, best_params).fit(X[final_features], y)
    final.save_model(MODELS_DIR / "xg_model.json")

    # ---- Figure 2: which features drive the predictions? (SHAP values) -----
    # A SHAP value is how much one feature pushed one prediction up or down.
    contributions = final.get_booster().predict(
        xgb.DMatrix(X[final_features]), pred_contribs=True)[:, :-1]
    importance = pd.Series(np.abs(contributions).mean(axis=0),
                           index=final_features).sort_values()
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.barh(importance.index, importance.values, color="#e63946")
    ax.set_xlabel("Average impact on the prediction (mean |SHAP|, log-odds)")
    ax.set_title("What the model pays attention to")
    fig.savefig(FIGURES_DIR / "feature_importance.png", dpi=150, bbox_inches="tight")

    (RESULTS_DIR / "metrics.json").write_text(json.dumps({
        "n_shots": int(len(shots)), "n_goals": int(y.sum()),
        "n_development_shots": int(len(X_dev)), "n_test_shots": int(len(X_test)),
        "test_competition": TEST_COMPETITION,
        "final_model": full, "final_features": final_features,
        "chosen_params": best_params, "tuning": tuning,
        "cross_validation": cv_table.to_dict(orient="index"),
        "held_out_test": test_table.to_dict(orient="index"),
        "test_log_loss_95_interval": ci,
        "test_correlation_with_statsbomb": round(corr, 4),
        "feature_importance": importance.sort_values(ascending=False).round(4).to_dict(),
    }, indent=2))
    print("\nSaved model, figures and results/metrics.json")


if __name__ == "__main__":
    main()
