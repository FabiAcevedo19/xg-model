"""Train several xG models, compare them fairly, and save the best one.

Run from the project folder:
    python src/train.py

How the comparison is kept honest
---------------------------------
* Cross-validation: the data is split into 5 parts. Each model is trained on
  4 parts and makes predictions for the part it has never seen. Repeating
  this 5 times gives a prediction for every shot, made by a model that did
  not train on it.
* Grouped by match: all shots from one match stay in the same part. Shots
  from the same match are related (same teams, same keeper), so splitting
  them between training and testing would be a subtle form of cheating.
* Benchmark: StatsBomb's own commercial xG is scored on exactly the same
  shots, so we know what "professional quality" looks like.
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

from features import ALL_FEATURES, BASIC_FEATURES, MONOTONE, build_features
from load_data import load_shots

ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = ROOT / "figures"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"


def logistic():
    """Logistic regression: a straight-line model squashed into 0..1."""
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(max_iter=1000))


def boosted_trees(features):
    """XGBoost: hundreds of small decision trees, each fixing the last one's errors."""
    # Shallow trees (max_depth=3) work best here: with only ~7,500 shots,
    # deeper trees start memorising noise. Chosen by trying five settings.
    return xgb.XGBClassifier(
        n_estimators=300, learning_rate=0.03, max_depth=3,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
        monotone_constraints=tuple(MONOTONE.get(f, 0) for f in features),
        eval_metric="logloss", random_state=0,
    )


def score(y, p):
    return {
        "log_loss": log_loss(y, p),        # punishes confident wrong answers. Lower is better.
        "brier": brier_score_loss(y, p),   # mean squared error of the probability. Lower is better.
        "roc_auc": roc_auc_score(y, p),    # can it rank goals above misses? Higher is better.
    }


def main():
    shots = load_shots()
    # Penalties are a fixed situation (about 76% score), so a model adds nothing.
    shots = shots[shots["shot_type"] != "Penalty"].reset_index(drop=True)

    print(f"Building features for {len(shots)} non-penalty shots...")
    X = build_features(shots)
    y = shots["is_goal"]
    matches = shots["match_id"]

    models = {
        "Baseline (always the average)": (DummyClassifier(strategy="prior"), BASIC_FEATURES),
        "Logistic regression (distance + angle)": (logistic(), ["distance", "angle_deg"]),
        "Logistic regression (all features)": (logistic(), ALL_FEATURES),
        "XGBoost (basic features)": (boosted_trees(BASIC_FEATURES), BASIC_FEATURES),
        "XGBoost (+ defender and keeper geometry)": (boosted_trees(ALL_FEATURES), ALL_FEATURES),
    }

    cv = GroupKFold(n_splits=5)
    predictions, results = {}, {}
    for name, (model, features) in models.items():
        p = cross_val_predict(model, X[features], y, groups=matches, cv=cv,
                              method="predict_proba")[:, 1]
        predictions[name] = p
        results[name] = score(y, p)

    predictions["StatsBomb xG (benchmark)"] = shots["statsbomb_xg"].to_numpy()
    results["StatsBomb xG (benchmark)"] = score(y, shots["statsbomb_xg"])

    table = pd.DataFrame(results).T.round(4)
    print("\nOut-of-sample results (5-fold cross-validation, grouped by match)\n")
    print(table.to_string())

    best = "XGBoost (+ defender and keeper geometry)"
    corr = np.corrcoef(predictions[best], shots["statsbomb_xg"])[0, 1]
    print(f"\nCorrelation between our xG and StatsBomb's: {corr:.3f}")
    print(f"Goals: {y.sum()}  |  our total xG: {predictions[best].sum():.1f}"
          f"  |  StatsBomb total xG: {shots['statsbomb_xg'].sum():.1f}")

    FIGURES_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    # ---- Figure 1: calibration. When we say 30%, do 30% go in? -------------
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.plot([0, 1], [0, 1], "--", color="grey", label="Perfect")
    for name, color in [("Logistic regression (distance + angle)", "#8d99ae"),
                        (best, "#e63946"),
                        ("StatsBomb xG (benchmark)", "#1d3557")]:
        observed, predicted = calibration_curve(y, predictions[name], n_bins=10,
                                                strategy="quantile")
        ax.plot(predicted, observed, "o-", color=color, label=name)
    ax.set_xlabel("Predicted goal probability")
    ax.set_ylabel("Share of shots that actually scored")
    ax.set_title("Calibration (out-of-sample)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.savefig(FIGURES_DIR / "calibration.png", dpi=150, bbox_inches="tight")

    # ---- Train the final model on every shot and save it -------------------
    final = boosted_trees(ALL_FEATURES).fit(X[ALL_FEATURES], y)
    final.save_model(MODELS_DIR / "xg_model.json")

    # ---- Figure 2: which features drive the predictions? (SHAP values) -----
    # A SHAP value is how much one feature pushed one prediction up or down.
    contributions = final.get_booster().predict(
        xgb.DMatrix(X[ALL_FEATURES]), pred_contribs=True)[:, :-1]
    importance = pd.Series(np.abs(contributions).mean(axis=0),
                           index=ALL_FEATURES).sort_values()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(importance.index, importance.values, color="#e63946")
    ax.set_xlabel("Average impact on the prediction (mean |SHAP|, log-odds)")
    ax.set_title("What the model pays attention to")
    fig.savefig(FIGURES_DIR / "feature_importance.png", dpi=150, bbox_inches="tight")

    (RESULTS_DIR / "metrics.json").write_text(json.dumps({
        "n_shots": int(len(shots)), "n_goals": int(y.sum()),
        "correlation_with_statsbomb": round(float(corr), 4),
        "models": table.to_dict(orient="index"),
        "feature_importance": importance.sort_values(ascending=False).round(4).to_dict(),
    }, indent=2))
    print("\nSaved model, figures and results/metrics.json")


if __name__ == "__main__":
    main()
