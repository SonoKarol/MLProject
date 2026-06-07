"""Supervised learning: classification and regression on wine quality.

Classification (target = ``good_quality``):
    * Logistic Regression
    * Random Forest Classifier
    * Gradient Boosting Classifier

Regression (target = ``quality`` score):
    * Linear Regression
    * Random Forest Regressor
    * Gradient Boosting Regressor

Each model is wrapped in a Pipeline with standardisation so the linear models
get scaled inputs. Trained estimators are persisted to ``models/``.
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
    root_mean_squared_error,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import (
    CLASSIFICATION_TARGET,
    MODELS_DIR,
    RANDOM_STATE,
    REGRESSION_TARGET,
    TEST_SIZE,
)
from .data import get_feature_matrix
from .utils import plot_bar_scores, plot_confusion, section


def _split(df: pd.DataFrame, target: str, stratify: bool):
    X = get_feature_matrix(df)
    y = df[target]
    strat = y if stratify else None
    return train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=strat
    )


# --------------------------------------------------------------------------- #
# Classification
# --------------------------------------------------------------------------- #
def run_classification(df: pd.DataFrame) -> pd.DataFrame:
    section("SUPERVISED — CLASSIFICATION  (good wine: quality >= 7)")
    X_train, X_test, y_train, y_test = _split(
        df, CLASSIFICATION_TARGET, stratify=True
    )
    print(
        f"Train/test: {len(X_train)}/{len(X_test)} | "
        f"positive rate (test): {y_test.mean():.3f}"
    )

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            random_state=RANDOM_STATE
        ),
    }

    rows = []
    for name, clf in models.items():
        pipe = Pipeline([("scaler", StandardScaler()), ("model", clf)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        proba = pipe.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, pred)
        f1 = f1_score(y_test, pred)
        auc = roc_auc_score(y_test, proba)
        rows.append(
            {"model": name, "accuracy": acc, "f1": f1, "roc_auc": auc}
        )
        print(f"  {name:<20} acc={acc:.3f}  f1={f1:.3f}  roc_auc={auc:.3f}")

        joblib.dump(pipe, MODELS_DIR / f"clf_{_slug(name)}.joblib")
        plot_confusion(
            confusion_matrix(y_test, pred),
            labels=["standard", "good"],
            title=f"Confusion — {name}",
            fname=f"clf_confusion_{_slug(name)}.png",
        )

    results = pd.DataFrame(rows).sort_values("f1", ascending=False)
    best = results.iloc[0]["model"]
    print(f"  >> best by F1: {best}")
    plot_bar_scores(
        dict(zip(results["model"], results["roc_auc"])),
        metric="ROC AUC",
        title="Classification — ROC AUC by model",
        fname="clf_roc_auc_comparison.png",
    )
    _save_feature_importance(
        models["Random Forest"], X_train, y_train, "classification"
    )
    return results


# --------------------------------------------------------------------------- #
# Regression
# --------------------------------------------------------------------------- #
def run_regression(df: pd.DataFrame) -> pd.DataFrame:
    section("SUPERVISED — REGRESSION  (predict quality score 0-10)")
    X_train, X_test, y_train, y_test = _split(
        df, REGRESSION_TARGET, stratify=False
    )
    print(f"Train/test: {len(X_train)}/{len(X_test)}")

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            random_state=RANDOM_STATE
        ),
    }

    rows = []
    for name, reg in models.items():
        pipe = Pipeline([("scaler", StandardScaler()), ("model", reg)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)

        rmse = root_mean_squared_error(y_test, pred)
        mae = mean_absolute_error(y_test, pred)
        r2 = r2_score(y_test, pred)
        rows.append({"model": name, "rmse": rmse, "mae": mae, "r2": r2})
        print(f"  {name:<20} rmse={rmse:.3f}  mae={mae:.3f}  r2={r2:.3f}")

        joblib.dump(pipe, MODELS_DIR / f"reg_{_slug(name)}.joblib")

    results = pd.DataFrame(rows).sort_values("rmse")
    best = results.iloc[0]["model"]
    print(f"  >> best by RMSE: {best}")
    plot_bar_scores(
        dict(zip(results["model"], results["r2"])),
        metric="R² (higher is better)",
        title="Regression — R² by model",
        fname="reg_r2_comparison.png",
    )
    _save_feature_importance(
        models["Random Forest"], X_train, y_train, "regression"
    )
    return results


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _slug(name: str) -> str:
    return name.lower().replace(" ", "_")


def _save_feature_importance(model, X_train, y_train, kind: str) -> None:
    """Fit a tree model on raw features and plot its importances."""
    model.fit(X_train, y_train)
    importances = pd.Series(
        model.feature_importances_, index=X_train.columns
    ).sort_values(ascending=True)
    plot_bar_scores(
        dict(zip(importances.index, importances.values)),
        metric="importance",
        title=f"Feature importance (Random Forest, {kind})",
        fname=f"feature_importance_{kind}.png",
    )
