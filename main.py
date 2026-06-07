"""End-to-end Wine Quality ML pipeline.

Run with:  python main.py

Steps:
  1. Load (download + cache) the UCI red+white wine dataset.
  2. Exploratory summary.
  3. Supervised classification  (3 models): good vs. standard wine.
  4. Supervised regression      (3 models): predict the 0-10 quality score.
  5. Unsupervised               (PCA + 3 clustering algorithms).
  6. Persist a metrics summary to reports/metrics_summary.csv.
"""
from __future__ import annotations

import json
import sys

import pandas as pd

# Ensure unicode (em-dashes, R²) prints cleanly on Windows consoles.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.config import CLASSIFICATION_TARGET, REPORTS_DIR
from src.data import load_wine_data
from src.supervised import run_classification, run_regression
from src.unsupervised import run_unsupervised
from src.utils import section


def explore(df: pd.DataFrame) -> None:
    section("DATASET OVERVIEW")
    print(f"Samples: {len(df)} | Features: {df.shape[1]}")
    if "wine_type" in df.columns:
        print("\nWine type counts:")
        print(df["wine_type"].value_counts().to_string())
    print("\nQuality score distribution:")
    print(df["quality"].value_counts().sort_index().to_string())
    pos = df[CLASSIFICATION_TARGET].mean()
    print(f"\n'Good' wines (quality >= 7): {pos * 100:.1f}% of samples")


def main() -> None:
    section("WINE QUALITY — MACHINE LEARNING PIPELINE")
    df = load_wine_data()
    explore(df)

    clf_results = run_classification(df)
    reg_results = run_regression(df)
    clu_results = run_unsupervised(df)

    # Persist a tidy summary.
    clf_results.assign(task="classification").to_csv(
        REPORTS_DIR / "classification_results.csv", index=False
    )
    reg_results.assign(task="regression").to_csv(
        REPORTS_DIR / "regression_results.csv", index=False
    )
    with open(REPORTS_DIR / "clustering_results.json", "w") as fh:
        json.dump(clu_results, fh, indent=2)

    section("DONE")
    print("Artifacts:")
    print("  models/                      trained model pipelines (.joblib)")
    print("  reports/figures/             plots (confusion, importances, clusters)")
    print("  reports/*_results.csv/json   metric tables")


if __name__ == "__main__":
    main()
