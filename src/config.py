"""Central configuration: paths and shared constants."""
from __future__ import annotations

from pathlib import Path

# --- Project paths -----------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

for _d in (DATA_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --- Reproducibility ---------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.2

# --- Target definitions ------------------------------------------------------
# Regression target: the raw 0-10 sensory quality score.
REGRESSION_TARGET = "quality"
# Classification target: a wine is "good" if its quality score is >= this value.
GOOD_QUALITY_THRESHOLD = 7
CLASSIFICATION_TARGET = "good_quality"
