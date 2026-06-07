"""Dataset loading and preparation.

Theme: **Wine Quality** (UCI Machine Learning Repository).

The dataset describes red and white "Vinho Verde" wine samples by 11
physicochemical properties (acidity, sugar, alcohol, ...) plus a sensory
``quality`` score (0-10). We combine both colours and engineer two targets:

* ``quality``       -> regression  (predict the numeric score)
* ``good_quality``  -> classification (quality >= threshold -> 1 else 0)

The raw CSVs are downloaded once and cached under ``data/``. If the machine is
offline and no cache exists, a realistic synthetic fallback is generated so the
pipeline always runs.
"""
from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    CLASSIFICATION_TARGET,
    DATA_DIR,
    GOOD_QUALITY_THRESHOLD,
    RANDOM_STATE,
    REGRESSION_TARGET,
)

_BASE_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/"
)
_FILES = {
    "red": "winequality-red.csv",
    "white": "winequality-white.csv",
}

FEATURE_COLUMNS = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
]


def _download_one(colour: str, dest: Path) -> bool:
    """Download a single colour's CSV to ``dest``. Returns True on success."""
    url = _BASE_URL + _FILES[colour]
    try:
        with urllib.request.urlopen(url, timeout=20) as resp:
            dest.write_bytes(resp.read())
        return True
    except Exception as exc:  # noqa: BLE001 - network is best-effort
        print(f"  [warn] could not download {colour} wine data: {exc}")
        return False


def _make_synthetic(n_per_colour: int = 1500) -> pd.DataFrame:
    """Generate a plausible synthetic fallback dataset (offline safety net)."""
    rng = np.random.default_rng(RANDOM_STATE)
    frames = []
    # Loose per-colour means for a few key features so clustering still works.
    profiles = {
        "red": dict(alcohol=10.4, volatile=0.53, sulfur=46, sugar=2.5),
        "white": dict(alcohol=10.5, volatile=0.28, sulfur=138, sugar=6.4),
    }
    for colour, p in profiles.items():
        n = n_per_colour
        alcohol = rng.normal(p["alcohol"], 1.1, n).clip(8, 15)
        df = pd.DataFrame(
            {
                "fixed acidity": rng.normal(7.5, 1.3, n).clip(4, 16),
                "volatile acidity": rng.normal(p["volatile"], 0.15, n).clip(0.08, 1.6),
                "citric acid": rng.normal(0.32, 0.15, n).clip(0, 1),
                "residual sugar": rng.normal(p["sugar"], 1.5, n).clip(0.6, 30),
                "chlorides": rng.normal(0.056, 0.02, n).clip(0.01, 0.4),
                "free sulfur dioxide": rng.normal(p["sulfur"] * 0.4, 10, n).clip(1, 120),
                "total sulfur dioxide": rng.normal(p["sulfur"], 25, n).clip(6, 350),
                "density": rng.normal(0.995, 0.003, n).clip(0.985, 1.01),
                "pH": rng.normal(3.2, 0.16, n).clip(2.7, 4.0),
                "sulphates": rng.normal(0.53, 0.14, n).clip(0.2, 2.0),
                "alcohol": alcohol,
            }
        )
        # Quality driven mostly by alcohol & (low) volatile acidity + noise.
        score = (
            0.5 * (alcohol - 10)
            - 3.0 * (df["volatile acidity"] - 0.4)
            + rng.normal(0, 0.7, n)
            + 5.6
        )
        df["quality"] = score.round().clip(3, 9).astype(int)
        df["wine_type"] = colour
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    return out.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)


def load_wine_data(force_download: bool = False) -> pd.DataFrame:
    """Load the combined red+white wine dataset as a tidy DataFrame.

    Adds a categorical ``wine_type`` column and the engineered
    ``good_quality`` classification target.
    """
    combined_path = DATA_DIR / "winequality-combined.csv"
    if combined_path.exists() and not force_download:
        df = pd.read_csv(combined_path)
        return _add_targets(df)

    frames = []
    for colour, fname in _FILES.items():
        raw_path = DATA_DIR / fname
        if force_download or not raw_path.exists():
            _download_one(colour, raw_path)
        if raw_path.exists():
            part = pd.read_csv(raw_path, sep=";")
            part["wine_type"] = colour
            frames.append(part)

    if frames:
        df = pd.concat(frames, ignore_index=True)
        print(f"  Loaded real UCI data: {len(df)} samples.")
    else:
        print("  [info] Falling back to synthetic dataset (offline).")
        df = _make_synthetic()

    df = df.drop_duplicates().reset_index(drop=True)
    df.to_csv(combined_path, index=False)
    return _add_targets(df)


def _add_targets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[CLASSIFICATION_TARGET] = (
        df[REGRESSION_TARGET] >= GOOD_QUALITY_THRESHOLD
    ).astype(int)
    return df


def get_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Return only the physicochemical feature columns present in ``df``."""
    cols = [c for c in FEATURE_COLUMNS if c in df.columns]
    return df[cols].copy()


if __name__ == "__main__":
    data = load_wine_data()
    print(data.head())
    print("\nShape:", data.shape)
    print("\nClass balance (good_quality):")
    print(data[CLASSIFICATION_TARGET].value_counts(normalize=True).round(3))
