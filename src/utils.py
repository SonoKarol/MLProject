"""Plotting and small shared helpers."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless-safe backend
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402

from .config import FIGURES_DIR  # noqa: E402

sns.set_theme(style="whitegrid", context="notebook")


def savefig(fig, name: str) -> None:
    """Save a figure into reports/figures and close it."""
    path = FIGURES_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"    figure -> {path.relative_to(FIGURES_DIR.parent.parent)}")


def plot_confusion(cm: np.ndarray, labels, title: str, fname: str) -> None:
    fig, ax = plt.subplots(figsize=(4.5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar=False,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    savefig(fig, fname)


def plot_bar_scores(scores: dict, metric: str, title: str, fname: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    names = list(scores.keys())
    vals = [scores[n] for n in names]
    sns.barplot(x=vals, y=names, hue=names, palette="viridis", legend=False, ax=ax)
    ax.set_xlabel(metric)
    ax.set_title(title)
    for i, v in enumerate(vals):
        ax.text(v, i, f" {v:.3f}", va="center")
    savefig(fig, fname)


def section(title: str) -> None:
    line = "=" * 70
    print(f"\n{line}\n{title}\n{line}")
