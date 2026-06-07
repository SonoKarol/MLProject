"""Unsupervised learning on the wine physicochemical features.

Algorithms:
    * PCA                    -> dimensionality reduction / visualisation
    * K-Means                -> partition into k clusters
    * Agglomerative (Ward)   -> hierarchical clustering
    * DBSCAN                 -> density-based clustering (finds outliers)

We evaluate clusterings with the silhouette score and check, using the known
``wine_type`` label, whether the unsupervised structure recovers the natural
red/white split.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    adjusted_rand_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler  # noqa: E402

from .config import RANDOM_STATE  # noqa: E402
from .data import get_feature_matrix  # noqa: E402
from .utils import savefig, section  # noqa: E402


def run_unsupervised(df: pd.DataFrame) -> dict:
    section("UNSUPERVISED — DIMENSIONALITY REDUCTION & CLUSTERING")

    X = get_feature_matrix(df)
    X_scaled = StandardScaler().fit_transform(X)
    has_type = "wine_type" in df.columns
    true_labels = (
        df["wine_type"].astype("category").cat.codes.to_numpy()
        if has_type
        else None
    )

    # --- PCA ---------------------------------------------------------------- #
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    coords = pca.fit_transform(X_scaled)
    evr = pca.explained_variance_ratio_
    print(
        f"PCA: 2 components explain "
        f"{evr.sum() * 100:.1f}% of variance "
        f"(PC1={evr[0] * 100:.1f}%, PC2={evr[1] * 100:.1f}%)"
    )
    _plot_pca(coords, df, has_type)

    results: dict = {"pca_explained_variance": evr.tolist()}

    # --- K-Means ------------------------------------------------------------ #
    k = _choose_k(X_scaled)
    kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    km_labels = kmeans.fit_predict(X_scaled)
    km_sil = silhouette_score(X_scaled, km_labels)
    print(f"K-Means (k={k}): silhouette={km_sil:.3f}")
    results["kmeans"] = {"k": k, "silhouette": km_sil}

    # --- Agglomerative (hierarchical) -------------------------------------- #
    agg = AgglomerativeClustering(n_clusters=k, linkage="ward")
    agg_labels = agg.fit_predict(X_scaled)
    agg_sil = silhouette_score(X_scaled, agg_labels)
    print(f"Agglomerative (k={k}, ward): silhouette={agg_sil:.3f}")
    results["agglomerative"] = {"k": k, "silhouette": agg_sil}

    # --- DBSCAN ------------------------------------------------------------- #
    eps = 2.5
    db = DBSCAN(eps=eps, min_samples=10)
    db_labels = db.fit_predict(X_scaled)
    n_clusters = len(set(db_labels)) - (1 if -1 in db_labels else 0)
    n_noise = int(np.sum(db_labels == -1))
    if n_clusters >= 2:
        mask = db_labels != -1
        db_sil = silhouette_score(X_scaled[mask], db_labels[mask])
    else:
        db_sil = float("nan")
    print(
        f"DBSCAN (eps={eps}): clusters={n_clusters}, "
        f"noise points={n_noise}, silhouette={db_sil:.3f}"
    )
    results["dbscan"] = {
        "clusters": n_clusters,
        "noise": n_noise,
        "silhouette": db_sil,
    }

    # --- Agreement with the true red/white label --------------------------- #
    if has_type:
        ari = adjusted_rand_score(true_labels, km_labels)
        print(
            f"K-Means vs. true wine type: adjusted Rand index={ari:.3f} "
            f"(1.0 = perfect recovery of red/white split)"
        )
        results["kmeans_vs_winetype_ari"] = ari

    # --- Visualise clusters on the PCA plane -------------------------------- #
    _plot_clusters(coords, km_labels, "K-Means", "cluster_kmeans_pca.png")
    _plot_clusters(coords, agg_labels, "Agglomerative", "cluster_agglomerative_pca.png")
    _plot_clusters(coords, db_labels, "DBSCAN", "cluster_dbscan_pca.png")

    return results


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _choose_k(X_scaled, k_range=range(2, 7)) -> int:
    """Pick k maximising the silhouette score (and draw the elbow curve)."""
    inertias, sils, ks = [], [], list(k_range)
    for k in ks:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        sils.append(silhouette_score(X_scaled, labels))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(ks, inertias, "o-")
    ax1.set(xlabel="k", ylabel="inertia", title="Elbow method")
    ax2.plot(ks, sils, "o-", color="darkorange")
    ax2.set(xlabel="k", ylabel="silhouette", title="Silhouette by k")
    savefig(fig, "cluster_k_selection.png")

    return ks[int(np.argmax(sils))]


def _plot_pca(coords, df, has_type) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5))
    if has_type:
        for colour in df["wine_type"].unique():
            m = (df["wine_type"] == colour).to_numpy()
            ax.scatter(coords[m, 0], coords[m, 1], s=8, alpha=0.4, label=colour)
        ax.legend(title="wine type")
    else:
        ax.scatter(coords[:, 0], coords[:, 1], s=8, alpha=0.4)
    ax.set(xlabel="PC1", ylabel="PC2", title="Wines projected onto 2 PCA components")
    savefig(fig, "pca_scatter_by_type.png")


def _plot_clusters(coords, labels, name, fname) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5))
    palette = sns.color_palette("tab10", len(set(labels)))
    for i, lab in enumerate(sorted(set(labels))):
        m = labels == lab
        tag = "noise" if lab == -1 else f"cluster {lab}"
        color = "lightgray" if lab == -1 else palette[i % len(palette)]
        ax.scatter(coords[m, 0], coords[m, 1], s=8, alpha=0.5, color=color, label=tag)
    ax.legend(markerscale=2, fontsize=8)
    ax.set(xlabel="PC1", ylabel="PC2", title=f"{name} clusters (PCA plane)")
    savefig(fig, fname)
