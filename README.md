# 🍷 Wine Quality — Machine Learning Project

![Status](https://img.shields.io/badge/status-complete-brightgreen)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

End-to-end machine learning project on the **UCI Wine Quality** dataset
(red + white "Vinho Verde" wines). A single command downloads the data, trains
and evaluates **6 supervised models** (classification + regression) and **4
unsupervised algorithms** (PCA + 3 clustering methods), and writes metrics and
figures to disk.

## 🎯 Problem & dataset

Each wine sample is described by 11 physicochemical measurements (acidity,
residual sugar, chlorides, sulphates, alcohol, …) and a sensory **quality**
score from 0 to 10. After de-duplication the combined dataset has **5,320
samples** (3,961 white, 1,359 red). From it we frame three tasks:

| Task | Type | Target |
|------|------|--------|
| Predict the quality score | **Regression** | `quality` (0–10) |
| Identify *good* wines | **Classification** | `good_quality` = 1 if `quality ≥ 7` |
| Discover natural groupings | **Unsupervised** | physicochemical features only |

> Data source: [UCI Machine Learning Repository — Wine Quality](https://archive.ics.uci.edu/dataset/186/wine+quality).
> The loader downloads and caches the CSVs automatically; if the machine is
> offline it falls back to a synthetic dataset so the pipeline always runs.

## 🧠 Models

**Supervised — Classification** (good vs. standard wine)
- Logistic Regression
- Random Forest Classifier
- Gradient Boosting Classifier

**Supervised — Regression** (predict the 0–10 score)
- Linear Regression
- Random Forest Regressor
- Gradient Boosting Regressor

**Unsupervised**
- **PCA** — dimensionality reduction & 2-D visualisation
- **K-Means** — partitional clustering (k chosen by silhouette)
- **Agglomerative** (Ward linkage) — hierarchical clustering
- **DBSCAN** — density-based clustering with outlier detection

All supervised models are wrapped in a `Pipeline` with `StandardScaler`, so the
linear models receive standardised inputs and there is no train/test leakage.

## 📊 Results

Hold-out test set (20%, stratified for classification).

**Classification** — the class is imbalanced (~19% "good"), so ROC AUC and F1
matter more than raw accuracy:

| Model | Accuracy | F1 | ROC AUC |
|-------|:--------:|:--:|:-------:|
| Random Forest | **0.844** | 0.435 | **0.879** |
| Gradient Boosting | 0.837 | 0.440 | 0.845 |
| Logistic Regression | 0.735 | **0.528** | 0.816 |

Random Forest has the best ranking ability (AUC 0.88). The class-balanced
Logistic Regression trades accuracy for the best F1 / recall on the minority
"good" class — useful if catching good wines is the priority.

**Regression** — predicting the exact 0–10 score:

| Model | RMSE ↓ | MAE ↓ | R² ↑ |
|-------|:------:|:-----:|:----:|
| **Random Forest** | **0.664** | **0.510** | **0.414** |
| Gradient Boosting | 0.685 | 0.528 | 0.375 |
| Linear Regression | 0.724 | 0.561 | 0.302 |

Random Forest wins, predicting the score to within ~0.5 points on average.
**Alcohol** and **volatile acidity** are consistently the most important
predictors of quality.

**Unsupervised** — clustering on physicochemical features alone:

| Algorithm | Clusters | Silhouette | Notes |
|-----------|:--------:|:----------:|-------|
| K-Means | 2 | 0.270 | **ARI = 0.93** vs. true red/white label |
| Agglomerative | 2 | 0.261 | Agrees with K-Means |
| DBSCAN | 2 + noise | 0.624 | Flags 72 outlier wines |

Key finding: **without ever seeing the red/white label, K-Means recovers it
almost perfectly** (adjusted Rand index = 0.93). The PCA projection shows why —
the two wine colours occupy clearly separated regions of feature space.

## 🗂️ Project structure

```
MLProject/
├── main.py                     # Run the whole pipeline
├── requirements.txt
├── src/
│   ├── config.py               # Paths, random seed, targets
│   ├── data.py                 # Download / cache / prepare dataset
│   ├── supervised.py           # Classification + regression
│   ├── unsupervised.py         # PCA + clustering
│   └── utils.py                # Plotting helpers
├── data/                       # Cached CSVs (gitignored)
├── models/                     # Saved model pipelines (.joblib)
└── reports/
    ├── figures/                # All generated plots (.png)
    ├── classification_results.csv
    ├── regression_results.csv
    └── clustering_results.json
```

## 🚀 Getting started

```bash
# (Recommended) create a virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the full pipeline
python main.py
```

Running `main.py` will:
1. Download & cache the dataset into `data/`.
2. Print an exploratory summary.
3. Train + evaluate all classification and regression models.
4. Run PCA and the three clustering algorithms.
5. Save trained models to `models/`, plots to `reports/figures/`, and metric
   tables to `reports/`.

You can also run a single stage, e.g. inspect the data loader:

```bash
python -m src.data
```

## 🖼️ Generated figures

`reports/figures/` includes, among others:
- `pca_scatter_by_type.png` — wines projected onto 2 PCA components
- `cluster_{kmeans,agglomerative,dbscan}_pca.png` — clusters on the PCA plane
- `cluster_k_selection.png` — elbow + silhouette curves for choosing *k*
- `clf_confusion_*.png` — per-model confusion matrices
- `clf_roc_auc_comparison.png`, `reg_r2_comparison.png` — model leaderboards
- `feature_importance_{classification,regression}.png`

## 🔁 Reproducibility

A fixed random seed (`RANDOM_STATE = 42`, see `src/config.py`) is used for all
splits and stochastic models, so results are deterministic across runs.

## 📄 License

MIT.
