import os
import platform

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import sklearn
from sklearn.datasets import load_breast_cancer, load_wine

SEEDS = [42, 52, 62]
FIG_DIR = "figures"
RESULTS_DIR = "results"


def make_folders():
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)


def save_fig(fig, filename):
    make_folders()
    fig.savefig(os.path.join(FIG_DIR, filename), dpi=150, bbox_inches="tight")


def package_versions():
    return pd.DataFrame({
        "package": ["python", "numpy", "pandas", "scikit-learn", "scipy",
                    "matplotlib", "seaborn"],
        "version": [platform.python_version(), np.__version__, pd.__version__,
                    sklearn.__version__, scipy.__version__,
                    matplotlib.__version__, sns.__version__],
    })


# Section 1: loading
def load_data(name):
    """Return X (DataFrame with feature names), y (Series) and the class names."""
    if name == "breast_cancer":
        data = load_breast_cancer()
    else:
        data = load_wine()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="target")      # original labels are kept
    class_names = [str(c) for c in data.target_names]
    return X, y, class_names


def class_table(y, class_names):
    counts = y.value_counts().sort_index()
    return pd.DataFrame({
        "label": counts.index,
        "class_name": [class_names[i] for i in counts.index],
        "n_samples": counts.values,
        "percent": (counts.values / len(y) * 100).round(2),
    })



# Section 2: quality checks
from scipy import stats


def missing_table(X):
    return pd.DataFrame({"feature": X.columns, "missing_count": X.isna().sum().values})


def count_duplicates(X, y):
    full = X.copy()
    full["target"] = y
    return int(full.duplicated().sum())


def dtype_table(X):
    return pd.DataFrame({"feature": X.columns, "dtype": X.dtypes.astype(str).values})


def outlier_table(X):
    """Count possible outliers with the IQR rule and |z| > 3. Nothing is deleted."""
    rows = []
    for col in X.columns:
        v = X[col]
        q1, q3 = v.quantile(0.25), v.quantile(0.75)
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        z = np.abs(stats.zscore(v))
        rows.append({"feature": col,
                     "iqr_lower": round(low, 4), "iqr_upper": round(high, 4),
                     "n_iqr_outliers": int(((v < low) | (v > high)).sum()),
                     "n_abs_z_gt_3": int((z > 3).sum())})
    return pd.DataFrame(rows).sort_values("n_iqr_outliers", ascending=False)

# Section 3: EDA
def describe_features(X):
    return pd.DataFrame({
        "mean": X.mean(), "median": X.median(), "min": X.min(), "max": X.max(),
        "std": X.std(), "Q1": X.quantile(0.25), "Q3": X.quantile(0.75),
    }).round(4)


def plot_class_distribution(y, class_names, dataset):
    counts = y.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    labels = [f"{i}: {class_names[i]}" for i in counts.index]
    bars = ax.bar(labels, counts.values, color=sns.color_palette("Set2", len(counts)))
    ax.bar_label(bars)
    ax.set_title(f"{dataset} - class distribution (full dataset)")
    ax.set_xlabel("Class (original label)")
    ax.set_ylabel("Number of samples")
    save_fig(fig, f"{dataset}_class_distribution.png")
    plt.show()


def plot_corr_heatmap(X, dataset):
    corr = X.corr(method="pearson")
    size = max(8, 0.45 * X.shape[1])
    fig, ax = plt.subplots(figsize=(size, size * 0.85))
    sns.heatmap(corr, cmap="coolwarm", vmin=-1, vmax=1, center=0, square=True,
                annot=X.shape[1] <= 15, fmt=".2f", annot_kws={"size": 7},
                cbar_kws={"shrink": 0.7}, ax=ax)
    ax.set_title(f"{dataset} - Pearson feature correlation (full dataset)")
    save_fig(fig, f"{dataset}_correlation_heatmap.png")
    plt.show()
    return corr


def top_corr_pairs(corr, n=3):
    """Strongest pairs by |r|. Only the upper triangle is used, so the diagonal
    and mirrored pairs (A-B and B-A) are not counted."""
    cols = corr.columns
    rows = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = corr.iloc[i, j]
            rows.append({"feature_1": cols[i], "feature_2": cols[j],
                         "pearson_r": round(r, 4), "abs_r": round(abs(r), 4)})
    return (pd.DataFrame(rows).sort_values("abs_r", ascending=False)
            .head(n).reset_index(drop=True))


def plot_boxplots(X, dataset, per_figure):
    """One small boxplot per feature, because the feature ranges are very different."""
    cols = list(X.columns)
    groups = [cols[i:i + per_figure] for i in range(0, len(cols), per_figure)]
    for k, group in enumerate(groups, start=1):
        nrows = int(np.ceil(len(group) / 5))
        fig, axes = plt.subplots(nrows, 5, figsize=(16, 3.2 * nrows))
        axes = np.array(axes).reshape(-1)
        for ax, col in zip(axes, group):
            ax.boxplot(X[col].values)
            ax.set_title(col, fontsize=9)
            ax.set_xticks([])
        for ax in axes[len(group):]:
            ax.axis("off")
        fig.suptitle(f"{dataset} - boxplots ({k}/{len(groups)}), full dataset", y=1.01)
        fig.tight_layout()
        save_fig(fig, f"{dataset}_boxplots_{k}.png")
        plt.show()