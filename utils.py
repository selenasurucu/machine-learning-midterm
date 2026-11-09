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

 #Section 4: splitting
from sklearn.model_selection import train_test_split

def make_splits(X, y):
    """1) Reserve ONE fixed stratified 20% test set (random_state=42).
    2) For every seed, split the remaining 80% into train/validation
       (test_size=0.125 -> about 70% / 10% of the full data).
    The original row indices are kept in the DataFrames/Series."""
    X_rest, X_test, y_rest, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42)
    splits = {}
    for seed in SEEDS:
        X_train, X_val, y_train, y_val = train_test_split(
            X_rest, y_rest, test_size=0.125, stratify=y_rest, random_state=seed)
        splits[seed] = {"X_train": X_train, "X_val": X_val,
                        "y_train": y_train, "y_val": y_val}
    return splits, X_test, y_test

def split_summary(splits, y_test, n_total, class_names):
    """9 rows: (train, validation, test) x 3 seeds, with counts, % and class counts."""
    rows = []
    for seed in SEEDS:
        for subset, ys in [("train", splits[seed]["y_train"]),
                           ("validation", splits[seed]["y_val"]),
                           ("test", y_test)]:
            row = {"seed": seed, "subset": subset, "n": len(ys),
                   "percent": round(len(ys) / n_total * 100, 2)}
            for c in range(len(class_names)):
                row[f"class_{c}_{class_names[c]}"] = int((ys == c).sum())
            rows.append(row)
    return pd.DataFrame(rows)

def check_splits(splits, X_test, n_total):
    """No overlap inside a seed, and train + val + test = full dataset."""
    test_idx = set(X_test.index)
    rows = []
    for seed in SEEDS:
        tr = set(splits[seed]["X_train"].index)
        va = set(splits[seed]["X_val"].index)
        rows.append({"seed": seed,
                     "train_val_overlap": len(tr & va),
                     "train_test_overlap": len(tr & test_idx),
                     "val_test_overlap": len(va & test_idx),
                     "total_rows": len(tr) + len(va) + len(test_idx),
                     "covers_full_dataset": len(tr | va | test_idx) == n_total})
    return pd.DataFrame(rows)

def save_split_indices(splits, X_test, dataset):
    """Save the original row indices of every seed/subset to a CSV file."""
    make_folders()
    rows = []
    for seed in SEEDS:
        for subset, idx in [("train", splits[seed]["X_train"].index),
                            ("validation", splits[seed]["X_val"].index),
                            ("test", X_test.index)]:
            rows.append({"dataset": dataset, "seed": seed, "subset": subset, "n": len(idx),
                         "original_row_indices": " ".join(str(i) for i in sorted(idx))})
    table = pd.DataFrame(rows)
    path = os.path.join(RESULTS_DIR, f"{dataset}_split_indices.csv")
    table.to_csv(path, index=False)
    return table, path



#Section 5: preprocessing and scaling
from sklearn.preprocessing import StandardScaler

def scale_data(splits, X_test):
    reps, scalers = {}, {}
    for seed in SEEDS:
        X_train, X_val = splits[seed]["X_train"], splits[seed]["X_val"]
        scaler = StandardScaler().fit(X_train)          # training data only
        X_train_scaled = pd.DataFrame(scaler.transform(X_train),
                                      columns=X_train.columns, index=X_train.index)
        X_val_scaled = pd.DataFrame(scaler.transform(X_val),
                                    columns=X_val.columns, index=X_val.index)
        scalers[seed] = scaler
        reps[seed] = {"unscaled": (X_train, X_val),
                      "scaled": (X_train_scaled, X_val_scaled)}
    X_test_scaled = pd.DataFrame(scalers[42].transform(X_test),
                                 columns=X_test.columns, index=X_test.index)
    test_reps = {"unscaled": X_test, "scaled": X_test_scaled}
    return reps, scalers, test_reps