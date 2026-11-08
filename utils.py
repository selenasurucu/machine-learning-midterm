# utils.py
# Functions shared by breast_cancer.ipynb and wine.ipynb.
# Every function works on one dataset, so the two datasets stay separate.

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