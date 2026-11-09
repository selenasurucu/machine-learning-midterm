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

#Section 7: training the classifiers
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score)

DT_DEPTHS = [2, 3, 5, 10, None]
KNN_KS = [1, 3, 5, 7, 11]
REPRESENTATIONS = ["unscaled", "scaled"]


def model_grid():
    """The 11 required settings: (algorithm, parameter name, value, short name)."""
    grid = []
    for d in DT_DEPTHS:
        grid.append(("decision_tree", "max_depth", d, f"dt_{{rep}}_d{d}"))
    for k in KNN_KS:
        grid.append(("knn", "n_neighbors", k, f"knn_{{rep}}_k{k}"))
    grid.append(("gaussian_nb", "default", "default", "gnb_{rep}"))
    return grid


def make_model(algorithm, value, seed):
    if algorithm == "decision_tree":
        return DecisionTreeClassifier(max_depth=value, random_state=seed)
    if algorithm == "knn":
        return KNeighborsClassifier(n_neighbors=value, weights="uniform", metric="euclidean")
    return GaussianNB()


def roc_auc(dataset, y_true, proba, model_classes):
    model_classes = list(model_classes)
    if dataset == "breast_cancer":
        y_eval = (np.asarray(y_true) == 0).astype(int)   # 1 = malignant
        p_malignant = proba[:, model_classes.index(0)]   # probability of class 0
        return roc_auc_score(y_eval, p_malignant)
    assert model_classes == [0, 1, 2], "probability columns must follow class order"
    return roc_auc_score(y_true, proba, multi_class="ovr", average="macro",
                         labels=model_classes)


def undefined_note(y_true, y_pred, labels):
    notes = []
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    for c in labels:
        if (y_pred == c).sum() == 0:
            notes.append(f"precision undefined for class {c}")
        if (y_true == c).sum() == 0:
            notes.append(f"recall undefined for class {c}")
    return "; ".join(notes) if notes else "none"


def get_metrics(dataset, model, X, y, prefix):
    y_pred = model.predict(X)
    proba = model.predict_proba(X)
    labels = list(model.classes_)
    return {
        f"{prefix}_accuracy": accuracy_score(y, y_pred),
        f"{prefix}_macro_precision": precision_score(y, y_pred, average="macro", zero_division=0),
        f"{prefix}_macro_recall": recall_score(y, y_pred, average="macro", zero_division=0),
        f"{prefix}_macro_f1": f1_score(y, y_pred, average="macro", zero_division=0),
        f"{prefix}_roc_auc": roc_auc(dataset, y, proba, model.classes_),
        f"{prefix}_undefined_metrics": undefined_note(y, y_pred, labels),
    }


def train_all_models(dataset, reps, splits):
    rows, models_42 = [], {}
    for seed in SEEDS:
        y_train, y_val = splits[seed]["y_train"], splits[seed]["y_val"]
        for rep in REPRESENTATIONS:
            X_train, X_val = reps[seed][rep]
            for algorithm, param, value, name in model_grid():
                config_id = f"{dataset}_" + name.format(rep=rep)
                run_id = f"{config_id}_s{seed}"
                model = make_model(algorithm, value, seed)
                model.fit(X_train, y_train)
                train_pred = model.predict(X_train)
                row = {"dataset": dataset, "run_id": run_id, "config_id": config_id,
                       "seed": seed, "algorithm": algorithm, "representation": rep,
                       "parameter": param, "parameter_value": str(value),
                       "train_accuracy": accuracy_score(y_train, train_pred),
                       "train_macro_f1": f1_score(y_train, train_pred, average="macro",
                                                  zero_division=0)}
                row.update(get_metrics(dataset, model, X_val, y_val, "val"))
                rows.append(row)
                if seed == 42:
                    models_42[config_id] = model
    return pd.DataFrame(rows), models_42

#Section 8: validation summary and model selection
METRICS = ["train_accuracy", "train_macro_f1", "val_accuracy", "val_macro_precision",
           "val_macro_recall", "val_macro_f1", "val_roc_auc"]


def summarize_runs(runs):
    """22 rows: mean and sample standard deviation (ddof=1) over the three seeds."""
    keys = ["dataset", "config_id", "algorithm", "representation", "parameter",
            "parameter_value"]
    groups = runs.groupby(keys, sort=False)[METRICS]
    mean = groups.mean().add_suffix("_mean")
    std = groups.std(ddof=1).add_suffix("_std")
    summary = pd.concat([mean, std], axis=1).reset_index()
    summary["n_seeds"] = groups.size().values
    ordered = keys + [c for m in METRICS for c in (m + "_mean", m + "_std")] + ["n_seeds"]
    return summary[ordered]


def select_config(summary):
    """Highest mean validation macro-F1. Ties: higher mean validation accuracy,
    then lower macro-F1 std, then alphabetical config ID."""
    ranked = summary.sort_values(
        ["val_macro_f1_mean", "val_accuracy_mean", "val_macro_f1_std", "config_id"],
        ascending=[False, False, True, True]).reset_index(drop=True)
    return ranked.iloc[0], ranked


def plot_param_curve(summary, dataset, algorithm, xlabel, filename):
    """Mean training and validation macro-F1 for every parameter value,
    for both representations, with the seed std as error bars."""
    sub = summary[summary["algorithm"] == algorithm]
    order = [str(v) for v in (DT_DEPTHS if algorithm == "decision_tree" else KNN_KS)]
    colors = {"unscaled": "tab:orange", "scaled": "tab:blue"}
    x = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(8, 5))
    for rep in REPRESENTATIONS:
        s = sub[sub["representation"] == rep].set_index("parameter_value").loc[order]
        shift = -0.06 if rep == "unscaled" else 0.06
        ax.errorbar(x + shift, s["train_macro_f1_mean"], yerr=s["train_macro_f1_std"],
                    marker="o", linestyle="--", capsize=4, color=colors[rep],
                    label=f"{rep} - training")
        ax.errorbar(x + shift, s["val_macro_f1_mean"], yerr=s["val_macro_f1_std"],
                    marker="s", linestyle="-", capsize=4, color=colors[rep],
                    label=f"{rep} - validation")
    ax.set_xticks(x)
    ax.set_xticklabels(order)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Macro-F1 (mean over seeds 42, 52, 62)")
    ax.set_title(f"{dataset} - {algorithm}: training vs validation macro-F1\n"
                 "error bars = sample std across the three seeds")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    save_fig(fig, filename)
    plt.show()

    #Section 9: final test evaluation
from sklearn.metrics import confusion_matrix, classification_report


def plot_confusion(cm, class_names, title, filename):
    """Counts, rows = Actual, columns = Predicted."""
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    save_fig(fig, filename)
    plt.show()


def manual_metrics(cm, class_index, class_names):
    cm = np.asarray(cm)
    tp = cm[class_index, class_index]
    fn = cm[class_index, :].sum() - tp     
    fp = cm[:, class_index].sum() - tp      
    tn = cm.sum() - tp - fn - fp
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")   # nan = undefined
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    if np.isnan(precision) or np.isnan(recall) or precision + recall == 0:
        f1 = float("nan")
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return {"class": f"{class_index} ({class_names[class_index]})",
            "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
            "precision = TP/(TP+FP)": precision, "recall = TP/(TP+FN)": recall,
            "F1 = 2PR/(P+R)": f1}

from sklearn.metrics import roc_curve

def plot_roc(dataset, model, X_test, y_test, class_names, filename):
    proba = model.predict_proba(X_test)
    classes = list(model.classes_)
    fig, ax = plt.subplots(figsize=(6, 5))
    aucs = {}
    if dataset == "breast_cancer":
        y_eval = (np.asarray(y_test) == 0).astype(int)
        p = proba[:, classes.index(0)]
        fpr, tpr, _ = roc_curve(y_eval, p)
        auc = roc_auc_score(y_eval, p)
        aucs["malignant (class 0) vs benign"] = auc
        ax.plot(fpr, tpr, lw=2, label=f"malignant vs benign (AUC = {auc:.4f})")
    else:
        styles = ["-", "--", ":"]
        for c in classes:
            y_bin = (np.asarray(y_test) == c).astype(int)
            p = proba[:, classes.index(c)]
            fpr, tpr, _ = roc_curve(y_bin, p)
            auc = roc_auc_score(y_bin, p)
            aucs[f"class {c} ({class_names[c]}) vs rest"] = auc
            ax.plot(fpr, tpr, lw=4 - c, linestyle=styles[c],
                    label=f"class {c} ({class_names[c]}) vs rest (AUC = {auc:.4f})")
        aucs["macro OvR AUC"] = roc_auc_score(y_test, proba, multi_class="ovr",
                                              average="macro", labels=classes)
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="random guess")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"{dataset} - ROC curve on the fixed test set (seed-42 model)")
    ax.legend(fontsize=8, loc="lower right")
    save_fig(fig, filename)
    plt.show()
    return aucs

def wrong_predictions(model, X_test, y_test, class_names):
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)
    rows = []
    for i in np.where(pred != np.asarray(y_test))[0]:
        row = {"original_row_index": int(X_test.index[i]),
               "actual": f"{y_test.iloc[i]} ({class_names[y_test.iloc[i]]})",
               "predicted": f"{pred[i]} ({class_names[pred[i]]})"}
        for j, c in enumerate(model.classes_):
            row[f"P(class {c})"] = round(proba[i, j], 4)
        rows.append(row)
    return pd.DataFrame(rows)