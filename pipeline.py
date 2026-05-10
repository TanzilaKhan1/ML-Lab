"""
Safety classifier pipeline — bus/legua door-hanging detection.

Labels:
  positive (label=0) — people NOT hanging on the door (safe)
  negative (label=1) — people ARE hanging on the door (UNSAFE)

Primary metric: Recall on the UNSAFE class (label=1).
  Missing an unsafe passenger is worse than a false alarm.

Pipeline design:
  - Group-aware train/val/test split: all augmented copies of a source
    image stay in the same partition (prevents data leakage).
  - Scaler is fitted inside each CV fold via sklearn Pipeline (prevents
    distribution leakage from scaler fit on full training set).
  - Learning curve averages over multiple stratified random subsets.
  - Train accuracy is reported alongside val/test for bias-variance diagnosis.
"""

import re
import time
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from PIL import Image
from skimage.feature import hog

from sklearn.base import clone as sk_clone
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (
    StratifiedGroupKFold, StratifiedShuffleSplit, cross_val_score,
)
from sklearn.metrics import (
    accuracy_score, recall_score, classification_report,
    confusion_matrix, roc_curve, auc, precision_recall_curve,
    average_precision_score, cohen_kappa_score,
    matthews_corrcoef, ConfusionMatrixDisplay,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR   = Path("Final_Dataset/raw_images")
RESULTS    = Path("results")
IMG_SIZE   = (128, 128)
HOG_PARAMS = dict(
    orientations=9,
    pixels_per_cell=(16, 16),
    cells_per_block=(2, 2),
    channel_axis=-1,
)
VAL_FRAC   = 0.15
TEST_FRAC  = 0.15
SEED       = 42

CLASSES     = {"positive": 0, "negative": 1}
CLASS_NAMES = ["positive (safe)", "negative (UNSAFE)"]
UNSAFE_CLS  = 1   # the safety-critical class

RESULTS.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# HOG feature extraction
# ---------------------------------------------------------------------------
AUG_RE = re.compile(r"^(.+)_aug_\d+$")

def source_stem(stem: str) -> str:
    m = AUG_RE.match(stem)
    return m.group(1) if m else stem


def hog_features(path: Path) -> np.ndarray | None:
    try:
        img = Image.open(path).convert("RGB").resize(IMG_SIZE, Image.BICUBIC)
        return hog(np.array(img, dtype=np.uint8), **HOG_PARAMS).astype(np.float32)
    except Exception as e:
        print(f"  SKIP {path.name}: {e}")
        return None


# ---------------------------------------------------------------------------
# 1. Load dataset
# ---------------------------------------------------------------------------
def load_dataset(data_dir: Path):
    X_list, y_list, group_list = [], [], []
    group_counter = 0
    group_map: dict[str, int] = {}

    for cls_name, label in CLASSES.items():
        folder = data_dir / cls_name
        paths  = sorted(folder.glob("*.png")) + sorted(folder.glob("*.jpg"))
        print(f"  {cls_name}: {len(paths)} files  (label={label})")
        for path in paths:
            feats = hog_features(path)
            if feats is None:
                continue
            src = f"{cls_name}/{source_stem(path.stem)}"
            if src not in group_map:
                group_map[src] = group_counter
                group_counter += 1
            X_list.append(feats)
            y_list.append(label)
            group_list.append(group_map[src])

    return (
        np.array(X_list,    dtype=np.float32),
        np.array(y_list,    dtype=np.int32),
        np.array(group_list, dtype=np.int32),
    )


# ---------------------------------------------------------------------------
# 2. Group-stratified split
# ---------------------------------------------------------------------------
def group_split(X, y, groups, val_frac=VAL_FRAC, test_frac=TEST_FRAC, seed=SEED):
    rng = np.random.default_rng(seed)

    class_groups: dict[int, list[int]] = defaultdict(list)
    for g, label in zip(groups, y):
        if g not in class_groups[label]:
            class_groups[label].append(g)

    train_gs, val_gs, test_gs = set(), set(), set()
    for label, gs in class_groups.items():
        gs = np.array(gs)
        rng.shuffle(gs)
        n   = len(gs)
        n_t = max(1, round(n * test_frac))
        n_v = max(1, round(n * val_frac))
        test_gs.update(gs[:n_t])
        val_gs.update(gs[n_t:n_t + n_v])
        train_gs.update(gs[n_t + n_v:])

    def mask(group_set):
        idx = np.where(np.isin(groups, list(group_set)))[0]
        return X[idx], y[idx], groups[idx]

    X_train, y_train, g_train = mask(train_gs)
    X_val,   y_val,   g_val   = mask(val_gs)
    X_test,  y_test,  g_test  = mask(test_gs)

    # Guard: both classes must be present in every partition
    for name_p, ys in [("train", y_train), ("val", y_val), ("test", y_test)]:
        assert len(np.unique(ys)) == 2, \
            f"Partition '{name_p}' contains only one class — change SEED"

    return (X_train, y_train, g_train), (X_val, y_val, g_val), (X_test, y_test, g_test)


# ---------------------------------------------------------------------------
# 3. Evaluate one fitted model (scaler already baked into sklearn Pipeline)
# ---------------------------------------------------------------------------
def evaluate(name: str, pipe,
             X_train, y_train,
             X_val,   y_val,
             X_test,  y_test) -> dict:

    tag = name.replace(" ", "_").replace("(", "").replace(")", "")
    print(f"\n{'='*62}")
    print(f"  {name}")
    print(f"{'='*62}")

    results = {}
    splits = [
        ("Train",      X_train, y_train),
        ("Validation", X_val,   y_val),
        ("Test",       X_test,  y_test),
    ]

    for split_name, Xs, ys in splits:
        preds    = pipe.predict(Xs)
        acc      = accuracy_score(ys, preds)
        recall_u = recall_score(ys, preds, pos_label=UNSAFE_CLS)
        mcc      = matthews_corrcoef(ys, preds)
        kappa    = cohen_kappa_score(ys, preds)

        print(f"\n  [{split_name}]  n={len(ys)}")
        print(f"    Accuracy : {acc*100:.2f}%")
        print(f"    Recall (UNSAFE class) : {recall_u*100:.2f}%  ← primary metric")
        print(f"    MCC      : {mcc:.4f}")
        print(f"    Cohen κ  : {kappa:.4f}")
        print(classification_report(
            ys, preds, target_names=CLASS_NAMES, digits=3, zero_division=0
        ))
        results[split_name] = {"acc": acc, "recall_unsafe": recall_u, "mcc": mcc}

        if split_name == "Train":
            continue   # skip plots for train

        # Confusion matrix
        cm  = confusion_matrix(ys, preds)
        fig, ax = plt.subplots(figsize=(5, 4))
        ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES).plot(
            ax=ax, colorbar=False, cmap="Blues"
        )
        ax.set_title(f"{name} — {split_name}\n"
                     f"Acc {acc*100:.1f}%  |  Unsafe Recall {recall_u*100:.1f}%")
        plt.xticks(rotation=15, ha="right")
        plt.tight_layout()
        plt.savefig(RESULTS / f"{tag}_{split_name.lower()}_cm.png", dpi=110)
        plt.close()

    # ROC + PR curves on test set
    ys_test = y_test
    if hasattr(pipe, "predict_proba"):
        scores = pipe.predict_proba(X_test)[:, UNSAFE_CLS]
    else:
        scores = pipe.decision_function(X_test)

    fpr, tpr, _ = roc_curve(ys_test, scores, pos_label=UNSAFE_CLS)
    roc_auc     = auc(fpr, tpr)

    prec, rec, _ = precision_recall_curve(ys_test, scores, pos_label=UNSAFE_CLS)
    avg_prec     = average_precision_score(ys_test, scores, pos_label=UNSAFE_CLS)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc:.3f}")
    ax1.plot([0, 1], [0, 1], "k--", lw=1)
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.set_title(f"ROC Curve — {name}")
    ax1.legend(loc="lower right")

    ax2.plot(rec, prec, lw=2, label=f"AP = {avg_prec:.3f}")
    ax2.axhline(y=ys_test.mean(), color="k", linestyle="--", lw=1,
                label="Baseline (random)")
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title(f"Precision-Recall — {name}")
    ax2.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(RESULTS / f"{tag}_curves.png", dpi=110)
    plt.close()

    print(f"  Test  ROC-AUC : {roc_auc:.4f}")
    print(f"  Test  Avg Prec: {avg_prec:.4f}")
    results["test_auc"] = roc_auc
    results["test_ap"]  = avg_prec
    return results


# ---------------------------------------------------------------------------
# 4. Learning curve — averaged over multiple stratified random subsets
# ---------------------------------------------------------------------------
def plot_learning_curve(name: str, pipe, X_train, y_train, X_test, y_test,
                        n_repeats: int = 5):
    tag   = name.replace(" ", "_").replace("(", "").replace(")", "")
    fracs = [0.10, 0.20, 0.35, 0.50, 0.65, 0.80, 1.00]
    n     = len(y_train)

    tr_means, te_means, tr_stds, te_stds, sizes_used = [], [], [], [], []

    for frac in fracs:
        size = max(4, int(n * frac))
        tr_fold, te_fold = [], []

        repeats = 1 if frac == 1.0 else n_repeats
        for rep in range(repeats):
            if size >= n:
                idx = np.arange(n)
            else:
                spl = StratifiedShuffleSplit(
                    n_splits=1, train_size=size, random_state=SEED + rep
                )
                idx, _ = next(spl.split(X_train, y_train))

            if len(np.unique(y_train[idx])) < 2:
                continue

            try:
                c = sk_clone(pipe)
                c.fit(X_train[idx], y_train[idx])
                tr_fold.append(accuracy_score(y_train[idx], c.predict(X_train[idx])))
                te_fold.append(accuracy_score(y_test, c.predict(X_test)))
            except Exception:
                continue

        if len(tr_fold) == 0:
            continue
        tr_means.append(np.mean(tr_fold))
        te_means.append(np.mean(te_fold))
        tr_stds.append(np.std(tr_fold))
        te_stds.append(np.std(te_fold))
        sizes_used.append(size)

    if len(sizes_used) < 2:
        return

    tr_means = np.array(tr_means)
    te_means = np.array(te_means)
    tr_stds  = np.array(tr_stds)
    te_stds  = np.array(te_stds)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(sizes_used, tr_means, "o-", label="Train accuracy")
    ax.fill_between(sizes_used, tr_means - tr_stds, tr_means + tr_stds, alpha=0.2)
    ax.plot(sizes_used, te_means, "s-", label="Test accuracy")
    ax.fill_between(sizes_used, te_means - te_stds, te_means + te_stds, alpha=0.2)
    ax.set_xlabel("Training samples used")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Learning Curve — {name}\n(averaged over {n_repeats} random subsets)")
    ax.legend()
    ax.set_ylim(0.3, 1.05)
    plt.tight_layout()
    plt.savefig(RESULTS / f"{tag}_learning_curve.png", dpi=110)
    plt.close()


# ---------------------------------------------------------------------------
# 5. Cross-validation (scaler inside Pipeline, so no leakage)
# ---------------------------------------------------------------------------
def cross_validate_pipeline(name: str, pipe, X_train, y_train, groups_train):
    cv = StratifiedGroupKFold(n_splits=3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scores = cross_val_score(
            pipe, X_train, y_train,
            cv=cv, groups=groups_train,
            scoring="accuracy", n_jobs=-1, error_score=np.nan,
        )
        recall_scores = cross_val_score(
            pipe, X_train, y_train,
            cv=cv, groups=groups_train,
            scoring="recall", n_jobs=-1, error_score=np.nan,
        )

    valid_acc    = scores[~np.isnan(scores)]
    valid_recall = recall_scores[~np.isnan(recall_scores)]

    acc_mean = float(np.mean(valid_acc))    if len(valid_acc)    else float("nan")
    acc_std  = float(np.std(valid_acc))     if len(valid_acc)    else float("nan")
    rec_mean = float(np.mean(valid_recall)) if len(valid_recall) else float("nan")
    rec_std  = float(np.std(valid_recall))  if len(valid_recall) else float("nan")

    print(f"  Group 3-fold CV  Accuracy: {acc_mean*100:.2f}% ± {acc_std*100:.2f}%"
          f"  [{', '.join(f'{s*100:.1f}' for s in scores)}]")
    print(f"  Group 3-fold CV  Recall  : {rec_mean*100:.2f}% ± {rec_std*100:.2f}%"
          f"  [{', '.join(f'{s*100:.1f}' for s in recall_scores)}]")
    return acc_mean, acc_std, rec_mean, rec_std


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Loading images and extracting HOG features…")
    t0 = time.time()
    X, y, groups = load_dataset(DATA_DIR)
    print(f"  Feature matrix : {X.shape}")
    print(f"  Class counts   : safe={np.sum(y==0)}, unsafe={np.sum(y==1)}")
    print(f"  Source groups  : {len(np.unique(groups))}")
    print(f"  Time           : {time.time()-t0:.1f}s")

    (X_train, y_train, g_train), \
    (X_val,   y_val,   g_val),   \
    (X_test,  y_test,  g_test)   = group_split(X, y, groups)

    print(f"\nGroup-aware split  (no source image spans two partitions):")
    for pname, ys, gs in [("train", y_train, g_train),
                           ("val",   y_val,   g_val),
                           ("test",  y_test,  g_test)]:
        print(f"  {pname:5s}: {len(ys):4d} samples  "
              f"(safe={np.sum(ys==0)}, unsafe={np.sum(ys==1)})  "
              f"source groups={len(np.unique(gs))}")

    # Base estimators
    base_models = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, C=1.0, random_state=SEED
        ),
        "Naive Bayes": GaussianNB(),
        "SVM (RBF kernel)": SVC(
            kernel="rbf", C=10, gamma="scale",
            probability=True, random_state=SEED
        ),
    }

    # Wrap each in a Pipeline so the scaler is fitted per CV fold — no leakage
    pipelines = {
        name: Pipeline([("scaler", StandardScaler()), ("clf", clf)])
        for name, clf in base_models.items()
    }

    summary = {}
    for name, pipe in pipelines.items():
        print(f"\n{'─'*62}")
        print(f"Training {name}…")
        t0 = time.time()
        pipe.fit(X_train, y_train)
        print(f"  Trained in {time.time()-t0:.1f}s")

        print("  Cross-validation (scaler fitted per fold — no leakage):")
        acc_mean, acc_std, rec_mean, rec_std = cross_validate_pipeline(
            name, pipe, X_train, y_train, g_train
        )

        metrics = evaluate(name, pipe,
                           X_train, y_train,
                           X_val,   y_val,
                           X_test,  y_test)

        plot_learning_curve(name, pipe, X_train, y_train, X_test, y_test)

        summary[name] = {
            "cv_acc_mean":    acc_mean,
            "cv_acc_std":     acc_std,
            "cv_rec_mean":    rec_mean,
            "cv_rec_std":     rec_std,
            "train_acc":      metrics["Train"]["acc"],
            "train_recall":   metrics["Train"]["recall_unsafe"],
            "val_acc":        metrics["Validation"]["acc"],
            "val_recall":     metrics["Validation"]["recall_unsafe"],
            "test_acc":       metrics["Test"]["acc"],
            "test_recall":    metrics["Test"]["recall_unsafe"],
            "test_mcc":       metrics["Test"]["mcc"],
            "test_auc":       metrics.get("test_auc", float("nan")),
            "test_ap":        metrics.get("test_ap",  float("nan")),
        }

    # ---------------------------------------------------------------------------
    # Final summary table
    # ---------------------------------------------------------------------------
    print(f"\n{'='*75}")
    print("  FINAL SUMMARY")
    print(f"{'='*75}")
    hdr = f"  {'Model':<25} {'Train Acc':>9} {'Val Acc':>8} {'Test Acc':>9} {'Unsafe Rec':>11} {'AUC':>7} {'MCC':>7}"
    print(hdr)
    print(f"  {'-'*72}")
    for name, m in sorted(summary.items(), key=lambda x: -x[1]["test_auc"]):
        print(f"  {name:<25} "
              f"{m['train_acc']*100:>8.2f}%"
              f"{m['val_acc']*100:>8.2f}%"
              f"{m['test_acc']*100:>9.2f}%"
              f"{m['test_recall']*100:>10.2f}%"
              f"{m['test_auc']:>8.4f}"
              f"{m['test_mcc']:>8.4f}")

    # Bar chart — test accuracy + unsafe recall side by side
    names  = list(summary.keys())
    colors = ["#4C72B0", "#DD8452", "#55A868"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    for ax, (metric_key, label, fmt) in zip(axes, [
        ("test_acc",     "Test Accuracy (%)",        lambda v: f"{v*100:.1f}%"),
        ("test_recall",  "Unsafe Recall (%) — primary", lambda v: f"{v*100:.1f}%"),
        ("test_auc",     "Test ROC-AUC",             lambda v: f"{v:.3f}"),
    ]):
        vals = [summary[n][metric_key] for n in names]
        if "%" in label:
            vals_plot = [v * 100 for v in vals]
            lo = max(0, min(vals_plot) - 12)
            ax.set_xlim(lo, 100)
            text_offset = 0.5
        else:
            vals_plot = vals
            lo = max(0, min(vals) - 0.12)
            ax.set_xlim(lo, 1.05)
            text_offset = 0.005
        bars = ax.barh(names, vals_plot, color=colors)
        ax.set_title(label, fontsize=9)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_width() + text_offset,
                    bar.get_y() + bar.get_height() / 2,
                    fmt(v), va="center", fontsize=8, clip_on=False)

    plt.suptitle("Classifier Comparison — Safety Detection", fontweight="bold")
    plt.tight_layout()
    plt.savefig(RESULTS / "summary.png", dpi=120)
    plt.close()
    print(f"\nAll outputs saved to {RESULTS}/")


if __name__ == "__main__":
    main()
