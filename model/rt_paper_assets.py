"""Generate the FULL set of paper assets — warm, polished, publication-grade —
from the new 432-image retrain.

Sources:
  - outputs_final/metrics_full.json   (train/val/test x balanced/high_recall/max_recall x 21 metrics)
  - outputs_final/probs_cache.npz     (per-image P(unsafe) on val+test for ROC/PR/tradeoff)
  - data/label_map.json               (dataset composition)

Writes into repo/paper_assets/ :
  figures/  fig01..fig12 (dataset, model comparison, ROC, PR, confusion grid,
            tradeoff, train/val/test error, generalization gap, metric heatmap,
            results table, operating modes, per-class recall)
  tables/   table_models_test.{csv,md}, table_train_val_test.{csv,md},
            table_operating_modes.md, table_dataset.md
  README.md

Run from repo/model:  python rt_paper_assets.py
"""
from __future__ import annotations
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score

import du

REPO = du.ROOT.parent
OUT = REPO / "paper_assets"
FIG = OUT / "figures"; TAB = OUT / "tables"
MET = du.ROOT / "outputs_final" / "metrics_full.json"
NPZ = du.ROOT / "outputs_final" / "probs_cache.npz"
LABELS = REPO.parent / "data" / "label_map.json"

# ───────────────────────── clean paper theme (white bg, muted palette) ───────
CREAM   = "#FFFFFF"   # white background
PANEL   = "#FFFFFF"
INK     = "#1A1A1A"   # near-black text
SUBINK  = "#555555"
GRIDC   = "#E8E8E8"
SAFE_C  = "#4C72B0"   # calm blue
UNSAFE_C= "#C44E52"   # muted red
ACCENT  = "#2F4B7C"   # deep navy (ensemble / highlight)

ORDER = ["Ensemble (best)", "ResNet50", "ConvNeXt-Tiny", "EfficientNet-B0",
         "ResNet18", "CNN", "SVM (RBF)", "Logistic Regression", "Naive Bayes"]
SHORT = {"Ensemble (best)": "Ensemble", "ResNet50": "ResNet50", "ConvNeXt-Tiny": "ConvNeXt-T",
         "EfficientNet-B0": "EffNet-B0", "ResNet18": "ResNet18", "CNN": "CNN",
         "SVM (RBF)": "SVM", "Logistic Regression": "LogReg", "Naive Bayes": "NaiveBayes"}
KEY = {"Ensemble (best)": "ensemble", "ResNet50": "resnet50", "ConvNeXt-Tiny": "convnext",
       "EfficientNet-B0": "efficientnet", "ResNet18": "resnet18", "CNN": "cnn",
       "SVM (RBF)": "svm", "Logistic Regression": "logreg", "Naive Bayes": "nb"}
COLOR = {"Ensemble (best)": "#2F4B7C", "ResNet50": "#4C72B0", "ConvNeXt-Tiny": "#55A868",
         "EfficientNet-B0": "#8172B3", "ResNet18": "#DD8452", "CNN": "#64B5CD",
         "SVM (RBF)": "#C44E52", "Logistic Regression": "#937860", "Naive Bayes": "#8C8C8C"}
# clean white→blue sequential for confusion / heatmap (replaces the warm map)
WARM = LinearSegmentedColormap.from_list("paperblue", ["#FFFFFF", "#C6DBEF", "#6BAED6", "#2171B5", "#08306B"])

plt.rcParams.update({
    "figure.facecolor": CREAM, "savefig.facecolor": CREAM, "axes.facecolor": PANEL,
    "figure.dpi": 150, "savefig.dpi": 200, "savefig.bbox": "tight",
    "font.family": "DejaVu Sans", "font.size": 11.5,
    "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": "#BBBBBB",
    "xtick.color": SUBINK, "ytick.color": SUBINK,
    "axes.titlesize": 14, "axes.titleweight": "bold", "axes.titlepad": 12,
    "axes.grid": True, "grid.color": GRIDC, "grid.linewidth": 0.9, "grid.alpha": 0.9,
    "legend.frameon": False, "axes.spines.top": False, "axes.spines.right": False,
})


def despine(ax):
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)


def suptitle(fig, title, sub=None):
    fig.suptitle(title, fontsize=16, fontweight="bold", color=INK, y=0.985)
    if sub:
        fig.text(0.5, 0.94, sub, ha="center", fontsize=10.5, color=SUBINK)


# ───────────────────────── load data ─────────────────────────
M = json.loads(MET.read_text())
D = M["dataset"]; MODELS = M["models"]
Z = np.load(NPZ)
yv, yt = Z["y_val"], Z["y_test"]
TEST_N = len(yt)


def bal(m, split):  # balanced-operating-point block
    return MODELS[m][split]["balanced"]


def test_p(m):
    return Z[f"p_{KEY[m]}_test"]


# ───────────────────────── FIG 1: dataset overview ─────────────────────────
def fig_dataset():
    recs = json.loads(LABELS.read_text())
    safe = sum(1 for r in recs if r["label"] == "safe")
    unsafe = sum(1 for r in recs if r["label"] == "unsafe")
    veh = Counter((r["vehicle"], r["label"]) for r in recs)
    sc = D["split_counts"]

    fig, ax = plt.subplots(2, 2, figsize=(13, 9.8))
    fig.subplots_adjust(hspace=0.42, wspace=0.28, top=0.87)

    # (a) donut safe/unsafe
    a = ax[0, 0]
    w, _, at = a.pie([safe, unsafe], colors=[SAFE_C, UNSAFE_C], startangle=90,
                     wedgeprops=dict(width=0.42, edgecolor=CREAM, linewidth=3),
                     autopct=lambda p: f"{p:.0f}%", pctdistance=0.79,
                     textprops=dict(color="white", fontweight="bold", fontsize=12))
    a.text(0, 0, f"{safe+unsafe}\nimages", ha="center", va="center", fontsize=14,
           fontweight="bold", color=INK)
    a.legend([f"safe  ({safe})", f"unsafe  ({unsafe})"], loc="lower center",
             bbox_to_anchor=(0.5, -0.16), ncol=2)
    a.set_title("Class balance (annotation-derived)")

    # (b) vehicle x class
    b = ax[0, 1]
    vehicles = ["bus", "legua"]
    s_vals = [veh[(v, "safe")] for v in vehicles]; u_vals = [veh[(v, "unsafe")] for v in vehicles]
    x = np.arange(2)
    b.bar(x - 0.2, s_vals, 0.38, label="safe", color=SAFE_C)
    b.bar(x + 0.2, u_vals, 0.38, label="unsafe", color=UNSAFE_C)
    for i, v in enumerate(s_vals): b.text(i - 0.2, v + 2, v, ha="center", fontweight="bold", color=INK)
    for i, v in enumerate(u_vals): b.text(i + 0.2, v + 2, v, ha="center", fontweight="bold", color=INK)
    b.set_xticks(x); b.set_xticklabels(["bus", "leguna"]); b.set_ylabel("images")
    b.set_title("Vehicle × class"); b.legend(); despine(b)

    # (c) split composition (val/test originals; train shown originals+aug)
    c = ax[1, 0]
    parts = ["train", "val", "test"]
    safe_o = [sc["train"]["0"] - 0, sc["val"]["0"], sc["test"]["0"]]
    # train balanced counts already include aug; show real split sizes
    safes = [sc[p]["0"] for p in parts]; unsafes = [sc[p]["1"] for p in parts]
    x = np.arange(3)
    c.bar(x, safes, 0.55, label="safe", color=SAFE_C)
    c.bar(x, unsafes, 0.55, bottom=safes, label="unsafe", color=UNSAFE_C)
    for i, p in enumerate(parts):
        c.text(i, safes[i] + unsafes[i] + 12, f"{safes[i]+unsafes[i]}", ha="center",
               fontweight="bold", color=INK)
    c.set_xticks(x); c.set_xticklabels([f"train\n(augmented)", "val", "test"]); c.set_ylabel("images")
    c.set_title("Split composition  (70 / 15 / 15, no leakage)"); c.legend(); despine(c)

    # (d) train augmentation growth
    d = ax[1, 1]
    labels = ["originals", "+ augmented"]
    vals = [D["train_originals"], D["train_total"]]
    bars = d.bar(labels, vals, color=["#A6BDDB", ACCENT], width=0.55)
    for r, v in zip(bars, vals):
        d.text(r.get_x() + r.get_width()/2, v + 25, f"{v}", ha="center", fontweight="bold", color=INK)
    d.annotate(f"×{D['train_total']/max(D['train_originals'],1):.1f}",
               xy=(1, vals[1]), xytext=(0.45, vals[1]*0.82), fontsize=15, fontweight="bold",
               color=ACCENT, ha="center")
    d.set_ylabel("train images"); d.set_title("A–Z augmentation (TRAIN only)"); despine(d)

    suptitle(fig, "Dataset Overview",
             f"{safe+unsafe} annotated bus/leguna images  ·  hanging-passenger safe vs unsafe")
    fig.savefig(FIG / "fig01_dataset_overview.png"); plt.close(fig)
    print("  fig01_dataset_overview")


# ───────────────────────── FIG 2: model comparison bars ─────────────────────────
def fig_model_comparison():
    metrics = [("accuracy", "Accuracy"), ("recall_unsafe", "Unsafe recall"),
               ("f1_unsafe", "F1 (unsafe)"), ("roc_auc", "ROC-AUC")]
    fig, axes = plt.subplots(1, 4, figsize=(19, 6.6))
    names = ORDER[::-1]
    for ax, (k, lab) in zip(axes, metrics):
        vals = [bal(m, "test").get(k) or 0 for m in names]
        cols = [COLOR[m] for m in names]
        bars = ax.barh(range(len(names)), vals, color=cols,
                       edgecolor=[INK if m == "Ensemble (best)" else "none" for m in names],
                       linewidth=[1.6 if m == "Ensemble (best)" else 0 for m in names])
        ax.set_yticks(range(len(names))); ax.set_yticklabels([SHORT[m] for m in names])
        ax.set_xlim(0, 1.0); ax.set_title(lab); despine(ax)
        ax.xaxis.grid(True); ax.yaxis.grid(False)
        for i, v in enumerate(vals):
            ax.text(min(v + 0.015, 0.99), i, f"{v:.3f}", va="center", fontsize=9.5,
                    fontweight="bold", color=INK)
    suptitle(fig, "Model Comparison — held-out TEST",
             f"n = {TEST_N}  ·  balanced operating point  ·  Ensemble = ResNet50 + ConvNeXt-Tiny")
    fig.tight_layout(rect=[0, 0, 1, 0.90]); fig.savefig(FIG / "fig02_model_comparison.png"); plt.close(fig)
    print("  fig02_model_comparison")


# ───────────────────────── FIG 3/4: ROC + PR ─────────────────────────
def fig_roc():
    fig, ax = plt.subplots(figsize=(7.6, 6.8))
    for m in ORDER:
        p = test_p(m); auc = roc_auc_score(yt, p)
        fpr, tpr, _ = roc_curve(yt, p)
        lw = 3.0 if m == "Ensemble (best)" else 1.7
        ax.plot(fpr, tpr, color=COLOR[m], lw=lw, label=f"{SHORT[m]}  ({auc:.3f})",
                zorder=5 if m == "Ensemble (best)" else 3)
    ax.plot([0, 1], [0, 1], "--", color="#999999", lw=1)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate (unsafe)")
    despine(ax)
    ax.legend(title="model (AUC)", fontsize=9.5, loc="lower right")
    suptitle(fig, "Receiver Operating Characteristic", f"held-out test, n = {TEST_N}")
    fig.tight_layout(rect=[0, 0, 1, 0.90]); fig.savefig(FIG / "fig03_roc_test.png"); plt.close(fig)
    print("  fig03_roc_test")


def fig_pr():
    base = float((yt == 1).mean())
    fig, ax = plt.subplots(figsize=(7.6, 6.8))
    for m in ORDER:
        p = test_p(m); ap = average_precision_score(yt, p)
        pr, rc, _ = precision_recall_curve(yt, p)
        lw = 3.0 if m == "Ensemble (best)" else 1.7
        ax.plot(rc, pr, color=COLOR[m], lw=lw, label=f"{SHORT[m]}  ({ap:.3f})",
                zorder=5 if m == "Ensemble (best)" else 3)
    ax.axhline(base, ls="--", color="#999999", lw=1, label=f"baseline ({base:.2f})")
    ax.set_xlabel("Recall (unsafe)"); ax.set_ylabel("Precision (unsafe)")
    despine(ax)
    ax.legend(title="model (AP)", fontsize=9.5, loc="lower left")
    suptitle(fig, "Precision–Recall", f"held-out test, n = {TEST_N}")
    fig.tight_layout(rect=[0, 0, 1, 0.90]); fig.savefig(FIG / "fig04_pr_curves.png"); plt.close(fig)
    print("  fig04_pr_curves")


# ───────────────────────── FIG 5: confusion grid ─────────────────────────
def fig_confusions():
    fig, axes = plt.subplots(3, 3, figsize=(12.5, 12))
    for ax, m in zip(axes.ravel(), ORDER):
        cm = np.array(bal(m, "test")["confusion_matrix"]["matrix"])
        im = ax.imshow(cm, cmap=WARM, vmin=0, vmax=cm.max())
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["safe", "unsafe"], fontsize=9); ax.set_yticklabels(["safe", "unsafe"], fontsize=9)
        ax.set_title(f"{SHORT[m]}\nacc {bal(m,'test')['accuracy']:.2f} · rec {bal(m,'test')['recall_unsafe']:.2f}",
                     fontsize=11)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=13,
                        fontweight="bold", color="white" if cm[i, j] > cm.max()*0.55 else INK)
        ax.set_xlabel("predicted", fontsize=8.5); ax.set_ylabel("true", fontsize=8.5)
        ax.grid(False)
    suptitle(fig, "Confusion Matrices — TEST", f"balanced operating point, n = {TEST_N}")
    fig.tight_layout(rect=[0, 0, 1, 0.90]); fig.savefig(FIG / "fig05_confusion_matrices.png"); plt.close(fig)
    print("  fig05_confusion_matrices")


# ───────────────────────── FIG 6: accuracy↔recall tradeoff ─────────────────────────
def fig_tradeoff():
    m = "Ensemble (best)"; p = test_p(m); y = yt
    ts = np.linspace(0.01, 0.99, 199)
    acc = [( (p >= t).astype(int) == y).mean() for t in ts]
    rec = [ ((p >= t).astype(int)[y == 1] == 1).mean() for t in ts]
    thr = MODELS[m]["operating_thresholds"]
    fig, ax = plt.subplots(figsize=(9.2, 6.2))
    ax.plot(ts, acc, color="#4C72B0", lw=2.4, label="accuracy")
    ax.plot(ts, rec, color="#C44E52", lw=2.4, label="unsafe recall")
    # mark the 3 operating points as legend markers (no overlapping vertical text)
    marks = [("balanced", thr["threshold_balanced"], "#2F4B7C", "o"),
             ("high-recall (deployed)", thr["threshold_high_recall"], "#55A868", "s"),
             ("max-recall (zero-miss)", thr["threshold_max_recall"], "#8172B3", "^")]
    for name, t, c, mk in marks:
        ax.axvline(t, color=c, ls=":", lw=1.2, alpha=0.7)
        ry = ((p >= t).astype(int)[y == 1] == 1).mean()
        ax.scatter([t], [ry], color=c, s=120, marker=mk, zorder=6, edgecolor="white",
                   linewidth=1.2, label=f"{name}  (t = {t:.2f})")
    ax.set_xlabel("decision threshold  P(unsafe)"); ax.set_ylabel("score")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.05); despine(ax)
    ax.legend(loc="lower center", fontsize=9.5, frameon=True, framealpha=0.95,
              edgecolor="#DDDDDD", ncol=1)
    suptitle(fig, "Operating-Point Trade-off", f"held-out test, n = {TEST_N}  ·  three deployable modes")
    fig.tight_layout(rect=[0, 0, 1, 0.90]); fig.savefig(FIG / "fig06_tradeoff.png"); plt.close(fig)
    print("  fig06_tradeoff")


# ───────────────────────── FIG 7: train/val/test ERROR ─────────────────────────
def fig_errors():
    names = ORDER
    err = {sp: [1 - bal(m, sp)["accuracy"] for m in names] for sp in ("train", "val", "test")}
    x = np.arange(len(names)); w = 0.26
    fig, ax = plt.subplots(figsize=(15, 7))
    ax.bar(x - w, err["train"], w, label="train error", color="#A6BDDB")
    ax.bar(x,     err["val"],   w, label="val error",   color="#6BAED6")
    ax.bar(x + w, err["test"],  w, label="test error",  color=ACCENT)
    for i in range(len(names)):
        for off, sp in [(-w, "train"), (0, "val"), (w, "test")]:
            ax.text(i + off, err[sp][i] + 0.006, f"{err[sp][i]*100:.0f}", ha="center",
                    fontsize=8, color=INK)
    ax.set_xticks(x); ax.set_xticklabels([SHORT[m] for m in names], rotation=20, ha="right")
    ax.set_ylabel("error rate  (1 − accuracy)"); ax.set_ylim(0, max(err["test"]) + 0.12)
    despine(ax); ax.legend()
    suptitle(fig, "Generalization — Error by Split", "balanced operating point  ·  lower is better")
    fig.tight_layout(rect=[0, 0, 1, 0.90]); fig.savefig(FIG / "fig07_train_val_test_error.png"); plt.close(fig)
    print("  fig07_train_val_test_error")


# ───────────────────────── FIG 8: generalization gap (dumbbell) ─────────────────────────
def fig_gap():
    names = ORDER[::-1]
    tr = [bal(m, "train")["accuracy"] for m in names]
    te = [bal(m, "test")["accuracy"] for m in names]
    y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(10.5, 7))
    for i in range(len(names)):
        ax.plot([te[i], tr[i]], [i, i], color="#CCCCCC", lw=3, zorder=1)
    ax.scatter(tr, y, s=90, color="#A6BDDB", label="train acc", zorder=3, edgecolor=INK, linewidth=0.6)
    ax.scatter(te, y, s=110, color=ACCENT, label="test acc", zorder=3, edgecolor="white", linewidth=0.8)
    for i in range(len(names)):
        ax.text((tr[i]+te[i])/2, i + 0.18, f"Δ{(tr[i]-te[i])*100:.0f}", ha="center", fontsize=8.4,
                color=SUBINK)
    ax.set_yticks(y); ax.set_yticklabels([SHORT[m] for m in names])
    ax.set_xlim(0.5, 1.02); ax.set_xlabel("accuracy")
    despine(ax); ax.xaxis.grid(True); ax.yaxis.grid(False); ax.legend(loc="lower left")
    suptitle(fig, "Overfitting View", "gap = train − test accuracy")
    fig.tight_layout(rect=[0, 0, 1, 0.90]); fig.savefig(FIG / "fig08_generalization_gap.png"); plt.close(fig)
    print("  fig08_generalization_gap")


# ───────────────────────── FIG 9: metric heatmap ─────────────────────────
def fig_heatmap():
    cols = [("accuracy", "Acc"), ("balanced_accuracy", "BalAcc"), ("recall_unsafe", "Recall"),
            ("precision_unsafe", "Prec"), ("specificity", "Spec"), ("f1_unsafe", "F1"),
            ("roc_auc", "AUC"), ("pr_auc", "PR-AUC"), ("mcc", "MCC")]
    data = np.array([[bal(m, "test").get(k) or 0 for k, _ in cols] for m in ORDER])
    fig, ax = plt.subplots(figsize=(11.5, 7))
    im = ax.imshow(data, cmap=WARM, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(cols))); ax.set_xticklabels([c for _, c in cols])
    ax.set_yticks(range(len(ORDER))); ax.set_yticklabels([SHORT[m] for m in ORDER])
    for i in range(len(ORDER)):
        for j in range(len(cols)):
            ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center", fontsize=9,
                    fontweight="bold", color="white" if data[i, j] > 0.62 else INK)
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02); cb.outline.set_visible(False)
    suptitle(fig, "All Metrics at a Glance — TEST", f"balanced operating point  ·  n = {TEST_N}")
    fig.tight_layout(rect=[0, 0, 1, 0.91]); fig.savefig(FIG / "fig09_metric_heatmap.png"); plt.close(fig)
    print("  fig09_metric_heatmap")


# ───────────────────────── FIG 10: results table ─────────────────────────
def fig_table():
    cols = ["Model", "Acc", "BalAcc", "Recall↑", "Prec", "Spec", "F1", "AUC", "PR-AUC", "MCC"]
    keys = ["accuracy", "balanced_accuracy", "recall_unsafe", "precision_unsafe",
            "specificity", "f1_unsafe", "roc_auc", "pr_auc", "mcc"]
    cell = []
    for m in ORDER:
        b = bal(m, "test")
        cell.append([SHORT[m]] + [f"{(b.get(k) or 0):.3f}" for k in keys])
    fig, ax = plt.subplots(figsize=(13.5, 4.4)); ax.axis("off")
    tbl = ax.table(cellText=cell, colLabels=cols, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1, 1.7)
    for j in range(len(cols)):
        tbl[0, j].set_facecolor(INK); tbl[0, j].set_text_props(color="white", fontweight="bold")
    for i, m in enumerate(ORDER, start=1):
        tbl[i, 0].set_facecolor(COLOR[m]); tbl[i, 0].set_text_props(color="white", fontweight="bold")
        for j in range(1, len(cols)):
            tbl[i, j].set_facecolor("#FFFFFF" if i % 2 else "#F0F3F8")
    ax.set_title("Model comparison — held-out test, balanced operating point",
                 fontweight="bold", fontsize=13, pad=16, color=INK)
    fig.savefig(FIG / "fig10_results_table.png", dpi=200); plt.close(fig)
    print("  fig10_results_table")


# ───────────────────────── FIG 11: operating modes (ensemble) ─────────────────────────
def fig_modes():
    m = "Ensemble (best)"; modes = ["balanced", "high_recall", "max_recall"]
    keys = [("accuracy", "Accuracy"), ("recall_unsafe", "Unsafe recall"),
            ("precision_unsafe", "Precision"), ("specificity", "Specificity")]
    x = np.arange(len(modes)); w = 0.2
    cols = ["#4C72B0", "#55A868", "#DD8452", "#8172B3"]
    fig, ax = plt.subplots(figsize=(10, 6.4))
    for i, (k, lab) in enumerate(keys):
        vals = [MODELS[m]["test"][mode].get(k) or 0 for mode in modes]
        bars = ax.bar(x + (i - 1.5)*w, vals, w, label=lab, color=cols[i])
        for r, v in zip(bars, vals):
            ax.text(r.get_x()+r.get_width()/2, v + 0.01, f"{v:.2f}", ha="center", fontsize=8.3, color=INK)
    ax.set_xticks(x); ax.set_xticklabels(["balanced", "high-recall\n(deployed)", "max-recall\n(zero-miss)"])
    ax.set_ylim(0, 1.12); ax.set_ylabel("score")
    despine(ax); ax.legend(ncol=2, loc="lower center")
    suptitle(fig, "Three Deployable Operating Modes", f"Ensemble on test, n = {TEST_N}")
    fig.tight_layout(rect=[0, 0, 1, 0.90]); fig.savefig(FIG / "fig11_operating_modes.png"); plt.close(fig)
    print("  fig11_operating_modes")


# ───────────────────────── FIG 12: per-class recall ─────────────────────────
def fig_per_class():
    names = ORDER
    sr = [bal(m, "test")["recall_safe"] for m in names]
    ur = [bal(m, "test")["recall_unsafe"] for m in names]
    x = np.arange(len(names)); w = 0.38
    fig, ax = plt.subplots(figsize=(14.5, 6.6))
    ax.bar(x - w/2, sr, w, label="safe recall (specificity)", color=SAFE_C)
    ax.bar(x + w/2, ur, w, label="unsafe recall (sensitivity)", color=UNSAFE_C)
    for i in range(len(names)):
        ax.text(i - w/2, sr[i] + 0.01, f"{sr[i]:.2f}", ha="center", fontsize=8, color=INK)
        ax.text(i + w/2, ur[i] + 0.01, f"{ur[i]:.2f}", ha="center", fontsize=8, color=INK)
    ax.set_xticks(x); ax.set_xticklabels([SHORT[m] for m in names], rotation=20, ha="right")
    ax.set_ylim(0, 1.16); ax.set_ylabel("recall")
    despine(ax); ax.legend()
    suptitle(fig, "Per-Class Recall", f"balanced operating point, n = {TEST_N}")
    fig.tight_layout(rect=[0, 0, 1, 0.90]); fig.savefig(FIG / "fig12_per_class_recall.png"); plt.close(fig)
    print("  fig12_per_class_recall")


# ───────────────────────── TABLES ─────────────────────────
def _md(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"]*len(headers)) + " |"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def tables():
    import csv
    # table 1: test metrics
    keys = [("accuracy", "Acc"), ("balanced_accuracy", "BalAcc"), ("recall_unsafe", "Recall"),
            ("precision_unsafe", "Prec"), ("specificity", "Spec"), ("f1_unsafe", "F1"),
            ("roc_auc", "AUC"), ("pr_auc", "PR-AUC"), ("mcc", "MCC")]
    headers = ["Model"] + [h for _, h in keys]
    rows = [[SHORT[m]] + [f"{(bal(m,'test').get(k) or 0):.3f}" for k, _ in keys] for m in ORDER]
    (TAB / "table_models_test.md").write_text("### Held-out TEST metrics (balanced op-point, n=%d)\n\n%s\n" % (TEST_N, _md(headers, rows)))
    with (TAB / "table_models_test.csv").open("w", newline="") as f:
        w = csv.writer(f); w.writerow(headers); w.writerows(rows)

    # table 2: train/val/test accuracy + error + test recall
    h2 = ["Model", "Train acc", "Train err", "Val acc", "Val err", "Test acc", "Test err", "Test recall"]
    r2 = []
    for m in ORDER:
        tr, va, te = bal(m, "train")["accuracy"], bal(m, "val")["accuracy"], bal(m, "test")["accuracy"]
        r2.append([SHORT[m], f"{tr:.3f}", f"{1-tr:.3f}", f"{va:.3f}", f"{1-va:.3f}",
                   f"{te:.3f}", f"{1-te:.3f}", f"{bal(m,'test')['recall_unsafe']:.3f}"])
    (TAB / "table_train_val_test.md").write_text("### Train / Val / Test accuracy & error\n\n" + _md(h2, r2) + "\n")
    with (TAB / "table_train_val_test.csv").open("w", newline="") as f:
        w = csv.writer(f); w.writerow(h2); w.writerows(r2)

    # table 3: ensemble operating modes
    m = "Ensemble (best)"
    h3 = ["Mode", "Threshold", "Test acc", "Unsafe recall", "Precision", "Specificity"]
    r3 = []
    tn = {"balanced": "threshold_balanced", "high_recall": "threshold_high_recall", "max_recall": "threshold_max_recall"}
    for mode in ("balanced", "high_recall", "max_recall"):
        b = MODELS[m]["test"][mode]
        r3.append([mode, f"{MODELS[m]['operating_thresholds'][tn[mode]]:.3f}",
                   f"{b['accuracy']:.3f}", f"{b['recall_unsafe']:.3f}",
                   f"{b['precision_unsafe']:.3f}", f"{b['specificity']:.3f}"])
    (TAB / "table_operating_modes.md").write_text("### Ensemble — selectable operating modes (test)\n\n" + _md(h3, r3) + "\n")

    # table 4: dataset
    h4 = ["Split", "Total", "Safe", "Unsafe"]
    sc = D["split_counts"]
    r4 = [["train (augmented)", D["train_total"], sc["train"]["0"], sc["train"]["1"]],
          ["val", D["val_total"], sc["val"]["0"], sc["val"]["1"]],
          ["test", D["test_total"], sc["test"]["0"], sc["test"]["1"]],
          ["originals (labeled)", D["total_labeled_images"], 292, 140]]
    (TAB / "table_dataset.md").write_text("### Dataset composition\n\n" + _md(h4, r4) + "\n")
    print("  tables: table_models_test.{md,csv}, table_train_val_test.{md,csv}, "
          "table_operating_modes.md, table_dataset.md")


def readme():
    best = bal("Ensemble (best)", "test")
    txt = f"""# Paper Assets — Hanging-Passenger (Safe/Unsafe) Classifier

Regenerated from the **432-image** retrain (annotation-derived labels; train-only
A–Z augmentation → 1600 train images; honest 65-image val & test holdouts).
Theme: warm publication palette. All numbers come from `model/outputs_final/metrics_full.json`.

**Headline:** Ensemble (ResNet50 + ConvNeXt-Tiny, TTA) — TEST accuracy
{best['accuracy']*100:.1f}%, unsafe-recall {best['recall_unsafe']*100:.1f}%, ROC-AUC {best['roc_auc']:.3f}.

## Figures (`figures/`)
| File | What |
|---|---|
| fig01_dataset_overview.png | class balance, vehicle×class, split composition, augmentation growth |
| fig02_model_comparison.png | accuracy / unsafe-recall / F1 / AUC across all 9 models |
| fig03_roc_test.png | ROC curves (test) |
| fig04_pr_curves.png | precision–recall curves (test) |
| fig05_confusion_matrices.png | confusion matrix grid (test, balanced) |
| fig06_tradeoff.png | accuracy↔recall trade-off with the 3 operating modes |
| fig07_train_val_test_error.png | **train / val / test error** per model |
| fig08_generalization_gap.png | train vs test accuracy (overfitting gap) |
| fig09_metric_heatmap.png | full metric heatmap (test) |
| fig10_results_table.png | rendered results table |
| fig11_operating_modes.png | ensemble balanced / high-recall / max-recall |
| fig12_per_class_recall.png | safe vs unsafe recall per model |

## Tables (`tables/`)
- `table_models_test.{{md,csv}}` — all test metrics
- `table_train_val_test.{{md,csv}}` — train/val/test accuracy & error
- `table_operating_modes.md` — ensemble operating modes
- `table_dataset.md` — dataset composition
"""
    (OUT / "README.md").write_text(txt)
    print("  README.md")


def main():
    import shutil
    for d in (FIG, TAB):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)
    # drop the stale tracked probs cache if present
    for stale in (OUT / "_test_probs.npz",):
        if stale.exists():
            stale.unlink()
    print("== figures ==")
    fig_dataset(); fig_model_comparison(); fig_roc(); fig_pr(); fig_confusions()
    fig_tradeoff(); fig_errors(); fig_gap(); fig_heatmap(); fig_table(); fig_modes(); fig_per_class()
    print("== tables ==")
    tables(); readme()
    print(f"\nassets -> {OUT}")


if __name__ == "__main__":
    main()
