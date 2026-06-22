"""Generate publication-quality figures + tables for all 9 deployed models.

Run from repo/model:  python make_paper_assets.py
Outputs -> repo/paper_assets/{figures, tables}

Evaluation sets:
  * Held-out TEST (n=58): common set for ALL 9 models -> ROC/PR/confusion/bars.
  * 5-fold CV OOF (n=327, deep models): more reliable -> supplementary ROC + table.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from sklearn.metrics import (accuracy_score, auc, average_precision_score,
                             confusion_matrix, f1_score, matthews_corrcoef,
                             precision_recall_curve, precision_score,
                             recall_score, roc_auc_score, roc_curve)

import du

REPO = du.ROOT.parent
PRED = REPO / "predictor"
sys.path.insert(0, str(PRED))
from predictor_app import inference as inf  # noqa: E402

OUT = REPO / "paper_assets"
FIG = OUT / "figures"; TAB = OUT / "tables"
FIG.mkdir(parents=True, exist_ok=True); TAB.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 200, "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
    "font.family": "DejaVu Sans",
})

# display order + colors by family
MODELS = [
    ("Logistic Regression", "Classical", "#60a5fa"),
    ("Naive Bayes",         "Classical", "#3b82f6"),
    ("SVM (RBF)",           "Classical", "#1d4ed8"),
    ("CNN",                 "CNN-scratch", "#34d399"),
    ("ResNet18",            "Transfer", "#fbbf24"),
    ("EfficientNet-B0",     "Transfer", "#fb923c"),
    ("ResNet50",            "Transfer", "#f97316"),
    ("ConvNeXt-Tiny",       "Transfer", "#ef4444"),
    ("Ensemble (best)",     "Ensemble", "#7c3aed"),
]
COL = {m: c for m, _, c in MODELS}
SHORT = {"Logistic Regression": "LogReg", "Naive Bayes": "NaiveBayes", "SVM (RBF)": "SVM",
         "CNN": "CNN", "ResNet18": "ResNet18", "EfficientNet-B0": "EffNet-B0",
         "ResNet50": "ResNet50", "ConvNeXt-Tiny": "ConvNeXt", "Ensemble (best)": "Ensemble"}


# --------------------------------------------------------------------------
def test_probs():
    """P(unsafe) + label for every model on the held-out test set (cached)."""
    cache = OUT / "_test_probs.npz"
    part = du.get_partition()
    tep, tel = part["test"]
    assert len(tep) > 0, "empty test set — run from repo/model so du paths resolve"
    y = np.array(tel)
    data = {"_labels": y}
    for name, _, _ in MODELS:
        ps = []
        for p in tep:
            pr = inf.predict(str(p), name)
            ps.append(pr.probabilities["positive (UNSAFE)"])
        data[name] = np.array(ps)
        print(f"  test probs: {name:20s} done")
    np.savez(cache, **{k.replace(" ", "~"): v for k, v in data.items()})
    return y, {name: data[name] for name, _, _ in MODELS}


def cv_oof():
    """Deep-model CV OOF P(unsafe) (n=327)."""
    hi = np.load(du.ROOT / "outputs_cv_hi" / "probs.npz")
    base = np.load(du.ROOT / "outputs_cv" / "probs.npz")
    y = hi["labels_tv"].astype(int)
    return y, {
        "ResNet50": hi["oof_resnet50"], "ConvNeXt-Tiny": hi["oof_convnext_tiny"],
        "EfficientNet-B0": base["oof_efficientnet_b0"], "Ensemble (best)": hi["oof_ensemble"],
    }


def deep_thresholds():
    th = {}
    for name, fn in [("ResNet50", "resnet50.joblib"), ("ConvNeXt-Tiny", "convnext_tiny.joblib"),
                     ("EfficientNet-B0", "efficientnet_b0.joblib")]:
        import joblib
        ck = joblib.load(inf.MODEL_DIR / fn)
        th[name] = float(ck.get("threshold_balanced", 0.5))
    ens = json.loads((inf.MODEL_DIR / "ensemble.json").read_text())
    th["Ensemble (best)"] = float(ens.get("threshold_balanced", 0.5))
    return th


def param_counts():
    from predictor_app.torch_models import _build_arch
    cnt = {}
    for n, bk in [("CNN", "smallcnn"), ("ResNet18", "resnet18"), ("ResNet50", "resnet50"),
                  ("EfficientNet-B0", "efficientnet_b0"), ("ConvNeXt-Tiny", "convnext_tiny")]:
        cnt[n] = sum(p.numel() for p in _build_arch(bk).parameters())
    return cnt


# =================== FIGURES ===================
def fig_dataset():
    recs = json.loads((REPO.parent / "data" / "label_map.json").read_text())
    from collections import Counter
    by = Counter((r["vehicle"], r["label"]) for r in recs)
    part = du.get_partition()
    sizes = {k: len(part[k][1]) for k in ("train", "val", "test")}
    split_pos = {k: int(sum(part[k][1])) for k in ("train", "val", "test")}

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    veh = ["bus", "legua"]
    safe = [by[(v, "safe")] for v in veh]; unsafe = [by[(v, "unsafe")] for v in veh]
    x = np.arange(len(veh)); w = 0.38
    ax[0].bar(x - w/2, safe, w, label="safe", color="#10b981")
    ax[0].bar(x + w/2, unsafe, w, label="unsafe", color="#ef4444")
    for i, (s, u) in enumerate(zip(safe, unsafe)):
        ax[0].text(i - w/2, s + 2, str(s), ha="center", fontsize=9)
        ax[0].text(i + w/2, u + 2, str(u), ha="center", fontsize=9)
    ax[0].set_xticks(x); ax[0].set_xticklabels(["Bus", "Leguna"])
    ax[0].set_ylabel("images"); ax[0].set_title("(a) Class distribution by vehicle")
    ax[0].legend()

    parts = ["train", "val", "test"]
    tot = [sizes[p] for p in parts]; pos = [split_pos[p] for p in parts]
    neg = [t - p for t, p in zip(tot, pos)]
    xb = np.arange(len(parts))
    ax[1].bar(xb, neg, 0.55, label="safe", color="#10b981")
    ax[1].bar(xb, pos, 0.55, bottom=neg, label="unsafe", color="#ef4444")
    for i, (t, p) in enumerate(zip(tot, pos)):
        ax[1].text(i, t + 3, f"{t}\n({p} unsafe)", ha="center", fontsize=9)
    ax[1].set_xticks(xb); ax[1].set_xticklabels([f"Train\n70%", f"Val\n15%", f"Test\n15%"])
    ax[1].set_ylabel("images"); ax[1].set_title("(b) Stratified 70-15-15 split")
    ax[1].legend()
    fig.suptitle("Dataset: 385 annotation-labelled images (287 safe / 98 unsafe)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG / "fig1_dataset_overview.png", bbox_inches="tight"); plt.close(fig)
    print("  fig1_dataset_overview.png")


def fig_bars(metrics):
    names = [m for m, _, _ in MODELS]
    acc = [metrics[m]["acc"] for m in names]
    rec = [metrics[m]["unsafe_recall"] for m in names]
    aucs = [metrics[m]["auc"] for m in names]
    x = np.arange(len(names)); w = 0.26
    fig, ax = plt.subplots(figsize=(13, 5.2))
    ax.bar(x - w, acc, w, label="Accuracy", color="#3b82f6")
    ax.bar(x, rec, w, label="Unsafe recall", color="#ef4444")
    ax.bar(x + w, aucs, w, label="ROC-AUC", color="#10b981")
    for i in range(len(names)):
        for off, v in [(-w, acc[i]), (0, rec[i]), (w, aucs[i])]:
            ax.text(i + off, v + 0.012, f"{v:.2f}", ha="center", fontsize=7.5, rotation=90)
    ax.set_xticks(x); ax.set_xticklabels([SHORT[n] for n in names], rotation=25, ha="right")
    ax.set_ylim(0, 1.08); ax.set_ylabel("score")
    ax.set_title("Model comparison on held-out test set (n=58)  ·  balanced operating point", fontweight="bold")
    ax.legend(ncol=3, loc="lower right")
    fig.tight_layout(); fig.savefig(FIG / "fig2_model_comparison.png", bbox_inches="tight"); plt.close(fig)
    print("  fig2_model_comparison.png")


def fig_roc(y, probs, fname, title):
    fig, ax = plt.subplots(figsize=(7, 6.2))
    order = sorted(probs, key=lambda m: -roc_auc_score(y, probs[m]))
    for m in order:
        fpr, tpr, _ = roc_curve(y, probs[m]); a = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=COL[m], lw=2, label=f"{SHORT[m]} (AUC={a:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate (unsafe recall)")
    ax.set_title(title, fontweight="bold"); ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout(); fig.savefig(FIG / fname, bbox_inches="tight"); plt.close(fig)
    print(f"  {fname}")


def fig_pr(y, probs):
    fig, ax = plt.subplots(figsize=(7, 6.2))
    order = sorted(probs, key=lambda m: -average_precision_score(y, probs[m]))
    for m in order:
        pr, rc, _ = precision_recall_curve(y, probs[m]); ap = average_precision_score(y, probs[m])
        ax.plot(rc, pr, color=COL[m], lw=2, label=f"{SHORT[m]} (AP={ap:.3f})")
    base = y.mean()
    ax.axhline(base, ls="--", color="k", alpha=0.5, lw=1, label=f"baseline ({base:.2f})")
    ax.set_xlabel("Recall (unsafe)"); ax.set_ylabel("Precision (unsafe)")
    ax.set_title("Precision-Recall — held-out test (n=58)", fontweight="bold")
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout(); fig.savefig(FIG / "fig4_pr_curves.png", bbox_inches="tight"); plt.close(fig)
    print("  fig4_pr_curves.png")


def fig_confusions(y, probs, thr):
    names = [m for m, _, _ in MODELS]
    fig = plt.figure(figsize=(13.5, 6))
    gs = GridSpec(2, 5, figure=fig, hspace=0.55, wspace=0.4)
    for k, m in enumerate(names):
        t = thr.get(m, 0.5)
        pred = (probs[m] >= t).astype(int)
        cm = confusion_matrix(y, pred, labels=[0, 1])
        ax = fig.add_subplot(gs[k // 5, k % 5])
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if cm[i, j] > cm.max()/2 else "black", fontsize=11)
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["safe", "uns"], fontsize=8); ax.set_yticklabels(["safe", "uns"], fontsize=8)
        acc = accuracy_score(y, pred)
        ax.set_title(f"{SHORT[m]}\nacc={acc:.2f} thr={t:.2f}", fontsize=9)
        ax.set_xlabel("pred", fontsize=8); ax.set_ylabel("true", fontsize=8)
    fig.add_subplot(gs[1, 4]).axis("off")
    fig.suptitle("Confusion matrices — held-out test (n=58)", fontweight="bold", y=0.98)
    fig.savefig(FIG / "fig5_confusion_matrices.png", bbox_inches="tight"); plt.close(fig)
    print("  fig5_confusion_matrices.png")


def fig_tradeoff():
    d = np.load(du.ROOT / "outputs_cv_hi" / "probs.npz")
    y = d["labels_tv"].astype(int); p = d["oof_ensemble"]
    ths = np.linspace(0.01, 0.99, 197)
    accs = [accuracy_score(y, (p >= t).astype(int)) for t in ths]
    recs = [recall_score(y, (p >= t).astype(int), pos_label=1, zero_division=0) for t in ths]
    precs = [precision_score(y, (p >= t).astype(int), pos_label=1, zero_division=0) for t in ths]
    fig, ax = plt.subplots(figsize=(8, 5.2))
    ax.plot(ths, accs, label="Accuracy", color="#3b82f6", lw=2)
    ax.plot(ths, recs, label="Unsafe recall", color="#ef4444", lw=2)
    ax.plot(ths, precs, label="Unsafe precision", color="#10b981", lw=2)
    for lbl, t, c in [("balanced", 0.57, "#7c3aed"), ("high-recall", 0.125, "#f59e0b"),
                      ("zero-miss", 0.017, "#111827")]:
        ax.axvline(t, ls="--", color=c, alpha=0.7)
        ax.text(t, 1.02, lbl, rotation=90, fontsize=8, color=c, va="bottom")
    ax.set_xlabel("decision threshold  P(unsafe)"); ax.set_ylabel("score")
    ax.set_title("Accuracy–recall tradeoff (Ensemble, 5-fold CV)", fontweight="bold")
    ax.legend(loc="center right"); ax.set_ylim(0, 1.12)
    fig.tight_layout(); fig.savefig(FIG / "fig6_tradeoff.png", bbox_inches="tight"); plt.close(fig)
    print("  fig6_tradeoff.png")


def fig_technique_progression():
    """AUC ceiling across the techniques tried for recall."""
    items = [("ResNet18\n224", 0.90), ("ResNet50\n224", 0.919), ("+320px", 0.919),
             ("Ensemble\n320", 0.929), ("+Tiling", 0.926), ("+Box-crops", 0.932)]
    labels = [i[0] for i in items]; vals = [i[1] for i in items]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.plot(range(len(items)), vals, "-o", color="#7c3aed", lw=2, ms=7)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.0015, f"{v:.3f}", ha="center", fontsize=9)
    ax.axhline(0.932, ls="--", color="#ef4444", alpha=0.6)
    ax.text(0.1, 0.9325, "ceiling ≈ 0.93 (data-limited)", color="#ef4444", fontsize=9)
    ax.set_xticks(range(len(items))); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Cross-validated ROC-AUC"); ax.set_ylim(0.88, 0.94)
    ax.set_title("AUC across recall-improvement techniques (CV)", fontweight="bold")
    fig.tight_layout(); fig.savefig(FIG / "fig7_technique_progression.png", bbox_inches="tight"); plt.close(fig)
    print("  fig7_technique_progression.png")


# =================== TABLES ===================
def metrics_from(y, probs, thr):
    out = {}
    for m in probs:
        t = thr.get(m, 0.5); pred = (probs[m] >= t).astype(int)
        out[m] = dict(
            acc=accuracy_score(y, pred),
            unsafe_recall=recall_score(y, pred, pos_label=1, zero_division=0),
            unsafe_precision=precision_score(y, pred, pos_label=1, zero_division=0),
            f1=f1_score(y, pred, pos_label=1, zero_division=0),
            auc=roc_auc_score(y, probs[m]) if len(set(y)) > 1 else float("nan"),
            ap=average_precision_score(y, probs[m]),
            mcc=matthews_corrcoef(y, pred) if len(set(pred)) > 1 else 0.0,
            threshold=t)
    return out


def write_tables(test_y, test_probs_d, thr, params, cv_y, cv_probs):
    tm = metrics_from(test_y, test_probs_d, thr)
    fam = {m: f for m, f, _ in MODELS}
    gpu = {"Logistic Regression": "CPU", "Naive Bayes": "CPU", "SVM (RBF)": "CPU"}
    # main CSV + MD
    cols = ["model", "family", "device", "params", "test_acc", "test_unsafe_recall",
            "test_unsafe_prec", "test_f1", "test_auc", "test_ap", "test_mcc"]
    rows = []
    for m, f, _ in MODELS:
        r = tm[m]
        rows.append([m, f, gpu.get(m, "GPU"),
                     f"{params[m]:,}" if m in params else "—",
                     f"{r['acc']:.3f}", f"{r['unsafe_recall']:.3f}", f"{r['unsafe_precision']:.3f}",
                     f"{r['f1']:.3f}", f"{r['auc']:.3f}", f"{r['ap']:.3f}", f"{r['mcc']:.3f}"])
    import csv
    with open(TAB / "table_models_test.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(cols); w.writerows(rows)
    md = ["# Model comparison — held-out test set (n=58), balanced operating point", "",
          "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        md.append("| " + " | ".join(str(x) for x in r) + " |")
    (TAB / "table_models_test.md").write_text("\n".join(md))

    # CV table (deep models)
    cm = metrics_from(cv_y, cv_probs, {k: 0.5 for k in cv_probs})  # AUC/AP threshold-free; acc@0.5 indicative
    # use balanced threshold from CV for acc
    cvb = {}
    for m in cv_probs:
        # balanced threshold on CV
        best = max(((np.mean([recall_score(cv_y,(cv_probs[m]>=t).astype(int),pos_label=0,zero_division=0),
                               recall_score(cv_y,(cv_probs[m]>=t).astype(int),pos_label=1,zero_division=0)]), t)
                    for t in np.linspace(0.05,0.95,181)))[1]
        cvb[m] = best
    cm = metrics_from(cv_y, cv_probs, cvb)
    cvcols = ["model", "cv_acc", "cv_unsafe_recall", "cv_unsafe_prec", "cv_f1", "cv_auc", "cv_ap", "cv_mcc"]
    cmd = ["# Cross-validated metrics (5-fold, n=327) — deep models, balanced threshold", "",
           "| " + " | ".join(cvcols) + " |", "|" + "---|" * len(cvcols)]
    for m in ["ResNet50", "ConvNeXt-Tiny", "EfficientNet-B0", "Ensemble (best)"]:
        r = cm[m]
        cmd.append(f"| {m} | {r['acc']:.3f} | {r['unsafe_recall']:.3f} | {r['unsafe_precision']:.3f} | "
                   f"{r['f1']:.3f} | {r['auc']:.3f} | {r['ap']:.3f} | {r['mcc']:.3f} |")
    (TAB / "table_models_cv.md").write_text("\n".join(cmd))

    # operating modes
    ens = json.loads((inf.MODEL_DIR / "ensemble.json").read_text())
    d = np.load(du.ROOT / "outputs_cv_hi" / "probs.npz"); yy = d["labels_tv"].astype(int); pp = d["oof_ensemble"]
    om = ["# Ensemble operating modes (CV, set via PREDICTOR_OP_MODE)", "",
          "| mode | threshold | cv_accuracy | cv_unsafe_recall | cv_unsafe_precision |",
          "|---|---|---|---|---|"]
    for mode, key in [("balanced", "threshold_balanced"), ("high_recall", "threshold_high_recall"),
                      ("max_recall (zero-miss)", "threshold_max_recall")]:
        t = float(ens[key]); pred = (pp >= t).astype(int)
        om.append(f"| {mode} | {t:.3f} | {accuracy_score(yy,pred):.3f} | "
                  f"{recall_score(yy,pred,pos_label=1,zero_division=0):.3f} | "
                  f"{precision_score(yy,pred,pos_label=1,zero_division=0):.3f} |")
    (TAB / "table_operating_modes.md").write_text("\n".join(om))
    print("  tables: table_models_test.{csv,md}, table_models_cv.md, table_operating_modes.md")
    return tm, rows, cols


def render_table_png(rows, cols):
    fig, ax = plt.subplots(figsize=(15, 3.8)); ax.axis("off")
    disp_cols = ["Model", "Family", "Dev", "Params", "Acc", "Recall↑", "Prec", "F1", "AUC", "AP", "MCC"]
    cell = [[SHORT[r[0]], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8], r[9], r[10]] for r in rows]
    tbl = ax.table(cellText=cell, colLabels=disp_cols, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1, 1.6)
    # widen the Model + Params columns, narrow the metric columns
    widths = [0.13, 0.11, 0.06, 0.11] + [0.066] * 7
    for j, wdt in enumerate(widths):
        for i in range(len(rows) + 1):
            tbl[i, j].set_width(wdt)
    for j in range(len(disp_cols)):
        tbl[0, j].set_facecolor("#1e293b"); tbl[0, j].set_text_props(color="white", fontweight="bold")
    for i, r in enumerate(rows, start=1):
        tbl[i, 0].set_facecolor(COL[r[0]]); tbl[i, 0].set_text_props(color="white", fontweight="bold")
    ax.set_title("Model comparison — held-out test (n=58), balanced operating point",
                 fontweight="bold", fontsize=12, pad=14)
    fig.savefig(FIG / "fig3_results_table.png", bbox_inches="tight", dpi=200); plt.close(fig)
    print("  fig3_results_table.png")


def main():
    print("== evaluating all models on test set ==")
    ty, tprobs = test_probs()
    cy, cprobs = cv_oof()
    params = param_counts()
    # thresholds: classical -> 0.5; deep -> balanced
    thr = {"Logistic Regression": 0.5, "Naive Bayes": 0.5, "SVM (RBF)": 0.5, "CNN": 0.5, "ResNet18": 0.5}
    thr.update(deep_thresholds())
    tm = metrics_from(ty, tprobs, thr)

    print("== figures ==")
    fig_dataset()
    fig_bars(tm)
    fig_roc(ty, tprobs, "fig3b_roc_test.png", "ROC — held-out test (n=58)")
    fig_roc(cy, cprobs, "fig8_roc_cv_deep.png", "ROC — 5-fold CV (deep models, n=327)")
    fig_pr(ty, tprobs)
    fig_confusions(ty, tprobs, thr)
    fig_tradeoff()
    fig_technique_progression()
    print("== tables ==")
    _, rows, cols = write_tables(ty, tprobs, thr, params, cy, cprobs)
    render_table_png(rows, cols)

    # console summary
    print("\n=== TEST-SET METRICS (n=58) ===")
    print(f"{'model':20s} {'acc':>6s} {'recall':>7s} {'prec':>6s} {'f1':>6s} {'auc':>6s} {'mcc':>6s}")
    for m, _, _ in MODELS:
        r = tm[m]
        print(f"{m:20s} {r['acc']:6.3f} {r['unsafe_recall']:7.3f} {r['unsafe_precision']:6.3f} "
              f"{r['f1']:6.3f} {r['auc']:6.3f} {r['mcc']:6.3f}")
    print(f"\nassets -> {OUT}")


if __name__ == "__main__":
    main()
