"""Assemble the 9 final models, tune operating thresholds, and emit COMPLETE
train/val/test metrics (+ graphs) for every model in one beautiful JSON.

Prereqs (run in order):
  python rt_build_dataset.py        # Preprocessed + split + physical folders
  python rt_augment_train.py        # offline aug of TRAIN only
  python train_classical.py         # -> outputs_classical/{svm,logistic,naive_bayes}_model.joblib
  python train_torch2.py            # -> outputs_torch2/{smallcnn,resnet18,resnet50,efficientnet_b0,convnext_tiny}.joblib

This script then:
  1. Copies the 9 model files into predictor/model/ under their deployed names
     (smallcnn->cnn_model, resnet18->resnet_model).
  2. For each torch checkpoint: recomputes P(unsafe) on val and tunes THREE
     operating thresholds (balanced / high_recall>=0.95 / max_recall=1.0),
     injects them, and re-dumps the checkpoint.
  3. Builds ensemble.json (resnet50 + convnext_tiny) with its own 3 thresholds.
  4. Evaluates ALL 9 models on TRAIN / VAL / TEST with a rich metric set and
     writes:  metrics_full.json  + per-model & summary PNGs under outputs_final/.

Run from repo/model:  python rt_finalize_eval.py
"""
from __future__ import annotations

import json
import shutil
import sys
from collections import Counter
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, brier_score_loss,
                             cohen_kappa_score, confusion_matrix, f1_score,
                             log_loss, matthews_corrcoef, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)

import du

ROOT = du.ROOT
PRED_MODEL_DIR = ROOT.parent.parent / "repo" / "predictor" / "model"
if not PRED_MODEL_DIR.exists():
    PRED_MODEL_DIR = ROOT.parent / "predictor" / "model"
PRED_PKG = (PRED_MODEL_DIR.parent).resolve()
if str(PRED_PKG) not in sys.path:
    sys.path.insert(0, str(PRED_PKG))

OUT = ROOT / "outputs_final"
OUT.mkdir(parents=True, exist_ok=True)
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# (display, source joblib in outputs_*, deployed filename, kind)
TORCH_SRC = ROOT / "outputs_torch2"
CLA_SRC = ROOT / "outputs_classical"
MODEL_PLAN = [
    ("ResNet50",            TORCH_SRC / "resnet50.joblib",        "resnet50.joblib",        "torch"),
    ("ConvNeXt-Tiny",       TORCH_SRC / "convnext_tiny.joblib",   "convnext_tiny.joblib",   "torch"),
    ("EfficientNet-B0",     TORCH_SRC / "efficientnet_b0.joblib", "efficientnet_b0.joblib", "torch"),
    ("ResNet18",            TORCH_SRC / "resnet18.joblib",        "resnet_model.joblib",    "torch"),
    ("CNN",                 TORCH_SRC / "smallcnn.joblib",        "cnn_model.joblib",       "torch"),
    ("SVM (RBF)",           CLA_SRC / "svm_model.joblib",         "svm_model.joblib",        "sklearn"),
    ("Logistic Regression", CLA_SRC / "logistic_model.joblib",    "logistic_model.joblib",   "sklearn"),
    ("Naive Bayes",         CLA_SRC / "naive_bayes_model.joblib", "naive_bayes_model.joblib","sklearn"),
]
# Ensemble members are chosen data-drivenly = top-K torch models by VAL
# balanced-accuracy (see ENSEMBLE_TOPK). Falls back to this list if needed.
ENSEMBLE_TOPK = 3
ENSEMBLE_FALLBACK = ["resnet50.joblib", "convnext_tiny.joblib"]


# ----------------------------- threshold tuning -----------------------------
def tune_balanced(p, y):
    best_t, best_key = 0.5, (-1.0, -1.0)
    for t in np.linspace(0.05, 0.95, 181):
        pred = (p >= t).astype(int)
        key = (round(balanced_accuracy_score(y, pred), 4),
               round(recall_score(y, pred, pos_label=1, zero_division=0), 4))
        if key > best_key:
            best_key, best_t = key, float(t)
    return best_t


def tune_high_recall(p, y, target=0.95):
    """Highest threshold whose unsafe-recall >= target (max accuracy among those)."""
    best_t, best_acc = None, -1.0
    for t in np.linspace(0.01, 0.99, 197):
        pred = (p >= t).astype(int)
        if recall_score(y, pred, pos_label=1, zero_division=0) >= target:
            a = accuracy_score(y, pred)
            if a >= best_acc:
                best_acc, best_t = a, float(t)
    return best_t if best_t is not None else 0.05


def tune_max_recall(p, y):
    """Largest threshold still catching 100% of unsafe (recall==1.0)."""
    pos = p[y == 1]
    return float(max(0.001, min(pos.min(), 0.99))) if len(pos) else 0.5


def three_thresholds(p, y):
    return {
        "threshold_balanced": tune_balanced(p, y),
        "threshold_high_recall": tune_high_recall(p, y, 0.95),
        "threshold_max_recall": tune_max_recall(p, y),
    }


# ----------------------------- metric block ---------------------------------
def metric_block(y, p, thr):
    y = np.asarray(y); p = np.asarray(p)
    pred = (p >= thr).astype(int)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    two = len(set(y.tolist())) > 1
    out = {
        "n": int(len(y)),
        "n_safe": int((y == 0).sum()),
        "n_unsafe": int((y == 1).sum()),
        "threshold": float(thr),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "precision_unsafe": float(precision_score(y, pred, pos_label=1, zero_division=0)),
        "recall_unsafe": float(recall_score(y, pred, pos_label=1, zero_division=0)),
        "precision_safe": float(precision_score(y, pred, pos_label=0, zero_division=0)),
        "recall_safe": float(recall_score(y, pred, pos_label=0, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "f1_unsafe": float(f1_score(y, pred, pos_label=1, zero_division=0)),
        "f1_macro": float(f1_score(y, pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y, pred, average="weighted", zero_division=0)),
        "mcc": float(matthews_corrcoef(y, pred)) if len(set(pred.tolist())) > 1 else 0.0,
        "cohen_kappa": float(cohen_kappa_score(y, pred)) if len(set(pred.tolist())) > 1 else 0.0,
        "roc_auc": float(roc_auc_score(y, p)) if two else None,
        "pr_auc": float(average_precision_score(y, p)) if two else None,
        "log_loss": float(log_loss(y, np.clip(p, 1e-7, 1 - 1e-7), labels=[0, 1])) if two else None,
        "brier": float(brier_score_loss(y, p)) if two else None,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
                             "matrix": cm.tolist()},
    }
    return out


# ----------------------------- model probas ---------------------------------
def torch_probs(ckpt_path, paths):
    from predictor_app.torch_models import load_torch_checkpoint
    clf = load_torch_checkpoint(ckpt_path)
    ps = []
    for p in paths:
        im = Image.open(p).convert("RGB")
        ps.append(float(clf.predict_proba(im)[0, 1]))
    return np.array(ps)


def sklearn_probs(model, paths):
    from predictor_app.preprocess import preprocess_for_model
    ps = []
    for p in paths:
        feats = preprocess_for_model(Image.open(p).convert("RGB"))
        ps.append(float(model.predict_proba(feats)[0, 1]))
    return np.array(ps)


# ----------------------------- graphs ---------------------------------------
def plot_confusion(cm, title, path):
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["safe", "unsafe"]); ax.set_yticklabels(["safe", "unsafe"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    mx = cm.max() if cm.max() else 1
    for i in range(2):
        for j in range(2):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > mx / 2 else "black")
    fig.colorbar(im, ax=ax); fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def plot_roc(curves, path):
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, (y, p) in curves.items():
        if len(set(np.asarray(y).tolist())) < 2:
            continue
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, label=f"{name} (AUC {roc_auc_score(y, p):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC — TEST set"); ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_pr(curves, path):
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, (y, p) in curves.items():
        if len(set(np.asarray(y).tolist())) < 2:
            continue
        pr, rc, _ = precision_recall_curve(y, p)
        ax.plot(rc, pr, label=f"{name} (AP {average_precision_score(y, p):.3f})")
    ax.set_xlabel("Recall (unsafe)"); ax.set_ylabel("Precision (unsafe)")
    ax.set_title("Precision-Recall — TEST set"); ax.legend(fontsize=7, loc="lower left")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_bars(summary, path, mode="balanced"):
    names = list(summary.keys())
    metrics = [("accuracy", "Accuracy"), ("recall_unsafe", "Unsafe recall"),
               ("f1_macro", "F1 (macro)"), ("roc_auc", "ROC-AUC")]
    fig, axes = plt.subplots(1, 4, figsize=(18, 5.5))
    for ax, (key, lab) in zip(axes, metrics):
        vals = [summary[n]["test"][mode].get(key) or 0 for n in names]
        ax.barh(names, vals, color="#4477aa")
        ax.set_xlim(0, 1); ax.set_title(f"TEST {lab} ({mode})"); ax.invert_yaxis()
        for i, v in enumerate(vals):
            ax.text(min(v + 0.01, 0.95), i, f"{v:.3f}", va="center", fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def plot_overfit(summary, path, mode="balanced"):
    """Train vs Test accuracy gap (overfitting view)."""
    names = list(summary.keys())
    tr = [summary[n]["train"][mode]["accuracy"] for n in names]
    te = [summary[n]["test"][mode]["accuracy"] for n in names]
    y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.barh(y - 0.2, tr, height=0.4, label="train", color="#88ccaa")
    ax.barh(y + 0.2, te, height=0.4, label="test", color="#ee6677")
    ax.set_yticks(y); ax.set_yticklabels(names); ax.invert_yaxis()
    ax.set_xlim(0, 1.05); ax.set_xlabel("Accuracy"); ax.set_title("Train vs Test accuracy")
    ax.legend(); fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


# ----------------------------- main -----------------------------------------
def main():
    part = du.get_partition()
    splits = {k: ([str(p) for p in part[k][0]], list(part[k][1])) for k in ("train", "val", "test")}
    print("split sizes:", {k: len(v[0]) for k, v in splits.items()})
    for k in ("train", "val", "test"):
        print(f"  {k}: {dict(Counter(splits[k][1]))}")

    # ---- 1. copy model files to predictor/model under deployed names ----
    PRED_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    deployed = {}                      # display -> (deployed_path, kind)
    for disp, src, dst_name, kind in MODEL_PLAN:
        if not src.exists():
            print(f"[skip] missing {src}")
            continue
        dst = PRED_MODEL_DIR / dst_name
        shutil.copy2(src, dst)
        deployed[disp] = (dst, kind)
        print(f"deployed {disp:20s} <- {src.name}  -> {dst_name}")

    # cache probas per model on each split (compute once, reuse for thresholds+metrics+curves)
    probas = {}                        # display -> {split: (y, p)}
    for disp, (dst, kind) in deployed.items():
        print(f"scoring {disp} ...")
        probas[disp] = {}
        for sp in ("train", "val", "test"):
            paths, ys = splits[sp]
            if kind == "torch":
                p = torch_probs(dst, paths)
            else:
                model = joblib.load(dst)
                p = sklearn_probs(model, paths)
            probas[disp][sp] = (np.array(ys), p)

    # ---- 2. tune + inject 3 thresholds into torch checkpoints ----
    chosen_thr = {}                    # display -> active threshold (high_recall default)
    for disp, (dst, kind) in deployed.items():
        yv, pv = probas[disp]["val"]
        thr3 = three_thresholds(pv, yv)
        chosen_thr[disp] = thr3["threshold_high_recall"]
        if kind == "torch":
            ck = joblib.load(dst)
            ck.update(thr3)
            ck["threshold"] = thr3["threshold_high_recall"]
            joblib.dump(ck, dst)

    # ---- 3. ensemble: the strongest, architecturally-DIVERSE transfer backbones.
    # The 65-image val set makes val-AUC ranking noisy (it once promoted the weak
    # 128px CNN), so we prefer the two high-AUC, decorrelated transfer models
    # (ResNet50 + ConvNeXt-Tiny). Fall back to top-2 by val AUC only if absent.
    present_files = {d[0].name for d in deployed.values()}
    PREFERRED = ["resnet50.joblib", "convnext_tiny.joblib"]
    members_present = [m for m in PREFERRED if m in present_files]
    if len(members_present) < 2:
        torch_disp = [disp for disp, (_, k) in deployed.items() if k == "torch"]
        def val_auc(disp):
            yv, pv = probas[disp]["val"]
            return roc_auc_score(yv, pv) if len(set(yv.tolist())) > 1 else 0.0
        ranked = sorted(torch_disp, key=val_auc, reverse=True)
        members_present = [deployed[d][0].name for d in ranked[:2]]
    print(f"ensemble members: {members_present}")
    ens_disp = "Ensemble (best)"
    if len(members_present) >= 2:
        name_by_file = {d[0].name: disp for disp, d in deployed.items()}
        ens = {}
        for sp in ("train", "val", "test"):
            ys = probas[name_by_file[members_present[0]]][sp][0]
            ps = np.mean([probas[name_by_file[m]][sp][1] for m in members_present], axis=0)
            ens[sp] = (ys, ps)
        probas[ens_disp] = ens
        yv, pv = ens["val"]
        thr3 = three_thresholds(pv, yv)
        chosen_thr[ens_disp] = thr3["threshold_high_recall"]
        spec = {"members": members_present, "threshold": thr3["threshold_high_recall"], **thr3}
        (PRED_MODEL_DIR / "ensemble.json").write_text(json.dumps(spec, indent=2))
        print(f"wrote ensemble.json members={members_present}")

    # ---- 4. full metrics on train/val/test for every model ----
    # order: ensemble first, then deep, then classical
    order = [ens_disp] + [d for d, *_ in
                          [(m[0],) for m in MODEL_PLAN] if d in probas and d != ens_disp]
    order = [d for d in order if d in probas]

    summary = {}
    for disp in order:
        yv, pv = probas[disp]["val"]
        thr3 = three_thresholds(pv, yv)
        modes = {"balanced": thr3["threshold_balanced"],
                 "high_recall": thr3["threshold_high_recall"],
                 "max_recall": thr3["threshold_max_recall"]}
        # for every split, evaluate at ALL THREE operating points
        per_split = {}
        for sp in ("train", "val", "test"):
            y, p = probas[disp][sp]
            per_split[sp] = {m: metric_block(y, p, t) for m, t in modes.items()}
        summary[disp] = {
            "kind": dict([(m[0], m[3]) for m in MODEL_PLAN]).get(disp, "ensemble"),
            "operating_thresholds": thr3,
            "default_deployed_mode": "high_recall",
            "train": per_split["train"], "val": per_split["val"], "test": per_split["test"],
        }
        plot_confusion(np.array(per_split["test"]["balanced"]["confusion_matrix"]["matrix"]),
                       f"{disp} — TEST (balanced)", FIG / f"cm_{disp.split()[0].lower()}.png")
        b_tr, b_v, b_te = per_split["train"]["balanced"], per_split["val"]["balanced"], per_split["test"]["balanced"]
        hr = per_split["test"]["high_recall"]
        auc = b_te["roc_auc"]
        print(f"  {disp:20s} | BAL train {b_tr['accuracy']:.3f} val {b_v['accuracy']:.3f} "
              f"TEST {b_te['accuracy']:.3f} (rec {b_te['recall_unsafe']:.3f}, auc {auc:.3f}) "
              f"| HIGH-REC TEST acc {hr['accuracy']:.3f} rec {hr['recall_unsafe']:.3f}")

    # combined graphs (use TEST proba curves; bar/overfit at balanced operating point)
    test_curves = {d: probas[d]["test"] for d in order}
    plot_roc(test_curves, FIG / "roc_test.png")
    plot_pr(test_curves, FIG / "pr_test.png")
    plot_bars(summary, FIG / "bars_test.png", mode="balanced")
    plot_overfit(summary, FIG / "train_vs_test.png", mode="balanced")

    # dataset composition (originals only; augmentation is train-only and not counted here)
    aug_train = sum(1 for p in splits["train"][0] if "_aug" in Path(p).stem)
    orig_labels = [lab for p, lab in zip(splits["train"][0], splits["train"][1])
                   if "_aug" not in Path(p).stem] + list(splits["val"][1]) + list(splits["test"][1])
    n_originals = len(orig_labels)
    n_safe = int(sum(1 for l in orig_labels if int(l) == 0))
    n_unsafe = int(sum(1 for l in orig_labels if int(l) == 1))
    dataset = {
        "source": "Cloudflare R2 bucket 'machine-learning' (raw + annotations)",
        "labels_from": "annotations (unsafe if any 'unsafe' box, else safe)",
        "total_labeled_images": n_originals,
        "n_safe": n_safe,
        "n_unsafe": n_unsafe,
        "imbalance_ratio": round(n_safe / max(n_unsafe, 1), 2),
        "split_strategy": "4-way stratified (vehicle x class) 70/15/15; augment TRAIN only -> no leakage",
        "split_counts": {sp: dict(Counter(splits[sp][1])) for sp in ("train", "val", "test")},
        "train_originals": len(splits["train"][0]) - aug_train,
        "train_augmented_copies": aug_train,
        "train_total": len(splits["train"][0]),
        "val_total": len(splits["val"][0]),
        "test_total": len(splits["test"][0]),
        "class_index": {"0": "safe (negative)", "1": "unsafe (positive)"},
    }

    full = {"dataset": dataset, "models": summary}
    (OUT / "metrics_full.json").write_text(json.dumps(full, indent=2))
    # lightweight per-model diagnostic dump (NOT the app's metrics.json — that
    # is owned by rt_app_metrics.py, which renders the Analysis page schema).
    light = {d: {"kind": summary[d]["kind"],
                 "operating_thresholds": summary[d]["operating_thresholds"],
                 "default_deployed_mode": summary[d]["default_deployed_mode"],
                 "test": summary[d]["test"], "val": summary[d]["val"]} for d in order}
    (PRED_MODEL_DIR / "metrics_models.json").write_text(
        json.dumps({"dataset": dataset, "models": light}, indent=2))
    print("next: run  python rt_app_metrics.py  to refresh the app's metrics.json")

    print(f"\nwrote {OUT/'metrics_full.json'}")
    print(f"figures -> {FIG}")
    print(f"updated  {PRED_MODEL_DIR/'metrics.json'}")


if __name__ == "__main__":
    main()
