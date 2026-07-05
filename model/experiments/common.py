"""Shared utilities for the imbalance + variance experiments (self-contained).

Reuses the canonical du split READ-ONLY but restricts TRAIN to the ORIGINAL
imbalanced images (204 safe / 98 unsafe) — augmented copies are excluded — so the
sampling / variance techniques are compared on a true-imbalance baseline. VAL/TEST
are the real held-out originals (65 each). Nothing here writes outside experiments/.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageFile
from skimage.color import rgb2gray
from skimage.feature import hog
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, confusion_matrix, f1_score,
                             matthews_corrcoef, precision_score, recall_score,
                             roc_auc_score)

ImageFile.LOAD_TRUNCATED_IMAGES = True
MODEL_DIR = Path(__file__).resolve().parent.parent          # repo/model
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))
import du  # noqa: E402

OUT = Path(__file__).resolve().parent / "outputs"
FIG = OUT / "figures"
OUT.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)

HOG_SIZE = 128
HOG_PARAMS = dict(orientations=9, pixels_per_cell=(16, 16),
                  cells_per_block=(2, 2), block_norm="L2-Hys")
SEED = 42


def load_splits(originals_only: bool = True):
    """split -> (paths:list[str], labels:np.ndarray[int]).  TRAIN = originals only."""
    part = du.get_partition()
    out = {}
    for k in ("train", "val", "test"):
        paths, labels = part[k]
        rows = [(str(p), y) for p, y in zip(paths, labels)
                if not (originals_only and "_aug" in Path(p).stem)]
        ps = [r[0] for r in rows]; ys = np.array([r[1] for r in rows], dtype=int)
        out[k] = (ps, ys)
    return out


# ----------------------------- HOG features (classical) -----------------------------
def _load_small(path):
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB").resize((HOG_SIZE, HOG_SIZE), Image.LANCZOS),
                          dtype=np.uint8)


def hog_features(paths):
    feats = []
    for p in paths:
        arr = _load_small(p).astype(np.float32) / 255.0
        feats.append(hog(rgb2gray(arr), **HOG_PARAMS).astype(np.float32))
    return np.stack(feats)


# ----------------------------- metrics -----------------------------
def tune_threshold(p, y, mode="balanced", target=0.95):
    """balanced -> max balanced-acc; high_recall -> highest thr with recall>=target."""
    best_t = 0.5
    if mode == "balanced":
        best_key = (-1.0, -1.0)
        for t in np.linspace(0.05, 0.95, 181):
            pred = (p >= t).astype(int)
            key = (round(balanced_accuracy_score(y, pred), 4),
                   round(recall_score(y, pred, pos_label=1, zero_division=0), 4))
            if key > best_key:
                best_key, best_t = key, float(t)
    else:
        best_acc = -1.0; best_t = 0.05
        for t in np.linspace(0.01, 0.99, 197):
            pred = (p >= t).astype(int)
            if recall_score(y, pred, pos_label=1, zero_division=0) >= target:
                a = accuracy_score(y, pred)
                if a >= best_acc:
                    best_acc, best_t = a, float(t)
    return best_t


def metric_block(y, p, thr):
    y = np.asarray(y); p = np.asarray(p)
    pred = (p >= thr).astype(int)
    cm = confusion_matrix(y, pred, labels=[0, 1]); tn, fp, fn, tp = cm.ravel()
    two = len(set(y.tolist())) > 1
    return {
        "threshold": round(float(thr), 4),
        "accuracy": round(float(accuracy_score(y, pred)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y, pred)), 4),
        "recall_unsafe": round(float(recall_score(y, pred, pos_label=1, zero_division=0)), 4),
        "precision_unsafe": round(float(precision_score(y, pred, pos_label=1, zero_division=0)), 4),
        "specificity": round(float(tn / (tn + fp)) if (tn + fp) else 0.0, 4),
        "f1_unsafe": round(float(f1_score(y, pred, pos_label=1, zero_division=0)), 4),
        "mcc": round(float(matthews_corrcoef(y, pred)), 4) if len(set(pred.tolist())) > 1 else 0.0,
        "roc_auc": round(float(roc_auc_score(y, p)), 4) if two else None,
        "pr_auc": round(float(average_precision_score(y, p)), 4) if two else None,
        "error": round(1 - float(accuracy_score(y, pred)), 4),
        "cm": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def summarize(name, family, probs):
    """probs = {'train':(y,p),'val':(y,p),'test':(y,p)} -> a result record.

    Threshold is tuned on VAL (balanced); applied to all splits.
    """
    yv, pv = probs["val"]
    thr = tune_threshold(pv, yv, "balanced")
    rec = {"name": name, "family": family, "threshold_balanced": round(thr, 4)}
    for sp in ("train", "val", "test"):
        y, p = probs[sp]
        rec[sp] = metric_block(y, p, thr)
    t, te = rec["train"], rec["test"]
    rec["gap_train_test"] = round(t["error"] - te["error"], 4)  # negative = test worse (variance)
    rec["variance_val"] = round(rec["val"]["error"] - t["error"], 4)
    return rec
