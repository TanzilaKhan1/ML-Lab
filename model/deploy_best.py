"""Pick operating points from CV out-of-fold probabilities, deploy the best
models (with a safety-oriented high-recall threshold) into the predictor, and
write the final results report.

"recall all perfectly" = a safety system must not miss a hanging passenger, so
we expose the recall/accuracy tradeoff and DEPLOY a high-recall threshold
(highest accuracy subject to CV unsafe-recall >= TARGET_RECALL).
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, precision_score, recall_score,
                             roc_auc_score)

import argparse

import du

_ap = argparse.ArgumentParser()
_ap.add_argument("--cv", default="outputs_cv_hi", help="CV output dir under model/")
_ap.add_argument("--target", type=float, default=0.90, help="deploy recall target")
_args = _ap.parse_args()

CV = du.ROOT / _args.cv
PRED_MODEL_DIR = du.ROOT.parent / "predictor" / "model"
TARGET_RECALL = _args.target  # deploy threshold catches >= this fraction of unsafe (CV)

probs = np.load(CV / "probs.npz")
ytv = probs["labels_tv"].astype(int)
yte = probs["labels_test"].astype(int)
results = json.loads((CV / "results.json").read_text())
backbones = results["ensemble"]["members"]
ALL = backbones + ["ensemble"]


def thr_grid():
    return np.linspace(0.02, 0.98, 193)


def op_balanced(p, y):
    best_t, best = 0.5, (-1, -1)
    for t in thr_grid():
        pred = (p >= t).astype(int)
        key = (round(balanced_accuracy_score(y, pred), 4),
               round(recall_score(y, pred, pos_label=1, zero_division=0), 4))
        if key > best:
            best, best_t = key, float(t)
    return best_t


def op_high_recall(p, y, target):
    """Lowest-cost threshold (max accuracy) subject to recall >= target."""
    cand = []
    for t in thr_grid():
        pred = (p >= t).astype(int)
        rec = recall_score(y, pred, pos_label=1, zero_division=0)
        if rec >= target:
            cand.append((accuracy_score(y, pred), float(t)))
    if not cand:
        return float(thr_grid()[0])
    cand.sort(key=lambda c: (-c[0], -c[1]))  # max acc, then highest threshold
    return cand[0][1]


def op_max_recall(p, y):
    """Highest threshold that still catches EVERY unsafe (100% recall, zero-miss)."""
    pos = p[y == 1]
    if len(pos) == 0:
        return 0.5
    return float(max(1e-4, pos.min() - 1e-4))


def summarize(p, y, t):
    pred = (p >= t).astype(int)
    return dict(
        thr=round(float(t), 3),
        acc=round(float(accuracy_score(y, pred)), 4),
        bal_acc=round(float(balanced_accuracy_score(y, pred)), 4),
        unsafe_recall=round(float(recall_score(y, pred, pos_label=1, zero_division=0)), 4),
        unsafe_prec=round(float(precision_score(y, pred, pos_label=1, zero_division=0)), 4),
        safe_recall=round(float(recall_score(y, pred, pos_label=0, zero_division=0)), 4),
        auc=round(float(roc_auc_score(y, p)), 4) if len(set(y)) > 1 else float("nan"),
        cm=confusion_matrix(y, pred, labels=[0, 1]).tolist(),
    )


def main():
    report = {}
    print(f"{'model':16s} {'point':12s} {'thr':>5s} {'CVacc':>6s} {'CVrec':>6s} "
          f"{'CVprec':>6s} | {'TESTacc':>7s} {'TESTrec':>7s}")
    deploy_thr = {}
    for name in ALL:
        oof = probs[f"oof_{name}"]
        tst = probs[f"test_{name}"]
        tb = op_balanced(oof, ytv)
        th = op_high_recall(oof, ytv, TARGET_RECALL)
        ops = {}
        for label, t in [("balanced", tb), (f"recall>={TARGET_RECALL}", th)]:
            cv = summarize(oof, ytv, t)
            te = summarize(tst, yte, t)
            ops[label] = {"cv": cv, "test": te}
            print(f"{name:16s} {label:12s} {t:5.2f} {cv['acc']*100:5.1f}% {cv['unsafe_recall']*100:5.1f}% "
                  f"{cv['unsafe_prec']*100:5.1f}% | {te['acc']*100:6.1f}% {te['unsafe_recall']*100:6.1f}%")
        # full recall-target tradeoff (CV)
        tradeoff = {}
        for target in (0.80, 0.90, 0.95, 0.99, 1.00):
            t = op_high_recall(oof, ytv, target)
            tradeoff[f"recall>={target}"] = summarize(oof, ytv, t)
        report[name] = {"operating_points": ops, "cv_tradeoff": tradeoff}
        deploy_thr[name] = th   # deploy the high-recall threshold

    # ---- deploy checkpoints with selectable operating-mode thresholds ----
    # default = high_recall (safety); balanced (max acc) and max_recall (zero-miss,
    # 100% recall) also stored so the predictor can switch via PREDICTOR_OP_MODE.
    PRED_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    def thr_set(p):
        return dict(threshold_balanced=float(op_balanced(p, ytv)),
                    threshold_high_recall=float(op_high_recall(p, ytv, TARGET_RECALL)),
                    threshold_max_recall=float(op_max_recall(p, ytv)))

    for name in backbones:
        ts = thr_set(probs[f"oof_{name}"])
        ck = joblib.load(CV / f"{name}.joblib")
        ck.update(ts)
        ck["threshold"] = ts["threshold_high_recall"]   # active default
        for dest_dir in (PRED_MODEL_DIR, du.ROOT):
            joblib.dump(ck, dest_dir / f"{name}.joblib")
        print(f"deployed {name}.joblib  bal={ts['threshold_balanced']:.2f} "
              f"hi_rec={ts['threshold_high_recall']:.2f} max_rec={ts['threshold_max_recall']:.3f}")
    ets = thr_set(probs["oof_ensemble"])
    ens = {"members": [f"{n}.joblib" for n in backbones],
           "threshold": ets["threshold_high_recall"], **ets}
    for dest_dir in (PRED_MODEL_DIR, du.ROOT):
        (dest_dir / "ensemble.json").write_text(json.dumps(ens, indent=2))
    print(f"deployed ensemble.json  bal={ets['threshold_balanced']:.2f} "
          f"hi_rec={ets['threshold_high_recall']:.2f} max_rec={ets['threshold_max_recall']:.3f}")

    (CV / "operating_points.json").write_text(json.dumps(report, indent=2))
    print(f"\nwrote {CV/'operating_points.json'}")


if __name__ == "__main__":
    main()
