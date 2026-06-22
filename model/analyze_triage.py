"""Find the tile-aggregation that maximises specificity at 100% sensitivity, i.e.
how many SAFE images we can auto-clear while NEVER missing an unsafe one.

This is the practical "recall all perfectly" for a safety system: a zero-miss
triage. Uses per-tile OOF probabilities from train_cv_crops.py.
"""
from __future__ import annotations

import numpy as np

import du

d = np.load(du.ROOT / "outputs_cv_crops" / "probs.npz")
y = d["labels_tv"].astype(int)
names = [k[len("oof_tiles_"):] for k in d.files if k.startswith("oof_tiles_")]
print(f"backbones with per-tile probs: {names} | n_unsafe={int((y==1).sum())} n_safe={int((y==0).sum())}")


def agg(tiles, how):
    if how == "max":
        return tiles.max(1)
    if how == "whole":
        return tiles[:, 0]
    if how == "mean":
        return tiles.mean(1)
    if how.startswith("top"):
        k = int(how[3:])
        return np.sort(tiles, 1)[:, -k:].mean(1)
    if how.startswith("p"):
        return np.percentile(tiles, int(how[1:]), axis=1)
    raise ValueError(how)


def spec_at_full_recall(score, y):
    """t_low just below the hardest unsafe; specificity = % safe below it."""
    pos = score[y == 1]
    t_low = float(pos.min()) - 1e-9
    spec = float((score[y == 0] < t_low).mean())
    return t_low, spec


HOWS = ["whole", "max", "top2", "top3", "top5", "mean", "p75", "p90"]
print("\n=== specificity @ 100% sensitivity (per backbone & ensemble) ===")
print(f"{'agg':6s} " + " ".join(f"{n[:10]:>11s}" for n in names) + f"{'ENSEMBLE':>11s}")
best = None
for how in HOWS:
    per = {n: agg(d[f"oof_tiles_{n}"], how) for n in names}
    row = []
    for n in names:
        _, s = spec_at_full_recall(per[n], y); row.append(s)
    ens = np.mean([per[n] for n in names], axis=0)
    t_low, s_ens = spec_at_full_recall(ens, y)
    print(f"{how:6s} " + " ".join(f"{v*100:10.1f}%" for v in row) + f"{s_ens*100:10.1f}%")
    if best is None or s_ens > best["spec"]:
        best = {"how": how, "spec": s_ens, "t_low": t_low}

print(f"\nBEST aggregation for zero-miss auto-clear: '{best['how']}' "
      f"-> auto-clears {best['spec']*100:.1f}% of safe images with 0 missed unsafe "
      f"(t_low={best['t_low']:.3f})")

# full triage with best agg: auto-safe / review / auto-unsafe
ens = np.mean([agg(d[f"oof_tiles_{n}"], best["how"]) for n in names], axis=0)
from sklearn.metrics import accuracy_score
t_low = best["t_low"]
print("\n=== zero-miss triage (best agg) — sweep auto-unsafe cutoff t_high ===")
print(f"{'t_high':>7s} {'review%':>8s} {'auto-dec acc':>12s} {'missed unsafe':>14s}")
for th in [0.5, 0.6, 0.7, 0.8, 0.9]:
    review = (ens >= t_low) & (ens < th)
    decided = ~review
    pred = (ens >= th).astype(int)
    acc = accuracy_score(y[decided], pred[decided]) if decided.sum() else float("nan")
    missed = int(((pred == 0) & (y == 1) & decided).sum())  # unsafe auto-cleared
    print(f"{th:7.2f} {review.mean()*100:7.1f}% {acc*100:11.1f}% {missed:13d}")
