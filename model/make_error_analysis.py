"""Error-analysis split for the group exercise.

Inference (held-out 5-fold CV) the validation examples, collect the ones any of
our deep models misclassifies, and split the pool into 6 per-member folders
(20 examples each, overlap allowed, every folder is 100% misclassified).
Each member also gets an error-analysis sheet (CSV) to fill in.

Output -> ML_lab/error_analysis/{member}/  +  index + README
"""
from __future__ import annotations

import csv
import json
import math
import shutil
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score

import du

PROJECT = du.ROOT.parent.parent                      # ML_lab/
DATA = PROJECT / "data"
ANN = DATA / "_annotations"
FLAT = DATA / "dataset_safeunsafe"                   # {safe,unsafe}/{vehicle}__{stem}.png
OUT = PROJECT / "error_analysis"
MEMBERS = ["asif", "tanzila", "taif", "amio", "tazkia", "walid"]
PER = 20
SEED = 42

LBL = {0: "safe", 1: "unsafe"}


def bal_thr(p, y):
    return max(((balanced_accuracy_score(y, (p >= t).astype(int)), t)
                for t in np.linspace(0.05, 0.95, 181)))[1]


def unsafe_box_fraction(stem, vehicle):
    """Max unsafe-box area fraction from the annotation (small => distant subject)."""
    for c in ANN.rglob(f"{stem}.json"):
        if "/_trash/" in str(c):
            continue
        d = json.loads(c.read_text())
        iw, ih = d.get("imageWidth") or 0, d.get("imageHeight") or 0
        if not iw or not ih:
            return None
        fr = [b["width"] * b["height"] / (iw * ih)
              for b in (d.get("annotations") or [])
              if b.get("label", "").lower() == "unsafe" and b.get("width")]
        return max(fr) if fr else None
    return None


def auto_hint(true, pred, frac):
    if true == 1 and pred == 0:  # missed unsafe (false negative)
        if frac is not None and frac < 0.05:
            return "small/distant hanger (tiny box) — model misses it"
        return "false negative — hanger present but predicted safe (occlusion/pose/clutter?)"
    if true == 0 and pred == 1:  # false alarm
        return "false positive — person near door but not hanging (ambiguous pose/crowd)?"
    return "review"


def main():
    part = du.get_partition()
    tvp = list(part["train"][0]) + list(part["val"][0])
    tvl = np.array(list(part["train"][1]) + list(part["val"][1]))
    hi = np.load(du.ROOT / "outputs_cv_hi" / "probs.npz")
    base = np.load(du.ROOT / "outputs_cv" / "probs.npz")
    assert np.array_equal(hi["labels_tv"].astype(int), tvl), "probs/order mismatch"
    oof = {"ensemble": hi["oof_ensemble"], "resnet50": hi["oof_resnet50"],
           "convnext_tiny": hi["oof_convnext_tiny"], "efficientnet_b0": base["oof_efficientnet_b0"]}
    thr = {m: bal_thr(p, tvl) for m, p in oof.items()}
    pred = {m: (oof[m] >= thr[m]).astype(int) for m in oof}
    wrong_by = {m: set(np.where(pred[m] != tvl)[0].tolist()) for m in oof}
    pool = sorted(set().union(*wrong_by.values()))
    print(f"misclassified pool (union of 4 deep models): {len(pool)}")

    # build records — report the prediction of a model that ACTUALLY misclassified it
    # (ensemble if it missed it, else the first model that did) so pred != true always.
    recs = []
    for eid, i in enumerate(pool):
        p = Path(tvp[i]); stem = p.stem; vehicle = p.parent.parent.name
        true = int(tvl[i])
        missed = [m for m in oof if i in wrong_by[m]]
        scored = "ensemble" if "ensemble" in missed else missed[0]
        spred = int(pred[scored][i]); sprob = float(oof[scored][i])
        frac = unsafe_box_fraction(stem, vehicle)
        recs.append(dict(eid=eid, stem=stem, vehicle=vehicle, true=true, pred=spred,
                         prob=sprob, scored=scored, missed=missed,
                         err="FN" if (true == 1 and spred == 0) else "FP",
                         hint=auto_hint(true, spred, frac),
                         src=FLAT / LBL[true] / f"{vehicle}__{stem}.png"))

    M = len(recs)
    order = list(range(M))
    np.random.RandomState(SEED).shuffle(order)
    stride = max(1, math.ceil((M - PER) / (len(MEMBERS) - 1))) if M > PER else 0

    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    member_assign = {}
    sheet_cols = ["example_id", "image_file", "true_label", "model_prediction",
                  "prob_unsafe", "scored_model", "error_type", "vehicle",
                  "models_that_missed", "auto_hint",
                  "reason:blurry", "reason:occlusion/crowd", "reason:small/distant",
                  "reason:lighting/glare", "reason:ambiguous_pose", "reason:possible_mislabel",
                  "reason:other", "comments", "proposed_fix"]

    for mi, name in enumerate(MEMBERS):
        start = min(mi * stride, max(0, M - PER))
        picks = [order[(start + k) % M] for k in range(PER)]
        # de-dup within a member while keeping 20 (wrap can repeat only if M<PER)
        seen = []
        for idx in picks:
            if idx not in seen:
                seen.append(idx)
        k = 0
        while len(seen) < PER and M > 0:
            cand = order[k % M]
            if cand not in seen:
                seen.append(cand)
            k += 1
        picks = seen[:PER]
        member_assign[name] = [recs[i]["eid"] for i in picks]

        mdir = OUT / name
        mdir.mkdir(parents=True)
        rows = []
        for j, i in enumerate(picks, 1):
            r = recs[i]
            fn = (f"{r['eid']:02d}_{r['err']}_true-{LBL[r['true']]}_pred-{LBL[r['pred']]}"
                  f"_p{r['prob']:.2f}_{r['vehicle']}_{r['stem']}.png")
            if r["src"].exists():
                shutil.copy2(r["src"], mdir / fn)
            rows.append([r["eid"], fn, LBL[r["true"]], LBL[r["pred"]], f"{r['prob']:.3f}",
                         r["scored"], r["err"], r["vehicle"], "+".join(r["missed"]), r["hint"],
                         "", "", "", "", "", "", "", "", ""])
        with open(mdir / f"{name}_error_analysis_sheet.csv", "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(sheet_cols); w.writerows(rows)
        n_fn = sum(1 for i in picks if recs[i]["err"] == "FN")
        n_fp = sum(1 for i in picks if recs[i]["err"] == "FP")
        print(f"  {name:8s}: {len(picks)} examples  (FN={n_fn}, FP={n_fp})  -> {mdir}")

    # index + README
    (OUT / "assignment_index.json").write_text(json.dumps(
        {"pool_size": M, "per_member": PER, "members": member_assign,
         "thresholds": thr}, indent=2, default=str))
    (OUT / "README.md").write_text(
        "# Error-analysis split (validation misclassifications)\n\n"
        f"Pool = **{M}** examples misclassified by >=1 deep model on the held-out "
        "5-fold CV validation predictions (the model never trained on the fold it was "
        "scored on). Each of the 6 members gets **20** examples (overlap allowed; every "
        "folder is 100% misclassified — FN = missed unsafe, FP = false alarm).\n\n"
        "## Each member folder contains\n"
        "- 20 image files named `id_ERR_true-X_pred-Y_pPROB_vehicle_stem.png`\n"
        "- `<name>_error_analysis_sheet.csv` — fill the `reason:*`, `comments`, "
        "`proposed_fix` columns (categories discussed in the theory class).\n\n"
        "## Counts\n"
        f"- False negatives in pool (unsafe predicted safe): "
        f"{sum(1 for r in recs if r['err']=='FN')}\n"
        f"- False positives in pool (safe predicted unsafe): "
        f"{sum(1 for r in recs if r['err']=='FP')}\n")
    print(f"\npool={M}; wrote {OUT}/ (folders + sheets + index + README)")


if __name__ == "__main__":
    main()
