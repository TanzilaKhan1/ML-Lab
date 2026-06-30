"""Rebuild the 6 member error-analysis folders — NON-LEAKED.

Each folder (20 examples) =
  * 13 from the VALIDATION + TEST sets (held-out, non-leaked predictions), plus
  * 7  TRAINING examples (the model's hard train cases, via out-of-fold preds)
Stratified so every folder contains BOTH classes (safe & unsafe) and BOTH
vehicles (bus & leguna). Overlap across members is allowed.

Non-leakage of predictions:
  * Deep CV models (ResNet50/ConvNeXt/EfficientNet/Ensemble): OOF preds for
    train+val (held out per fold), held-out test preds for test.
  * Classical / CNN / ResNet18: trained on TRAIN only -> non-leaked on val+test
    (in-sample only on the few train rows, which are flagged by the `split` col).
"""
from __future__ import annotations

import csv
import json
import shutil
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
PROJECT = REPO.parent
sys.path.insert(0, str(REPO / "predictor"))
import du  # noqa: E402
from predictor_app import inference as inf  # noqa: E402

EA = PROJECT / "error_analysis"
FLAT = PROJECT / "data" / "dataset_safeunsafe"
MEMBERS = ["asif", "tanzila", "taif", "amio", "tazkia", "walid"]
HELD_PER, TRAIN_PER = 13, 7
SEED = 42
LBL = {0: "safe", 1: "unsafe"}

DEEP = {"ResNet50": ("outputs_cv_hi", "resnet50"), "ConvNeXt-Tiny": ("outputs_cv_hi", "convnext_tiny"),
        "Ensemble (best)": ("outputs_cv_hi", "ensemble"), "EfficientNet-B0": ("outputs_cv", "efficientnet_b0")}
SHALLOW = ["Logistic Regression", "Naive Bayes", "SVM (RBF)", "CNN", "ResNet18"]
MODEL_ORDER = ["Logistic Regression", "Naive Bayes", "SVM (RBF)", "CNN", "ResNet18",
               "ResNet50", "ConvNeXt-Tiny", "EfficientNet-B0", "Ensemble (best)"]
SHORT = {"Logistic Regression": "LogReg", "Naive Bayes": "NaiveBayes", "SVM (RBF)": "SVM",
         "CNN": "CNN", "ResNet18": "ResNet18", "ResNet50": "ResNet50",
         "ConvNeXt-Tiny": "ConvNeXt", "EfficientNet-B0": "EffNetB0", "Ensemble (best)": "Ensemble"}


def deep_lookup():
    part = du.get_partition()
    tv = list(part["train"][0]) + list(part["val"][0])
    te = list(part["test"][0])
    hi = np.load(du.ROOT / "outputs_cv_hi" / "probs.npz")
    cv = np.load(du.ROOT / "outputs_cv" / "probs.npz")
    src = {"outputs_cv_hi": hi, "outputs_cv": cv}
    lk = {m: {} for m in DEEP}
    for m, (d, k) in DEEP.items():
        s = src[d]
        for i, p in enumerate(tv):
            lk[m][str(p)] = float(s[f"oof_{k}"][i])
        for j, p in enumerate(te):
            lk[m][str(p)] = float(s[f"test_{k}"][j])
    return lk


_shallow_cache: dict = {}
def shallow_prob(path, model):
    key = (str(path), model)
    if key not in _shallow_cache:
        _shallow_cache[key] = float(inf.predict(str(path), model).probabilities["positive (UNSAFE)"])
    return _shallow_cache[key]


def take_stratified(records, n, offset):
    groups: dict = {}
    for r in records:
        groups.setdefault((r["class"], r["vehicle"]), []).append(r)
    keys = sorted(groups)
    for ki, k in enumerate(keys):
        rng = np.random.RandomState(SEED + ki)
        g = groups[k]; idx = list(range(len(g))); rng.shuffle(idx)
        groups[k] = [g[i] for i in idx]
    base, rem = divmod(n, len(keys))
    quota = {k: base + (1 if i < rem else 0) for i, k in enumerate(keys)}
    picked, seen = [], set()
    for k in keys:
        g = groups[k]; cnt = quota[k]
        for j in range(cnt):
            r = g[(offset * cnt + j) % len(g)]
            if r["id"] not in seen:
                seen.add(r["id"]); picked.append(r)
    # backfill to n distinct
    allidx = list(range(len(records)))
    np.random.RandomState(SEED + 100 + offset).shuffle(allidx)
    for i in allidx:
        if len(picked) >= n:
            break
        if records[i]["id"] not in seen:
            seen.add(records[i]["id"]); picked.append(records[i])
    return picked[:n]


def meta(path, label):
    p = Path(path)
    return dict(stem=p.stem, vehicle=p.parent.parent.name, true=int(label),
                cls=LBL[int(label)], src=FLAT / LBL[int(label)] / f"{p.parent.parent.name}__{p.stem}.png")


def main():
    part = du.get_partition()
    lk = deep_lookup()
    val_p, val_y = part["val"]; test_p, test_y = part["test"]; tr_p, tr_y = part["train"]

    # ---- HELD pool: val+test misclassified by ANY of the 9 models @0.5 (non-leaked)
    held = []
    for split, (ps, ys) in (("val", (val_p, val_y)), ("test", (test_p, test_y))):
        for p, y in zip(ps, ys):
            preds = {}
            for m in DEEP:
                preds[m] = int(lk[m][str(p)] >= 0.5)
            for m in SHALLOW:
                preds[m] = int(shallow_prob(p, m) >= 0.5)
            wrong = [m for m in MODEL_ORDER if preds[m] != int(y)]
            if wrong:
                mt = meta(p, y)
                held.append(dict(path=str(p), split=split, vehicle=mt["vehicle"],
                                 stem=mt["stem"], true=int(y), **{"class": mt["cls"]},
                                 src=mt["src"], ens_prob=lk["Ensemble (best)"][str(p)],
                                 wrong=wrong))
    # ---- TRAIN pool: train misclassified by ANY deep OOF model @0.5 (held-out per fold)
    train = []
    for p, y in zip(tr_p, tr_y):
        wrong_deep = [m for m in DEEP if int(lk[m][str(p)] >= 0.5) != int(y)]
        if wrong_deep:
            mt = meta(p, y)
            train.append(dict(path=str(p), split="train", vehicle=mt["vehicle"],
                              stem=mt["stem"], true=int(y), **{"class": mt["cls"]},
                              src=mt["src"], ens_prob=lk["Ensemble (best)"][str(p)],
                              wrong=wrong_deep))
    # global ids
    for gid, r in enumerate(held + train):
        r["id"] = gid
    print(f"HELD pool (val+test, non-leaked): {len(held)}  "
          f"[{sum(r['class']=='safe' for r in held)} safe / {sum(r['class']=='unsafe' for r in held)} unsafe]")
    print(f"TRAIN pool (OOF hard cases): {len(train)}  "
          f"[{sum(r['class']=='safe' for r in train)} safe / {sum(r['class']=='unsafe' for r in train)} unsafe]")

    # ---- assign
    assign = {}
    for mi, name in enumerate(MEMBERS):
        picks = take_stratified(held, HELD_PER, mi) + take_stratified(train, TRAIN_PER, mi)
        assign[name] = picks

    # ---- compute shallow for any selected TRAIN images (for the CSV)
    for name in MEMBERS:
        for r in assign[name]:
            if r["split"] == "train":
                for m in SHALLOW:
                    shallow_prob(r["path"], m)

    def prob_of(path, model):
        return lk[model][path] if model in DEEP else shallow_prob(path, model)

    # ---- write folders, sheets, inference CSVs
    if EA.exists():
        shutil.rmtree(EA)
    EA.mkdir(parents=True)
    sheet_cols = ["example_id", "image_file", "split", "true_label", "vehicle",
                  "ensemble_pred", "ensemble_prob_unsafe", "models_that_missed", "auto_hint",
                  "reason:blurry", "reason:occlusion/crowd", "reason:small/distant",
                  "reason:lighting/glare", "reason:ambiguous_pose", "reason:possible_mislabel",
                  "reason:other", "comments", "proposed_fix"]
    inf_cols = ["image", "split", "vehicle", "true_label"]
    for m in MODEL_ORDER:
        inf_cols += [f"{SHORT[m]}_pred", f"{SHORT[m]}_p_unsafe"]
    inf_cols += ["n_models_unsafe", "majority_vote", "ensemble_pred", "ensemble_correct"]

    comp = {}
    for name in MEMBERS:
        mdir = EA / name; mdir.mkdir(parents=True)
        srows, irows = [], []
        c = {"safe": 0, "unsafe": 0, "bus": 0, "legua": 0, "train": 0, "val": 0, "test": 0}
        for r in assign[name]:
            ens_pred = "unsafe" if r["ens_prob"] >= 0.5 else "safe"
            err = "FN" if (r["true"] == 1 and ens_pred == "safe") else (
                  "FP" if (r["true"] == 0 and ens_pred == "unsafe") else
                  ("FN" if r["true"] == 1 else "FP"))
            frac_hint = ("small/distant hanger?" if (r["true"] == 1) else
                         "person near door but not hanging (ambiguous)?")
            fn = (f"{r['id']:03d}_{r['split']}_{err}_true-{LBL[r['true']]}_pred-{ens_pred}"
                  f"_p{r['ens_prob']:.2f}_{r['vehicle']}_{r['stem']}.png")
            if r["src"].exists():
                shutil.copy2(r["src"], mdir / fn)
            srows.append([r["id"], fn, r["split"], LBL[r["true"]], r["vehicle"],
                          ens_pred, f"{r['ens_prob']:.3f}", "+".join(r["wrong"]), frac_hint,
                          "", "", "", "", "", "", "", "", ""])
            # inference row (all 9 models, non-leaked)
            row = [fn, r["split"], r["vehicle"], LBL[r["true"]]]
            n_unsafe = 0
            for m in MODEL_ORDER:
                pu = prob_of(r["path"], m); pd = "unsafe" if pu >= 0.5 else "safe"
                if pd == "unsafe":
                    n_unsafe += 1
                row += [pd, f"{pu:.3f}"]
            maj = "unsafe" if n_unsafe >= len(MODEL_ORDER) / 2 else "safe"
            row += [n_unsafe, maj, ens_pred, ens_pred == LBL[r["true"]]]
            irows.append(row)
            c[r["class"]] += 1; c[r["vehicle"]] += 1; c[r["split"]] += 1
        with open(mdir / f"{name}_error_analysis_sheet.csv", "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(sheet_cols); w.writerows(srows)
        for dest in (mdir / f"{name}_all_models_inference.csv", EA / f"{name}_all_models_inference.csv"):
            with open(dest, "w", newline="") as fh:
                w = csv.writer(fh); w.writerow(inf_cols); w.writerows(irows)
        comp[name] = c

    (EA / "assignment_index.json").write_text(json.dumps(
        {"per_member": 20, "held_per": HELD_PER, "train_per": TRAIN_PER,
         "members": {n: [r["id"] for r in assign[n]] for n in MEMBERS},
         "composition": comp}, indent=2))
    print("\n=== per-member composition (each = 20) ===")
    print(f"{'member':9s} {'safe':>5s} {'unsafe':>7s} {'bus':>5s} {'legua':>6s} "
          f"{'train':>6s} {'val':>5s} {'test':>5s}")
    for n in MEMBERS:
        c = comp[n]
        print(f"{n:9s} {c['safe']:5d} {c['unsafe']:7d} {c['bus']:5d} {c['legua']:6d} "
              f"{c['train']:6d} {c['val']:5d} {c['test']:5d}")
    print(f"\nwrote {EA}/ (folders + sheets + per-folder inference CSVs)")


if __name__ == "__main__":
    main()
