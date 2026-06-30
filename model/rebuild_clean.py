"""Rebuild the error-analysis split using ONLY val+test images with NON-LEAKED
predictions, then write per-folder all-model inference CSVs.

Leakage handling (deployed models):
  - LogReg/NB/SVM, CNN, ResNet18  -> trained on TRAIN only  -> clean on val & test
  - ResNet50/ConvNeXt/EffNet/Ensemble (retrained on train+val) -> LEAKED on val,
    so for val images we use their 5-fold OUT-OF-FOLD probs; for test we use the
    held-out test probs. Both are non-leaked.

Output -> ML_lab/error_analysis/{member}/  (20 misclassified val/test examples each)
         + per-folder <member>_all_models_inference.csv (all 9 models, non-leaked)
"""
from __future__ import annotations

import csv
import json
import math
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
PER, SEED = 20, 42
LBL = {0: "safe", 1: "unsafe"}

DEEP = {"ResNet50": ("outputs_cv_hi", "resnet50"), "ConvNeXt-Tiny": ("outputs_cv_hi", "convnext_tiny"),
        "Ensemble (best)": ("outputs_cv_hi", "ensemble"), "EfficientNet-B0": ("outputs_cv", "efficientnet_b0")}
SHALLOW = ["Logistic Regression", "Naive Bayes", "SVM (RBF)", "CNN", "ResNet18"]
ORDER = ["Logistic Regression", "Naive Bayes", "SVM (RBF)", "CNN", "ResNet18",
         "ResNet50", "EfficientNet-B0", "ConvNeXt-Tiny", "Ensemble (best)"]
SHORT = {"Logistic Regression": "LogReg", "Naive Bayes": "NaiveBayes", "SVM (RBF)": "SVM",
         "CNN": "CNN", "ResNet18": "ResNet18", "ResNet50": "ResNet50",
         "EfficientNet-B0": "EffNetB0", "ConvNeXt-Tiny": "ConvNeXt", "Ensemble (best)": "Ensemble"}


def build_probs():
    part = du.get_partition()
    ntr, nval = len(part["train"][0]), len(part["val"][0])
    paths = list(part["val"][0]) + list(part["test"][0])
    labels = np.array(list(part["val"][1]) + list(part["test"][1]))
    splits = ["val"] * nval + ["test"] * len(part["test"][0])
    prob = {}
    for name, (d, k) in DEEP.items():
        src = np.load(du.ROOT / d / "probs.npz")
        prob[name] = np.concatenate([src[f"oof_{k}"][ntr:ntr + nval], src[f"test_{k}"]])
    for m in SHALLOW:
        prob[m] = np.array([inf.predict(str(p), m).probabilities["positive (UNSAFE)"] for p in paths])
    return paths, labels, splits, prob


def main():
    paths, labels, splits, prob = build_probs()
    M = len(paths)
    pred = {m: (prob[m] >= 0.5).astype(int) for m in prob}
    wrong = {m: set(np.where(pred[m] != labels)[0]) for m in prob}
    pool = sorted(set().union(*wrong.values()))
    print(f"non-leaked val+test pool (any of 9 models wrong @0.5): {len(pool)}")

    # per-pool-item metadata
    meta = {}
    for i in pool:
        p = Path(paths[i]); stem, veh = p.stem, p.parent.parent.name
        true = int(labels[i])
        scored = "Ensemble (best)" if i in wrong["Ensemble (best)"] else \
            next(m for m in ORDER if i in wrong[m])
        spred = int(pred[scored][i]); sp = float(prob[scored][i])
        missed = [SHORT[m] for m in ORDER if i in wrong[m]]
        err = "FN" if (true == 1 and spred == 0) else "FP"
        meta[i] = dict(stem=stem, veh=veh, true=true, scored=scored, spred=spred, sp=sp,
                       missed=missed, err=err, split=splits[i],
                       src=FLAT / LBL[true] / f"{veh}__{stem}.png")

    # divide: 6 members, 20 each, sliding window over shuffled pool -> full coverage
    order = list(range(len(pool)))
    np.random.RandomState(SEED).shuffle(order)
    stride = max(1, math.ceil((len(pool) - PER) / (len(MEMBERS) - 1))) if len(pool) > PER else 0

    if EA.exists():
        shutil.rmtree(EA)
    EA.mkdir(parents=True)

    sheet_cols = ["example_id", "image_file", "split", "true_label", "model_prediction",
                  "prob_unsafe", "scored_model", "error_type", "vehicle", "models_that_missed",
                  "reason:blurry", "reason:occlusion/crowd", "reason:small/distant",
                  "reason:lighting/glare", "reason:ambiguous_pose", "reason:possible_mislabel",
                  "reason:other", "comments", "proposed_fix"]
    inf_cols = ["split", "image", "true_label"] + \
        sum([[f"{SHORT[m]}_pred", f"{SHORT[m]}_p_unsafe"] for m in ORDER], []) + \
        ["n_models_unsafe", "majority_vote", "ensemble_pred", "ensemble_correct"]

    assign = {}
    for mi, name in enumerate(MEMBERS):
        start = min(mi * stride, max(0, len(pool) - PER))
        picks_local = [order[(start + k) % len(pool)] for k in range(PER)]
        seen = []
        for x in picks_local:
            if x not in seen:
                seen.append(x)
        k = 0
        while len(seen) < PER:
            if order[k % len(pool)] not in seen:
                seen.append(order[k % len(pool)])
            k += 1
        picks = [pool[x] for x in seen[:PER]]
        assign[name] = picks

        mdir = EA / name; mdir.mkdir()
        sheet_rows, inf_rows = [], []
        for i in picks:
            mt = meta[i]
            fn = (f"{mt['split']}_{mt['err']}_true-{LBL[mt['true']]}_pred-{LBL[mt['spred']]}"
                  f"_p{mt['sp']:.2f}_{mt['veh']}_{mt['stem']}.png")
            if mt["src"].exists():
                shutil.copy2(mt["src"], mdir / fn)
            sheet_rows.append([i, fn, mt["split"], LBL[mt["true"]], LBL[mt["spred"]],
                               f"{mt['sp']:.3f}", SHORT[mt["scored"]], mt["err"], mt["veh"],
                               "+".join(mt["missed"])] + [""] * 9)
            # inference row (all 9 models, non-leaked)
            n_uns = sum(1 for m in ORDER if pred[m][i] == 1)
            maj = "unsafe" if n_uns >= len(ORDER) / 2 else "safe"
            enspred = LBL[pred["Ensemble (best)"][i]]
            r = [mt["split"], fn, LBL[mt["true"]]]
            for m in ORDER:
                r += [LBL[pred[m][i]], f"{prob[m][i]:.3f}"]
            r += [n_uns, maj, enspred, enspred == LBL[mt["true"]]]
            inf_rows.append(r)

        with open(mdir / f"{name}_error_analysis_sheet.csv", "w", newline="") as fh:
            csv.writer(fh).writerows([sheet_cols] + sheet_rows)
        with open(mdir / f"{name}_all_models_inference.csv", "w", newline="") as fh:
            csv.writer(fh).writerows([inf_cols] + inf_rows)
        with open(EA / f"{name}_all_models_inference.csv", "w", newline="") as fh:
            csv.writer(fh).writerows([inf_cols] + inf_rows)
        c_split = {"val": sum(1 for i in picks if meta[i]["split"] == "val"),
                   "test": sum(1 for i in picks if meta[i]["split"] == "test")}
        c_err = {"FN": sum(1 for i in picks if meta[i]["err"] == "FN"),
                 "FP": sum(1 for i in picks if meta[i]["err"] == "FP")}
        print(f"  {name:8s}: 20 imgs  split={c_split}  err={c_err}")

    cov = set().union(*[set(assign[m]) for m in MEMBERS])
    (EA / "assignment_index.json").write_text(json.dumps(
        {"pool_size": len(pool), "coverage": len(cov), "per_member": PER,
         "members": {m: [int(i) for i in assign[m]] for m in MEMBERS},
         "source": "val+test only (non-leaked)"}, indent=2))
    (EA / "README.md").write_text(
        "# Error-analysis split — NON-LEAKED (validation + test only)\n\n"
        f"Pool = **{len(pool)}** images from the **validation + test** splits "
        "(NO training images) that >=1 of the 9 models misclassifies. Predictions are "
        "non-leaked: deep CV models use out-of-fold probs on val and held-out probs on "
        "test; classical/CNN/ResNet18 trained on train only.\n\n"
        "Each member: 20 misclassified examples (overlap allowed; filenames start with "
        "`val_`/`test_`). Files: `<name>_error_analysis_sheet.csv` (fill reason columns) "
        "and `<name>_all_models_inference.csv` (all 9 models' safe/unsafe + prob).\n")
    print(f"\npool={len(pool)} coverage={len(cov)} -> wrote {EA}")


if __name__ == "__main__":
    main()
