"""Separate the images Naive Bayes misclassified (from the all-385 run), copy them
to a folder, RE-RUN all 9 models on them, and write a per-model-classification CSV.
"""
from __future__ import annotations

import csv
import shutil
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
PROJECT = REPO.parent
sys.path.insert(0, str(REPO / "predictor"))
from predictor_app import inference as inf  # noqa: E402
from predictor_app.preprocess import preprocess_for_model, standardize_image  # noqa: E402

EA = PROJECT / "error_analysis"
FLAT = PROJECT / "data" / "dataset_safeunsafe"
ALL385 = EA / "all385_all_models_inference.csv"
DEST = EA / "naive_bayes_misclassified"
ORDER = ["Logistic Regression", "Naive Bayes", "SVM (RBF)", "CNN", "ResNet18",
         "ResNet50", "ConvNeXt-Tiny", "EfficientNet-B0", "Ensemble (best)"]
SHORT = {"Logistic Regression": "LogReg", "Naive Bayes": "NaiveBayes", "SVM (RBF)": "SVM",
         "CNN": "CNN", "ResNet18": "ResNet18", "ResNet50": "ResNet50",
         "ConvNeXt-Tiny": "ConvNeXt", "EfficientNet-B0": "EffNetB0", "Ensemble (best)": "Ensemble"}


def p_unsafe(model, kind, feats, pil):
    if kind == "sklearn":
        if hasattr(model, "predict_proba"):
            return float(model.predict_proba(feats)[0, 1])
        s = float(np.atleast_1d(model.decision_function(feats))[0])
        return 1.0 / (1.0 + np.exp(-s))
    return float(model.predict_proba(pil)[0, 1])


def main():
    # 1. find NB-misclassified images from the all-385 CSV
    rows = list(csv.DictReader(open(ALL385)))
    nb_wrong = [r for r in rows if r["NaiveBayes_pred"] != r["true_label"]]
    print(f"Naive Bayes misclassified: {len(nb_wrong)} / {len(rows)}")

    # 2. separate them into a folder
    if DEST.exists():
        shutil.rmtree(DEST)
    DEST.mkdir(parents=True)
    srcs = []
    for r in nb_wrong:
        src = FLAT / r["true_label"] / r["image"]
        nbp = r["NaiveBayes_pred"]
        out = DEST / f"true-{r['true_label']}_NBpred-{nbp}_{r['split']}_{r['image']}"
        if src.exists():
            shutil.copy2(src, out)
        srcs.append((src, r["image"], r["split"], r["vehicle"], r["true_label"]))

    # 3. RE-RUN all 9 models on these images
    loaded = {m: (inf.load_model(m), inf.AVAILABLE_MODELS[m][1]) for m in ORDER}
    header = ["image", "split", "vehicle", "true_label"]
    for m in ORDER:
        header += [f"{SHORT[m]}_pred", f"{SHORT[m]}_p_unsafe"]
    header += ["n_models_correct", "n_models_unsafe", "majority_vote", "ensemble_pred"]

    out_rows = []
    correct = {m: 0 for m in ORDER}
    for src, name, split, veh, true in srcs:
        pil = standardize_image(str(src)); feats = preprocess_for_model(pil)
        row = [name, split, veh, true]
        n_correct = n_unsafe = 0; ens_pred = "safe"
        for m in ORDER:
            model, kind = loaded[m]
            pu = p_unsafe(model, kind, feats, pil)
            pred = "unsafe" if pu >= 0.5 else "safe"
            if m == "Ensemble (best)":
                ens_pred = pred
            if pred == "unsafe":
                n_unsafe += 1
            if pred == true:
                n_correct += 1; correct[m] += 1
            row += [pred, f"{pu:.3f}"]
        maj = "unsafe" if n_unsafe >= len(ORDER) / 2 else "safe"
        row += [n_correct, n_unsafe, maj, ens_pred]
        out_rows.append(row)

    out_csv = EA / "naive_bayes_misclassified_inference.csv"
    for dest in (out_csv, DEST / "naive_bayes_misclassified_inference.csv"):
        with open(dest, "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(header); w.writerows(out_rows)

    N = len(out_rows)
    print(f"\n=== how each model does on the {N} NB-misclassified images ===")
    print(f"{'Model':11s} {'correct':>8s} {'wrong':>6s} {'acc':>5s}")
    for m in ORDER:
        c = correct[m]; print(f"{SHORT[m]:11s} {c:8d} {N-c:6d} {c/N*100:4.0f}%")
    print(f"\nseparated images -> {DEST}/ ({N} images)")
    print(f"CSV -> {out_csv}")


if __name__ == "__main__":
    main()
