"""Run ALL models on every image in asif's and tanzila's error-analysis folders
and write a careful per-image / per-model safe-vs-unsafe CSV.

Probability = model's P(unsafe). Prediction = unsafe if P(unsafe) >= 0.5 (uniform
argmax across all models for a fair comparison). True label parsed from filename.
"""
from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PROJECT = REPO.parent
sys.path.insert(0, str(REPO / "predictor"))
from predictor_app import inference as inf  # noqa: E402

EA = PROJECT / "error_analysis"
MEMBERS = ["asif", "tanzila"]
# fixed model order: classical -> deep -> ensemble
MODEL_ORDER = ["Logistic Regression", "Naive Bayes", "SVM (RBF)",
               "CNN", "ResNet18", "ResNet50", "EfficientNet-B0", "ConvNeXt-Tiny",
               "Ensemble (best)"]
SHORT = {"Logistic Regression": "LogReg", "Naive Bayes": "NaiveBayes", "SVM (RBF)": "SVM",
         "CNN": "CNN", "ResNet18": "ResNet18", "ResNet50": "ResNet50",
         "EfficientNet-B0": "EffNetB0", "ConvNeXt-Tiny": "ConvNeXt", "Ensemble (best)": "Ensemble"}

TRUE_RE = re.compile(r"true-(safe|unsafe)")


def main():
    avail = inf.list_models()
    models = [m for m in MODEL_ORDER if m in avail]
    print(f"models: {[SHORT[m] for m in models]}")

    # cache loaded models + probabilities per image
    header = ["folder", "image", "true_label"]
    for m in models:
        header += [f"{SHORT[m]}_pred", f"{SHORT[m]}_p_unsafe"]
    header += ["n_models_unsafe", "majority_vote", "ensemble_pred", "ensemble_correct"]

    rows = []
    for member in MEMBERS:
        folder = EA / member
        imgs = sorted(folder.glob("*.png"))
        for img in imgs:
            tm = TRUE_RE.search(img.name)
            true = tm.group(1) if tm else "?"
            row = [member, img.name, true]
            n_unsafe = 0
            probs = {}
            for m in models:
                pr = inf.predict(str(img), m)
                p_unsafe = float(pr.probabilities["positive (UNSAFE)"])
                pred = "unsafe" if p_unsafe >= 0.5 else "safe"
                probs[m] = (pred, p_unsafe)
                if pred == "unsafe":
                    n_unsafe += 1
                row += [pred, f"{p_unsafe:.3f}"]
            maj = "unsafe" if n_unsafe >= (len(models) / 2) else "safe"
            ens_pred = probs["Ensemble (best)"][0] if "Ensemble (best)" in probs else maj
            ens_correct = (ens_pred == true) if true in ("safe", "unsafe") else ""
            row += [n_unsafe, maj, ens_pred, ens_correct]
            rows.append(row)
        print(f"  {member}: {len(imgs)} images inferenced")

    # remove any old combined file (user wants per-folder only)
    old = EA / "asif_tanzila_all_models_inference.csv"
    if old.exists():
        old.unlink()

    # write SEPARATE CSV per folder (both inside the folder and at error_analysis/)
    for member in MEMBERS:
        sub = [r for r in rows if r[0] == member]
        for dest in (EA / member / f"{member}_all_models_inference.csv",
                     EA / f"{member}_all_models_inference.csv"):
            with open(dest, "w", newline="") as fh:
                w = csv.writer(fh); w.writerow(header); w.writerows(sub)
        print(f"\n=== {member}: {len(sub)} images — per-model accuracy (pred@0.5 vs true) ===")
        for mi, m in enumerate(models):
            pcol = 3 + mi * 2
            correct = sum(1 for r in sub if r[pcol] == r[2])
            print(f"  {SHORT[m]:10s}: {correct}/{len(sub)} correct")
        print(f"  -> wrote {member}_all_models_inference.csv")


if __name__ == "__main__":
    main()
