"""Run all 9 deployed models on ALL 385 images (efficient: load each image once).
Per-image CSV + correct/misclassified summary. Prediction = unsafe if P(unsafe) >= 0.5.

NOTE: deployed models are in-sample on data they trained on (classical/CNN/ResNet18 on
train; deep CV models on train+val). The `split` column isolates the held-out `test` rows.
"""
from __future__ import annotations

import csv
import gc
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
PROJECT = REPO.parent
sys.path.insert(0, str(REPO / "predictor"))
import du  # noqa: E402
from predictor_app import inference as inf  # noqa: E402
from predictor_app.preprocess import preprocess_for_model, standardize_image  # noqa: E402

FLAT = PROJECT / "data" / "dataset_safeunsafe"
OUT = PROJECT / "error_analysis"
ORDER = ["Logistic Regression", "Naive Bayes", "SVM (RBF)", "CNN", "ResNet18",
         "ResNet50", "ConvNeXt-Tiny", "EfficientNet-B0", "Ensemble (best)"]
SHORT = {"Logistic Regression": "LogReg", "Naive Bayes": "NaiveBayes", "SVM (RBF)": "SVM",
         "CNN": "CNN", "ResNet18": "ResNet18", "ResNet50": "ResNet50",
         "ConvNeXt-Tiny": "ConvNeXt", "EfficientNet-B0": "EffNetB0", "Ensemble (best)": "Ensemble"}


def split_map():
    part = du.get_partition(); mp = {}
    for s in ("train", "val", "test"):
        for p in part[s][0]:
            p = Path(p); mp[(p.parent.parent.name, p.parent.name, p.stem)] = s
    return mp


def p_unsafe(model, kind, feats, pil):
    if kind == "sklearn":
        if hasattr(model, "predict_proba"):
            return float(model.predict_proba(feats)[0, 1])
        s = float(np.atleast_1d(model.decision_function(feats))[0])
        return 1.0 / (1.0 + np.exp(-s))
    return float(model.predict_proba(pil)[0, 1])


def main():
    smap = split_map()
    items = []
    for lbl in ("safe", "unsafe"):
        for f in sorted((FLAT / lbl).glob("*.png")):
            veh, stem = f.stem.split("__", 1)
            cls = "negative" if lbl == "safe" else "positive"
            items.append((f, lbl, veh, stem, smap.get((veh, cls, stem), "?")))
    print(f"images: {len(items)}", flush=True)

    loaded = {m: (inf.load_model(m), inf.AVAILABLE_MODELS[m][1]) for m in ORDER}
    print("models loaded", flush=True)

    header = ["image", "split", "vehicle", "true_label"]
    for m in ORDER:
        header += [f"{SHORT[m]}_pred", f"{SHORT[m]}_p_unsafe"]
    header += ["n_models_unsafe", "majority_vote", "ensemble_pred", "ensemble_correct"]

    rows = []
    correct = {m: 0 for m in ORDER}
    by_split = {s: {m: [0, 0] for m in ORDER} for s in ("train", "val", "test")}
    for k, (f, true, veh, stem, split) in enumerate(items):
        pil = standardize_image(str(f))
        feats = preprocess_for_model(pil)
        row = [f.name, split, veh, true]
        n_unsafe = 0
        ens_pred = "safe"
        for m in ORDER:
            model, kind = loaded[m]
            pu = p_unsafe(model, kind, feats, pil)
            pred = "unsafe" if pu >= 0.5 else "safe"
            if m == "Ensemble (best)":
                ens_pred = pred
            if pred == "unsafe":
                n_unsafe += 1
            if pred == true:
                correct[m] += 1
                if split in by_split:
                    by_split[split][m][0] += 1
            if split in by_split:
                by_split[split][m][1] += 1
            row += [pred, f"{pu:.3f}"]
        maj = "unsafe" if n_unsafe >= len(ORDER) / 2 else "safe"
        row += [n_unsafe, maj, ens_pred, ens_pred == true]
        rows.append(row)
        del pil, feats
        if (k + 1) % 50 == 0:
            gc.collect(); torch.cuda.empty_cache()
            print(f"  {k+1}/{len(items)}", flush=True)

    with open(OUT / "all385_all_models_inference.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(header); w.writerows(rows)

    N = len(items)
    print("\n=== correct / misclassified out of 385 (pred@0.5) ===", flush=True)
    print(f"{'Model':11s} {'correct':>8s} {'wrong':>6s} {'acc':>5s} | "
          f"{'train':>10s} {'val':>10s} {'test':>10s}")
    smd = ["# All-385 inference — correct vs misclassified (pred@0.5)", "",
           "All 9 deployed models on all 385 images. `train`/`val` rows are in-sample for the "
           "models trained on them; `test` (n=58) is held out for every model.", "",
           "| Model | Correct/385 | Misclassified/385 | Acc | Train acc | Val acc | Test acc |",
           "|---|---|---|---|---|---|---|"]
    for m in ORDER:
        c = correct[m]
        def pc(s):
            a, t = by_split[s][m]; return f"{a}/{t}={a/t*100:.0f}%" if t else "-"
        def ac(s):
            a, t = by_split[s][m]; return f"{a/t*100:.0f}%" if t else "-"
        print(f"{SHORT[m]:11s} {c:8d} {N-c:6d} {c/N*100:4.0f}% | "
              f"{pc('train'):>10s} {pc('val'):>10s} {pc('test'):>10s}")
        smd.append(f"| {SHORT[m]} | {c} | {N-c} | {c/N*100:.0f}% | {ac('train')} | {ac('val')} | {ac('test')} |")
    (OUT / "all385_summary.md").write_text("\n".join(smd))
    print(f"\nwrote {OUT}/all385_all_models_inference.csv and all385_summary.md", flush=True)


if __name__ == "__main__":
    main()
