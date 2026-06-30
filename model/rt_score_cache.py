"""Score all 9 deployed models on VAL + TEST and cache P(unsafe) to npz.

Shared input for the paper-asset figures and the misclassified workbook, so the
(slowish) per-image scoring runs only once. Train ERROR is read from
metrics_full.json (no need to re-score the 1600 train images).

Writes: outputs_final/probs_cache.npz  (+ probs_cache_meta.json with paths/order)
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import joblib
from PIL import Image
import du

PRED = (du.ROOT.parent / "predictor").resolve()
if str(PRED) not in sys.path:
    sys.path.insert(0, str(PRED))
MODEL_DIR = PRED / "model"
OUT = du.ROOT / "outputs_final"; OUT.mkdir(parents=True, exist_ok=True)

# display -> (deployed file, kind)
MODELS = [
    ("Ensemble (best)",     "ensemble.json",            "ensemble"),
    ("ResNet50",            "resnet50.joblib",           "torch"),
    ("ConvNeXt-Tiny",       "convnext_tiny.joblib",      "torch"),
    ("EfficientNet-B0",     "efficientnet_b0.joblib",    "torch"),
    ("ResNet18",            "resnet_model.joblib",        "torch"),
    ("CNN",                 "cnn_model.joblib",           "torch"),
    ("SVM (RBF)",           "svm_model.joblib",           "sklearn"),
    ("Logistic Regression", "logistic_model.joblib",      "sklearn"),
    ("Naive Bayes",         "naive_bayes_model.joblib",   "sklearn"),
]
KEY = {"Ensemble (best)": "ensemble", "ResNet50": "resnet50", "ConvNeXt-Tiny": "convnext",
       "EfficientNet-B0": "efficientnet", "ResNet18": "resnet18", "CNN": "cnn",
       "SVM (RBF)": "svm", "Logistic Regression": "logreg", "Naive Bayes": "nb"}


def torch_p(path, paths):
    from predictor_app.torch_models import load_torch_checkpoint
    clf = load_torch_checkpoint(path)
    return np.array([float(clf.predict_proba(Image.open(p).convert("RGB"))[0, 1]) for p in paths])


def ens_p(path, paths):
    from predictor_app.torch_models import load_ensemble
    clf = load_ensemble(path)
    return np.array([float(clf.predict_proba(Image.open(p).convert("RGB"))[0, 1]) for p in paths])


def sk_p(path, paths):
    from predictor_app.preprocess import preprocess_for_model
    m = joblib.load(path)
    return np.array([float(m.predict_proba(preprocess_for_model(Image.open(p).convert("RGB")))[0, 1])
                     for p in paths])


def main():
    part = du.get_partition()
    arrays, meta = {}, {"models": [], "splits": {}}
    for sp in ("val", "test"):
        paths = [str(p) for p in part[sp][0]]
        ys = np.array(part[sp][1])
        arrays[f"y_{sp}"] = ys
        meta["splits"][sp] = {"paths": paths, "y": ys.tolist()}
    for disp, fn, kind in MODELS:
        meta["models"].append(disp)
        for sp in ("val", "test"):
            paths = [str(p) for p in part[sp][0]]
            print(f"scoring {disp} on {sp} ({len(paths)})")
            if kind == "ensemble":
                p = ens_p(MODEL_DIR / fn, paths)
            elif kind == "torch":
                p = torch_p(MODEL_DIR / fn, paths)
            else:
                p = sk_p(MODEL_DIR / fn, paths)
            arrays[f"p_{KEY[disp]}_{sp}"] = p
    np.savez(OUT / "probs_cache.npz", **arrays)
    (OUT / "probs_cache_meta.json").write_text(json.dumps(meta, indent=1))
    print(f"\nwrote {OUT/'probs_cache.npz'} and meta")


if __name__ == "__main__":
    main()
