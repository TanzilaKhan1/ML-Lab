"""Regenerate metrics.json for the app's Model Analysis tab.

This is the single source of truth behind the app's analysis tab. It evaluates
each model on the real train/val/test splits, measures train error, and writes
the per-model numbers the app reads. Runs anywhere the dataset (model/Preprocessed)
is present — the cluster, or locally after build_local_dataset.py.

Workflow after you retrain a model:
    python model/export_metrics.py            # refresh every model
    python model/export_metrics.py resnet50.joblib   # refresh just one
    python model/export_metrics.py --hlp-sample 50   # print human-level sample
  then commit the updated metrics.json (both copies) and reboot the Streamlit app.

Methodology (follows Andrew Ng, *Machine Learning Yearning* — the standard
avoidable-bias/variance framework):
  - avoidable bias = train_error - human_level_error   (human-level ~ Bayes proxy)
  - variance       = dev_error   - train_error
  - train/dev/test error use ACCURACY (1 - accuracy), so they are comparable to
    the human-level *accuracy*. Every model is scored at the BALANCED decision
    rule (argmax / P(unsafe)>=0.5), NOT each checkpoint's tuned high-recall
    threshold — mixing thresholds would make the error numbers incomparable.
  - train error is measured on the ORIGINAL (un-augmented) train images.
    Augmentation changes the loss landscape and would bias the decomposition
    (see arXiv:2105.13343); un-augmented originals give a clean train-vs-dev gap.

It PRESERVES the human-level, error_analysis, and diagnosis sections of the
existing metrics.json (those are human judgements), and only overwrites the
measured per-model numbers + the `generated` stamp.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
from PIL import Image
from skimage.color import rgb2gray
from skimage.feature import hog

import du  # shared split + dataset scan

# Reuse the evaluation helpers (arch registry, transforms, inference).
from eval_train_error import load_checkpoint, predict_partition, scores

ROOT = du.ROOT
# Map joblib filename stem -> the display name used in the app + metrics.json.
STEM_TO_NAME = {
    "resnet50": "ResNet50",
    "convnext_tiny": "ConvNeXt-Tiny",
    "efficientnet_b0": "EfficientNet-B0",
    "resnet_model": "ResNet18",
    "cnn_model": "CNN",
}

# Classical sklearn pipelines consume HOG features, not raw images.
CLASSICAL_STEM_TO_NAME = {
    "svm_model": "SVM (RBF)",
    "logistic_model": "Logistic Regression",
    "naive_bayes_model": "Naive Bayes",
}

# Models whose deployable checkpoint was retrained on train+val (train_cv.py),
# so their VAL error is contaminated -> the app uses test as their held-out.
# (CNN/ResNet18 from train_torch.py and the classical models train on train
#  only, so their val stays a valid held-out.)
TRAIN_VAL_REFIT_STEMS = {"resnet50", "convnext_tiny", "efficientnet_b0"}

# HOG config — MUST match train_classical.py exactly (128x128 gray, 1764-dim).
HOG_SIZE = 128
HOG_PARAMS = dict(orientations=9, pixels_per_cell=(16, 16),
                  cells_per_block=(2, 2), block_norm="L2-Hys")


def _hog_features(paths) -> np.ndarray:
    """HOG feature matrix for a list of image paths (no augmentation).

    We deliberately use the ORIGINAL (un-augmented) train images so train error
    is comparable to dev/test error for a clean bias/variance reading.
    """
    feats = []
    for p in paths:
        with Image.open(p) as im:
            arr = np.asarray(im.convert("RGB").resize((HOG_SIZE, HOG_SIZE), Image.LANCZOS),
                             dtype=np.float32) / 255.0
        feats.append(hog(rgb2gray(arr), **HOG_PARAMS).astype(np.float32))
    return np.stack(feats)
# Where to write (cluster model dir + the deployed predictor copy if present).
OUT_PATHS = [
    ROOT / "metrics.json",
    ROOT.parent / "predictor" / "model" / "metrics.json",
]


def err(acc):
    return None if acc is None else round(1.0 - acc, 4)


# Bias/variance is an accuracy concept compared against human-level ACCURACY,
# so every model must use the same BALANCED decision rule. For 2-class softmax,
# P(unsafe) >= 0.5 is exactly argmax. We deliberately ignore each checkpoint's
# tuned high-recall threshold (~0.12) here — that operating point is for
# deployment recall, not for measuring train/dev/test accuracy comparably.
BALANCED_THRESHOLD = 0.5


def evaluate_torch(path: Path) -> dict:
    net, tfm, _tuned_thr, backbone = load_checkpoint(path)
    part = du.get_partition()
    res = {}
    for split in ("train", "val", "test"):
        paths, labels = part[split]
        if not paths:
            continue
        y_pred, _ = predict_partition(net, tfm, BALANCED_THRESHOLD, paths)
        acc, rec = scores(labels, y_pred)
        res[split] = {"acc": round(float(acc), 4), "unsafe_recall": round(float(rec), 4)}
    return res


def evaluate_classical(path: Path) -> dict:
    """Evaluate an sklearn HOG pipeline (StandardScaler->PCA->clf) per split."""
    pipe = joblib.load(path)
    part = du.get_partition()
    res = {}
    for split in ("train", "val", "test"):
        paths, labels = part[split]
        if not paths:
            continue
        X = _hog_features(paths)
        y_pred = pipe.predict(X)
        acc, rec = scores(labels, y_pred)
        res[split] = {"acc": round(float(acc), 4), "unsafe_recall": round(float(rec), 4)}
    return res


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]

    if "--hlp-sample" in sys.argv:
        i = sys.argv.index("--hlp-sample")
        n = int(sys.argv[i + 1]) if i + 1 < len(sys.argv) else 50
        from eval_train_error import print_hlp_sample
        print_hlp_sample(n)

    # Load the existing metrics.json so we keep the human-written sections.
    base_path = next((p for p in OUT_PATHS if p.exists()), None)
    data = json.loads(base_path.read_text()) if base_path else {"models": []}
    by_name = {m["name"]: m for m in data.get("models", [])}

    # Search dirs: the model/ tree and the deployed predictor/model/ copy.
    # The big transfer joblibs live under predictor/model/, the rest under model/.
    search_dirs = [ROOT, ROOT.parent / "predictor" / "model"]

    def _resolve(stem: str):
        for d in search_dirs:
            p = d / f"{stem}.joblib"
            if p.exists():
                return p
        return ROOT / f"{stem}.joblib"  # report the canonical miss path

    all_stems = list(STEM_TO_NAME) + list(CLASSICAL_STEM_TO_NAME)
    targets = ([Path(a) if Path(a).is_absolute() else ROOT / a for a in args] if args
               else [_resolve(stem) for stem in all_stems])
    for t in targets:
        if not t.exists():
            print(f"[skip] not found: {t}")
            continue
        is_classical = t.stem in CLASSICAL_STEM_TO_NAME
        name = (CLASSICAL_STEM_TO_NAME if is_classical else STEM_TO_NAME).get(t.stem, t.stem)
        family = "classical (HOG)" if is_classical else "deep"
        # train_cv.py retrains these on ALL train+val before saving, so their val
        # error is contaminated -> the app uses TEST as their held-out instead.
        contaminated = t.stem in TRAIN_VAL_REFIT_STEMS
        print(f"evaluating {name} ({t.name}) ...")
        r = evaluate_classical(t) if is_classical else evaluate_torch(t)
        entry = by_name.setdefault(name, {"name": name, "family": family})
        entry["val_contaminated"] = contaminated
        if "train" in r:
            entry["train_error"] = err(r["train"]["acc"])
        if "val" in r:
            entry["dev_error"] = err(r["val"]["acc"])
            entry["dev_method"] = ("val (CONTAMINATED — trained on train+val)"
                                   if contaminated else "val holdout")
            entry["accuracy"] = r["val"]["acc"]
            entry["unsafe_recall"] = r["val"]["unsafe_recall"]
        if "test" in r:
            entry["test_error"] = err(r["test"]["acc"])
        print(f"  train_err={entry.get('train_error')} dev_err={entry.get('dev_error')} "
              f"test_err={entry.get('test_error')}"
              f"{'  [val contaminated -> held-out=test]' if contaminated else ''}")

    data["models"] = list(by_name.values())
    data["generated"] = "measured by model/export_metrics.py"

    for out in OUT_PATHS:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(data, indent=2))
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
