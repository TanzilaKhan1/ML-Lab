"""Regenerate metrics.json for the app's Model Analysis tab — RUN ON THE CLUSTER.

This is the single source of truth behind the app's analysis tab. It evaluates
each model on the real train/val/test splits (the dataset only exists here),
measures train error, and writes the per-model numbers the app reads.

Workflow after you retrain a model:
    python model/export_metrics.py            # refresh every model
    python model/export_metrics.py resnet50.joblib   # refresh just one
    python model/export_metrics.py --hlp-sample 50   # print human-level sample
  then commit the updated metrics.json (both copies) and reboot the Streamlit app.

It PRESERVES the human-level, error_analysis, and diagnosis sections of the
existing metrics.json (those are human judgements), and only overwrites the
measured per-model numbers + the `generated` stamp.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

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
# Where to write (cluster model dir + the deployed predictor copy if present).
OUT_PATHS = [
    ROOT / "metrics.json",
    ROOT.parent / "predictor" / "model" / "metrics.json",
]


def err(acc):
    return None if acc is None else round(1.0 - acc, 4)


def evaluate_model(path: Path) -> dict:
    net, tfm, thr, backbone = load_checkpoint(path)
    part = du.get_partition()
    res = {}
    for split in ("train", "val", "test"):
        paths, labels = part[split]
        if not paths:
            continue
        y_pred, _ = predict_partition(net, tfm, thr, paths)
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

    targets = ([ROOT / a for a in args] if args
               else [ROOT / f"{stem}.joblib" for stem in STEM_TO_NAME])
    for t in targets:
        if not t.exists():
            print(f"[skip] not found: {t}")
            continue
        name = STEM_TO_NAME.get(t.stem, t.stem)
        print(f"evaluating {name} ({t.name}) ...")
        r = evaluate_model(t)
        entry = by_name.setdefault(name, {"name": name, "family": "deep"})
        if "train" in r:
            entry["train_error"] = err(r["train"]["acc"])
        if "val" in r:
            entry["dev_error"] = err(r["val"]["acc"])
            entry["dev_method"] = "val holdout (measured by export_metrics.py)"
            entry["accuracy"] = r["val"]["acc"]
            entry["unsafe_recall"] = r["val"]["unsafe_recall"]
        if "test" in r:
            entry["test_error"] = err(r["test"]["acc"])
        print(f"  train_err={entry.get('train_error')} dev_err={entry.get('dev_error')} "
              f"test_err={entry.get('test_error')}")

    data["models"] = list(by_name.values())
    data["generated"] = "measured by model/export_metrics.py on the cluster"

    for out in OUT_PATHS:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(data, indent=2))
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
