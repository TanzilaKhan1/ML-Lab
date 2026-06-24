"""Run all locally available models on Tanzila's 20 error-analysis images.

Run from inside the predictor environment:
    cd /Users/nasrin/Documents/GitHub/ML-Lab/predictor
    poetry run python ../error-analysis-lab/run_all_models.py

Outputs:
  - a table printed to stdout
  - error-analysis-lab/predictions_all_models.csv
"""
import csv
import sys
from pathlib import Path

# ── path setup so we can import the predictor package ──────────────────────
SCRIPT_DIR    = Path(__file__).resolve().parent          # error-analysis-lab/
PREDICTOR_DIR = SCRIPT_DIR.parent / "predictor"         # predictor/
sys.path.insert(0, str(PREDICTOR_DIR))

IMAGE_DIR  = SCRIPT_DIR / "tanzila"
SHEET_CSV  = SCRIPT_DIR / "error_analysis_sheet.csv"
OUTPUT_CSV = SCRIPT_DIR / "predictions_all_models.csv"

# ── models to run (only those whose .joblib is locally present) ─────────────
# Maps display_name -> (filename, kind)  — mirrors inference.AVAILABLE_MODELS
MODELS = {
    "Ensemble (best)":     ("ensemble.json",            "ensemble"),
    "ResNet50":            ("resnet50.joblib",           "torch"),
    "ConvNeXt-Tiny":       ("convnext_tiny.joblib",      "torch"),
    "EfficientNet-B0":     ("efficientnet_b0.joblib",    "torch"),
    "ResNet18":            ("resnet_model.joblib",       "resnet"),
    "CNN":                 ("cnn_model.joblib",          "cnn"),
    "SVM (RBF)":           ("svm_model.joblib",          "sklearn"),
    "Logistic Regression": ("logistic_model.joblib",     "sklearn"),
    "Naive Bayes":         ("naive_bayes_model.joblib",  "sklearn"),
}
MODEL_DIR = PREDICTOR_DIR / "model"


# ── load metadata from the analysis sheet ──────────────────────────────────
def load_sheet(csv_path: Path) -> list[dict]:
    with csv_path.open(newline="") as f:
        return list(csv.DictReader(f))


# ── load a model ──────────────────────────────────────────────────────────
def load_model(fname: str, kind: str):
    path = MODEL_DIR / fname
    if not path.exists():
        return None
    if kind == "ensemble":
        from predictor_app.torch_models import load_ensemble
        return load_ensemble(path)
    if kind in ("cnn", "resnet", "torch"):
        from predictor_app.torch_models import load_torch_checkpoint
        return load_torch_checkpoint(path, kind=kind)
    import joblib
    return joblib.load(path)


# ── run one model on one image ─────────────────────────────────────────────
def run_model(model, kind: str, image_path: Path) -> tuple[str, float]:
    """Returns (predicted_label_str, prob_unsafe)."""
    import numpy as np
    from PIL import Image as PILImage
    from predictor_app.preprocess import standardize_image, preprocess_for_model

    img = PILImage.open(image_path)

    if kind in ("cnn", "resnet", "torch", "ensemble"):
        pil   = standardize_image(img)
        proba = model.predict_proba(pil)[0]
        pred  = int(model.predict(pil)[0])
    else:
        feats = preprocess_for_model(img)
        pred  = int(model.predict(feats)[0])
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(feats)[0]
        else:
            s  = float(model.decision_function(feats)[0])
            p1 = 1.0 / (1.0 + np.exp(-s))
            proba = [1 - p1, p1]

    label = "unsafe" if pred == 1 else "safe"
    return label, float(proba[1])


# ── main ───────────────────────────────────────────────────────────────────
def main():
    rows = load_sheet(SHEET_CSV)
    if not rows:
        print(f"No rows found in {SHEET_CSV}")
        sys.exit(1)

    print(f"\nLoading {len(MODELS)} models...")
    loaded = {}
    for name, (fname, kind) in MODELS.items():
        m = load_model(fname, kind)
        if m is None:
            print(f"  SKIP  {name:<25} — {fname} not found")
        else:
            loaded[name] = (m, kind)
            print(f"  OK    {name}")

    model_names = list(loaded.keys())
    print(f"\nRunning on {len(rows)} images...\n")

    # ── column widths ──────────────────────────────────────────────────────
    COL = 9
    results = []

    header = (
        f"{'ID':>4}  {'Image':>12}  {'True':>6}  {'Type':>4}  {'Veh':>5}"
        + "".join(f"  {n[:COL]:>{COL}}" for n in model_names)
    )
    print(header)
    print("-" * len(header))

    for row in rows:
        ex_id      = row["example_id"]
        image_file = row["image_file"]
        true_label = row["true_label"]
        err_type   = row["error_type"]
        vehicle    = row["vehicle"]
        img_path   = IMAGE_DIR / image_file

        if not img_path.exists():
            print(f"  MISSING: {img_path}")
            continue

        preds, probs = {}, {}
        for name, (model, kind) in loaded.items():
            try:
                pred, prob = run_model(model, kind, img_path)
            except Exception as e:
                pred, prob = "ERR", 0.0
                print(f"  ERROR {image_file} / {name}: {e}")
            preds[name] = pred
            probs[name] = prob

        # build printed row — mark wrong predictions with *
        cells = []
        for n in model_names:
            p = preds.get(n, "?")
            marker = "*" if p != true_label else " "
            cells.append(f"{(marker + p)[:COL]:>{COL}}")

        short_name = image_file.replace(".png", "")
        print(
            f"{ex_id:>4}  {short_name:>12}  {true_label:>6}  {err_type:>4}  {vehicle:>5}"
            + "  ".join([""] + cells)
        )

        results.append({
            "ex_id": ex_id, "image_file": image_file,
            "true_label": true_label, "err_type": err_type, "vehicle": vehicle,
            **{f"{n}_pred": preds.get(n, "") for n in model_names},
            **{f"{n}_prob_unsafe": f"{probs.get(n, 0):.3f}" for n in model_names},
        })

    # ── save CSV ───────────────────────────────────────────────────────────
    if results:
        fieldnames = list(results[0].keys())
        with OUTPUT_CSV.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(results)
        print(f"\n* = model predicted wrong label")
        print(f"Saved → {OUTPUT_CSV}\n")

    # ── per-model accuracy summary ─────────────────────────────────────────
    if not results:
        return
    print("── Accuracy on these 20 images ───────────────────────")
    print(f"  {'Model':<25}  {'Correct':>8}  {'Acc':>6}")
    for name in model_names:
        correct = sum(1 for r in results if r.get(f"{name}_pred") == r["true_label"])
        total   = len(results)
        print(f"  {name:<25}  {correct:>5}/{total:<3}  {correct/total*100:>5.1f}%")


if __name__ == "__main__":
    main()
