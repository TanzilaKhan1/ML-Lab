"""Aggregate classical + torch results, deploy best models into the predictor,
and sanity-check predictions on held-out TEST images.

- Reads outputs_classical/results.json and outputs_torch/results.json
- Copies the 5 trained model files into predictor/model/ and model/ (backing up
  any existing files to model_backup_<ts>/ first)
- Writes RESULTS.md (comparison table) + combined results.json
- Loads every deployed model through the predictor's own inference stack and
  predicts on a balanced sample of TEST images to confirm end-to-end safe/unsafe
  prediction works
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import du

MODEL_DIR = du.ROOT                      # repo/model
REPO_ROOT = MODEL_DIR.parent             # repo
PRED_MODEL_DIR = REPO_ROOT / "predictor" / "model"
CLASSICAL = MODEL_DIR / "outputs_classical"
TORCH = MODEL_DIR / "outputs_torch"

MODEL_FILES = {
    "Logistic Regression": ("logistic_model.joblib", CLASSICAL),
    "SVM (RBF)": ("svm_model.joblib", CLASSICAL),
    "Naive Bayes": ("naive_bayes_model.joblib", CLASSICAL),
    "CNN": ("cnn_model.joblib", TORCH),
    "ResNet18": ("resnet_model.joblib", TORCH),
}


def deploy():
    PRED_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    backup = MODEL_DIR / "model_backup_pretrained"
    for name, (fname, srcdir) in MODEL_FILES.items():
        src = srcdir / fname
        if not src.exists():
            print(f"  ! missing trained file for {name}: {src}")
            continue
        for dest_dir in (PRED_MODEL_DIR, MODEL_DIR):
            dest = dest_dir / fname
            if dest.exists():
                backup.mkdir(parents=True, exist_ok=True)
                shutil.copy2(dest, backup / f"{dest_dir.name}__{fname}")
            shutil.copy2(src, dest)
        print(f"  deployed {name}: {fname}")


def load_results():
    cl = json.loads((CLASSICAL / "results.json").read_text()) if (CLASSICAL / "results.json").exists() else {}
    to = json.loads((TORCH / "results.json").read_text()) if (TORCH / "results.json").exists() else {}
    return cl, to


def build_table(cl, to):
    rows = []
    name_map = {"logistic": "Logistic Regression", "svm": "SVM (RBF)",
                "naive_bayes": "Naive Bayes", "cnn": "CNN (scratch)",
                "resnet": "ResNet18 (transfer)"}
    for key, disp in [("logistic", None), ("svm", None), ("naive_bayes", None)]:
        if key in cl:
            r = cl[key]
            rows.append((name_map[key], r.get("cv_acc"), r["val"]["acc"], r["test"]["acc"],
                         r["test"]["unsafe_recall"], r["test"]["auc"], r["test"]["mcc"]))
    for key in ("cnn", "resnet"):
        if key in to:
            r = to[key]
            rows.append((name_map[key], r["val_acc"], r["val_acc"], r["test"]["acc"],
                         r["test"]["unsafe_recall"], r["test"]["auc"], r["test"]["mcc"]))
    return rows


def write_md(rows, cl, to):
    lines = ["# Safe / Unsafe Classifier — Results (annotation-labelled, 70-15-15, GPU)\n",
             "Label: **0 = safe (negative)**, **1 = unsafe (positive)** — derived from "
             "annotation boxes (ground truth), not folder names.\n",
             "Dataset: 385 images (287 safe / 98 unsafe). Split: train 269 / val 58 / test 58 "
             "(stratified). Augmentation: offline (classical, train-only) + online (CNN/ResNet).\n",
             "\n## Test-set comparison\n",
             "| Model | CV/Val acc | Val acc | **Test acc** | Unsafe recall | Test AUC | Test MCC |",
             "|---|---|---|---|---|---|---|"]
    for (name, cv, va, te, ur, auc, mcc) in sorted(rows, key=lambda r: -r[3]):
        def f(x):
            return "—" if x is None or (isinstance(x, float) and x != x) else f"{x*100:.2f}%" if x <= 1 else f"{x:.3f}"
        def g(x):
            return "—" if x is None or (isinstance(x, float) and x != x) else f"{x:.3f}"
        lines.append(f"| {name} | {f(cv)} | {f(va)} | **{f(te)}** | {f(ur)} | {g(auc)} | {g(mcc)} |")
    lines.append("\n> Primary safety metric = recall on the **unsafe** class (a missed unsafe "
                 "passenger is worse than a false alarm).\n")
    (REPO_ROOT / "RESULTS.md").write_text("\n".join(lines))
    combined = {"classical": cl, "torch": to}
    (MODEL_DIR / "all_results.json").write_text(json.dumps(combined, indent=2, default=str))
    print(f"  wrote {REPO_ROOT/'RESULTS.md'} and {MODEL_DIR/'all_results.json'}")


def smoke_test():
    """Run each deployed model through the predictor inference stack on TEST images."""
    sys.path.insert(0, str(REPO_ROOT / "predictor"))
    from predictor_app import inference as inf
    part = du.get_partition()
    tep, tel = part["test"]
    # pick 4 safe + 4 unsafe test images
    safe = [p for p, y in zip(tep, tel) if y == 0][:4]
    unsafe = [p for p, y in zip(tep, tel) if y == 1][:4]
    samples = [(p, 0) for p in safe] + [(p, 1) for p in unsafe]
    print("\n=== predictor smoke test (held-out TEST images) ===")
    models = inf.list_models()
    print(f"models visible to predictor: {list(models)}")
    for mname in models:
        correct = 0
        for p, y in samples:
            pred = inf.predict(mname, p) if hasattr(inf, "predict") else None
            if pred is None:
                # fall back to load_model + manual predict
                model = inf.load_model(mname)
                fname, kind = inf.AVAILABLE_MODELS[mname]
                if kind == "sklearn":
                    from predictor_app.preprocess import preprocess_for_model
                    feats = preprocess_for_model(p)
                    label = int(model.predict(feats)[0])
                else:
                    from PIL import Image
                    label = int(model.predict(Image.open(p).convert("RGB"))[0])
            else:
                label = pred.label
            correct += int(label == y)
        print(f"  {mname:22s}: {correct}/{len(samples)} correct on sampled test images")


def main():
    print("deploying trained models into predictor...")
    deploy()
    cl, to = load_results()
    rows = build_table(cl, to)
    write_md(rows, cl, to)
    try:
        smoke_test()
    except Exception as e:
        import traceback
        print(f"smoke test skipped/failed: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
