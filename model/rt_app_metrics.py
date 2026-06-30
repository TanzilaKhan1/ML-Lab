"""Generate the predictor app's metrics.json (the schema the Analysis page reads)
from the new 432-image retrain, so the Data/Analysis page shows latest numbers.

Reads  outputs_final/metrics_full.json + outputs_final/probs_cache.npz
Writes predictor/model/metrics.json  and  model/metrics.json
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import du

OUTF = du.ROOT / "outputs_final"
M = json.loads((OUTF / "metrics_full.json").read_text())
D, MM = M["dataset"], M["models"]
Z = np.load(OUTF / "probs_cache.npz")
META = json.loads((OUTF / "probs_cache_meta.json").read_text())

ORDER = ["Ensemble (best)", "ResNet50", "ConvNeXt-Tiny", "EfficientNet-B0", "ResNet18",
         "CNN", "SVM (RBF)", "Logistic Regression", "Naive Bayes"]
FAMILY = {"Ensemble (best)": "deep", "ResNet50": "deep", "ConvNeXt-Tiny": "deep",
          "EfficientNet-B0": "deep", "ResNet18": "deep", "CNN": "deep",
          "SVM (RBF)": "classical", "Logistic Regression": "classical", "Naive Bayes": "classical"}
KEY = {"Ensemble (best)": "ensemble", "ResNet50": "resnet50", "ConvNeXt-Tiny": "convnext",
       "EfficientNet-B0": "efficientnet", "ResNet18": "resnet18", "CNN": "cnn",
       "SVM (RBF)": "svm", "Logistic Regression": "logreg", "Naive Bayes": "nb"}

N_IMG, N_SAFE, N_UNSAFE = 432, 292, 140
BASELINE = round(N_SAFE / N_IMG, 4)


def b(name, split):
    return MM[name][split]["balanced"]


def models_block():
    out = []
    for name in ORDER:
        tr, va, te = b(name, "train"), b(name, "val"), b(name, "test")
        out.append({
            "name": name, "family": FAMILY[name],
            # headline = VALIDATION (selection set); test_* = untouched test
            "accuracy": round(va["accuracy"], 4), "unsafe_recall": round(va["recall_unsafe"], 4),
            "safe_recall": round(va["recall_safe"], 4), "auc": round(va["roc_auc"], 4),
            "test_accuracy": round(te["accuracy"], 4), "test_unsafe_recall": round(te["recall_unsafe"], 4),
            "test_auc": round(te["roc_auc"], 4),
            "train_error": round(1 - tr["accuracy"], 4),
            "dev_error": round(1 - va["accuracy"], 4), "dev_method": "Val (n=65)",
            "test_error": round(1 - te["accuracy"], 4),
            "val_contaminated": False,
        })
    return out


def error_analysis_block():
    # per-model FP/FN on the held-out TEST (balanced), straight from the confusion matrices
    pmc = []
    for name in ORDER:
        cm = b(name, "test")["confusion_matrix"]
        pmc.append({"model": name, "fp": int(cm["fp"]), "fn": int(cm["fn"])})

    # concrete FN / FP examples for the Ensemble on the test set
    paths = META["splits"]["test"]["paths"]; y = np.array(META["splits"]["test"]["y"])
    p = Z["p_ensemble_test"]; thr = MM["Ensemble (best)"]["operating_thresholds"]["threshold_balanced"]
    pred = (p >= thr).astype(int)

    def rel(path):
        q = Path(path); return f"{q.parent.parent.name}/{q.parent.name}/{q.stem}"

    fn_ex, fp_ex = [], []
    for i, path in enumerate(paths):
        if pred[i] != y[i]:
            ex = {"path": rel(path), "unsafe_prob": round(float(p[i]), 3),
                  "note": "missed hanger" if y[i] == 1 else "false alarm"}
            (fn_ex if y[i] == 1 else fp_ex).append(ex)
    fn_ex.sort(key=lambda e: e["unsafe_prob"]); fp_ex.sort(key=lambda e: -e["unsafe_prob"])

    ens = b("Ensemble (best)", "test")["confusion_matrix"]
    return {
        "model": "Ensemble (best)",
        "summary": (f"On the untouched 65-image test set the Ensemble makes "
                    f"{ens['fp'] + ens['fn']} errors ({ens['fn']} missed-unsafe + {ens['fp']} false-alarms) "
                    f"at the balanced operating point. Misses are tiny/distant hangers; the linear/HOG "
                    f"models make many more errors, confirming the deep features help."),
        "image_base_url": "https://ml-lab-bmiv.onrender.com",
        "per_model_counts": {
            "_note": ("Per-model errors on the held-out TEST (n=65), balanced operating point, "
                      "split into false-positive (safe->unsafe) and false-negative "
                      "(unsafe->safe, the safety-critical misses)."),
            "pool_size": 65, "models": pmc,
        },
        "categories": [
            {"category": "False negatives - missed unsafe (safety-critical)",
             "direction": "unsafe -> model says safe", "count": str(ens["fn"]),
             "fixable": "Hard - tiny/distant hangers need higher-res / crop inference",
             "ceiling": "Direct unsafe-recall gain", "examples": fn_ex[:6]},
            {"category": "False positives - false alarms",
             "direction": "safe -> model says unsafe", "count": str(ens["fp"]),
             "fixable": "Partly - threshold tuning + more varied safe images",
             "ceiling": "Accuracy / precision gain", "examples": fp_ex[:6]},
        ],
    }


def main():
    data = {
        "_README": ("App Analysis-page metrics. Regenerate with `python model/rt_app_metrics.py` "
                    "after a retrain; commit predictor/model/metrics.json and reboot the app."),
        "generated": ("regenerated 2026-06-30 from the 432-image retrain "
                      "(model/outputs_final/metrics_full.json); train/val/test errors from "
                      "rt_finalize_eval.py. Re-run model/rt_app_metrics.py to refresh."),
        "task": {
            "name": "Door-Hanging Safety Classifier",
            "description": ("Binary image classification: is a passenger hanging on the door of a "
                            "bus/leguna (unsafe) or not (safe)?"),
            "n_images": N_IMG, "n_safe": N_SAFE, "n_unsafe": N_UNSAFE,
            "imbalance": "2.1:1 (safe:unsafe)",
            "split": ("70/15/15 stratified by vehicle x class (train 302 originals -> 1600 after "
                      "train-only A-Z augmentation / val 65 / test 65). Val & test are real "
                      "originals - no leakage."),
            "majority_baseline_accuracy": BASELINE,
            "primary_metric_note": (f"Because data is imbalanced 2.1:1, watch UNSAFE-RECALL (catching "
                                    f"violations), not just accuracy. An 'always safe' model already "
                                    f"scores {BASELINE*100:.1f}% accuracy while catching 0 violations."),
        },
        "human_level": {
            "error": 0.04, "accuracy": 0.96,
            "definition": ("Careful expert reviewing the full-resolution image (can zoom). "
                           "Used as the Bayes-error proxy."),
            "why_not_zero": ("Tiny / distant hangers occupy only 1-8% of a cluttered frame; some "
                             "borderline poses are genuinely ambiguous (a passenger leaning vs hanging); "
                             "and inter-annotator agreement on subjective safety labels is only ~0.70-0.85. "
                             "So neither a human nor a model reaches 0% error."),
            "sota": ("Controlled bus boarding/posture detection reaches 96-98% (MDPI Sensors 2026; "
                     "MDPI Appl. Sci. 2025); real-world clutter drops naive systems to 72-75%; "
                     "inter-annotator agreement for subjective safety labels is ~0.70-0.85 "
                     "(Claru; PaSBench-Video). => clear images ~97-98%, borderline subset ~80-85%, "
                     "blended ~96%."),
        },
        "models": models_block(),
        "error_analysis": error_analysis_block(),
        "diagnosis": {
            "verdict": "Improved by more data - recall ceiling broken; now mild overfitting.",
            "reasoning": ("The expanded set added unsafe images (98 -> 140), which lifted the best "
                          "ROC-AUC past the old ~0.93 plateau to 0.949 (Ensemble) and raised the "
                          "zero-miss (100%-unsafe-recall) operating point from ~26% to ~57-69% "
                          "accuracy. Deep nets still fit train ~0% with a 13-18 pt train->test gap, "
                          "so the remaining error is variance-leaning - more (and more varied) unsafe "
                          "images would help further, but the hard data ceiling no longer binds."),
        },
    }
    out = json.dumps(data, indent=1)
    (du.ROOT.parent / "predictor" / "model" / "metrics.json").write_text(out)
    (du.ROOT / "metrics.json").write_text(out)
    print("wrote predictor/model/metrics.json and model/metrics.json")
    print(f"  task: {N_IMG} imgs ({N_SAFE} safe / {N_UNSAFE} unsafe), baseline {BASELINE*100:.1f}%")
    print(f"  models: {len(ORDER)} | error_analysis pool=65")


if __name__ == "__main__":
    main()
