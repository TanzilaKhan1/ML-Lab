"""Render outputs_final/metrics_full.json into a readable RESULTS.md."""
import json
from pathlib import Path

OUT = Path(__file__).parent / "outputs_final"
M = json.loads((OUT / "metrics_full.json").read_text())
d, models = M["dataset"], M["models"]


def row(vals):
    return "| " + " | ".join(vals) + " |"


def pct(x):
    return "—" if x is None else f"{100*x:.1f}%"


def f3(x):
    return "—" if x is None else f"{x:.3f}"


lines = []
A = lines.append
A("# Safe / Unsafe (Hanging-Passenger) Classifier — Retrained Results")
A("")
A("Binary image classification: passengers hanging on **bus / leguna** doors "
  "(`unsafe`) vs not (`safe`). Dhaka road-safety task.")
A("")
A("## Dataset")
A(f"- Source: {d['source']}")
A(f"- Labels: {d['labels_from']}")
# true ORIGINAL class balance from the authoritative annotation-derived label map
_lm = json.loads((Path(__file__).parent.parent.parent / "data" / "label_map.json").read_text())
_safe = sum(1 for r in _lm if r["label"] == "safe")
_unsafe = sum(1 for r in _lm if r["label"] == "unsafe")
A(f"- **Total labeled images: {d['total_labeled_images']}** "
  f"(originals: safe {_safe} / unsafe {_unsafe})")
A(f"- Split strategy: {d['split_strategy']}")
A(f"- **Train**: {d['train_total']} images = {d['train_originals']} originals + "
  f"**{d['train_augmented_copies']} offline augmentations** "
  f"(safe {d['split_counts']['train']['0']} / unsafe {d['split_counts']['train']['1']}, balanced)")
A(f"- **Val**: {d['val_total']} (safe {d['split_counts']['val']['0']} / unsafe {d['split_counts']['val']['1']}) — real originals only")
A(f"- **Test**: {d['test_total']} (safe {d['split_counts']['test']['0']} / unsafe {d['split_counts']['test']['1']}) — untouched holdout, real originals only")
A("- Val & Test are 4-way stratified so each contains every category (bus-safe, bus-unsafe, legua-safe, legua-unsafe).")
A("- Augmentation is applied to **TRAIN only** → no data leakage.")
A("")

order = list(models)

A("## 1. Main results — BALANCED operating point (max accuracy)")
A("")
A(row(["Model", "Train acc", "Val acc", "Test acc", "Test recall (unsafe)",
       "Test prec (unsafe)", "Test F1 (unsafe)", "Test ROC-AUC", "Test PR-AUC", "Test MCC"]))
A(row(["---"]*10))
for name in order:
    b = models[name]
    tr, va, te = b["train"]["balanced"], b["val"]["balanced"], b["test"]["balanced"]
    A(row([f"**{name}**", pct(tr["accuracy"]), pct(va["accuracy"]), pct(te["accuracy"]),
           pct(te["recall_unsafe"]), pct(te["precision_unsafe"]), f3(te["f1_unsafe"]),
           f3(te["roc_auc"]), f3(te["pr_auc"]), f3(te["mcc"])]))
A("")

A("## 2. HIGH-RECALL operating point (catch >=95% of unsafe — safety default)")
A("")
A(row(["Model", "Test acc", "Test recall (unsafe)", "Test precision (unsafe)", "Test specificity"]))
A(row(["---"]*5))
for name in order:
    te = models[name]["test"]["high_recall"]
    A(row([f"**{name}**", pct(te["accuracy"]), pct(te["recall_unsafe"]),
           pct(te["precision_unsafe"]), pct(te["specificity"])]))
A("")

A("## 3. MAX-RECALL / zero-miss operating point (100% unsafe recall)")
A("")
A(row(["Model", "Test acc", "Test recall (unsafe)", "Test precision (unsafe)"]))
A(row(["---"]*4))
for name in order:
    te = models[name]["test"]["max_recall"]
    A(row([f"**{name}**", pct(te["accuracy"]), pct(te["recall_unsafe"]), pct(te["precision_unsafe"])]))
A("")

A("## 4. Operating thresholds per model (tuned on validation)")
A("")
A(row(["Model", "balanced", "high_recall", "max_recall", "deployed default"]))
A(row(["---"]*5))
for name in order:
    t = models[name]["operating_thresholds"]
    A(row([name, f3(t["threshold_balanced"]), f3(t["threshold_high_recall"]),
           f3(t["threshold_max_recall"]), models[name]["default_deployed_mode"]]))
A("")

A("## 5. Train vs Test (overfitting view, balanced point)")
A("")
A(row(["Model", "Train acc", "Test acc", "Gap"]))
A(row(["---"]*4))
for name in order:
    tr = models[name]["train"]["balanced"]["accuracy"]
    te = models[name]["test"]["balanced"]["accuracy"]
    A(row([name, pct(tr), pct(te), f"{100*(tr-te):.1f} pts"]))
A("")

A("## Figures (`outputs_final/figures/`)")
for f in ["roc_test.png", "pr_test.png", "bars_test.png", "train_vs_test.png",
          "cm_ensemble.png", "cm_resnet50.png", "cm_convnext-tiny.png",
          "cm_efficientnet-b0.png", "cm_resnet18.png", "cm_cnn.png",
          "cm_svm.png", "cm_logistic.png", "cm_naive.png"]:
    A(f"- `{f}`")
A("")
A("Full per-metric data (21 metrics x 3 splits x 3 operating points x 9 models): "
  "`outputs_final/metrics_full.json`.")

(OUT / "RESULTS.md").write_text("\n".join(lines))
print(f"wrote {OUT/'RESULTS.md'}")
