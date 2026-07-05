"""Aggregate classical + deep (all backbones) experiment results into comparison
tables, charts, and a recommendation. Reads outputs/results_*.json; writes outputs/."""
from __future__ import annotations
import json, csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import common as C

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "savefig.dpi": 160, "font.size": 10, "axes.grid": True, "grid.color": "#E8E8E8",
    "axes.spines.top": False, "axes.spines.right": False, "savefig.bbox": "tight",
})
BLUE, RED, GREEN = "#4C72B0", "#C44E52", "#55A868"


def flat(rec):
    te, tr, va = rec["test"], rec["train"], rec["val"]
    bb = rec["family"].split("/")[-1]
    grp = rec["family"].split("/")[0]
    label = rec["name"] if grp == "classical" else f"{bb}:{rec['name']}" + ("·emb" if grp == "embed" else "")
    return {
        "name": rec["name"], "family": rec["family"], "backbone": bb, "group": grp,
        "label": label,
        "test_acc": te["accuracy"], "test_recall": te["recall_unsafe"],
        "test_prec": te["precision_unsafe"], "test_f1": te["f1_unsafe"],
        "test_auc": te["roc_auc"], "test_prauc": te["pr_auc"], "test_mcc": te["mcc"],
        "train_err": tr["error"], "val_err": va["error"], "test_err": te["error"],
        "gap": rec["gap_train_test"],
    }


def md_table(rows, cols, headers):
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for r in rows:
        out.append("| " + " | ".join(f"{r[c]:.3f}" if isinstance(r[c], float) else str(r[c]) for c in cols) + " |")
    return "\n".join(out)


def hbar(ax, rows, key, title, color):
    rows = sorted(rows, key=lambda r: r[key] if r[key] is not None else -1)
    ax.barh(range(len(rows)), [r[key] or 0 for r in rows], color=color)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r["label"] for r in rows], fontsize=7)
    ax.set_xlim(0, 1); ax.set_title(title)
    for i, r in enumerate(rows):
        ax.text(min((r[key] or 0) + 0.01, 0.97), i, f"{(r[key] or 0):.2f}", va="center", fontsize=6.5)


def main():
    classical = [flat(r) for r in json.loads((C.OUT / "results_classical.json").read_text())] \
        if (C.OUT / "results_classical.json").exists() else []
    deep = []
    for f in sorted(C.OUT.glob("results_deep*.json")):
        deep += [flat(r) for r in json.loads(f.read_text())]
    embed = []
    for f in sorted(C.OUT.glob("results_embed_*.json")):
        embed += [flat(r) for r in json.loads(f.read_text())]
    allrows = classical + deep + embed
    backbones = sorted({r["backbone"] for r in deep})

    cols = ["label", "test_acc", "test_recall", "test_prec", "test_f1", "test_auc",
            "test_prauc", "test_mcc", "train_err", "test_err", "gap"]
    headers = ["Technique", "Acc", "Recall", "Prec", "F1", "AUC", "PR-AUC", "MCC",
               "TrainErr", "TestErr", "Gap"]
    with (C.OUT / "comparison.csv").open("w", newline="") as fh:
        w = csv.writer(fh); w.writerow(headers + ["family"])
        for r in sorted(allrows, key=lambda r: -(r["test_prauc"] or 0)):
            w.writerow([r[c] for c in cols] + [r["family"]])

    by_pr = sorted(allrows, key=lambda r: -(r["test_prauc"] or 0))
    # best technique per backbone (by a safety score = .6*recall + .4*PR-AUC)
    def score(r): return 0.6 * (r["test_recall"] or 0) + 0.4 * (r["test_prauc"] or 0)
    best_per_bb = []
    for bb in backbones:
        rows = [r for r in deep if r["backbone"] == bb]
        best_per_bb.append(max(rows, key=score))

    md = ["# Imbalance + Variance experiments — full comparison\n",
          "Base: original imbalanced train (204 safe / 98 unsafe); held-out val/test = 65 each. "
          "Threshold tuned on val (balanced).",
          f"Backbones: {', '.join(backbones)} (13 techniques each) + classical HOG ({len(classical)} runs).\n",
          "## Best technique per deep backbone (by 0.6·recall + 0.4·PR-AUC)\n",
          md_table(sorted(best_per_bb, key=lambda r: -score(r)), cols, headers), "",
          "## Top 20 overall by PR-AUC (test)\n", md_table(by_pr[:20], cols, headers), "",
          "## Embedding-space sampling (frozen backbone + sklearn) — ranked by PR-AUC\n",
          md_table(sorted(embed, key=lambda r: -(r["test_prauc"] or 0)), cols, headers), "",
          "## Classical (HOG) — ranked by PR-AUC\n",
          md_table(sorted(classical, key=lambda r: -(r["test_prauc"] or 0)), cols, headers)]
    (C.OUT / "comparison.md").write_text("\n".join(md))

    # charts: top-18 deep by PR-AUC; backbone baseline comparison; classical
    if deep:
        top = sorted(deep, key=lambda r: -(r["test_prauc"] or 0))[:18]
        fig, ax = plt.subplots(1, 2, figsize=(15, 8))
        hbar(ax[0], top, "test_prauc", "Deep — top-18 TEST PR-AUC", BLUE)
        hbar(ax[1], top, "test_recall", "Deep (same set) — TEST recall", RED)
        fig.suptitle("Deep techniques across backbones (ResNet18/50, ConvNeXt-Tiny)",
                     fontsize=13, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.96]); fig.savefig(C.FIG / "deep_all_backbones.png"); plt.close(fig)

        # backbone effect on the shared baseline + best technique
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        for j, key in enumerate(("test_prauc", "test_recall")):
            base = sorted([r for r in deep if r["name"] == "baseline"], key=lambda r: r["backbone"])
            ax[j].bar([r["backbone"] for r in base], [r[key] or 0 for r in base], color=[BLUE, GREEN, RED][:len(base)])
            ax[j].set_ylim(0, 1); ax[j].set_title(f"baseline {key}")
            for i, r in enumerate(base):
                ax[j].text(i, (r[key] or 0) + 0.01, f"{(r[key] or 0):.2f}", ha="center", fontsize=8)
        fig.suptitle("Backbone effect (baseline)", fontsize=12, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.94]); fig.savefig(C.FIG / "backbone_effect.png"); plt.close(fig)
    if embed:
        top = sorted(embed, key=lambda r: -(r["test_prauc"] or 0))[:18]
        fig, ax = plt.subplots(1, 2, figsize=(15, 8))
        hbar(ax[0], top, "test_prauc", "Embedding sampling — top TEST PR-AUC", BLUE)
        hbar(ax[1], top, "test_recall", "Embedding sampling (same set) — TEST recall", RED)
        fig.suptitle("Feature-space (frozen-backbone) sampling — DeepSMOTE-style",
                     fontsize=13, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.96]); fig.savefig(C.FIG / "embed_sampling.png"); plt.close(fig)
    if classical:
        fig, ax = plt.subplots(1, 2, figsize=(13, 7))
        hbar(ax[0], classical, "test_recall", "Classical — TEST recall", RED)
        hbar(ax[1], classical, "test_prauc", "Classical — TEST PR-AUC", BLUE)
        fig.suptitle("Classical sampling techniques (HOG)", fontsize=13, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(C.FIG / "classical_techniques.png"); plt.close(fig)

    # recommendation
    pool = deep + embed
    bpr = max(allrows, key=lambda r: (r["test_prauc"] or 0))
    bacc = max(allrows, key=lambda r: (r["test_acc"] or 0))
    safe = max(pool, key=score) if pool else None
    lowvar = max([r for r in pool if (r["test_recall"] or 0) >= 0.80] or pool, key=lambda r: r["gap"]) if pool else None
    bemb = max(embed, key=lambda r: (r["test_prauc"] or 0)) if embed else None
    bclf = max(classical, key=lambda r: (r["test_prauc"] or 0)) if classical else None
    rl = ["# Recommendation\n"]
    rl.append(f"- **Best PR-AUC overall:** `{bpr['label']}` (PR-AUC {bpr['test_prauc']}, recall {bpr['test_recall']}, acc {bpr['test_acc']}, AUC {bpr['test_auc']}).")
    rl.append(f"- **Best accuracy overall:** `{bacc['label']}` (acc {bacc['test_acc']}, recall {bacc['test_recall']}, PR-AUC {bacc['test_prauc']}, AUC {bacc['test_auc']}).")
    if safe: rl.append(f"- **Best safety (0.6·recall+0.4·PR-AUC):** `{safe['label']}` (recall {safe['test_recall']}, PR-AUC {safe['test_prauc']}, acc {safe['test_acc']}).")
    if lowvar: rl.append(f"- **Lowest variance (gap, recall≥0.80):** `{lowvar['label']}` (gap {lowvar['gap']:+.3f}, test_err {lowvar['test_err']}, recall {lowvar['test_recall']}).")
    if bemb: rl.append(f"- **Best embedding-sampling:** `{bemb['label']}` (PR-AUC {bemb['test_prauc']}, recall {bemb['test_recall']}, acc {bemb['test_acc']}, AUC {bemb['test_auc']}).")
    if bclf: rl.append(f"- **Best classical:** `{bclf['label']}` (PR-AUC {bclf['test_prauc']}, recall {bclf['test_recall']}, acc {bclf['test_acc']}).")
    for bb in backbones:
        b = max([r for r in deep if r["backbone"] == bb], key=score)
        rl.append(f"  - best **{bb}**: `{b['name']}` (recall {b['test_recall']}, PR-AUC {b['test_prauc']}, AUC {b['test_auc']}, acc {b['test_acc']}).")
    (C.OUT / "RECOMMENDATION.md").write_text("\n".join(rl) + "\n")
    print("wrote comparison.{md,csv}, RECOMMENDATION.md, figures/")
    print("\n".join(rl))


if __name__ == "__main__":
    main()
