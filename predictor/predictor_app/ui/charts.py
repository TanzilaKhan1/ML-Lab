"""Matplotlib figures for the Model Analysis tab.

Every figure is built from ``metrics.json`` data (passed in by the caller) — no
numbers are hard-coded here. The charts operationalise the assignment's three
deliverables and Andrew Ng's *Machine Learning Yearning* ch. 28–31:

  * ``human_level_ladder``        — Human-Level Performance (the "desired
    performance" line, Ng ch. 28).
  * ``bias_variance_dumbbell``    — avoidable bias & variance read at the
    full-data point (Ng ch. 27–28).
  * ``learning_curve_schematic``  — the high-variance learning-curve template
    (Ng ch. 29–31) annotated with our measured endpoints.
  * ``auc_plateau``               — AUC vs. stacked techniques: the "plateau"
    test (Ng ch. 28) → data ceiling.
  * ``error_fp_fn``               — manual 6-reviewer error audit, FP vs FN.

All return a Matplotlib ``Figure``; the caller renders it with ``st.pyplot``
and closes it.
"""
from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from ..config import PALETTE

# Shared colours
_TRAIN = PALETTE.accent       # indigo  — training error
_HELD = PALETTE.unsafe        # red     — held-out / dev error
_HUMAN = PALETTE.safe         # emerald — human-level / desired performance
_FP = "#f59e0b"               # amber   — false positives (safe→unsafe)
_FN = PALETTE.unsafe          # red     — false negatives (unsafe→safe)
_MUTED = PALETTE.text_muted
_TEXT = PALETTE.text
_GRID = PALETTE.border


def _style(ax) -> None:
    """Common axis cosmetics: hide top/right spines, soft grid, muted ticks."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_GRID)
    ax.tick_params(colors=_MUTED, labelsize=9)
    ax.set_axisbelow(True)


def _new(figsize):
    fig, ax = plt.subplots(figsize=figsize, facecolor="white")
    ax.set_facecolor("white")
    _style(ax)
    return fig, ax


# ---------------------------------------------------------------------------
# 1. Human-level performance ladder  (assignment bullet 3 · Ng ch. 28)
# ---------------------------------------------------------------------------
def human_level_ladder(rows: list[tuple[str, float, str]], human_error: float):
    """rows = [(label, error_fraction, kind)]; kind ∈ {human, model, baseline}.
    Horizontal error bars, ascending, with the human-level line marked."""
    rows = sorted(rows, key=lambda r: r[1])
    labels = [r[0] for r in rows]
    vals = [r[1] * 100 for r in rows]
    color_by_kind = {"human": _HUMAN, "model": _TRAIN, "baseline": _MUTED}
    colors = [color_by_kind.get(r[2], _TRAIN) for r in rows]

    fig, ax = _new((6.6, 0.5 * len(rows) + 1.2))
    y = np.arange(len(rows))
    ax.barh(y, vals, color=colors, height=0.6, zorder=3)
    for yi, v in zip(y, vals):
        ax.text(v + 0.6, yi, f"{v:.1f}%", va="center", ha="left",
                fontsize=9, color=_TEXT)
    ax.axvline(human_error * 100, color=_HUMAN, ls="--", lw=1.6, zorder=2)
    ax.text(human_error * 100, len(rows) - 0.3,
            f"  human-level (desired) {human_error*100:.0f}%",
            color=_HUMAN, fontsize=8.5, va="bottom", ha="left")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9, color=_TEXT)
    ax.set_xlabel("error rate  (%, lower is better)", fontsize=9, color=_MUTED)
    ax.set_xlim(0, max(vals) * 1.18)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 2. Bias / variance dumbbell  (assignment bullet 4 · Ng ch. 27–28)
# ---------------------------------------------------------------------------
def bias_variance_dumbbell(items: list[dict], human_error: float):
    """items = [{name, train, held, has_train}] sorted by caller.
    Draws train→held-out segment (variance) per model with the human line."""
    items = sorted(items, key=lambda d: (d["held"] if d["held"] is not None else 1))
    n = len(items)
    fig, ax = _new((7.2, 0.52 * n + 1.4))
    y = np.arange(n)

    for yi, it in zip(y, items):
        held = it["held"]
        tr = it["train"]
        if tr is not None and held is not None:
            ax.plot([tr * 100, held * 100], [yi, yi], color=_GRID, lw=3,
                    solid_capstyle="round", zorder=2)
            ax.scatter([tr * 100], [yi], color=_TRAIN, s=55, zorder=4)
        if held is not None:
            ax.scatter([held * 100], [yi], color=_HELD, s=55, zorder=4)
            ax.text(held * 100 + 0.7, yi, f"{held*100:.1f}%", va="center",
                    ha="left", fontsize=8.5, color=_TEXT)
        if tr is None:
            ax.text(0.6, yi, "train n/a", va="center", ha="left",
                    fontsize=8, color=_MUTED, style="italic")

    ax.axvline(human_error * 100, color=_HUMAN, ls="--", lw=1.6, zorder=1)
    ax.text(human_error * 100, n - 0.3, f"  human {human_error*100:.0f}%",
            color=_HUMAN, fontsize=8.5, va="bottom", ha="left")
    ax.set_yticks(y)
    ax.set_yticklabels([it["name"] for it in items], fontsize=9, color=_TEXT)
    ax.set_xlabel("error rate  (%)", fontsize=9, color=_MUTED)
    ax.set_xlim(-1, max((it["held"] or 0) for it in items) * 100 * 1.15 + 2)

    # legend
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=_TRAIN,
               markersize=8, label="train error"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=_HELD,
               markersize=8, label="held-out error"),
        Line2D([0], [0], color=_GRID, lw=3, label="variance (the gap)"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=8, frameon=False)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 3. Learning-curve schematic  (Ng ch. 29–31, high-variance template)
# ---------------------------------------------------------------------------
def learning_curve_schematic(train_end: float, dev_end: float, human_error: float):
    """High-variance learning-curve TEMPLATE (Ng): curve shapes are illustrative,
    the right-edge endpoints are our measured train/dev errors."""
    fig, ax = _new((6.6, 4.0))
    m = np.linspace(0.04, 1.0, 200)

    # dev error: starts high, decays toward the measured endpoint (plateaus above human)
    dev = dev_end + (0.34 - dev_end) * np.exp(-3.6 * m)
    # training error: rises from ~0 toward the measured endpoint
    train = train_end * (1 - np.exp(-5.0 * m))

    ax.plot(m, dev * 100, color=_HELD, lw=2.2, label="dev / held-out error")
    ax.plot(m, train * 100, color=_TRAIN, lw=2.2, label="training error")
    ax.axhline(human_error * 100, color=_HUMAN, ls="--", lw=1.6,
               label="desired performance (human-level)")

    # shade the train↔dev gap at the right edge = variance
    ax.fill_between(m, train * 100, dev * 100, color=_HELD, alpha=0.06, zorder=0)

    # endpoint annotations
    ax.scatter([1.0], [dev_end * 100], color=_HELD, s=40, zorder=5)
    ax.scatter([1.0], [train_end * 100], color=_TRAIN, s=40, zorder=5)
    ax.annotate(f"dev ≈ {dev_end*100:.1f}%", (1.0, dev_end * 100),
                xytext=(-6, 8), textcoords="offset points", ha="right",
                fontsize=8.5, color=_HELD)
    ax.annotate(f"train ≈ {train_end*100:.0f}–2%", (1.0, train_end * 100),
                xytext=(-6, -12), textcoords="offset points", ha="right",
                fontsize=8.5, color=_TRAIN)
    # variance bracket label
    mid = dev_end * 100 - (dev_end * 100 - train_end * 100) / 2
    ax.annotate("large gap\n= high variance\n⇒ more data helps",
                (0.62, 0.5 * (dev[120] + train[120]) * 100),
                fontsize=8.5, color=_MUTED, ha="center", va="center")

    ax.set_xlabel("training-set size  →", fontsize=9, color=_MUTED)
    ax.set_ylabel("error  (%)", fontsize=9, color=_MUTED)
    ax.set_xticks([])
    ax.set_ylim(0, 36)
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 4. AUC plateau across techniques  (Ng ch. 28 "has it plateaued?")
# ---------------------------------------------------------------------------
def auc_plateau(points: list[dict], ceiling: float):
    """points = [{label, auc}] in stacking order; horizontal ceiling line."""
    labels = [p["label"] for p in points]
    vals = [p["auc"] for p in points]
    x = np.arange(len(points))

    fig, ax = _new((7.0, 4.0))
    ax.plot(x, vals, color=_TRAIN, lw=2, marker="o", markersize=7, zorder=3)
    for xi, v in zip(x, vals):
        ax.text(xi, v + 0.0015, f"{v:.3f}", ha="center", va="bottom",
                fontsize=8.5, color=_TEXT)
    ax.axhline(ceiling, color=_HELD, ls="--", lw=1.6, zorder=2)
    ax.text(0, ceiling, f"ceiling ≈ {ceiling:.2f}  ",
            color=_HELD, fontsize=8.5, va="bottom", ha="left")

    lo = min(vals) - 0.02
    hi = max(max(vals), ceiling) + 0.012
    ax.set_ylim(lo, hi)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8, color=_TEXT, rotation=20, ha="right")
    ax.set_ylabel("5-fold CV ROC-AUC", fontsize=9, color=_MUTED)
    ax.grid(axis="y", color=_GRID, lw=0.6, alpha=0.7)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 5. Manual error analysis: FP vs FN per model  (assignment bullet 1)
# ---------------------------------------------------------------------------
def error_fp_fn(models: list[dict], pool_size: Optional[int] = None):
    """models = [{model, fp, fn}]; stacked horizontal bars, sorted by total."""
    models = sorted(models, key=lambda d: d["fp"] + d["fn"])
    names = [m["model"] for m in models]
    fp = np.array([m["fp"] for m in models])
    fn = np.array([m["fn"] for m in models])
    y = np.arange(len(models))

    fig, ax = _new((6.8, 0.5 * len(models) + 1.4))
    ax.barh(y, fn, color=_FN, height=0.6, zorder=3,
            label="false negative (unsafe→safe)")
    ax.barh(y, fp, left=fn, color=_FP, height=0.6, zorder=3,
            label="false positive (safe→unsafe)")
    for yi, (a, b) in zip(y, zip(fn, fp)):
        ax.text(a + b + 0.3, yi, f"{a+b}", va="center", ha="left",
                fontsize=9, color=_TEXT)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9, color=_TEXT)
    ax.set_xlabel("misclassified images" +
                  (f"  (pool of {pool_size})" if pool_size else ""),
                  fontsize=9, color=_MUTED)
    ax.set_xlim(0, (fp + fn).max() * 1.16)
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    fig.tight_layout()
    return fig
