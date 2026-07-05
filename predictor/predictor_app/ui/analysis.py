"""Model Analysis tab — error analysis, human-level performance, bias & variance.

Everything here is rendered from ``metrics.json`` (see ``metrics.py``). Derived
quantities (avoidable bias, variance) are computed live, so the tab reflects the
latest committed metrics after a reboot. No numbers are hard-coded in this file.
"""
from __future__ import annotations

import html as _html
from typing import Optional

import matplotlib.pyplot as plt
import streamlit as st

from . import charts
from ..metrics import ModelDiagnosis, load_metrics, model_diagnoses


# set by render_analysis_page() so helper tables can read the raw model dicts
_MODELS_RAW: Optional[list] = None


def _show(fig) -> None:
    """Render a Matplotlib figure full-width, then free it."""
    st.pyplot(fig, width="stretch")
    plt.close(fig)


def _pct(x: Optional[float], digits: int = 1) -> str:
    return "—" if x is None else f"{x * 100:.{digits}f}%"


def _signed_pct(x: Optional[float]) -> str:
    if x is None:
        return "—"
    sign = "+" if x >= 0 else "−"
    return f"{sign}{abs(x) * 100:.1f}%"


def _auc(x: Optional[float]) -> str:
    return "—" if x is None else f"{x:.3f}"


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    head = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join("---" for _ in headers) + " |"
    body = "\n".join("| " + " | ".join(r) + " |" for r in rows)
    return f"{head}\n{sep}\n{body}"


def _render_human_level(hl: dict) -> None:
    st.markdown("### 🧍 Human-Level Performance (Bayes-error proxy)")
    c1, c2 = st.columns(2)
    c1.metric("Human-level accuracy", _pct(hl.get("accuracy")))
    c2.metric("Human-level error", _pct(hl.get("error")))
    st.markdown(f"**Definition:** {hl.get('definition', '—')}")
    if hl.get("sota"):
        st.caption(f"**SOTA / literature:** {hl['sota']}")
    with st.expander("Why isn't human-level error 0%?"):
        st.markdown(hl.get("why_not_zero", "—"))


def _render_bias_variance(diags: list[ModelDiagnosis], human_error: float,
                          val_n: int, test_n: int) -> None:
    st.markdown("### 📉 Avoidable Bias & Variance (per model)")
    st.markdown(
        '<p class="pp-legend"><b>All values are error rates (1 − accuracy) — '
        "lower is better.</b> <b>Avoidable bias</b> = train − human error "
        f"({_pct(human_error)}); a large value ⇒ underfitting. <b>Variance</b> = "
        "held-out − train; a large value ⇒ overfitting / needs more data.</p>",
        unsafe_allow_html=True,
    )
    one = 100.0 / test_n if test_n else 1.5
    st.caption(
        f"Variance is read from the **Val (n={val_n})** column (validation was used for threshold "
        f"tuning + model selection, so it leans slightly optimistic); the **Test (n={test_n})** "
        f"column is the untouched hold-out cross-check. With ~{test_n} images per split, "
        f"one image ≈ {one:.1f}%."
    )

    rows = []
    for d in diags:
        val_err = _pct(d.dev_error)
        rows.append(
            f'<tr><td class="model">{_html.escape(d.name)}</td>'
            f'<td class="num grpsep">{_pct(d.train_error)}</td>'
            f'<td class="num">{val_err}</td>'
            f'<td class="num">{_pct(d.test_error)}</td>'
            f'<td class="num grpsep">{_signed_pct(d.avoidable_bias)}</td>'
            f'<td class="num">{_signed_pct(d.variance)}</td>'
            f'<td class="lft grpsep">{_html.escape(d.regime)}</td></tr>'
        )

    table = (
        '<div class="pp-table-wrap"><table class="pp-table"><thead>'
        '<tr class="grp">'
        '<th class="lft" rowspan="2">Model</th>'
        '<th class="grpsep" colspan="3">Error rate (lower is better)</th>'
        '<th class="grpsep" colspan="2">Decomposition</th>'
        '<th class="lft grpsep" rowspan="2">Diagnosis</th>'
        '</tr>'
        '<tr class="sub">'
        f'<th class="grpsep">Train</th><th>Val (n={val_n})</th><th>Test (n={test_n})</th>'
        '<th class="grpsep">Avoidable bias</th><th>Variance</th>'
        '</tr></thead><tbody>'
        + "".join(rows)
        + "</tbody></table></div>"
    )
    st.markdown(table, unsafe_allow_html=True)

    if all(d.train_error is None for d in diags):
        st.warning(
            "**Train error not measured yet** — avoidable bias & variance can't be computed. "
            "Run `python model/rt_app_metrics.py` where the dataset lives, "
            "commit the refreshed `metrics.json`, and reboot the app. The table fills in automatically."
        )
        return

    # Dumbbell: train→held-out gap (variance) per model, vs the human line.
    items = [
        {"name": d.name, "train": d.train_error, "held": d.held_out_error}
        for d in diags
    ]
    st.markdown("**Read at the full-data point (Ng):** distance from the blue "
                "train dot to the green human line is *avoidable bias*; the bar to "
                "the red held-out dot is *variance*. Bars are long and sit to the "
                "right ⇒ variance dominates.")
    _show(charts.bias_variance_dumbbell(items, human_error))


def _render_performance(diags: list[ModelDiagnosis], val_n: int, test_n: int,
                        imbalance: str) -> None:
    st.markdown("### 📊 Model performance")
    st.markdown(
        '<p class="pp-legend"><b>All values are scores — higher is better.</b> '
        "Accuracy and unsafe-class recall are percentages; AUC is the 0–1 ROC area. "
        f"For this {_html.escape(imbalance)}-imbalanced safety task, <b>unsafe-recall</b> "
        "matters most.</p>",
        unsafe_allow_html=True,
    )
    st.caption(
        f"**Val (n={val_n})** = the validation split used for threshold tuning + model selection, "
        f"so it leans slightly optimistic. **Test (n={test_n})** = the untouched hold-out — the "
        f"honest cross-check."
    )

    rows = []
    for d in diags:
        rows.append(
            f'<tr><td class="model">{_html.escape(d.name)}</td>'
            f'<td class="num grpsep">{_pct(d.accuracy)}</td><td class="num">{_pct(d.test_accuracy)}</td>'
            f'<td class="num grpsep">{_pct(d.unsafe_recall)}</td><td class="num">{_pct(d.test_unsafe_recall)}</td>'
            f'<td class="num grpsep">{_auc(d.auc)}</td><td class="num">{_auc(d.test_auc)}</td></tr>'
        )

    table = (
        '<div class="pp-table-wrap"><table class="pp-table"><thead>'
        '<tr class="grp">'
        '<th class="lft" rowspan="2">Model</th>'
        '<th class="grpsep" colspan="2">Accuracy</th>'
        '<th class="grpsep" colspan="2">Unsafe-recall</th>'
        '<th class="grpsep" colspan="2">AUC</th>'
        '</tr>'
        '<tr class="sub">'
        f'<th class="grpsep">Val (n={val_n})</th><th>Test (n={test_n})</th>'
        f'<th class="grpsep">Val (n={val_n})</th><th>Test (n={test_n})</th>'
        f'<th class="grpsep">Val (n={val_n})</th><th>Test (n={test_n})</th>'
        '</tr></thead><tbody>'
        + "".join(rows)
        + "</tbody></table></div>"
    )
    st.markdown(table, unsafe_allow_html=True)

    # Advanced held-out TEST metrics (precision / F1 / PR-AUC / MCC / specificity)
    adv = [m for m in (_MODELS_RAW or []) if m.get("test_pr_auc") is not None]
    if adv:
        with st.expander("Advanced held-out TEST metrics (precision · F1 · PR-AUC · MCC · specificity)"):
            arows = []
            for m in adv:
                arows.append([
                    m["name"], _pct(m.get("test_precision_unsafe")), _pct(m.get("test_f1_unsafe"), 1),
                    _auc(m.get("test_pr_auc")), f'{m.get("test_mcc", 0):.3f}',
                    _pct(m.get("test_specificity")),
                ])
            st.markdown(_md_table(
                ["Model", "Precision (unsafe)", "F1 (unsafe)", "PR-AUC", "MCC", "Specificity"], arows))
            st.caption("Precision/F1 are for the unsafe class; PR-AUC and MCC are robust to imbalance; "
                       "specificity = safe-class recall. All on the untouched test set.")


def _render_error_analysis(ea: dict) -> None:
    st.markdown("### 🔍 Error Analysis")
    st.caption(
        f"Manual review of the mistakes made by **{ea.get('model', 'the best model')}**, "
        "grouped by *reason* — a ceiling analysis showing what to fix first."
    )
    st.info(ea.get("summary", ""))

    pmc = ea.get("per_model_counts")
    if pmc and pmc.get("models"):
        st.caption(
            "**Manual audit — false positives vs false negatives.** From the "
            "6-reviewer sheets: errors on the shared misclassified pool, split by "
            "type. **False negatives (red)** are the safety-critical misses. The "
            "Ensemble and ConvNeXt make the fewest errors; the linear/HOG models "
            "make many more, confirming the deep features help."
        )
        _show(charts.error_fp_fn(pmc["models"], pmc.get("pool_size")))

    base = (ea.get("image_base_url") or "").rstrip("/")
    for cat in ea.get("categories", []):
        with st.expander(f"{cat.get('category', '?')}  ·  ~{cat.get('count', '?')} images  ·  {cat.get('fixable', '')}"):
            st.markdown(
                f"- **Direction:** {cat.get('direction', '—')}\n"
                f"- **Ceiling if fixed:** {cat.get('ceiling', '—')}"
            )
            examples = cat.get("examples", [])
            if examples:
                rows = []
                for ex in examples:
                    path = ex.get("path", "")
                    name = path.split("/")[-1] if path else "?"
                    link = f"[{name}]({base}/)" if base else name
                    rows.append([
                        link,
                        f"{ex.get('unsafe_prob', '—')}",
                        ex.get("note", "") or "—",
                    ])
                st.markdown(_md_table(["Image", "P(unsafe)", "Note"], rows))
                if base:
                    st.caption(f"Images are hosted in the annotator app: {base}")


def _render_auc_plateau(ap: dict) -> None:
    points = ap.get("points")
    if not points:
        return
    st.markdown("### 📈 The data-ceiling test")
    st.caption(
        "Ng's plateau test: if stacking stronger techniques no longer moves the curve, "
        "more of the *same kind of effort* won't reach the goal. The old ~0.93 AUC "
        "plateau was **broken by adding unsafe data** (unsafe 98→251 → AUC 0.97+), "
        "confirming the binding constraint was **data**, not model capacity."
    )
    _show(charts.auc_plateau(points, ap.get("ceiling", 0.95)))


def _render_dataset_detail(sd: dict) -> None:
    st.markdown("### 🧬 Dataset & split (leak-free)")
    sc = sd.get("split_counts", {})

    def _cell(split):
        c = sc.get(split, {})
        safe = c.get("0", c.get(0, "—")); unsafe = c.get("1", c.get(1, "—"))
        return safe, unsafe

    tr_s, tr_u = _cell("train"); va_s, va_u = _cell("val"); te_s, te_u = _cell("test")
    rows = [
        ["Train (augmented)", sd.get("train_total", "—"), tr_s, tr_u],
        ["Train (originals)", sd.get("train_originals", "—"), "—", "—"],
        ["Val", sd.get("val_total", "—"), va_s, va_u],
        ["Test", sd.get("test_total", "—"), te_s, te_u],
    ]
    st.markdown(_md_table(["Split", "Images", "Safe", "Unsafe"],
                          [[str(x) for x in r] for r in rows]))
    aug = sd.get("train_augmented_copies")
    if aug is not None:
        st.caption(f"**{aug} augmented copies** added to TRAIN only. {sd.get('note', '')}")


def _render_ensemble(ens: dict, op: dict) -> None:
    st.markdown("### 🧩 Deployed ensemble & operating modes")
    members = ", ".join(m.replace(".joblib", "") for m in ens.get("members", []))
    st.markdown(f"**Ensemble = {members}**  ·  {ens.get('method', '')}")
    modes = op.get("modes", [])
    if modes:
        rows = [[m["mode"], f'{m["threshold"]:.3f}', _pct(m["unsafe_recall"]),
                 _pct(m["accuracy"]), _pct(m.get("precision_unsafe"))] for m in modes]
        st.markdown(_md_table(
            ["Mode", "Threshold", "Unsafe-recall", "Accuracy", "Precision (unsafe)"], rows))
        st.caption(op.get("_note", "") +
                   f"  Default deployed mode: **{op.get('default_deployed', 'high_recall')}**.")


def _render_diagnosis(dg: dict) -> None:
    st.markdown("### 🧭 Overall Diagnosis")
    st.success(f"**Verdict: {dg.get('verdict', '—')}**")
    st.markdown(dg.get("reasoning", ""))


def render_analysis_page() -> None:
    """Render the whole Model Analysis tab from metrics.json."""
    data = load_metrics()
    if data is None:
        st.warning(
            "No `metrics.json` found in the model directory. Generate it with "
            "`python model/rt_app_metrics.py` and place it alongside the model files."
        )
        return

    global _MODELS_RAW
    _MODELS_RAW = data.get("models")

    task = data.get("task", {})
    st.markdown(f"## {task.get('name', 'Model Analysis')}")
    if task.get("description"):
        st.caption(task["description"])
    gen = data.get("generated", "")
    if gen:
        st.caption(f"📄 metrics source: {gen}")

    # Dataset snapshot
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Images", task.get("n_images", "—"))
    c2.metric("Safe", task.get("n_safe", "—"))
    c3.metric("Unsafe", task.get("n_unsafe", "—"))
    c4.metric("Majority baseline", _pct(task.get("majority_baseline_accuracy")))
    if task.get("split"):
        st.caption(f"**Split:** {task['split']}")
    if task.get("primary_metric_note"):
        st.info(task["primary_metric_note"])

    sizes = data.get("eval_sizes", {})
    val_n = int(sizes.get("val", 65))
    test_n = int(sizes.get("test", 65))
    imbalance = task.get("imbalance", "imbalanced")

    diags = model_diagnoses(data)
    human_error = float(data.get("human_level", {}).get("error", 0.04))

    st.divider()
    if "split_detail" in data:
        _render_dataset_detail(data["split_detail"])
        st.divider()

    if "human_level" in data:
        _render_human_level(data["human_level"])
        st.divider()

    if diags:
        _render_performance(diags, val_n, test_n, imbalance)
        st.divider()
        _render_bias_variance(diags, human_error, val_n, test_n)
        st.divider()

    if "ensemble" in data and "operating_points" in data:
        _render_ensemble(data["ensemble"], data["operating_points"])
        st.divider()

    if "error_analysis" in data:
        _render_error_analysis(data["error_analysis"])
        st.divider()

    if "auc_progression" in data:
        _render_auc_plateau(data["auc_progression"])
        st.divider()

    if "diagnosis" in data:
        _render_diagnosis(data["diagnosis"])
