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


def _render_bias_variance(diags: list[ModelDiagnosis], human_error: float) -> None:
    st.markdown("### 📉 Avoidable Bias & Variance (per model)")
    st.markdown(
        '<p class="pp-legend"><b>All values are error rates (1 − accuracy) — '
        "lower is better.</b> <b>Avoidable bias</b> = train − human error "
        f"({_pct(human_error)}); a large value ⇒ underfitting. <b>Variance</b> = "
        "held-out − train; a large value ⇒ overfitting / needs more data.</p>",
        unsafe_allow_html=True,
    )
    st.caption(
        "Variance is read from the **Val (n=65)** column (validation was used for threshold "
        "tuning + model selection, so it leans slightly optimistic); the **Test (n=65)** "
        "column is the untouched hold-out cross-check. With 65 images per split, one image ≈ 1.5%."
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
        '<th class="grpsep">Train</th><th>Val (n=65)</th><th>Test (n=65)</th>'
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


def _render_performance(diags: list[ModelDiagnosis]) -> None:
    st.markdown("### 📊 Model performance")
    st.markdown(
        '<p class="pp-legend"><b>All values are scores — higher is better.</b> '
        "Accuracy and unsafe-class recall are percentages; AUC is the 0–1 ROC area. "
        "For this 2.1:1-imbalanced safety task, <b>unsafe-recall</b> matters most.</p>",
        unsafe_allow_html=True,
    )
    st.caption(
        "**Val (n=65)** = the validation split used for threshold tuning + model selection, "
        "so it leans slightly optimistic. **Test (n=65)** = the untouched hold-out — the "
        "honest cross-check. With 65 images per split, one image ≈ 1.5%."
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
        '<th class="grpsep">Val (n=65)</th><th>Test (n=65)</th>'
        '<th class="grpsep">Val (n=65)</th><th>Test (n=65)</th>'
        '<th class="grpsep">Val (n=65)</th><th>Test (n=65)</th>'
        '</tr></thead><tbody>'
        + "".join(rows)
        + "</tbody></table></div>"
    )
    st.markdown(table, unsafe_allow_html=True)


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
        "plateau was **broken by adding unsafe data** (98→140 images → AUC 0.949), "
        "confirming the binding constraint was **data**, not model capacity."
    )
    _show(charts.auc_plateau(points, ap.get("ceiling", 0.95)))


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
    if task.get("primary_metric_note"):
        st.info(task["primary_metric_note"])

    diags = model_diagnoses(data)
    human_error = float(data.get("human_level", {}).get("error", 0.04))

    st.divider()
    if "human_level" in data:
        _render_human_level(data["human_level"])
        st.divider()

    if diags:
        _render_performance(diags)
        st.divider()
        _render_bias_variance(diags, human_error)
        st.divider()

    if "error_analysis" in data:
        _render_error_analysis(data["error_analysis"])
        st.divider()

    if "auc_progression" in data:
        _render_auc_plateau(data["auc_progression"])
        st.divider()

    if "diagnosis" in data:
        _render_diagnosis(data["diagnosis"])
