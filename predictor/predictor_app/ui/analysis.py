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
        "Held-out error = **CV (n=327)** for the deep transfer models (the n=58 test is "
        "noisy — one image ≈ 6.7%) and the **Test (n=58)** error for the rest, so variance "
        "is computed from the CV column where present and the Test column otherwise. CV is "
        "measured on the train+val images, so the deep-model variance here is a lower bound; "
        "the Test column is the untouched-hold-out cross-check."
    )

    rows = []
    for d in diags:
        cv_err = _pct(d.dev_error) if d.headline_is_cv else "—"
        rows.append(
            f'<tr><td class="model">{_html.escape(d.name)}</td>'
            f'<td class="num grpsep">{_pct(d.train_error)}</td>'
            f'<td class="num">{cv_err}</td>'
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
        '<th class="grpsep">Train</th><th>CV (n=327)</th><th>Test (n=58)</th>'
        '<th class="grpsep">Avoidable bias</th><th>Variance</th>'
        '</tr></thead><tbody>'
        + "".join(rows)
        + "</tbody></table></div>"
    )
    st.markdown(table, unsafe_allow_html=True)

    if all(d.train_error is None for d in diags):
        st.warning(
            "**Train error not measured yet** — avoidable bias & variance can't be computed. "
            "Run `python model/export_metrics.py` where the dataset lives, "
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
        "For this 2.9:1-imbalanced safety task, <b>unsafe-recall</b> matters most.</p>",
        unsafe_allow_html=True,
    )
    st.caption(
        "**CV (n=327)** = 5-fold cross-validation over the train+val images — the "
        "deep-model headline, used because one image moves the small n=58 test score "
        "~6.7%. Because CV is measured on data the models were selected on, it is "
        "optimistic-leaning. **Test (n=58)** = the untouched hold-out — the honest "
        "cross-check. Models without a CV run show “—” in that column."
    )

    rows = []
    for d in diags:
        is_cv = d.headline_is_cv
        acc_cv = _pct(d.accuracy) if is_cv else "—"
        acc_t = _pct(d.test_accuracy) if is_cv else _pct(d.accuracy)
        rec_cv = _pct(d.unsafe_recall) if is_cv else "—"
        rec_t = _pct(d.test_unsafe_recall) if is_cv else _pct(d.unsafe_recall)
        auc_cv = _auc(d.auc) if is_cv else "—"
        auc_t = _auc(d.test_auc) if is_cv else _auc(d.auc)
        rows.append(
            f'<tr><td class="model">{_html.escape(d.name)}</td>'
            f'<td class="num grpsep">{acc_cv}</td><td class="num">{acc_t}</td>'
            f'<td class="num grpsep">{rec_cv}</td><td class="num">{rec_t}</td>'
            f'<td class="num grpsep">{auc_cv}</td><td class="num">{auc_t}</td></tr>'
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
        '<th class="grpsep">CV (n=327)</th><th>Test (n=58)</th>'
        '<th class="grpsep">CV (n=327)</th><th>Test (n=58)</th>'
        '<th class="grpsep">CV (n=327)</th><th>Test (n=58)</th>'
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
    st.markdown("### 📈 Has it plateaued? (the data-ceiling test)")
    st.caption(
        "Ng's plateau test: if stacking stronger techniques no longer moves the "
        "curve, more of the *same kind of effort* won't reach the goal. Across six "
        "recall-improvement techniques the CV AUC never clears the **~0.93 ceiling** "
        "— so the binding constraint is **data** (label noise + too few clear unsafe "
        "images), not model capacity."
    )
    _show(charts.auc_plateau(points, ap.get("ceiling", 0.93)))


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
            "`python model/export_metrics.py` and place it alongside the model files."
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
