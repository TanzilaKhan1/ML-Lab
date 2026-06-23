"""Model Analysis tab — error analysis, human-level performance, bias & variance.

Everything here is rendered from ``metrics.json`` (see ``metrics.py``). Derived
quantities (avoidable bias, variance) are computed live, so the tab reflects the
latest committed metrics after a reboot. No numbers are hard-coded in this file.
"""
from __future__ import annotations

from typing import Optional

import streamlit as st

from ..metrics import ModelDiagnosis, load_metrics, model_diagnoses


def _pct(x: Optional[float], digits: int = 1) -> str:
    return "—" if x is None else f"{x * 100:.{digits}f}%"


def _signed_pct(x: Optional[float]) -> str:
    if x is None:
        return "—"
    sign = "+" if x >= 0 else "−"
    return f"{sign}{abs(x) * 100:.1f}%"


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    head = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join("---" for _ in headers) + " |"
    body = "\n".join("| " + " | ".join(r) + " |" for r in rows)
    return f"{head}\n{sep}\n{body}"


def _render_human_level(hl: dict) -> None:
    st.markdown("### 🧍 Human-Level Performance")
    st.caption(
        "Used as the **Bayes-error proxy** — the best score anyone could realistically get. "
        "It is a property of the *task*, so it is the **same for every model**."
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Human-level accuracy", _pct(hl.get("accuracy")))
    c2.metric("Human-level error", _pct(hl.get("error")))
    c3.metric("Quick-glance error", _pct(hl.get("alt_quick_glance_error")))
    st.markdown(f"**Definition:** {hl.get('definition', '—')}")
    with st.expander("Why isn't human-level error 0%?"):
        st.markdown(hl.get("why_not_zero", "—"))
        st.info(hl.get("how_to_measure", ""))


def _render_bias_variance(diags: list[ModelDiagnosis], human_error: float) -> None:
    st.markdown("### 📉 Avoidable Bias & Variance (per model)")
    st.caption(
        f"**Avoidable bias** = train error − human error ({_pct(human_error)}). Large ⇒ underfitting.  \n"
        "**Variance** = held-out error − train error. Large ⇒ overfitting / needs more data.  \n"
        "Computed **per model**. Held-out error uses **val** by default; for models marked "
        "⚠️ (retrained on train+val, so their val is contaminated) it uses **test** instead."
    )
    headers = ["Model", "Train", "Val", "Test", "Held-out", "Avoid. bias", "Variance", "Diagnosis"]
    rows = []
    any_contam = False
    for d in diags:
        flag = " ⚠️" if d.val_contaminated else ""
        if d.val_contaminated:
            any_contam = True
        rows.append([
            f"**{d.name}**",
            _pct(d.train_error),
            _pct(d.dev_error),
            _pct(d.test_error),
            f"{_pct(d.held_out_error)}{flag}",
            _signed_pct(d.avoidable_bias),
            _signed_pct(d.variance),
            d.regime,
        ])
    st.markdown(_md_table(headers, rows))
    if any_contam:
        st.caption(
            "⚠️ **ResNet50 / ConvNeXt / EfficientNet** were retrained on **train+val** before "
            "saving (`train_cv.py`), so their val error (~0%) is really *training* error, not a "
            "held-out estimate. For an honest variance figure these models use **test** as the "
            "held-out set; all other models use **val**."
        )
    if all(d.train_error is None for d in diags):
        st.warning(
            "**Train error not measured yet** — avoidable bias & variance can't be computed. "
            "Run `python model/export_metrics.py` where the dataset lives, "
            "commit the refreshed `metrics.json`, and reboot the app. The table fills in automatically."
        )


def _render_performance(diags: list[ModelDiagnosis]) -> None:
    st.markdown("### 📊 Model performance")
    st.caption("Headline metrics per model. For this 2.9:1-imbalanced safety task, **unsafe-recall** matters most.")
    headers = ["Model", "Family", "Accuracy", "Unsafe-recall", "AUC", "Dev estimate"]
    rows = []
    for d in diags:
        rows.append([
            f"**{d.name}**", d.family, _pct(d.accuracy), _pct(d.unsafe_recall),
            "—" if d.auc is None else f"{d.auc:.3f}", d.dev_method or "—",
        ])
    st.markdown(_md_table(headers, rows))


def _render_error_analysis(ea: dict) -> None:
    st.markdown("### 🔍 Error Analysis")
    st.caption(
        f"Manual review of the mistakes made by **{ea.get('model', 'the best model')}**, "
        "grouped by *reason* — a ceiling analysis showing what to fix first."
    )
    st.info(ea.get("summary", ""))
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


def _render_diagnosis(dg: dict) -> None:
    st.markdown("### 🧭 Overall Diagnosis")
    st.success(f"**Verdict: {dg.get('verdict', '—')}**")
    st.markdown(dg.get("reasoning", ""))
    actions = dg.get("next_actions", [])
    if actions:
        st.markdown("**Next actions (in priority order):**")
        st.markdown("\n".join(f"{i+1}. {a}" for i, a in enumerate(actions)))
    if dg.get("note_classical_vs_deep"):
        st.caption(dg["note_classical_vs_deep"])
    if dg.get("methodology_note"):
        with st.expander("Methodology — how train/held-out error & bias/variance are computed"):
            st.markdown(dg["methodology_note"])


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

    st.divider()
    if "human_level" in data:
        _render_human_level(data["human_level"])
        st.divider()

    diags = model_diagnoses(data)
    human_error = float(data.get("human_level", {}).get("error", 0.04))
    if diags:
        _render_performance(diags)
        st.divider()
        _render_bias_variance(diags, human_error)
        st.divider()

    if "error_analysis" in data:
        _render_error_analysis(data["error_analysis"])
        st.divider()

    if "diagnosis" in data:
        _render_diagnosis(data["diagnosis"])
