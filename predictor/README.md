# Predictor — Door-Hanging Safety Classifier UI

A Streamlit application that **classifies whether passengers are hanging on a
bus/leguna door** in an uploaded image, and **explains its decision** with a LIME
region-importance heatmap.

The app loads the joblib pipelines trained in [`../model/`](../model/) — no
retraining is performed here. The poetry environment is fully scoped to this
folder so it stays independent of the training/research env at the repo root.

![flow](https://img.shields.io/badge/flow-upload%20%E2%86%92%20preprocess%20%E2%86%92%20predict%20%E2%86%92%20explain-blue)

---

## What the app does

1. **Upload** — accepts JPG, PNG, WEBP, BMP, TIFF, HEIC/HEIF (iPhone photos work).
2. **Preprocess** — EXIF-orient → RGB → resize short side to 512 → center-crop
   to 512×512, exactly mirroring the training-time preprocessing in
   `../model/preprocess.py`.
3. **Feature extract** — resize 512×512 → 128×128 → grayscale → HOG
   (9 orientations, 16×16 cells per block, L2-Hys normalisation), producing
   the same 1,764-dim feature vector the models were trained on.
4. **Predict** — runs the chosen sklearn `Pipeline` (`StandardScaler → PCA →
   classifier`) from `../model/*.joblib` and emits a label + per-class
   probabilities:

   | Label | Class | Meaning |
   |---|---|---|
   | `0` | negative (safe) | Passengers are **NOT** hanging on the door |
   | `1` | positive (UNSAFE) | Passengers **ARE** hanging on the door |

   This ordering matches the alphabetical class assignment used by the
   training scripts (`model/train_*.py`) and is confirmed by the saved
   `GaussianNB.class_prior_` of the majority class.

5. **Explain (LIME)** — see below.

### Available models

| Display name | Weights file | Kind |
|---|---|---|
| Ensemble (best) | `model/ensemble.json` | torch ensemble (ResNet50 + ConvNeXt) |
| ResNet50 | `model/resnet50.joblib` | torch transfer |
| ConvNeXt-Tiny | `model/convnext_tiny.joblib` | torch transfer |
| EfficientNet-B0 | `model/efficientnet_b0.joblib` | torch transfer |
| CNN | `model/cnn_model.joblib` | torch (custom SmallCNN) |
| ResNet18 | `model/resnet_model.joblib` | torch transfer |
| SVM (RBF) | `model/svm_model.joblib` | sklearn (HOG features) |
| Logistic Regression | `model/logistic_model.joblib` | sklearn (HOG features) |
| Naive Bayes | `model/naive_bayes_model.joblib` | sklearn (HOG features) |

Switch between them from the sidebar; predictions and the explanation update
live. Torch models consume the raw 512×512 image; the sklearn models consume
the HOG feature vector described above. Only the model currently selected is
held in memory (see `load_model`'s `lru_cache(maxsize=1)`), which keeps the
app within small cloud hosts' memory limits.

> **Deployment note:** `resnet50.joblib` and `convnext_tiny.joblib` (hence the
> Ensemble) are tracked via **Git LFS**. Hosts that don't fetch LFS (e.g.
> Streamlit Community Cloud) clone pointer files instead of the real weights, so
> those three options error if selected there until the weights are made
> available another way. The Model Analysis tab is unaffected.

---

## The explainer — LIME

For every prediction the app also runs **LIME** (Local Interpretable
Model-agnostic Explanations) so you can see *which parts of the image* pushed
the model toward its decision. LIME is model-agnostic: it treats the sklearn
pipeline as a black box.

**How it works here:**

1. The standardised 512×512 image is segmented into ~80 *superpixels*
   (small connected regions) via SLIC.
2. LIME generates a few hundred perturbed copies of the image, each with a
   random subset of superpixels masked out, and queries the model for class
   probabilities on every copy. Perturbed superpixels are filled with the
   segment's **mean colour** (not black) — critical for HOG models, because
   a zero fill would manufacture artificial high-gradient edges at every
   segment boundary and bias the surrogate.
3. A small **ridge surrogate** is fit in this perturbation space *for the
   predicted class only* (LIME's default fits one per class — wasted work for
   binary). Its coefficients tell us how much each superpixel pushed the
   prediction toward that class.
4. We render three views in the UI (as tabs):
   - **Overlay** — top supporting superpixels washed in **red**, top opposing
     in **green**, with thin **white outlines** marking each segment boundary.
   - **Heatmap** — continuous per-pixel weight from the surrogate; **red**
     regions pushed the model *toward* the predicted class, **green** regions
     pushed it *against* (matplotlib `RdYlGn_r` colormap).
   - **Top regions** — only the strongest positive superpixels are shown
     (others masked) so you see the clean "what convinced the model" picture.

**Robustness & performance notes:**

- The classifier wrapper handles `predict_proba`, binary `decision_function`
  (1-D margin → sigmoid), and multi-class `decision_function` (softmax) so any
  of the three saved pipelines plugs in without special-casing.
- Perturbations are run in batches of 64 (LIME's default is 10), which cuts
  Python-side overhead on the HOG feature extractor.
- The signed per-pixel heatmap is built via an O(pixels) segment-id lookup
  rather than an O(segments × pixels) mask loop.
- A fixed `random_state=42` is passed to both LIME and SLIC for reproducible
  explanations across runs of the same image.

Perturbation count is adjustable in the sidebar (100 – 2000). More samples →
smoother map, slower (a few seconds → ~30s). The default of 200 keeps memory
and latency low on small cloud hosts; raise it for a smoother map.

---

## The Model Analysis tab (dynamic)

The app has a second tab, **📊 Model Analysis**, that presents the project
through the standard ML-debugging lens — **error analysis, human-level
performance, and avoidable bias vs variance** — for every model.

**What it shows**

- **Human-level performance** — the Bayes-error proxy (a property of the *task*,
  so it is shared across all models).
- **Per-model performance** — accuracy, unsafe-recall, AUC.
- **Avoidable bias & variance** — `avoidable_bias = train_error − human_error`
  and `variance = dev_error − train_error`, computed **per model**, with a
  plain-language read of each model's regime (under/over-fitting).
- **Error analysis** — the model's mistakes grouped by *reason* (label noise,
  tiny/distant hanger, occlusion), with example images linking to the
  annotator app.
- **Overall diagnosis** — the headline verdict and prioritised next actions.

**How it works (high level)**

The app is deployed **without the dataset**, so it never recomputes errors from
the model files. Instead it reads a single data file, `model/metrics.json`, and
computes only the *derived* quantities (avoidable bias, variance) live. This
keeps the tab **dynamic, not hard-coded**:

```
retrain a model
   └─► python model/export_metrics.py        # ON THE CLUSTER (where the data lives):
                                              #   measures train/val/test error per model
                                              #   and rewrites model/metrics.json
        └─► commit the refreshed metrics.json
             └─► reboot the app  ─►  the tab updates automatically
```

So the responsibilities split cleanly:

| Concern | Where | Why |
|---|---|---|
| Measuring train/dev/test error | `model/export_metrics.py` (cluster) | only the cluster has the dataset |
| Storing the numbers | `model/metrics.json` (committed) | single source of truth, lightweight |
| Reading + deriving bias/variance | `predictor_app/metrics.py` | no dataset needed, runs in-app |
| Rendering the tab | `predictor_app/ui/analysis.py` | pure presentation |

The human-level value and the error-analysis categories are human judgements,
so `export_metrics.py` **preserves** those sections and only overwrites the
measured per-model numbers.

---

## Project layout

```
predictor/
├── pyproject.toml             # poetry config (scoped to this folder)
├── poetry.lock                # generated by `poetry install`
├── poetry.toml                # local config: virtualenvs.in-project = true
├── .venv/                     # generated by `poetry install`
├── README.md                  # this file
├── app_streamlit.py           # thin entry point: wires UI ↔ backend
└── predictor_app/             # the actual library
    ├── __init__.py            # public exports
    ├── config.py              # paths, defaults, colour palette
    ├── preprocess.py          # standardize_image + HOG feature extraction
    ├── inference.py           # load_model + predict (joblib pipelines + torch wrappers)
    ├── torch_models.py        # torch checkpoint loaders (CNN/ResNet/transfer/ensemble)
    ├── explain.py             # LIME wrapper returning a LimeExplanation dataclass
    ├── metrics.py             # reads metrics.json, derives avoidable bias / variance
    └── ui/                    # presentation layer (Streamlit only)
        ├── theme.py           # page_config() + apply_theme() (CSS)
        ├── components.py      # render_sidebar / header / upload / prediction / explanation
        └── analysis.py        # the 📊 Model Analysis tab
```

Model-analysis data + tooling live in [`../model/`](../model/):
`metrics.json` (the data the tab reads) and `export_metrics.py` (regenerates it
on the cluster after retraining).

The domain layer (`preprocess`, `inference`, `explain`) has **zero Streamlit
imports**, so the same code is reusable from a FastAPI route, a CLI, or a
notebook.

---

## One-time setup

```bash
cd predictor

# 1) Ensure poetry is available. If you don't have it:
pip install --user poetry
# (or: brew install poetry / pipx install poetry)
# If `poetry` is not on PATH, prefix every command with `python3 -m`.

# 2) Keep the virtualenv inside this folder
python3 -m poetry config virtualenvs.in-project true --local

# 3) Install dependencies (creates predictor/.venv and predictor/poetry.lock)
python3 -m poetry install --no-root
```

## Run the app

```bash
python3 -m poetry run streamlit run app_streamlit.py
```

Opens at <http://localhost:8501>.

## Streamlit Community Cloud model weights

Streamlit Community Cloud may clone Git LFS pointers instead of the real large
weight files. This app therefore supports downloading the large weights at
startup and caching them under `predictor/model/`.

Use this flow for `resnet50.joblib` and `convnext_tiny.joblib`:

1. The two files are hosted as assets on the GitHub Release
   <https://github.com/TanzilaKhan1/ML-Lab/releases/tag/model-weights-v1>.
2. In Streamlit Community Cloud, open the app settings and add these secrets:
   ```toml
   MODEL_BASE_URL = "https://github.com/TanzilaKhan1/ML-Lab/releases/download/model-weights-v1"
   RESNET50_JOBLIB_SHA256 = "e012f9466f9273f15448d4dffc2eec5a52f5180b69ffecd46660c02c77469d35"
   CONVNEXT_TINY_JOBLIB_SHA256 = "075f6823a489243831773de080997c5b4f4a68eb64659ab7e3f4e1992167d371"
   ```
   If the files are moved to unrelated URLs later, set direct URLs instead:
   ```toml
   RESNET50_JOBLIB_URL = "https://..."
   CONVNEXT_TINY_JOBLIB_URL = "https://..."
   ```

`efficientnet_b0.joblib` is small enough to stay as a normal repository file.
Only change the SHA256 values above if you upload newly trained weights.

## Configuration tips

- To silence Streamlit's first-run email prompt, create `~/.streamlit/credentials.toml`:
  ```toml
  [general]
  email = ""
  ```
- To bind to a different port:
  `python3 -m poetry run streamlit run app_streamlit.py --server.port 8600`
- For deployment (e.g. Render), bind to `$PORT` and `0.0.0.0`:
  `streamlit run app_streamlit.py --server.port=$PORT --server.address=0.0.0.0`

---

## Extending the app

- **Add a new model** — drop the `.joblib` into `../model/`, then add one line
  to `AVAILABLE_MODELS` in `predictor_app/inference.py`. The pipeline must
  accept the same 1,764-dim HOG feature vector.
- **Add a new explainer** (SHAP, GradCAM, …) — create a sibling module to
  `predictor_app/explain.py` returning a similar dataclass, then add a new tab
  in `predictor_app/ui/components.py::render_explanation`.
- **Add a new page** — Streamlit's `pages/` convention works; the components
  in `predictor_app/ui/` are reusable.

---

## Dependencies (key)

| Package | Role |
|---|---|
| `streamlit` | web UI framework |
| `scikit-learn` | runs the trained pipelines |
| `scikit-image` | HOG features + SLIC superpixels |
| `lime` | model-agnostic explainability |
| `joblib` | loads saved model pipelines |
| `pillow` + `pillow-heif` | image decoding incl. iPhone HEIC |
| `matplotlib` | heatmap rendering |
