# ML-Lab — Bus & Leguna Door-Hanging Safety Classifier

A complete computer-vision project for detecting whether passengers are
**hanging on the doors of buses and legunas** — a common and dangerous safety
violation on Dhaka roads. The pipeline goes end-to-end from data collection
and annotation to a deployable prediction UI with model-agnostic explanations.

> Full methodology, dataset stats, training details, and per-model evaluation
> live in **[REPORT.md](REPORT.md)**. This README is the navigation map.

---

## What's in the repo

```
ML-Lab/
├── annotator/                  # Web-based image annotation tool (R2-backed)
├── augment.py                  # Image augmentation entry point
├── augment_images.py           #   ↳ 32-transform augmentation pipeline
├── convert_dataset_to_png.py   # Standardise raw images → 512×512 PNG
├── pipeline.py                 # End-to-end training pipeline (HOG + classical ML)
├── model/                      # Training scripts + saved best-weight .joblib files
│   ├── train_logistic.py
│   ├── train_svm.py
│   ├── train_naive_bayes.py
│   ├── logistic_model.joblib
│   ├── svm_model.joblib
│   └── naive_bayes_model.joblib
├── predictor/                  # ★ Streamlit prediction UI + LIME explainer
├── upload_bus_to_r2.py         # Pushes raw images to Cloudflare R2
├── pyproject.toml              # Research-side poetry env
└── REPORT.md                   # Full project report
```

---

## Pipeline at a glance

```
   Raw images (R2)
         │
         ▼
 [annotator]  ─── web UI for labelling positive / negative
         │
         ▼
 [convert_dataset_to_png.py]  ─── EXIF-orient · RGB · 512×512 centre-crop
         │
         ▼
 [augment.py / augment_images.py]  ─── 32× transforms → 1,100 images
         │
         ▼
 [pipeline.py · model/train_*.py]  ─── HOG · StandardScaler · PCA · {LR, SVM, NB}
         │                              group-aware split, CV per fold
         ▼
   model/*.joblib  (saved best weights)
         │
         ▼
 [predictor/]  ─── Streamlit UI: upload → predict → LIME explanation
```

---

## Components

### `annotator/`
Custom web-based image annotation tool. Backed by Cloudflare R2 for image
storage. Supports bounding-box labelling, soft-trash, YOLO/COCO export, and
rotation. Used to produce the labelled dataset.

### `pipeline.py` and `model/`
The research/training side. `pipeline.py` is the end-to-end script with
group-aware splits and learning curves; `model/train_*.py` are per-classifier
trainers that save best-weight `.joblib` pipelines. All three classical
models (Logistic Regression, SVM-RBF, Gaussian Naive Bayes) operate on
1,764-dim HOG feature vectors. See [REPORT.md](REPORT.md) for full results
(test accuracy 75–86%, unsafe recall 69–75%, SVM-RBF AUC 0.94).

### `predictor/` — the prediction app ★
A Streamlit application that lets anyone upload an image and get a prediction
**plus a visual explanation** of why the model decided what it did:

- **Upload** an image (JPG, PNG, WEBP, HEIC/HEIF — iPhone photos work)
- **Preprocess** identically to training (EXIF-orient → 512×512 crop → 128×128
  grayscale → HOG features)
- **Predict** using any of the three saved `.joblib` pipelines (selectable in
  the sidebar) — returns SAFE / UNSAFE with per-class probabilities
- **Explain with LIME** — segments the image into superpixels, perturbs them,
  and fits a local linear surrogate to surface the regions that pushed the
  model toward its decision. Rendered as an overlay, a continuous red/green
  heatmap, and a top-regions-only view.

The app has its **own poetry environment** scoped under `predictor/.venv`, so
the deployable UI stays isolated from the research/training environment at
the repo root.

```bash
cd predictor
python3 -m poetry install --no-root
python3 -m poetry run streamlit run app_streamlit.py
# → http://localhost:8501
```

Full documentation, architecture, and extension notes:
[**predictor/README.md**](predictor/README.md).

---

## Labels (important)

The saved models use the alphabetical convention from `scan_dataset` and the
training scripts in `model/`:

| Label | Class | Meaning |
|---|---|---|
| `0` | negative (safe) | Passengers are **NOT** hanging on the door |
| `1` | positive (UNSAFE) | Passengers **ARE** hanging on the door |

This matches the standard ML convention (positive = event detected) and the
majority-class evidence in `GaussianNB.class_prior_ = [0.6677, 0.3322]`.
`REPORT.md` describes the *intended* mapping at design time; the predictor
follows the *actual* mapping baked into the saved weights.

The primary metric throughout the project is **recall on the unsafe class** —
missing an unsafe passenger is worse than a false alarm.

---

## Quick-start by use case

| Want to… | Go to |
|---|---|
| Run the prediction UI on your machine | [`predictor/`](predictor/README.md) |
| Re-train a model on the dataset | `model/train_*.py` or `pipeline.py` |
| Annotate new images | [`annotator/`](annotator/) |
| Read the full methodology & results | [REPORT.md](REPORT.md) |

---

## Environments

There are **two independent poetry environments** on purpose:

- **Repo-root** (`pyproject.toml`, `poetry.lock`) — research/training:
  numpy, scikit-learn, scikit-image, matplotlib, etc.
- **`predictor/`** (`predictor/pyproject.toml`, `predictor/poetry.lock`) —
  the deployable Streamlit + LIME UI plus everything needed to load the
  joblib pipelines, isolated so it can be deployed independently
  (e.g. Render / Streamlit Cloud) without dragging in the training stack.
