"""Gaussian Naive Bayes on HOG features.

Pipeline: PNG → grayscale → resize 128x128 → HOG → StandardScaler → PCA → GaussianNB.
Stratified 80/20 split. GridSearchCV over var_smoothing.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage.color import rgb2gray
from skimage.feature import hog
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from data_utils import CLASS_NAMES, SEED, scan_dataset, stratified_split

ROOT = Path(__file__).parent
DATA_ROOT = ROOT / "Preprocessed"
OUT_DIR = ROOT / "outputs_naive_bayes"
OUT_DIR.mkdir(parents=True, exist_ok=True)

HOG_SIZE = 128
HOG_PARAMS = dict(orientations=9, pixels_per_cell=(16, 16),
                  cells_per_block=(2, 2), block_norm="L2-Hys")


def extract_features(paths: list[Path]) -> np.ndarray:
    feats = []
    for p in tqdm(paths, desc="HOG", dynamic_ncols=True):
        with Image.open(p) as im:
            im = im.convert("RGB").resize((HOG_SIZE, HOG_SIZE), Image.LANCZOS)
            arr = np.asarray(im, dtype=np.float32) / 255.0
        gray = rgb2gray(arr)
        feats.append(hog(gray, **HOG_PARAMS))
    return np.stack(feats).astype(np.float32)


def plot_confusion(cm: np.ndarray, classes: list[str], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Gaussian Naive Bayes confusion matrix (test)")
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> None:
    paths, labels = scan_dataset(DATA_ROOT)
    print(f"total: {len(paths)} | pos: {sum(labels)} | neg: {len(labels) - sum(labels)}")

    X = extract_features(paths)
    y = np.asarray(labels, dtype=np.int64)
    print(f"HOG dim: {X.shape[1]}")

    train_idx, test_idx = stratified_split(labels, test_size=0.20, seed=SEED)
    X_tr, X_te = X[train_idx], X[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]
    print(f"train: {len(train_idx)} | test: {len(test_idx)}")

    n_neg = int((y_tr == 0).sum())
    n_pos = int((y_tr == 1).sum())
    priors = [n_neg / len(y_tr), n_pos / len(y_tr)]

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=0.95, random_state=SEED)),
        ("nb", GaussianNB(priors=priors)),
    ])
    grid = {
        "nb__var_smoothing": np.logspace(-12, -2, 11).tolist(),
        "pca__n_components": [0.9, 0.95, 0.99],
    }
    search = GridSearchCV(pipe, grid, cv=5, n_jobs=-1, scoring="accuracy", verbose=1)
    print("grid search...")
    search.fit(X_tr, y_tr)
    print(f"best params: {search.best_params_}")
    print(f"best CV acc: {search.best_score_:.4f}")

    model = search.best_estimator_
    y_pred = model.predict(X_te)
    acc = float((y_pred == y_te).mean())
    cm = confusion_matrix(y_te, y_pred)
    report = classification_report(y_te, y_pred, target_names=CLASS_NAMES, digits=4)
    print(f"\ntest acc: {acc:.4f}")
    print("confusion matrix:")
    print(cm)
    print(report)

    joblib.dump(model, OUT_DIR / "naive_bayes_model.joblib")
    plot_confusion(cm, CLASS_NAMES, OUT_DIR / "confusion_matrix.png")
    (OUT_DIR / "report.txt").write_text(
        f"best params: {search.best_params_}\n"
        f"best CV acc: {search.best_score_:.4f}\n"
        f"test acc: {acc:.4f}\n\n"
        f"confusion matrix:\n{cm}\n\n{report}\n")
    (OUT_DIR / "best_params.json").write_text(json.dumps(
        {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
         for k, v in search.best_params_.items()}, indent=2))
    print(f"\nsaved: {OUT_DIR}/{{naive_bayes_model.joblib,confusion_matrix.png,report.txt,best_params.json}}")


if __name__ == "__main__":
    main()
