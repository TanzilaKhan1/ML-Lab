"""Train the three classical models (LogReg, SVM-RBF, Gaussian NB) on HOG
features with offline augmentation of the TRAIN split only.

- 70/15/15 canonical split (du.py)
- Offline augmentation balances classes (safe x2, unsafe x6) -> ~braced 50/50
- HOG (128x128 grayscale, 1764-dim) — identical to predictor preprocess
- Pipeline: StandardScaler -> PCA(0.95) -> classifier
- GridSearchCV (5-fold) hyper-parameter search; best refit
- Evaluated on val + test; saves predictor-compatible joblib pipelines
"""
from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

import albumentations as A
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage.color import rgb2gray
from skimage.feature import hog
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, matthews_corrcoef,
                             recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import du

OUT = du.ROOT / "outputs_classical"
OUT.mkdir(parents=True, exist_ok=True)

HOG_SIZE = 128
HOG_PARAMS = dict(orientations=9, pixels_per_cell=(16, 16),
                  cells_per_block=(2, 2), block_norm="L2-Hys")
N_EXTRA = {0: 1, 1: 5}   # extra augmented copies per class (label -> n)
SEED = du.SEED

_AUG = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=12, border_mode=0, p=0.6),
    A.RandomResizedCrop(size=(HOG_SIZE, HOG_SIZE), scale=(0.75, 1.0),
                        ratio=(0.85, 1.15), p=0.7),
    A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.6),
    A.HueSaturationValue(hue_shift_limit=12, sat_shift_limit=25, val_shift_limit=20, p=0.4),
    A.OneOf([A.MotionBlur(blur_limit=(3, 7)), A.GaussianBlur(blur_limit=(3, 7))], p=0.3),
    A.GaussNoise(p=0.2),
])


def hog_from_arr(arr_uint8: np.ndarray) -> np.ndarray:
    """arr_uint8: HxWx3 uint8 at HOG_SIZE -> 1764-dim HOG float32."""
    arr = arr_uint8.astype(np.float32) / 255.0
    gray = rgb2gray(arr)
    return hog(gray, **HOG_PARAMS).astype(np.float32)


def load_small(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        im = im.convert("RGB").resize((HOG_SIZE, HOG_SIZE), Image.LANCZOS)
        return np.asarray(im, dtype=np.uint8)


def build_features(paths, labels, augment: bool, seed: int = SEED):
    rng = np.random.RandomState(seed)
    feats, ys = [], []
    for p, y in zip(paths, labels):
        base = load_small(p)
        feats.append(hog_from_arr(base))
        ys.append(y)
        if augment:
            for k in range(N_EXTRA.get(y, 1)):
                # vary the per-call RNG so copies differ
                A_seed = int(rng.randint(0, 2**31 - 1))
                import random as _r
                _r.seed(A_seed)
                np.random.seed(A_seed)
                out = _AUG(image=base)["image"]
                if out.shape[:2] != (HOG_SIZE, HOG_SIZE):
                    out = np.asarray(Image.fromarray(out).resize((HOG_SIZE, HOG_SIZE)))
                feats.append(hog_from_arr(out))
                ys.append(y)
    return np.stack(feats).astype(np.float32), np.asarray(ys, dtype=np.int64)


def plot_confusion(cm, title, out_path):
    fig, ax = plt.subplots(figsize=(4.5, 3.8))
    im = ax.imshow(cm, cmap="Blues")
    ticks = ["safe", "unsafe"]
    ax.set_xticks(range(2)); ax.set_yticks(range(2))
    ax.set_xticklabels(ticks); ax.set_yticklabels(ticks)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax); fig.tight_layout()
    fig.savefig(out_path, dpi=120); plt.close(fig)


def eval_split(model, X, y):
    pred = model.predict(X)
    try:
        prob = model.predict_proba(X)[:, 1]
        auc = float(roc_auc_score(y, prob)) if len(set(y)) > 1 else float("nan")
    except Exception:
        auc = float("nan")
    return {
        "acc": float(accuracy_score(y, pred)),
        "unsafe_recall": float(recall_score(y, pred, pos_label=1, zero_division=0)),
        "mcc": float(matthews_corrcoef(y, pred)),
        "auc": auc,
        "cm": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
        "report": classification_report(y, pred, target_names=["safe", "unsafe"],
                                        digits=4, zero_division=0),
        "pred": pred.tolist(),
    }


MODELS = {
    "logistic": dict(
        filename="logistic_model.joblib",
        estimator=LogisticRegression(class_weight="balanced", max_iter=5000, random_state=SEED),
        prefix="lr",
        grid=[
            {"lr__C": [0.01, 0.1, 1.0, 10.0], "lr__penalty": ["l2"], "lr__solver": ["lbfgs"]},
            {"lr__C": [0.1, 1.0, 10.0], "lr__penalty": ["l1"], "lr__solver": ["liblinear"]},
        ],
    ),
    "svm": dict(
        filename="svm_model.joblib",
        estimator=SVC(class_weight="balanced", probability=True, random_state=SEED),
        prefix="svm",
        grid={"svm__C": [0.5, 1.0, 4.0, 16.0], "svm__gamma": ["scale", 0.01, 0.001],
              "svm__kernel": ["rbf", "linear"]},
    ),
    "naive_bayes": dict(
        filename="naive_bayes_model.joblib",
        estimator=GaussianNB(),
        prefix="nb",
        grid={"nb__var_smoothing": np.logspace(-12, -2, 11).tolist(),
              "pca__n_components": [0.9, 0.95, 0.99]},
    ),
}


def main():
    t0 = time.time()
    part = du.get_partition()
    (trp, trl), (vap, val), (tep, tel) = part["train"], part["val"], part["test"]
    print(f"split sizes -> train {len(trp)}  val {len(vap)}  test {len(tep)}")

    print("extracting HOG (train with augmentation, val/test originals)...")
    Xtr, ytr = build_features(trp, trl, augment=True)
    Xva, yva = build_features(vap, val, augment=False)
    Xte, yte = build_features(tep, tel, augment=False)
    print(f"train feats: {Xtr.shape}  class balance: {dict(Counter(ytr.tolist()))}")
    print(f"val feats:   {Xva.shape}   test feats: {Xte.shape}")

    results = {}
    for name, cfg in MODELS.items():
        print(f"\n===== {name} =====")
        steps = [("scaler", StandardScaler()),
                 ("pca", PCA(n_components=0.95, random_state=SEED)),
                 (cfg["prefix"], cfg["estimator"])]
        pipe = Pipeline(steps)
        search = GridSearchCV(pipe, cfg["grid"], cv=5, n_jobs=-1,
                              scoring="accuracy", verbose=0)
        search.fit(Xtr, ytr)
        model = search.best_estimator_
        print(f"best params: {search.best_params_}")
        print(f"CV acc: {search.best_score_:.4f}")
        val_m = eval_split(model, Xva, yva)
        test_m = eval_split(model, Xte, yte)
        print(f"VAL  acc {val_m['acc']:.4f}  unsafe_recall {val_m['unsafe_recall']:.4f}")
        print(f"TEST acc {test_m['acc']:.4f}  unsafe_recall {test_m['unsafe_recall']:.4f}  "
              f"auc {test_m['auc']:.4f}  mcc {test_m['mcc']:.4f}")
        print(test_m["report"])

        joblib.dump(model, OUT / cfg["filename"])
        plot_confusion(np.array(test_m["cm"]), f"{name} (test)",
                       OUT / f"{name}_test_cm.png")
        results[name] = {
            "best_params": {k: (str(v) if hasattr(v, "__name__") else v)
                            for k, v in search.best_params_.items()},
            "cv_acc": float(search.best_score_),
            "val": {k: v for k, v in val_m.items() if k != "pred"},
            "test": {k: v for k, v in test_m.items() if k != "pred"},
            "kind": "sklearn",
        }

    (OUT / "results.json").write_text(json.dumps(results, indent=2))
    print(f"\nsaved classical results -> {OUT}/results.json  ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
