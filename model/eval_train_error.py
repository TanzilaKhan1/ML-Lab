"""Measure TRAIN vs VAL vs TEST error for the deployed models.

Run this ON THE CLUSTER (where Preprocessed/ + the .joblib checkpoints live).
It fills the one missing number in LAB_ERROR_ANALYSIS.md Part 3:
    avoidable bias = train_error - human_level_error
    variance       = dev(val)_error - train_error

Usage:
    python eval_train_error.py                      # all torch checkpoints found
    python eval_train_error.py resnet50.joblib      # a specific one
    python eval_train_error.py --hlp-sample 50      # also print a human-level labelling sample

Notes:
- Uses the SAME 70/15/15 split (du.load_split) every model was trained on.
- Uses the SAME eval transform as inference (Resize 1.15x -> CenterCrop -> ImageNet norm).
- "error" = 1 - accuracy. We also print unsafe-recall (the safety-critical metric).
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import models as tvm
from torchvision import transforms

import du  # noqa: E402  (shared split + dataset scan)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- architecture registry (mirrors predictor_app/torch_models._build_arch) ---
def build_arch(backbone: str, num_classes: int = 2) -> nn.Module:
    if backbone in ("smallcnn", "cnn"):
        from train_customised_cnn import SmallCNN
        return SmallCNN(num_classes=num_classes)
    if backbone in ("resnet18", "resnet"):
        m = tvm.resnet18(weights=None)
        m.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(m.fc.in_features, num_classes))
        return m
    if backbone == "resnet50":
        m = tvm.resnet50(weights=None)
        m.fc = nn.Sequential(nn.Dropout(0.4), nn.Linear(m.fc.in_features, num_classes))
        return m
    if backbone == "efficientnet_b0":
        m = tvm.efficientnet_b0(weights=None)
        m.classifier = nn.Sequential(nn.Dropout(0.4),
                                     nn.Linear(m.classifier[1].in_features, num_classes))
        return m
    if backbone == "convnext_tiny":
        m = tvm.convnext_tiny(weights=None)
        m.classifier[2] = nn.Linear(m.classifier[2].in_features, num_classes)
        return m
    raise ValueError(f"Unknown backbone {backbone!r}")


def eval_transform(backbone: str, img_size: int):
    if backbone in ("smallcnn", "cnn"):
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize(int(img_size * 1.15)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def load_checkpoint(path: Path):
    ck = joblib.load(path)
    classes = ck.get("classes", ["negative", "positive"])
    backbone = ck.get("backbone") or ("cnn" if "cnn" in path.stem else "resnet")
    default_img = 128 if backbone in ("cnn", "smallcnn") else 224
    img_size = int(ck.get("img_size", default_img))
    threshold = float(ck.get("threshold", 0.5))
    net = build_arch(backbone, num_classes=len(classes)).to(DEVICE).eval()
    net.load_state_dict(ck["model"])
    return net, eval_transform(backbone, img_size), threshold, backbone


@torch.inference_mode()
def predict_partition(net, tfm, threshold, paths):
    preds, probs = [], []
    for p in paths:
        x = tfm(Image.open(p).convert("RGB")).unsqueeze(0).to(DEVICE)
        pr = torch.softmax(net(x), 1)[0, 1].item()  # P(unsafe)
        probs.append(pr)
        preds.append(int(pr >= threshold))
    return np.array(preds), np.array(probs)


def scores(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    acc = (y_true == y_pred).mean()
    unsafe = y_true == 1
    recall = (y_pred[unsafe] == 1).mean() if unsafe.any() else float("nan")
    return acc, recall


def evaluate(path: Path):
    net, tfm, thr, backbone = load_checkpoint(path)
    part = du.get_partition()
    print(f"\n=== {path.name}  (backbone={backbone}, threshold={thr:.3f}) ===")
    print(f"{'split':6s} {'n':>4s} {'error':>8s} {'accuracy':>9s} {'unsafe-recall':>14s}")
    row = {}
    for name in ("train", "val", "test"):
        paths, labels = part[name]
        if not paths:
            continue
        y_pred, _ = predict_partition(net, tfm, thr, paths)
        acc, rec = scores(labels, y_pred)
        row[name] = (1 - acc, acc, rec)
        print(f"{name:6s} {len(paths):4d} {(1-acc)*100:7.1f}% {acc*100:8.1f}% {rec*100:13.1f}%")
    if "train" in row and "val" in row:
        train_err, val_err = row["train"][0], row["val"][0]
        print(f"  -> variance (val_err - train_err) = {(val_err - train_err)*100:.1f}%")
        print(f"  -> with human-level ~4%: avoidable bias = {(train_err*100 - 4):.1f}%")
    return row


def print_hlp_sample(n: int):
    """Print a random class-balanced sample for inter-annotator (human-level) labelling."""
    rng = np.random.default_rng(du.SEED)
    paths, labels, _ = du.scan_dataset()
    labels = np.array(labels)
    per = n // 2
    pick = []
    for cls in (0, 1):
        idx = np.where(labels == cls)[0]
        pick += list(rng.choice(idx, size=min(per, len(idx)), replace=False))
    print(f"\n=== Human-level labelling sample ({len(pick)} images) ===")
    print("Have 2-3 teammates label these blind; disagreement rate ~= human-level error.")
    for i in sorted(pick):
        print(f"  {paths[i]}")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if "--hlp-sample" in sys.argv:
        i = sys.argv.index("--hlp-sample")
        n = int(sys.argv[i + 1]) if i + 1 < len(sys.argv) else 50
        print_hlp_sample(n)

    targets = ([Path(a) if Path(a).is_absolute() else du.ROOT / a for a in args]
               if args else
               sorted(p for p in du.ROOT.glob("*.joblib")
                      if p.stem in ("resnet50", "convnext_tiny", "efficientnet_b0",
                                    "cnn_model", "resnet_model")))
    if not targets:
        print("No torch checkpoints found. Pass one explicitly, e.g. resnet50.joblib")
    for t in targets:
        if t.exists():
            evaluate(t)
        else:
            print(f"[skip] not found: {t}")
