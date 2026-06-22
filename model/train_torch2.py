"""Enhanced GPU training to maximise BOTH accuracy and unsafe-recall for the
bus/leguna hanging-passenger classifier.

Improvements over train_torch.py:
- Multiple pretrained backbones (resnet18/50, efficientnet_b0, convnext_tiny)
  + the from-scratch SmallCNN, each two-phase fine-tuned with discriminative LRs.
- "Proper" task-aware augmentation: conservative RandomResizedCrop (keeps the
  door-edge passenger in frame), HFlip, mild rotation, RandAugment, ColorJitter,
  RandomErasing.
- Class imbalance handled by WeightedRandomSampler + class-weighted CE (focal
  optional) so the minority UNSAFE class is learned, not ignored.
- Model selection by VALIDATION balanced-accuracy (mean of per-class recall),
  which rewards getting BOTH classes right.
- Decision-THRESHOLD tuning on validation (the direct lever for recall) +
  test-time augmentation (TTA, hflip) for a free accuracy bump.
- Ensemble of the top models (probability averaging) with its own tuned threshold.
- Saves predictor-compatible checkpoints incl. {backbone, threshold}.

Run: python train_torch2.py            (all backbones + ensemble)
     python train_torch2.py --quick    (fewer configs)
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageFile
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             classification_report, confusion_matrix,
                             matthews_corrcoef, recall_score, roc_auc_score)
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from torchvision import models as tvm

import du
from train_customised_cnn import SmallCNN

ImageFile.LOAD_TRUNCATED_IMAGES = True
OUT = du.ROOT / "outputs_torch2"
OUT.mkdir(parents=True, exist_ok=True)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


# ----------------------------- backbones -----------------------------
def build_backbone(name, dropout=0.3):
    """Return (model, img_size, default_weights_meta) with a 2-class head."""
    if name == "smallcnn":
        return SmallCNN(num_classes=2), 128, None
    if name == "resnet18":
        w = tvm.ResNet18_Weights.IMAGENET1K_V1
        m = tvm.resnet18(weights=w)
        m.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(m.fc.in_features, 2))
        return m, 224, w
    if name == "resnet50":
        w = tvm.ResNet50_Weights.IMAGENET1K_V2
        m = tvm.resnet50(weights=w)
        m.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(m.fc.in_features, 2))
        return m, 224, w
    if name == "efficientnet_b0":
        w = tvm.EfficientNet_B0_Weights.IMAGENET1K_V1
        m = tvm.efficientnet_b0(weights=w)
        m.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(m.classifier[1].in_features, 2))
        return m, 224, w
    if name == "convnext_tiny":
        w = tvm.ConvNeXt_Tiny_Weights.IMAGENET1K_V1
        m = tvm.convnext_tiny(weights=w)
        in_f = m.classifier[2].in_features
        m.classifier[2] = nn.Linear(in_f, 2)
        return m, 224, w
    raise ValueError(name)


def head_params(name, model):
    if name == "smallcnn":
        return list(model.parameters())
    if name in ("resnet18", "resnet50"):
        return list(model.fc.parameters())
    if name in ("efficientnet_b0", "convnext_tiny"):
        return list(model.classifier.parameters())


def set_backbone_trainable(name, model, flag):
    if name == "smallcnn":
        return
    head_ids = {id(p) for p in head_params(name, model)}
    for p in model.parameters():
        if id(p) not in head_ids:
            p.requires_grad = flag


MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def make_transforms(img, strong=True):
    aug = [
        transforms.RandomResizedCrop(img, scale=(0.8, 1.0), ratio=(0.8, 1.25)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
    ]
    if strong:
        aug.append(transforms.RandAugment(num_ops=2, magnitude=7))
    aug += [
        transforms.ColorJitter(0.25, 0.25, 0.25, 0.05),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ]
    if strong:
        aug.append(transforms.RandomErasing(p=0.25, scale=(0.02, 0.12)))
    train = transforms.Compose(aug)
    ev = transforms.Compose([
        transforms.Resize(int(img * 1.15)),
        transforms.CenterCrop(img),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    return train, ev


class PathDS(Dataset):
    def __init__(self, paths, labels, tf):
        self.paths, self.labels, self.tf = paths, labels, tf

    def __len__(self): return len(self.paths)

    def __getitem__(self, i):
        return self.tf(Image.open(self.paths[i]).convert("RGB")), self.labels[i]


def loaders(img, bs, use_sampler):
    part = du.get_partition()
    (trp, trl) = part["train"]; (vap, val) = part["val"]; (tep, tel) = part["test"]
    ttf, etf = make_transforms(img)
    tr = PathDS(trp, trl, ttf); va = PathDS(vap, val, etf); te = PathDS(tep, tel, etf)
    pin = DEV.type == "cuda"
    if use_sampler:
        counts = np.bincount(trl, minlength=2); w = 1.0 / counts
        sw = [w[y] for y in trl]
        sampler = WeightedRandomSampler(sw, num_samples=len(sw), replacement=True)
        trl_loader = DataLoader(tr, batch_size=bs, sampler=sampler, num_workers=6, pin_memory=pin)
    else:
        trl_loader = DataLoader(tr, batch_size=bs, shuffle=True, num_workers=6, pin_memory=pin)
    return (trl_loader,
            DataLoader(va, batch_size=64, shuffle=False, num_workers=6, pin_memory=pin),
            DataLoader(te, batch_size=64, shuffle=False, num_workers=6, pin_memory=pin),
            trl)


class FocalLoss(nn.Module):
    def __init__(self, weight, gamma=1.5, ls=0.0):
        super().__init__(); self.weight = weight; self.gamma = gamma; self.ls = ls

    def forward(self, logits, target):
        ce = F.cross_entropy(logits, target, weight=self.weight,
                             reduction="none", label_smoothing=self.ls)
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


@torch.no_grad()
def probs(model, loader, tta=True):
    model.eval()
    ps, ys = [], []
    for imgs, labels in loader:
        imgs = imgs.to(DEV)
        logit = model(imgs)
        p = torch.softmax(logit, 1)[:, 1]
        if tta:
            p2 = torch.softmax(model(torch.flip(imgs, dims=[3])), 1)[:, 1]
            p = (p + p2) / 2
        ps.extend(p.cpu().tolist()); ys.extend(labels.tolist())
    return np.array(ps), np.array(ys)


def tune_threshold(p, y):
    """Pick threshold maximising balanced accuracy; tie-break by higher unsafe recall."""
    best_t, best_key = 0.5, (-1, -1)
    for t in np.linspace(0.05, 0.95, 181):
        pred = (p >= t).astype(int)
        ba = balanced_accuracy_score(y, pred)
        rec = recall_score(y, pred, pos_label=1, zero_division=0)
        key = (round(ba, 4), round(rec, 4))
        if key > best_key:
            best_key, best_t = key, float(t)
    return best_t


def metrics_at(p, y, t):
    pred = (p >= t).astype(int)
    auc = float(roc_auc_score(y, p)) if len(set(y.tolist())) > 1 else float("nan")
    return {
        "threshold": float(t),
        "acc": float(accuracy_score(y, pred)),
        "balanced_acc": float(balanced_accuracy_score(y, pred)),
        "unsafe_recall": float(recall_score(y, pred, pos_label=1, zero_division=0)),
        "safe_recall": float(recall_score(y, pred, pos_label=0, zero_division=0)),
        "mcc": float(matthews_corrcoef(y, pred)) if len(set(pred.tolist())) > 1 else 0.0,
        "auc": auc,
        "cm": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
        "report": classification_report(y, pred, target_names=["safe", "unsafe"],
                                        digits=4, zero_division=0),
    }


def train_run(name, cfg):
    seed_all(cfg["seed"])
    model, img, _ = build_backbone(name, cfg.get("dropout", 0.3))
    model = model.to(DEV)
    tr, va, te, trl = loaders(img, cfg["bs"], cfg.get("sampler", True))
    counts = np.bincount(trl, minlength=2)
    cw = torch.tensor(len(trl) / (2.0 * counts), dtype=torch.float32, device=DEV)
    if cfg.get("focal"):
        crit = FocalLoss(cw, gamma=cfg.get("gamma", 1.5), ls=cfg.get("ls", 0.0))
    else:
        crit = nn.CrossEntropyLoss(weight=cw, label_smoothing=cfg.get("ls", 0.05))

    transfer = name != "smallcnn"
    if transfer:
        set_backbone_trainable(name, model, False)
        opt = torch.optim.AdamW(head_params(name, model), lr=cfg["lr_head"], weight_decay=cfg["wd"])
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])
    amp = DEV.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    best_ba, best_state = -1.0, None
    for ep in range(1, cfg["epochs"] + 1):
        if transfer and ep == cfg["warmup"] + 1:
            set_backbone_trainable(name, model, True)
            opt = torch.optim.AdamW([
                {"params": head_params(name, model), "lr": cfg["lr_finetune"] * 5},
                {"params": [p for p in model.parameters()
                            if id(p) not in {id(q) for q in head_params(name, model)}],
                 "lr": cfg["lr_finetune"]},
            ], weight_decay=cfg["wd"])
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"] - cfg["warmup"])
        model.train()
        for imgs, labels in tr:
            imgs, labels = imgs.to(DEV), labels.to(DEV)
            opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=amp):
                loss = crit(model(imgs), labels)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        sched.step()
        pv, yv = probs(model, va, tta=False)
        ba = balanced_accuracy_score(yv, (pv >= 0.5).astype(int))
        if ba > best_ba:
            best_ba = ba
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    # threshold tuning on val (with TTA), evaluate test (with TTA)
    pv, yv = probs(model, va, tta=True)
    t = tune_threshold(pv, yv)
    pte, yte = probs(model, te, tta=True)
    val_m = metrics_at(pv, yv, t)
    test_m = metrics_at(pte, yte, t)
    return dict(name=name, img=img, state=best_state, threshold=t,
                val=val_m, test=test_m, val_probs=pv.tolist(), val_y=yv.tolist(),
                test_probs=pte.tolist(), test_y=yte.tolist())


CONFIGS = {
    "smallcnn": [dict(seed=42, bs=32, epochs=60, lr=8e-4, wd=2e-4, ls=0.05, sampler=True),
                 dict(seed=7, bs=32, epochs=70, lr=1e-3, wd=1e-4, ls=0.05, sampler=True, focal=True, gamma=1.5)],
    "resnet18": [dict(seed=42, bs=32, epochs=45, warmup=5, lr_head=1e-3, lr_finetune=1e-4, wd=1e-4, dropout=0.4, ls=0.05, sampler=True),
                 dict(seed=7, bs=32, epochs=50, warmup=5, lr_head=1e-3, lr_finetune=1e-4, wd=2e-4, dropout=0.4, ls=0.0, sampler=True, focal=True, gamma=1.5)],
    "resnet50": [dict(seed=42, bs=24, epochs=45, warmup=5, lr_head=1e-3, lr_finetune=6e-5, wd=1e-4, dropout=0.4, ls=0.05, sampler=True),
                 dict(seed=123, bs=24, epochs=50, warmup=6, lr_head=1e-3, lr_finetune=5e-5, wd=2e-4, dropout=0.5, ls=0.05, sampler=True, focal=True, gamma=2.0)],
    "efficientnet_b0": [dict(seed=42, bs=32, epochs=45, warmup=5, lr_head=1e-3, lr_finetune=8e-5, wd=1e-4, dropout=0.4, ls=0.05, sampler=True),
                        dict(seed=7, bs=32, epochs=55, warmup=5, lr_head=1e-3, lr_finetune=6e-5, wd=2e-4, dropout=0.4, ls=0.05, sampler=True, focal=True, gamma=1.5)],
    "convnext_tiny": [dict(seed=42, bs=24, epochs=45, warmup=5, lr_head=1e-3, lr_finetune=5e-5, wd=5e-5, dropout=0.4, ls=0.05, sampler=True)],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    print(f"device: {DEV} ({torch.cuda.get_device_name(0) if DEV.type=='cuda' else 'cpu'})")

    all_best = {}
    for name, cfgs in CONFIGS.items():
        if args.quick:
            cfgs = cfgs[:1]
        print(f"\n########## {name} ({len(cfgs)} configs) ##########")
        best = None
        for i, cfg in enumerate(cfgs):
            t0 = time.time()
            r = train_run(name, cfg)
            dt = time.time() - t0
            print(f"  [{name} {i+1}/{len(cfgs)}] val_bal_acc {r['val']['balanced_acc']:.4f} "
                  f"thr {r['threshold']:.2f} | TEST acc {r['test']['acc']:.4f} "
                  f"unsafe_rec {r['test']['unsafe_recall']:.4f} bal_acc {r['test']['balanced_acc']:.4f} ({dt:.0f}s)")
            if best is None or r["val"]["balanced_acc"] > best["val"]["balanced_acc"]:
                best = r
        all_best[name] = best
        print(f"  >>> best {name}: val_bal_acc {best['val']['balanced_acc']:.4f} "
              f"TEST acc {best['test']['acc']:.4f} unsafe_rec {best['test']['unsafe_recall']:.4f}")

    # ---- ensemble: average val/test probs of top-3 by val balanced acc ----
    ranked = sorted(all_best.values(), key=lambda r: -r["val"]["balanced_acc"])
    top = ranked[:3]
    print(f"\n########## ENSEMBLE of {[r['name'] for r in top]} ##########")
    pv = np.mean([np.array(r["val_probs"]) for r in top], axis=0)
    yv = np.array(top[0]["val_y"])
    pte = np.mean([np.array(r["test_probs"]) for r in top], axis=0)
    yte = np.array(top[0]["test_y"])
    et = tune_threshold(pv, yv)
    ens_val = metrics_at(pv, yv, et); ens_test = metrics_at(pte, yte, et)
    print(f"  ENSEMBLE val_bal_acc {ens_val['balanced_acc']:.4f} thr {et:.2f} | "
          f"TEST acc {ens_test['acc']:.4f} unsafe_rec {ens_test['unsafe_recall']:.4f} "
          f"bal_acc {ens_test['balanced_acc']:.4f}")
    print(ens_test["report"])

    # ---- save checkpoints (predictor-compatible) + results ----
    summary = {}
    for name, r in all_best.items():
        ckpt = {"model": r["state"], "classes": du.CLASS_NAMES, "img_size": r["img"],
                "backbone": name, "threshold": r["threshold"], "tta": True}
        joblib.dump(ckpt, OUT / f"{name}.joblib")
        summary[name] = {"backbone": name, "img_size": r["img"], "threshold": r["threshold"],
                         "val": r["val"], "test": r["test"]}
    summary["ensemble"] = {"members": [r["name"] for r in top], "threshold": et,
                           "val": ens_val, "test": ens_test}
    (OUT / "results.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nsaved -> {OUT}/results.json and per-backbone .joblib checkpoints")


if __name__ == "__main__":
    main()
