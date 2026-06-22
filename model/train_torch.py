"""Train GPU models — SmallCNN (from scratch) and ResNet18 (transfer) — on the
annotation-labelled safe/unsafe dataset.

- 70/15/15 canonical split (du.py)
- Online augmentation in the dataloader (train only)
- Class-weighted CE (+ optional WeightedRandomSampler) for the 287/98 imbalance
- AdamW + cosine LR, AMP mixed precision on CUDA
- Multi-config search: keep the checkpoint with the best VALIDATION accuracy
  ("run until best"); TEST is evaluated once, only on the chosen model
- Saves predictor-compatible checkpoints {model, classes, img_size}

Run:
  python train_torch.py --model cnn
  python train_torch.py --model resnet
  python train_torch.py --model both   (default)
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
from PIL import Image, ImageFile
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, matthews_corrcoef,
                             recall_score, roc_auc_score)
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18

import du
from train_customised_cnn import SmallCNN  # reuse exact arch for predictor compat

ImageFile.LOAD_TRUNCATED_IMAGES = True
OUT = du.ROOT / "outputs_torch"
OUT.mkdir(parents=True, exist_ok=True)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PathDataset(Dataset):
    def __init__(self, paths, labels, tf):
        self.paths, self.labels, self.tf = paths, labels, tf

    def __len__(self): return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return self.tf(img), self.labels[i]


def cnn_transforms(img=128):
    train = transforms.Compose([
        transforms.RandomResizedCrop(img, scale=(0.7, 1.0), ratio=(0.85, 1.15)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.25, 0.25, 0.25, 0.05),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    ev = transforms.Compose([
        transforms.Resize((img, img)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train, ev


def resnet_transforms(weights, img=224):
    pre = weights.transforms()
    mean, std = list(pre.mean), list(pre.std)
    train = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(img, scale=(0.65, 1.0), ratio=(0.85, 1.15)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.25, 0.25, 0.25, 0.05),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    ev = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(img),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return train, ev


def build_resnet(dropout=0.3):
    weights = ResNet18_Weights.IMAGENET1K_V1
    model = resnet18(weights=weights)
    for p in model.parameters():
        p.requires_grad = False
    model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(model.fc.in_features, 2))
    return model, weights


def make_loaders(kind, cfg, weights=None):
    part = du.get_partition()
    (trp, trl) = part["train"]; (vap, val) = part["val"]; (tep, tel) = part["test"]
    if kind == "cnn":
        ttf, etf = cnn_transforms(cfg["img"])
    else:
        ttf, etf = resnet_transforms(weights, cfg["img"])
    tr_ds = PathDataset(trp, trl, ttf)
    va_ds = PathDataset(vap, val, etf)
    te_ds = PathDataset(tep, tel, etf)
    pin = device().type == "cuda"
    if cfg.get("sampler"):
        counts = np.bincount(trl, minlength=2)
        w = 1.0 / counts
        sw = [w[y] for y in trl]
        sampler = WeightedRandomSampler(sw, num_samples=len(sw), replacement=True)
        tr_loader = DataLoader(tr_ds, batch_size=cfg["bs"], sampler=sampler,
                               num_workers=4, pin_memory=pin, drop_last=False)
    else:
        tr_loader = DataLoader(tr_ds, batch_size=cfg["bs"], shuffle=True,
                               num_workers=4, pin_memory=pin)
    va_loader = DataLoader(va_ds, batch_size=64, shuffle=False, num_workers=4, pin_memory=pin)
    te_loader = DataLoader(te_ds, batch_size=64, shuffle=False, num_workers=4, pin_memory=pin)
    return tr_loader, va_loader, te_loader, trl


@torch.no_grad()
def evaluate(model, loader, dev):
    model.eval()
    yt, yp, ypr = [], [], []
    for imgs, labels in loader:
        imgs = imgs.to(dev)
        logits = model(imgs)
        prob = torch.softmax(logits, 1)[:, 1]
        pred = logits.argmax(1)
        yt.extend(labels.tolist()); yp.extend(pred.cpu().tolist()); ypr.extend(prob.cpu().tolist())
    return yt, yp, ypr


def metrics(yt, yp, ypr):
    auc = float(roc_auc_score(yt, ypr)) if len(set(yt)) > 1 else float("nan")
    return {
        "acc": float(accuracy_score(yt, yp)),
        "unsafe_recall": float(recall_score(yt, yp, pos_label=1, zero_division=0)),
        "mcc": float(matthews_corrcoef(yt, yp)),
        "auc": auc,
        "cm": confusion_matrix(yt, yp, labels=[0, 1]).tolist(),
        "report": classification_report(yt, yp, target_names=["safe", "unsafe"],
                                        digits=4, zero_division=0),
    }


def train_one(kind, cfg, dev, verbose=False):
    seed_all(cfg["seed"])
    if kind == "cnn":
        model = SmallCNN(num_classes=2).to(dev)
        weights = None
    else:
        model, weights = build_resnet(cfg["dropout"])
        model = model.to(dev)
    tr_loader, va_loader, te_loader, trl = make_loaders(kind, cfg, weights)

    counts = np.bincount(trl, minlength=2)
    cw = torch.tensor(len(trl) / (2.0 * counts), dtype=torch.float32, device=dev)
    criterion = nn.CrossEntropyLoss(weight=cw, label_smoothing=cfg.get("ls", 0.05))

    if kind == "resnet":
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                lr=cfg["lr_head"], weight_decay=cfg["wd"])
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])
    use_amp = dev.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_val, best_state, best_ep = -1.0, None, -1
    for ep in range(1, cfg["epochs"] + 1):
        if kind == "resnet" and ep == cfg["warmup"] + 1:
            for p in model.layer4.parameters(): p.requires_grad = True
            for p in model.fc.parameters(): p.requires_grad = True
            opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                    lr=cfg["lr_finetune"], weight_decay=cfg["wd"])
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"] - cfg["warmup"])
        model.train()
        for imgs, labels in tr_loader:
            imgs, labels = imgs.to(dev), labels.to(dev)
            opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = criterion(model(imgs), labels)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
        sched.step()
        yt, yp, ypr = evaluate(model, va_loader, dev)
        va = accuracy_score(yt, yp)
        if verbose:
            print(f"  ep{ep:02d} val_acc {va:.4f}")
        if va > best_val:
            best_val = va; best_ep = ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    # load best, eval test
    model.load_state_dict(best_state)
    yt, yp, ypr = evaluate(model, te_loader, dev)
    test_m = metrics(yt, yp, ypr)
    return best_val, best_ep, best_state, test_m


CNN_CONFIGS = [
    dict(img=128, bs=32, epochs=40, lr=1e-3, wd=1e-4, ls=0.05, sampler=False, seed=42),
    dict(img=128, bs=32, epochs=50, lr=8e-4, wd=2e-4, ls=0.05, sampler=True, seed=42),
    dict(img=128, bs=16, epochs=60, lr=6e-4, wd=1e-4, ls=0.1, sampler=True, seed=7),
    dict(img=128, bs=32, epochs=60, lr=1e-3, wd=5e-4, ls=0.05, sampler=True, seed=123),
]
RESNET_CONFIGS = [
    dict(img=224, bs=32, epochs=30, warmup=5, lr_head=1e-3, lr_finetune=1e-4,
         wd=1e-4, dropout=0.3, ls=0.05, sampler=False, seed=42),
    dict(img=224, bs=32, epochs=40, warmup=5, lr_head=1e-3, lr_finetune=1.5e-4,
         wd=1e-4, dropout=0.4, ls=0.05, sampler=True, seed=42),
    dict(img=224, bs=16, epochs=45, warmup=6, lr_head=1e-3, lr_finetune=8e-5,
         wd=2e-4, dropout=0.4, ls=0.1, sampler=True, seed=7),
    dict(img=224, bs=32, epochs=50, warmup=5, lr_head=1.2e-3, lr_finetune=1e-4,
         wd=1e-4, dropout=0.3, ls=0.05, sampler=True, seed=123),
]


def search(kind, configs, dev):
    print(f"\n########## {kind.upper()} — searching {len(configs)} configs ##########")
    best = None
    trials = []
    for i, cfg in enumerate(configs):
        t0 = time.time()
        val, ep, state, test_m = train_one(kind, cfg, dev)
        dt = time.time() - t0
        print(f"[{kind} cfg {i+1}/{len(configs)}] val_acc {val:.4f} (ep{ep})  "
              f"-> test_acc {test_m['acc']:.4f} unsafe_recall {test_m['unsafe_recall']:.4f}  ({dt:.0f}s)")
        trials.append({"cfg": cfg, "val_acc": val, "best_epoch": ep,
                       "test": {k: v for k, v in test_m.items()}})
        if best is None or val > best["val_acc"]:
            best = {"cfg": cfg, "val_acc": val, "best_epoch": ep,
                    "state": state, "test": test_m, "img": cfg["img"]}
    return best, trials


def save_model(kind, best):
    fname = "cnn_model.joblib" if kind == "cnn" else "resnet_model.joblib"
    joblib.dump({"model": best["state"], "classes": du.CLASS_NAMES,
                 "img_size": best["img"]}, OUT / fname)
    return OUT / fname


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["cnn", "resnet", "both"], default="both")
    args = ap.parse_args()
    dev = device()
    print(f"device: {dev}  ({torch.cuda.get_device_name(0) if dev.type=='cuda' else 'cpu'})")

    summary = {}
    todo = ["cnn", "resnet"] if args.model == "both" else [args.model]
    for kind in todo:
        configs = CNN_CONFIGS if kind == "cnn" else RESNET_CONFIGS
        best, trials = search(kind, configs, dev)
        path = save_model(kind, best)
        print(f"\n>>> BEST {kind}: val_acc {best['val_acc']:.4f}  "
              f"test_acc {best['test']['acc']:.4f}  unsafe_recall {best['test']['unsafe_recall']:.4f}")
        print(best["test"]["report"])
        print(f"saved -> {path}")
        summary[kind] = {
            "kind": kind,
            "best_config": best["cfg"],
            "val_acc": best["val_acc"],
            "best_epoch": best["best_epoch"],
            "img_size": best["img"],
            "test": {k: v for k, v in best["test"].items()},
            "all_trials": trials,
        }
    (OUT / "results.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nsaved torch results -> {OUT}/results.json")


if __name__ == "__main__":
    main()
