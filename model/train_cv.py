"""Robust training via 5-fold stratified CV — maximise accuracy AND unsafe recall.

Protocol (keeps the 15% TEST set untouched for an honest final number):
  - train+val (327 imgs, ~83 unsafe) used for 5-fold stratified CV.
  - Per backbone: out-of-fold (OOF) probabilities collected over all 327 imgs,
    so accuracy/recall are estimated over ~83 unsafe (not 15) -> reliable.
  - Decision threshold tuned on the pooled OOF probs (balanced-accuracy, tie-break
    higher recall). TTA (hflip) at every prediction.
  - ENSEMBLE = mean of per-backbone OOF probs; its threshold tuned on OOF too.
  - Each backbone retrained on ALL 327 train+val imgs -> final deployable model,
    then evaluated once on the 58-img holdout TEST with the CV threshold.

Reuses helpers from train_torch2.py.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, WeightedRandomSampler

import du
import train_torch2 as T2

OUT = du.ROOT / "outputs_cv"
OUT.mkdir(parents=True, exist_ok=True)
DEV = T2.DEV

# strong, diverse backbones (top performers from the ranking run)
BACKBONES = {
    "resnet50": dict(bs=24, epochs=40, warmup=5, lr_head=1e-3, lr_finetune=6e-5,
                     wd=1e-4, dropout=0.4, ls=0.05, sampler=True, seed=42),
    "convnext_tiny": dict(bs=24, epochs=40, warmup=5, lr_head=1e-3, lr_finetune=5e-5,
                          wd=5e-5, dropout=0.4, ls=0.05, sampler=True, seed=42),
    "efficientnet_b0": dict(bs=32, epochs=42, warmup=5, lr_head=1e-3, lr_finetune=8e-5,
                            wd=1e-4, dropout=0.4, ls=0.05, sampler=True, seed=42),
}
N_FOLDS = 5
SEED = 42


def make_loader(paths, labels, img, bs, train, sampler):
    ttf, etf = T2.make_transforms(img)
    ds = T2.PathDS(paths, labels, ttf if train else etf)
    pin = DEV.type == "cuda"
    if train and sampler:
        counts = np.bincount(labels, minlength=2); w = 1.0 / counts
        sw = [w[y] for y in labels]
        smp = WeightedRandomSampler(sw, num_samples=len(sw), replacement=True)
        return DataLoader(ds, batch_size=bs, sampler=smp, num_workers=6, pin_memory=pin)
    return DataLoader(ds, batch_size=(bs if train else 64), shuffle=train,
                      num_workers=6, pin_memory=pin)


def train_model(name, cfg, tr_paths, tr_labels):
    """Train one model on the given data (fixed epochs, two-phase). Returns model."""
    T2.seed_all(cfg["seed"])
    model, img, _ = T2.build_backbone(name, cfg.get("dropout", 0.3))
    img = cfg.get("img_override") or img
    model = model.to(DEV)
    loader = make_loader(tr_paths, tr_labels, img, cfg["bs"], train=True, sampler=cfg.get("sampler", True))
    counts = np.bincount(tr_labels, minlength=2)
    cw = torch.tensor(len(tr_labels) / (2.0 * counts), dtype=torch.float32, device=DEV)
    crit = nn.CrossEntropyLoss(weight=cw, label_smoothing=cfg.get("ls", 0.05))

    T2.set_backbone_trainable(name, model, False)
    opt = torch.optim.AdamW(T2.head_params(name, model), lr=cfg["lr_head"], weight_decay=cfg["wd"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])
    amp = DEV.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    for ep in range(1, cfg["epochs"] + 1):
        if ep == cfg["warmup"] + 1:
            T2.set_backbone_trainable(name, model, True)
            head_ids = {id(p) for p in T2.head_params(name, model)}
            opt = torch.optim.AdamW([
                {"params": T2.head_params(name, model), "lr": cfg["lr_finetune"] * 5},
                {"params": [p for p in model.parameters() if id(p) not in head_ids],
                 "lr": cfg["lr_finetune"]},
            ], weight_decay=cfg["wd"])
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"] - cfg["warmup"])
        model.train()
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEV), labels.to(DEV)
            opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=amp):
                loss = crit(model(imgs), labels)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        sched.step()
    return model, img


TILING = False
from PIL import Image as _PILImage
from torchvision.transforms import functional as _TF


def _tiles(im):
    """Full image + square crops + vertical door-strips + a fine grid to zoom
    into small/distant hangers (some unsafe boxes are only 1-8% of the frame)."""
    W, H = im.size
    crops = [im]
    s = int(min(W, H) * 0.65)
    crops += [im.crop(b) for b in [
        (0, 0, s, s), (W - s, 0, W, s), (0, H - s, s, H), (W - s, H - s, W, H),
        ((W - s) // 2, (H - s) // 2, (W + s) // 2, (H + s) // 2)]]
    tw = int(W * 0.5)
    crops += [im.crop(b) for b in [
        (0, 0, tw, H), ((W - tw) // 2, 0, (W + tw) // 2, H), (W - tw, 0, W, H)]]
    # fine grid: overlapping tiles at 0.4 and 0.28 scale to catch tiny hangers
    for frac in (0.40, 0.28):
        cw, ch = int(W * frac), int(H * frac)
        if cw < 8 or ch < 8:
            continue
        xs = [0, (W - cw) // 2, W - cw]
        ys = [0, (H - ch) // 2, H - ch]
        for yy in ys:
            for xx in xs:
                crops.append(im.crop((xx, yy, xx + cw, yy + ch)))
    return crops


@torch.inference_mode()
def tiled_probs(model, paths, labels, img):
    """Max P(unsafe) over tiles + hflip — weakly-supervised 'hanging anywhere?'."""
    model.eval()
    norm = __import__("torchvision").transforms.Normalize(T2.MEAN, T2.STD)
    out = []
    for p in paths:
        im = _PILImage.open(p).convert("RGB")
        ts = [norm(_TF.to_tensor(c.resize((img, img), _PILImage.LANCZOS))) for c in _tiles(im)]
        x = torch.stack(ts).to(DEV)
        x = torch.cat([x, torch.flip(x, dims=[3])], 0)
        pr = torch.softmax(model(x), 1)[:, 1]
        out.append(float(pr.max().cpu()))
    return np.array(out), np.array(labels)


def predict_paths(model, paths, labels, img):
    if TILING:
        return tiled_probs(model, paths, labels, img)
    loader = make_loader(paths, labels, img, 64, train=False, sampler=False)
    p, y = T2.probs(model, loader, tta=True)
    return p, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", type=int, default=0, help="override input resolution")
    ap.add_argument("--out", default="outputs_cv", help="output subdir under model/")
    ap.add_argument("--backbones", default="", help="comma list subset")
    ap.add_argument("--tiling", action="store_true", help="region/tiling max-pool inference")
    args = ap.parse_args()
    global OUT, BACKBONES, TILING
    TILING = args.tiling
    OUT = du.ROOT / args.out
    OUT.mkdir(parents=True, exist_ok=True)
    if args.backbones:
        BACKBONES = {k: v for k, v in BACKBONES.items() if k in args.backbones.split(",")}
    if args.img:
        for c in BACKBONES.values():
            c["img_override"] = args.img

    print(f"device: {DEV} | out={OUT.name} | img={args.img or 'default'} | backbones={list(BACKBONES)}")
    part = du.get_partition()
    trp, trl = part["train"]; vap, val = part["val"]; tep, tel = part["test"]
    tv_paths = list(trp) + list(vap)
    tv_labels = list(trl) + list(val)
    tv_labels_arr = np.array(tv_labels)
    print(f"train+val: {len(tv_paths)} ({int((tv_labels_arr==1).sum())} unsafe) | test: {len(tep)} "
          f"({sum(tel)} unsafe)")

    per_backbone = {}
    for name, cfg in BACKBONES.items():
        print(f"\n########## {name}: {N_FOLDS}-fold CV ##########")
        t0 = time.time()
        oof = np.full(len(tv_paths), np.nan)
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        for fold, (tri, vai) in enumerate(skf.split(tv_paths, tv_labels_arr)):
            ftr_p = [tv_paths[i] for i in tri]; ftr_l = [tv_labels[i] for i in tri]
            fva_p = [tv_paths[i] for i in vai]; fva_l = [tv_labels[i] for i in vai]
            model, img = train_model(name, cfg, ftr_p, ftr_l)
            p, _ = predict_paths(model, fva_p, fva_l, img)
            oof[vai] = p
            print(f"  fold {fold+1}/{N_FOLDS} done")
        thr = T2.tune_threshold(oof, tv_labels_arr)
        cv_m = T2.metrics_at(oof, tv_labels_arr, thr)
        print(f"  CV: acc {cv_m['acc']:.4f} bal_acc {cv_m['balanced_acc']:.4f} "
              f"unsafe_rec {cv_m['unsafe_recall']:.4f} safe_rec {cv_m['safe_recall']:.4f} "
              f"auc {cv_m['auc']:.4f} thr {thr:.2f}  ({time.time()-t0:.0f}s)")
        # retrain on ALL train+val -> final deployable model
        final_model, img = train_model(name, cfg, tv_paths, tv_labels)
        ptest, ytest = predict_paths(final_model, tep, tel, img)
        test_m = T2.metrics_at(ptest, np.array(tel), thr)
        print(f"  HOLDOUT TEST @thr {thr:.2f}: acc {test_m['acc']:.4f} "
              f"unsafe_rec {test_m['unsafe_recall']:.4f} bal_acc {test_m['balanced_acc']:.4f}")
        state = {k: v.detach().cpu().clone() for k, v in final_model.state_dict().items()}
        joblib.dump({"model": state, "classes": du.CLASS_NAMES, "img_size": img,
                     "backbone": name, "threshold": thr, "tta": True},
                    OUT / f"{name}.joblib")
        per_backbone[name] = dict(oof=oof.tolist(), threshold=thr, cv=cv_m, test=test_m,
                                  test_probs=ptest.tolist(), img=img)

    # ---- ensemble over OOF + test ----
    names = list(per_backbone)
    oof_stack = np.mean([np.array(per_backbone[n]["oof"]) for n in names], axis=0)
    ens_thr = T2.tune_threshold(oof_stack, tv_labels_arr)
    ens_cv = T2.metrics_at(oof_stack, tv_labels_arr, ens_thr)
    test_stack = np.mean([np.array(per_backbone[n]["test_probs"]) for n in names], axis=0)
    ens_test = T2.metrics_at(test_stack, np.array(tel), ens_thr)
    print(f"\n########## ENSEMBLE {names} ##########")
    print(f"  CV: acc {ens_cv['acc']:.4f} bal_acc {ens_cv['balanced_acc']:.4f} "
          f"unsafe_rec {ens_cv['unsafe_recall']:.4f} safe_rec {ens_cv['safe_recall']:.4f} "
          f"auc {ens_cv['auc']:.4f} thr {ens_thr:.2f}")
    print(f"  HOLDOUT TEST: acc {ens_test['acc']:.4f} unsafe_rec {ens_test['unsafe_recall']:.4f} "
          f"bal_acc {ens_test['balanced_acc']:.4f}")
    print(ens_test["report"])

    summary = {n: {"backbone": n, "threshold": per_backbone[n]["threshold"],
                   "img_size": per_backbone[n]["img"],
                   "cv": per_backbone[n]["cv"], "test": per_backbone[n]["test"]}
               for n in names}
    summary["ensemble"] = {"members": names, "threshold": ens_thr,
                           "cv": ens_cv, "test": ens_test}
    (OUT / "results.json").write_text(json.dumps(summary, indent=2, default=str))

    # persist OOF + test probabilities so threshold/operating-point analysis can
    # run without retraining
    save = {"labels_tv": tv_labels_arr, "labels_test": np.array(tel),
            "oof_ensemble": oof_stack, "test_ensemble": test_stack}
    for n in names:
        save[f"oof_{n}"] = np.array(per_backbone[n]["oof"])
        save[f"test_{n}"] = np.array(per_backbone[n]["test_probs"])
    np.savez(OUT / "probs.npz", **save)
    print(f"\nsaved -> {OUT}/results.json + probs.npz + per-backbone checkpoints")


if __name__ == "__main__":
    main()
