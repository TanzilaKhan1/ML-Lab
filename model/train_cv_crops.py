"""Crop-augmented 5-fold CV — the genuine attempt at 100% recall @ usable accuracy.

Training set per fold = full images (fold-train) + box-supervised crops of those
same images (hanger close-ups for unsafe, door-region for safe). Crops follow
their source image's fold (group-aware) and are used for TRAINING ONLY.
Evaluation = full images (fold-val / test) via matched multi-scale tiling.

This teaches the model to recognise small/distant hangers (the cases that block
perfect recall) and matches the scale tiling presents at inference.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import torch
import torchvision
from PIL import Image, ImageFile
from sklearn.model_selection import StratifiedKFold
from torchvision.transforms import functional as TF

import du
import train_torch2 as T2
import train_cv as TC

ImageFile.LOAD_TRUNCATED_IMAGES = True
OUT = du.ROOT / "outputs_cv_crops"
OUT.mkdir(parents=True, exist_ok=True)
DEV = TC.DEV
SEED = 42
IMG = 320

CROPS = json.loads((du.ROOT / "crops_index.json").read_text())
crops_by_stem: dict[str, list] = {}
for path, info in CROPS.items():
    lbl = 0 if info["label"] == "safe" else 1
    crops_by_stem.setdefault(info["source_stem"], []).append((Path(path), lbl))

BACKBONES = {
    "resnet50": dict(bs=24, epochs=40, warmup=5, lr_head=1e-3, lr_finetune=6e-5,
                     wd=1e-4, dropout=0.4, ls=0.05, sampler=True, seed=42, img_override=IMG),
    "convnext_tiny": dict(bs=24, epochs=40, warmup=5, lr_head=1e-3, lr_finetune=5e-5,
                          wd=5e-5, dropout=0.4, ls=0.05, sampler=True, seed=42, img_override=IMG),
}
NORM = torchvision.transforms.Normalize(T2.MEAN, T2.STD)


def tiles(im):
    W, H = im.size
    cr = [im]
    s = int(min(W, H) * 0.65)
    cr += [im.crop(b) for b in [
        (0, 0, s, s), (W - s, 0, W, s), (0, H - s, s, H), (W - s, H - s, W, H),
        ((W - s) // 2, (H - s) // 2, (W + s) // 2, (H + s) // 2)]]
    tw = int(W * 0.5)
    cr += [im.crop(b) for b in [
        (0, 0, tw, H), ((W - tw) // 2, 0, (W + tw) // 2, H), (W - tw, 0, W, H)]]
    # medium 3x3 grid at 0.4 for distant hangers (now in-distribution thanks to crops)
    cw, ch = int(W * 0.4), int(H * 0.4)
    xs = [0, (W - cw) // 2, W - cw]; ys = [0, (H - ch) // 2, H - ch]
    cr += [im.crop((x, y, x + cw, y + ch)) for y in ys for x in xs]
    return cr


@torch.inference_mode()
def tiled_probs(model, paths, img):
    """Return max-tile P(unsafe) (for ensemble/threshold) — back-compat scalar/image."""
    return tiled_tileprobs(model, paths, img).max(axis=1)


@torch.inference_mode()
def tiled_tileprobs(model, paths, img):
    """Return per-image per-tile P(unsafe), shape [N, n_tiles] (flip-maxed per tile)."""
    model.eval(); out = []
    for p in paths:
        im = Image.open(p).convert("RGB")
        cr = tiles(im)
        ts = [NORM(TF.to_tensor(c.resize((img, img), Image.LANCZOS))) for c in cr]
        x = torch.stack(ts).to(DEV)
        xf = torch.flip(x, dims=[3])
        pr = torch.softmax(model(x), 1)[:, 1]
        prf = torch.softmax(model(xf), 1)[:, 1]
        per_tile = torch.maximum(pr, prf).cpu().numpy()   # [n_tiles]
        out.append(per_tile)
    return np.array(out)   # [N, n_tiles]


def add_crops(stems):
    extra_p, extra_l = [], []
    for st in stems:
        for cp, cl in crops_by_stem.get(st, []):
            if cp.exists():
                extra_p.append(cp); extra_l.append(cl)
    return extra_p, extra_l


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbones", default="resnet50,convnext_tiny")
    args = ap.parse_args()
    bks = {k: v for k, v in BACKBONES.items() if k in args.backbones.split(",")}
    print(f"device {DEV} | img {IMG} | backbones {list(bks)} | crops {len(CROPS)}")

    part = du.get_partition()
    tvp = list(part["train"][0]) + list(part["val"][0])
    tvl = list(part["train"][1]) + list(part["val"][1])
    tep, tel = part["test"]
    tv_stems = [Path(p).stem for p in tvp]
    y = np.array(tvl)

    per = {}
    n_tiles = None
    for name, cfg in bks.items():
        print(f"\n########## {name} (crop-augmented 5-fold CV) ##########")
        t0 = time.time()
        oof_tiles = None
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        for fold, (tri, vai) in enumerate(skf.split(tvp, y)):
            ftr_p = [tvp[i] for i in tri]; ftr_l = [tvl[i] for i in tri]
            ep, el = add_crops([tv_stems[i] for i in tri])
            ftr_p += ep; ftr_l += el
            model, img = TC.train_model(name, cfg, ftr_p, ftr_l)
            tp = tiled_tileprobs(model, [tvp[i] for i in vai], img)
            if oof_tiles is None:
                n_tiles = tp.shape[1]; oof_tiles = np.full((len(tvp), n_tiles), np.nan)
            oof_tiles[vai] = tp
            print(f"  fold {fold+1}/5 done (+{len(ep)} crops)")
        oof = oof_tiles.max(axis=1)
        thr = T2.tune_threshold(oof, y)
        m = T2.metrics_at(oof, y, thr)
        from sklearn.metrics import roc_auc_score
        print(f"  CV: acc {m['acc']:.4f} bal_acc {m['balanced_acc']:.4f} unsafe_rec "
              f"{m['unsafe_recall']:.4f} safe_rec {m['safe_recall']:.4f} AUC {m['auc']:.4f}  "
              f"({time.time()-t0:.0f}s)")
        # retrain on all train+val + their crops
        allp = list(tvp); alll = list(tvl)
        ep, el = add_crops(tv_stems); allp += ep; alll += el
        fmodel, img = TC.train_model(name, cfg, allp, alll)
        test_tiles = tiled_tileprobs(fmodel, tep, img)
        state = {k: v.detach().cpu().clone() for k, v in fmodel.state_dict().items()}
        joblib.dump({"model": state, "classes": du.CLASS_NAMES, "img_size": img,
                     "backbone": name, "threshold": thr, "tta": True}, OUT / f"{name}.joblib")
        per[name] = dict(oof_tiles=oof_tiles, test_tiles=test_tiles,
                         oof=oof, test_probs=test_tiles.max(axis=1))

    names = list(per)
    oof_e = np.mean([per[n]["oof"] for n in names], axis=0)
    test_e = np.mean([per[n]["test_probs"] for n in names], axis=0)
    save = {"labels_tv": y, "labels_test": np.array(tel),
            "oof_ensemble": oof_e, "test_ensemble": test_e}
    for n in names:
        save[f"oof_{n}"] = per[n]["oof"]; save[f"test_{n}"] = per[n]["test_probs"]
        save[f"oof_tiles_{n}"] = per[n]["oof_tiles"]; save[f"test_tiles_{n}"] = per[n]["test_tiles"]
    np.savez(OUT / "probs.npz", **save)
    # ensemble results.json (members list for deploy_best)
    (OUT / "results.json").write_text(json.dumps({"ensemble": {"members": names}}, default=str))
    from sklearn.metrics import roc_auc_score
    print(f"\nENSEMBLE CV AUC {roc_auc_score(y, oof_e):.4f}")
    et = T2.tune_threshold(oof_e, y)
    em = T2.metrics_at(oof_e, y, et)
    print(f"ENSEMBLE CV: acc {em['acc']:.4f} bal_acc {em['balanced_acc']:.4f} "
          f"unsafe_rec {em['unsafe_recall']:.4f} AUC {em['auc']:.4f}")
    print(f"saved -> {OUT}/probs.npz + checkpoints")


if __name__ == "__main__":
    main()
