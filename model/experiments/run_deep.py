"""Deep imbalance + variance experiments — one fixed backbone (ResNet18), one
recipe, toggling a single technique at a time so effects are comparable.

Base train = original imbalanced images (204 safe / 98 unsafe); val/test = real
held-out originals (65 each). Measures train/val/test error, unsafe-recall, PR-AUC
and the train->test gap (variance). Nothing is written outside experiments/.

Techniques: loss {ce, weighted_ce, focal, class_balanced}; sampler {none,
weighted, random_over, random_under}; mixup; cutmix; freeze {none, linear_probe,
linear_then_finetune}; plus a strong-regularization combo.

Run from experiments/:  TORCH_HOME=<proj>/.torch_cache python run_deep.py
"""
from __future__ import annotations
import json, random, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models as tvm, transforms
from PIL import Image, ImageFile

import common as C

ImageFile.LOAD_TRUNCATED_IMAGES = True
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
IMG, EPOCHS = 224, 30
BACKBONE = sys.argv[1] if len(sys.argv) > 1 else "resnet18"
BS = 24 if BACKBONE in ("resnet50", "convnext_tiny") else 32


def seed_all(s=C.SEED):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


class DS(Dataset):
    def __init__(self, paths, labels, tf):
        self.paths, self.labels, self.tf = paths, list(labels), tf
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        return self.tf(Image.open(self.paths[i]).convert("RGB")), self.labels[i]


def tfs():
    tr = transforms.Compose([
        transforms.Resize(int(IMG * 1.15)),
        transforms.RandomResizedCrop(IMG, scale=(0.8, 1.0), ratio=(0.8, 1.25)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.25, 0.25, 0.25, 0.05),
        transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    ev = transforms.Compose([
        transforms.Resize(int(IMG * 1.15)), transforms.CenterCrop(IMG),
        transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    return tr, ev


class Focal(nn.Module):
    def __init__(self, weight=None, gamma=2.0):
        super().__init__(); self.w = weight; self.g = gamma
    def forward(self, logits, y):
        ce = F.cross_entropy(logits, y, weight=self.w, reduction="none")
        return ((1 - torch.exp(-ce)) ** self.g * ce).mean()


def class_weights(labels, scheme):
    c = np.bincount(labels, minlength=2).astype(float)
    if scheme == "weighted_ce":
        w = len(labels) / (2.0 * c)
    elif scheme == "class_balanced":          # Cui 2019 effective number, beta=0.999
        beta = 0.999; eff = (1 - np.power(beta, c)) / (1 - beta); w = (1 / eff)
        w = w / w.sum() * 2
    else:
        return None
    return torch.tensor(w, dtype=torch.float32, device=DEV)


def make_loader(paths, labels, tr, sampler):
    ds = DS(paths, labels, tr)
    if sampler == "weighted":
        c = np.bincount(labels, minlength=2); wpc = 1.0 / c
        sw = [wpc[y] for y in labels]
        smp = WeightedRandomSampler(sw, num_samples=len(sw), replacement=True)
        return DataLoader(ds, batch_size=BS, sampler=smp, num_workers=6, pin_memory=True)
    if sampler in ("random_over", "random_under"):
        idx = resample_indices(labels, sampler)
        ds2 = DS([paths[i] for i in idx], [labels[i] for i in idx], tr)
        return DataLoader(ds2, batch_size=BS, shuffle=True, num_workers=6, pin_memory=True)
    return DataLoader(ds, batch_size=BS, shuffle=True, num_workers=6, pin_memory=True)


def resample_indices(labels, mode):
    labels = np.array(labels); idx0 = np.where(labels == 0)[0]; idx1 = np.where(labels == 1)[0]
    rng = np.random.RandomState(C.SEED)
    if mode == "random_over":
        n = max(len(idx0), len(idx1))
        a = rng.choice(idx0, n, replace=True); b = rng.choice(idx1, n, replace=True)
    else:  # random_under
        n = min(len(idx0), len(idx1))
        a = rng.choice(idx0, n, replace=False); b = rng.choice(idx1, n, replace=False)
    out = np.concatenate([a, b]); rng.shuffle(out); return out.tolist()


def build_model(backbone, dropout):
    if backbone == "resnet18":
        m = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1)
        m.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(m.fc.in_features, 2))
    elif backbone == "resnet50":
        m = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2)
        m.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(m.fc.in_features, 2))
    elif backbone == "convnext_tiny":
        m = tvm.convnext_tiny(weights=tvm.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        m.classifier[2] = nn.Sequential(nn.Dropout(dropout),
                                        nn.Linear(m.classifier[2].in_features, 2))
    else:
        raise ValueError(backbone)
    return m.to(DEV)


def _head_params(m, backbone):
    return (m.classifier.parameters() if backbone == "convnext_tiny" else m.fc.parameters())


def set_backbone_trainable(m, backbone, flag):
    head = {id(p) for p in _head_params(m, backbone)}
    for p in m.parameters():
        if id(p) not in head:
            p.requires_grad = flag


def mix_batch(x, y, mode, alpha=0.2):
    lam = np.random.beta(alpha, alpha)
    perm = torch.randperm(x.size(0), device=x.device)
    if mode == "mixup":
        x = lam * x + (1 - lam) * x[perm]
    else:  # cutmix
        H, W = x.size(2), x.size(3)
        rh, rw = int(H * np.sqrt(1 - lam)), int(W * np.sqrt(1 - lam))
        cy, cx = np.random.randint(H), np.random.randint(W)
        y1, y2 = max(cy - rh // 2, 0), min(cy + rh // 2, H)
        x1, x2 = max(cx - rw // 2, 0), min(cx + rw // 2, W)
        x[:, :, y1:y2, x1:x2] = x[perm, :, y1:y2, x1:x2]
        lam = 1 - ((x2 - x1) * (y2 - y1) / (H * W))
    return x, y, y[perm], lam


@torch.no_grad()
def probs(model, loader):
    model.eval(); ps, ys = [], []
    for xb, yb in loader:
        p = torch.softmax(model(xb.to(DEV)), 1)[:, 1]
        ps.extend(p.cpu().tolist()); ys.extend(yb.tolist())
    return np.array(ys), np.array(ps)


def train_one(cfg, data):
    seed_all()
    tr_tf, ev_tf = tfs()
    (trp, trl), (vap, val), (tep, tel) = data["train"], data["val"], data["test"]
    loader = make_loader(trp, trl, tr_tf, cfg.get("sampler", "none"))
    eval_tr = DataLoader(DS(trp, trl, ev_tf), batch_size=64, num_workers=6)
    eval_va = DataLoader(DS(vap, val, ev_tf), batch_size=64, num_workers=6)
    eval_te = DataLoader(DS(tep, tel, ev_tf), batch_size=64, num_workers=6)

    model = build_model(BACKBONE, cfg.get("dropout", 0.3))
    w = class_weights(trl, cfg.get("loss", "ce"))
    crit = Focal(w, gamma=2.0) if cfg.get("loss") == "focal" else nn.CrossEntropyLoss(weight=w)

    freeze = cfg.get("freeze", "none")
    if freeze in ("linear_probe", "linear_then_finetune"):
        set_backbone_trainable(model, BACKBONE, False)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=cfg.get("lr", 2e-4), weight_decay=cfg.get("wd", 1e-4))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    scaler = torch.amp.GradScaler("cuda", enabled=DEV.type == "cuda")
    use_mix = cfg.get("mixup") or cfg.get("cutmix")
    mode = "mixup" if cfg.get("mixup") else "cutmix"

    best_ba, best_state = -1, None
    for ep in range(1, EPOCHS + 1):
        if freeze == "linear_then_finetune" and ep == EPOCHS // 2 + 1:
            set_backbone_trainable(model, BACKBONE, True)
            opt = torch.optim.AdamW(model.parameters(), lr=cfg.get("lr", 2e-4) / 10,
                                    weight_decay=cfg.get("wd", 1e-4))
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS - ep + 1)
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(DEV), yb.to(DEV)
            opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=DEV.type == "cuda"):
                if use_mix and np.random.rand() < 0.5:
                    xb, ya, ybb, lam = mix_batch(xb, yb, mode)
                    out = model(xb); loss = lam * crit(out, ya) + (1 - lam) * crit(out, ybb)
                else:
                    loss = crit(model(xb), yb)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        sched.step()
        from sklearn.metrics import balanced_accuracy_score
        yv, pv = probs(model, eval_va)
        ba = balanced_accuracy_score(yv, (pv >= 0.5).astype(int))
        if ba > best_ba:
            best_ba = ba; best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    pr = {"train": probs(model, eval_tr), "val": probs(model, eval_va), "test": probs(model, eval_te)}
    return C.summarize(cfg["name"], f"deep/{BACKBONE}", pr)


EXPERIMENTS = [
    {"name": "baseline"},
    {"name": "weighted_ce", "loss": "weighted_ce"},
    {"name": "focal", "loss": "focal", "dropout": 0.4},
    {"name": "class_balanced", "loss": "class_balanced"},
    {"name": "weighted_sampler", "sampler": "weighted"},
    {"name": "random_over", "sampler": "random_over"},
    {"name": "random_under", "sampler": "random_under"},
    {"name": "mixup", "mixup": True, "dropout": 0.4},
    {"name": "cutmix", "cutmix": True, "dropout": 0.4},
    {"name": "linear_probe", "freeze": "linear_probe", "lr": 1e-3},
    {"name": "linear_then_finetune", "freeze": "linear_then_finetune", "lr": 1e-3},
    {"name": "strong_reg", "wd": 5e-3, "dropout": 0.5, "mixup": True},
    {"name": "sampler+focal", "sampler": "weighted", "loss": "focal", "dropout": 0.4},
]


def main():
    sp = C.load_splits(originals_only=True)
    data = {k: (sp[k][0], list(sp[k][1])) for k in ("train", "val", "test")}
    out_name = "results_deep.json" if BACKBONE == "resnet18" else f"results_deep_{BACKBONE}.json"
    print(f"backbone {BACKBONE} | device {DEV} | train {len(data['train'][0])} "
          f"(unsafe {sum(data['train'][1])}) | val {len(data['val'][0])} | test {len(data['test'][0])}")
    results = []
    for cfg in EXPERIMENTS:
        t0 = time.time()
        rec = train_one(cfg, data)
        results.append(rec)
        te, tr = rec["test"], rec["train"]
        print(f"  [{BACKBONE}] {cfg['name']:22s} train_err {tr['error']:.3f} TEST acc {te['accuracy']:.3f} "
              f"rec {te['recall_unsafe']:.3f} PR-AUC {te['pr_auc']} AUC {te['roc_auc']} "
              f"gap {rec['gap_train_test']:+.3f} ({time.time()-t0:.0f}s)")
        (C.OUT / out_name).write_text(json.dumps(results, indent=2))
    print(f"\nwrote {C.OUT/out_name} ({len(results)} runs)")


if __name__ == "__main__":
    main()
