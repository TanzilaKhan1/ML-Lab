"""Binary classifier: positive vs negative (bus + leguna combined).

Small CNN from scratch in PyTorch. Stratified 80/20 train/test split.
HEIC support via pillow_heif. Mac MPS / CUDA / CPU autodetect.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pillow_heif  # noqa: F401  side-effect import for plugin registration
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageFile, UnidentifiedImageError
from pillow_heif import register_heif_opener
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from tqdm import tqdm

register_heif_opener()
ImageFile.LOAD_TRUNCATED_IMAGES = True

ROOT = Path(__file__).parent
DATA_ROOT = ROOT / "Preprocessed"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 30
LR = 1e-3
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 2
CLASS_NAMES = ["negative", "positive"]
EXTS = {".png"}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class BinaryImageDataset(Dataset):
    """Walks Merged/{bus,leguna}/{positive,negative}. Label = leaf folder name."""

    def __init__(self, root: Path, transform=None):
        self.transform = transform
        self.samples: list[tuple[Path, int]] = []
        for vehicle_dir in sorted(root.iterdir()):
            if not vehicle_dir.is_dir():
                continue
            for cls_dir in sorted(vehicle_dir.iterdir()):
                if not cls_dir.is_dir():
                    continue
                cls_name = cls_dir.name.lower()
                if cls_name not in CLASS_NAMES:
                    continue
                label = CLASS_NAMES.index(cls_name)
                for p in cls_dir.iterdir():
                    if p.suffix.lower() in EXTS and p.is_file():
                        self.samples.append((p, label))
        if not self.samples:
            raise RuntimeError(f"No images found under {root}")

    def __len__(self) -> int:
        return len(self.samples)

    def get_targets(self) -> list[int]:
        return [lbl for _, lbl in self.samples]

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
        except (UnidentifiedImageError, OSError) as e:
            raise RuntimeError(f"Failed to load {path}: {e}") from e
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def build_transforms():
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tf, eval_tf


class SmallCNN(nn.Module):
    """4 conv blocks → GAP → FC. ~600K params. From scratch."""

    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.features = nn.Sequential(
            self._block(3, 32),
            self._block(32, 64),
            self._block(64, 128),
            self._block(128, 256),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    @staticmethod
    def _block(in_c: int, out_c: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        return self.classifier(x)


class TransformSubset(Dataset):
    """Wraps a Subset to apply a transform at access time (so train/test can differ)."""

    def __init__(self, base: BinaryImageDataset, indices: list[int], transform):
        self.base = base
        self.indices = indices
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        path, label = self.base.samples[self.indices[i]]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


def stratified_split(targets: list[int], test_size: float, seed: int):
    idx = list(range(len(targets)))
    return train_test_split(idx, test_size=test_size, stratify=targets, random_state=seed)


def train_one_epoch(model, loader, criterion, optimizer, device, desc: str):
    model.train()
    total, correct, loss_sum = 0, 0, 0.0
    pbar = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True)
    for imgs, labels in pbar:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * imgs.size(0)
        pred = logits.argmax(1)
        correct += (pred == labels).sum().item()
        total += imgs.size(0)
        pbar.set_postfix(loss=f"{loss_sum/total:.4f}", acc=f"{correct/total:.4f}")
    return loss_sum / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device, desc: str = "eval"):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    all_pred, all_true = [], []
    pbar = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True)
    for imgs, labels in pbar:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss_sum += loss.item() * imgs.size(0)
        pred = logits.argmax(1)
        correct += (pred == labels).sum().item()
        total += imgs.size(0)
        all_pred.extend(pred.cpu().tolist())
        all_true.extend(labels.cpu().tolist())
        pbar.set_postfix(loss=f"{loss_sum/total:.4f}", acc=f"{correct/total:.4f}")
    return loss_sum / total, correct / total, all_true, all_pred


def plot_curves(history: dict, out_path: Path) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(history["train_loss"], label="train")
    ax[0].plot(history["test_loss"], label="test")
    ax[0].set_title("Loss")
    ax[0].set_xlabel("epoch")
    ax[0].legend()
    ax[1].plot(history["train_acc"], label="train")
    ax[1].plot(history["test_acc"], label="test")
    ax[1].set_title("Accuracy")
    ax[1].set_xlabel("epoch")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_confusion(cm: np.ndarray, classes: list[str], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix (test)")
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> None:
    seed_all(SEED)
    device = pick_device()
    print(f"device: {device}")

    train_tf, eval_tf = build_transforms()
    base_ds = BinaryImageDataset(DATA_ROOT, transform=None)
    targets = base_ds.get_targets()
    n_pos = sum(targets)
    n_neg = len(targets) - n_pos
    print(f"total: {len(targets)} | positive: {n_pos} | negative: {n_neg}")

    train_idx, test_idx = stratified_split(targets, test_size=0.20, seed=SEED)
    print(f"train: {len(train_idx)} | test: {len(test_idx)}")

    train_ds = TransformSubset(base_ds, train_idx, train_tf)
    test_ds = TransformSubset(base_ds, test_idx, eval_tf)

    pin = device.type == "cuda"
    persist = NUM_WORKERS > 0
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=pin,
                              persistent_workers=persist)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=pin,
                             persistent_workers=persist)

    class_counts = np.bincount([targets[i] for i in train_idx], minlength=2)
    weights = torch.tensor(len(train_idx) / (2.0 * class_counts), dtype=torch.float32, device=device)
    print(f"class weights: {weights.tolist()}")

    model = SmallCNN(num_classes=2).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params: {n_params:,}")

    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    history = {"train_loss": [], "test_loss": [], "train_acc": [], "test_acc": []}
    best_acc = 0.0
    best_path = OUT_DIR / "best_model.joblib"

    epoch_bar = tqdm(range(1, EPOCHS + 1), desc="epochs", dynamic_ncols=True)
    for epoch in epoch_bar:
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device,
                                          desc=f"ep{epoch:02d}")
        te_loss, te_acc, _, _ = evaluate(model, test_loader, criterion, device)
        epoch_bar.set_postfix(tr=f"{tr_acc:.3f}", te=f"{te_acc:.3f}", best=f"{max(best_acc, te_acc):.3f}")
        scheduler.step()
        history["train_loss"].append(tr_loss)
        history["test_loss"].append(te_loss)
        history["train_acc"].append(tr_acc)
        history["test_acc"].append(te_acc)
        marker = " *" if te_acc > best_acc else ""
        print(f"epoch {epoch:02d}/{EPOCHS} | "
              f"train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
              f"test loss {te_loss:.4f} acc {te_acc:.4f}{marker}")
        if te_acc > best_acc:
            best_acc = te_acc
            cpu_state = {k: v.cpu() for k, v in model.state_dict().items()}
            joblib.dump({"model": cpu_state, "classes": CLASS_NAMES,
                         "img_size": IMG_SIZE}, best_path)

    ckpt = joblib.load(best_path)
    model.load_state_dict(ckpt["model"])
    _, final_acc, y_true, y_pred = evaluate(model, test_loader, criterion, device)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4)
    print("\n=== final test ===")
    print(f"best test acc: {best_acc:.4f}")
    print(f"reloaded test acc: {final_acc:.4f}")
    print("confusion matrix:")
    print(cm)
    print(report)

    plot_curves(history, OUT_DIR / "curves.png")
    plot_confusion(cm, CLASS_NAMES, OUT_DIR / "confusion_matrix.png")
    (OUT_DIR / "history.json").write_text(json.dumps(history, indent=2))
    (OUT_DIR / "report.txt").write_text(
        f"best test acc: {best_acc:.4f}\nfinal test acc: {final_acc:.4f}\n\n"
        f"confusion matrix:\n{cm}\n\n{report}\n")
    print(f"\nsaved: {best_path}, curves.png, confusion_matrix.png, report.txt, history.json")


if __name__ == "__main__":
    main()
