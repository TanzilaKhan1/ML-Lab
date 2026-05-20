"""Transfer-learning CNN: ResNet18 (ImageNet pretrained) → binary head.

Frozen backbone for first warmup epochs, then unfreeze last block.
Stratified 80/20. Class-weighted CE. Cosine LR. tqdm progress.
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
import torch
import torch.nn as nn
from PIL import Image, ImageFile
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18
from tqdm import tqdm

from data_utils import CLASS_NAMES, SEED, scan_dataset, stratified_split

ImageFile.LOAD_TRUNCATED_IMAGES = True

ROOT = Path(__file__).parent
DATA_ROOT = ROOT / "Preprocessed"
OUT_DIR = ROOT / "outputs_resnet"
OUT_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = 224  # ResNet18 native
BATCH_SIZE = 16
EPOCHS = 25
WARMUP_EPOCHS = 5  # train head only
LR_HEAD = 1e-3
LR_FINETUNE = 1e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 4


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


class PathDataset(Dataset):
    def __init__(self, paths, labels, transform):
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        img = Image.open(self.paths[i]).convert("RGB")
        return self.transform(img), self.labels[i]


def build_transforms(weights: ResNet18_Weights):
    pre = weights.transforms()
    norm_mean = list(pre.mean)
    norm_std = list(pre.std)
    train_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std),
    ])
    return train_tf, eval_tf


def build_model() -> nn.Module:
    weights = ResNet18_Weights.IMAGENET1K_V1
    model = resnet18(weights=weights)
    for p in model.parameters():
        p.requires_grad = False
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, 2),
    )
    return model, weights


def unfreeze_last_block(model: nn.Module) -> None:
    for p in model.layer4.parameters():
        p.requires_grad = True
    for p in model.fc.parameters():
        p.requires_grad = True


def train_one_epoch(model, loader, criterion, optimizer, device, desc: str):
    model.train()
    total, correct, loss_sum = 0, 0, 0.0
    pbar = tqdm(loader, desc=desc, leave=False)
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
def evaluate(model, loader, criterion, device):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    all_pred, all_true = [], []
    for imgs, labels in tqdm(loader, desc="eval", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss_sum += loss.item() * imgs.size(0)
        pred = logits.argmax(1)
        correct += (pred == labels).sum().item()
        total += imgs.size(0)
        all_pred.extend(pred.cpu().tolist())
        all_true.extend(labels.cpu().tolist())
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
    ax.set_title("ResNet18 confusion matrix (test)")
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

    paths, labels = scan_dataset(DATA_ROOT)
    print(f"total: {len(paths)} | pos: {sum(labels)} | neg: {len(labels) - sum(labels)}")
    train_idx, test_idx = stratified_split(labels, test_size=0.20, seed=SEED)
    print(f"train: {len(train_idx)} | test: {len(test_idx)}")

    model, weights = build_model()
    model = model.to(device)
    train_tf, eval_tf = build_transforms(weights)

    train_paths = [paths[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    test_paths = [paths[i] for i in test_idx]
    test_labels = [labels[i] for i in test_idx]
    train_ds = PathDataset(train_paths, train_labels, train_tf)
    test_ds = PathDataset(test_paths, test_labels, eval_tf)

    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=pin)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=pin)

    class_counts = np.bincount(train_labels, minlength=2)
    cw = torch.tensor(len(train_labels) / (2.0 * class_counts), dtype=torch.float32, device=device)
    print(f"class weights: {cw.tolist()}")
    criterion = nn.CrossEntropyLoss(weight=cw, label_smoothing=0.05)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR_HEAD, weight_decay=WEIGHT_DECAY,
    )

    history = {"train_loss": [], "test_loss": [], "train_acc": [], "test_acc": []}
    best_acc = 0.0
    best_path = OUT_DIR / "best_model.joblib"
    scheduler = None

    for epoch in range(1, EPOCHS + 1):
        if epoch == WARMUP_EPOCHS + 1:
            unfreeze_last_block(model)
            optimizer = torch.optim.AdamW(
                [p for p in model.parameters() if p.requires_grad],
                lr=LR_FINETUNE, weight_decay=WEIGHT_DECAY,
            )
            remain = EPOCHS - WARMUP_EPOCHS
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=remain)
            print(">>> unfroze layer4 + head, switched to fine-tune LR")

        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            desc=f"ep{epoch:02d}",
        )
        te_loss, te_acc, _, _ = evaluate(model, test_loader, criterion, device)
        if scheduler is not None:
            scheduler.step()
        history["train_loss"].append(tr_loss)
        history["test_loss"].append(te_loss)
        history["train_acc"].append(tr_acc)
        history["test_acc"].append(te_acc)
        marker = " *" if te_acc > best_acc else ""
        print(f"epoch {epoch:02d}/{EPOCHS} | train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
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
    print(f"\nsaved: {OUT_DIR}/{{best_model.joblib,curves.png,confusion_matrix.png,report.txt,history.json}}")


if __name__ == "__main__":
    main()
