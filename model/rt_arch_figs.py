"""Regenerate the three architecture flow diagrams used in the paper:

  ML-Report/arch_svm.png     classical HOG pipeline (LogReg / SVM / Naive Bayes)
  ML-Report/arch_cnn.png     scratch SmallCNN
  ML-Report/arch_resnet.png  two-phase transfer-learning scheme (ResNet18 shown)

All facts are taken from the actual training code so the figures stay faithful:
  - SmallCNN: 4 double-conv blocks (32/64/128/256) -> GAP(256) -> head with
    Dropout(0.3), Linear 256->128, ReLU, Dropout(0.3), Linear 128->2  (~1.2M params)
    [train_customised_cnn.py]
  - Transfer models: ImageNet-pretrained backbone, 5-epoch head-only warm-up then
    deepest stage unfrozen, new head Dropout(0.4) -> Linear -> 2  [train_torch2.py]
  - Classical: PNG -> grayscale/resize 128 -> HOG (9 orient, 16px cells, 1764-d)
    -> StandardScaler -> PCA(95% var) -> classifier; SVM tuned by 5-fold GridSearchCV.

Run from repo/model:  python rt_arch_figs.py
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

OUT = Path(__file__).resolve().parent.parent / "ML-Report"

# muted pastel palette shared with the fig01-fig12 assets
BLUE   = "#d9e7fb"   # input / output
AMBER  = "#fce7c3"   # feature extraction / conv blocks
GREEN  = "#d4efc9"   # pooling / scaling / reduction
PINK   = "#f7d7ef"   # classifier / trainable head
GREY   = "#e6e6e6"   # frozen (pretrained) stages
ORANGE = "#f9cd8b"   # unfrozen-after-warmup stage
EDGE   = "#222222"
TITLE_FS, BOX_FS, NOTE_FS = 15, 11, 10

plt.rcParams["font.family"] = "DejaVu Sans"


def _arrow(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                 mutation_scale=14, lw=1.3, color=EDGE))


# ───────────────────────── classical pipeline (horizontal) ─────────────────────────
def classical():
    boxes = [
        ("Input image\nPNG 512×512×3", BLUE),
        ("Grayscale\n+ resize 128", AMBER),
        ("HOG features\n9 orient · 16px cells\n1764-dim", AMBER),
        ("Standard\nScaler", GREEN),
        ("PCA\n95% variance", GREEN),
        ("Classifier\nLogReg · SVM (RBF)\nNaive Bayes", PINK),
        ("Predict\n{safe, unsafe}", BLUE),
    ]
    fig, ax = plt.subplots(figsize=(15, 4.2))
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
    n = len(boxes); bw, gap = 11.5, 2.0
    x0 = (100 - (n * bw + (n - 1) * gap)) / 2
    cy, bh = 52, 30
    for i, (txt, col) in enumerate(boxes):
        left = x0 + i * (bw + gap)
        ax.add_patch(Rectangle((left, cy - bh / 2), bw, bh, facecolor=col,
                               edgecolor=EDGE, lw=1.2))
        ax.text(left + bw / 2, cy, txt, ha="center", va="center", fontsize=BOX_FS)
        if i:
            pl = x0 + (i - 1) * (bw + gap) + bw
            _arrow(ax, pl, cy, left, cy)
    ax.text(50, 90, "Classical machine-learning pipeline",
            ha="center", va="center", fontsize=TITLE_FS, fontweight="bold")
    ax.text(50, 20,
            "SVM tuned by 5-fold GridSearchCV:  "
            "C ∈ {0.5, 1, 4, 16}  ×  γ ∈ {scale, 0.01, 0.001}  "
            "×  kernel ∈ {rbf, linear}\n"
            "class_weight = 'balanced'   ·   scoring = accuracy",
            ha="center", va="center", fontsize=NOTE_FS, style="italic", color="#444")
    fig.tight_layout()
    fig.savefig(OUT / "arch_svm.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote arch_svm.png")


# ───────────────────────── vertical stack helper ─────────────────────────
def _vstack(ax, boxes, x_center, box_w, top=90, bottom=6, gap=1.8):
    weights = [b.get("lines", b["text"].count("\n") + 1) for b in boxes]
    avail = top - bottom - gap * (len(boxes) - 1)
    unit = avail / sum(weights)
    y = top
    for i, (b, w) in enumerate(zip(boxes, weights)):
        h = unit * w
        ax.add_patch(Rectangle((x_center - box_w / 2, y - h), box_w, h,
                               facecolor=b["color"], edgecolor=EDGE, lw=1.2))
        ax.text(x_center, y - h / 2, b["text"], ha="center", va="center", fontsize=BOX_FS)
        if b.get("side"):
            ax.text(x_center + box_w / 2 + 2.5, y - h / 2, b["side"], ha="left",
                    va="center", fontsize=NOTE_FS, style="italic", color="#555")
        if i < len(boxes) - 1:
            _arrow(ax, x_center, y - h, x_center, y - h - gap)
        y -= h + gap


# ───────────────────────── scratch CNN (vertical) ─────────────────────────
def cnn():
    B = "Conv 3×3 ({a}→{b}) · BN · ReLU\nConv 3×3 ({b}→{b}) · BN · ReLU\nMaxPool 2  →  {b} × {s} × {s}"
    boxes = [
        {"text": "Input   3 × 128 × 128", "color": BLUE, "lines": 1},
        {"text": B.format(a=3, b=32, s=64), "color": AMBER, "lines": 3},
        {"text": B.format(a=32, b=64, s=32), "color": AMBER, "lines": 3},
        {"text": B.format(a=64, b=128, s=16), "color": AMBER, "lines": 3},
        {"text": B.format(a=128, b=256, s=8), "color": AMBER, "lines": 3},
        {"text": "Global average pool  →  256", "color": GREEN, "lines": 1},
        {"text": "Dropout(0.3)\nLinear 256 → 128 · ReLU\nDropout(0.3)\nLinear 128 → 2",
         "color": PINK, "lines": 4},
        {"text": "Logits  →  Softmax  →  {safe, unsafe}", "color": BLUE, "lines": 1},
    ]
    fig, ax = plt.subplots(figsize=(8.6, 11.6))
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
    ax.text(50, 96, "Scratch CNN  —  about 1.2M parameters",
            ha="center", va="center", fontsize=TITLE_FS, fontweight="bold")
    _vstack(ax, boxes, x_center=50, box_w=80, top=91, bottom=4)
    fig.savefig(OUT / "arch_cnn.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote arch_cnn.png")


# ───────────────────────── transfer learning (vertical) ─────────────────────────
def resnet():
    boxes = [
        {"text": "Input   3 × 224 × 224", "color": BLUE, "lines": 1},
        {"text": "Conv 7×7 stride 2  (3→64)\nBN · ReLU · MaxPool 3×3 stride 2  →  64 × 56 × 56",
         "color": GREY, "lines": 2, "side": "frozen"},
        {"text": "Layer 1: 2 × BasicBlock (64ch)\nresidual: x + Conv→Conv  →  64 × 56 × 56",
         "color": GREY, "lines": 2, "side": "frozen"},
        {"text": "Layer 2: 2 × BasicBlock (128ch, stride 2)  →  128 × 28 × 28",
         "color": GREY, "lines": 1, "side": "frozen"},
        {"text": "Layer 3: 2 × BasicBlock (256ch, stride 2)  →  256 × 14 × 14",
         "color": GREY, "lines": 1, "side": "frozen"},
        {"text": "Layer 4: 2 × BasicBlock (512ch, stride 2)  →  512 × 7 × 7",
         "color": ORANGE, "lines": 1, "side": "unfreeze after warm-up (epoch 6+)"},
        {"text": "Global average pool  →  512", "color": GREY, "lines": 1},
        {"text": "New head:\nDropout(0.4)  ·  Linear 512 → 2", "color": PINK, "lines": 2,
         "side": "trained from scratch"},
        {"text": "Logits  →  Softmax  →  {safe, unsafe}", "color": BLUE, "lines": 1},
    ]
    fig, ax = plt.subplots(figsize=(11.4, 12.4))
    ax.set_xlim(0, 118); ax.set_ylim(0, 100); ax.axis("off")
    ax.text(43, 97, "Two-phase transfer learning  (ResNet18 shown)  —  "
                    "about 11M parameters, ImageNet-pretrained",
            ha="center", va="center", fontsize=13.5, fontweight="bold")
    _vstack(ax, boxes, x_center=43, box_w=70, top=92, bottom=9)
    # legend
    leg = [(GREY, "frozen (ImageNet weights)"), (ORANGE, "unfrozen after warm-up"),
           (PINK, "new head (random init)")]
    lx = 6
    for col, lab in leg:
        ax.add_patch(Rectangle((lx, 2.2), 4, 3, facecolor=col, edgecolor=EDGE, lw=1.0))
        ax.text(lx + 5, 3.7, lab, ha="left", va="center", fontsize=NOTE_FS)
        lx += 6 + len(lab) * 1.75
    fig.savefig(OUT / "arch_resnet.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote arch_resnet.png")


if __name__ == "__main__":
    classical(); cnn(); resnet()
    print("done ->", OUT)
