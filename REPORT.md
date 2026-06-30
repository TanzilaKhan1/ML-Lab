# Hanging-Passenger (Safe / Unsafe) Classifier — Technical Report

Detecting passengers hanging on the doors of **buses / legunas** — a Dhaka road-safety violation — as a binary image-classification task. This report reflects the **432-image** retrain.

## 1. Dataset

- **432 annotated images**, labels derived from the image **annotations** (a box labelled `unsafe` ⇒ unsafe; else `safe`; `license` ignored) — not the bucket folder names.
- Class balance: **292 safe / 140 unsafe** (≈2.1 : 1).
- Source: Cloudflare R2 bucket `machine-learning` (raw images + annotations).

## 2. Method — split & augmentation (no leakage)

- **4-way stratified (vehicle × class) 70 / 15 / 15** split so val & test each carry every category (bus-safe, bus-unsafe, legua-safe, legua-unsafe).
- **Train** = 1600 images = 302 originals + **1298 offline A–Z augmentations**, class-balanced (816 safe / 784 unsafe).
- **Val** = 65, **Test** = 65 — real originals only (augmentation applied to TRAIN only → no data leakage).
- Deep nets: ImageNet-pretrained, two-phase fine-tune, online aug + WeightedRandomSampler, threshold tuned on val, hflip TTA. Classical: HOG → StandardScaler → PCA → classifier. GPU = RTX 5090.

## 3. Results — held-out TEST (balanced operating point)

| Model | Acc | Bal-Acc | Recall (unsafe) | Precision | F1 | ROC-AUC | PR-AUC | MCC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Ensemble (best)** | 0.862 | 0.873 | 0.905 | 0.731 | 0.809 | 0.949 | 0.919 | 0.712 |
| **ResNet50** | 0.846 | 0.812 | 0.714 | 0.789 | 0.750 | 0.943 | 0.914 | 0.641 |
| **ConvNeXt-Tiny** | 0.815 | 0.826 | 0.857 | 0.667 | 0.750 | 0.944 | 0.915 | 0.619 |
| **EfficientNet-B0** | 0.769 | 0.792 | 0.857 | 0.600 | 0.706 | 0.909 | 0.887 | 0.548 |
| **ResNet18** | 0.831 | 0.813 | 0.762 | 0.727 | 0.744 | 0.892 | 0.877 | 0.618 |
| **CNN** | 0.815 | 0.777 | 0.667 | 0.737 | 0.700 | 0.866 | 0.778 | 0.569 |
| **SVM (RBF)** | 0.846 | 0.824 | 0.762 | 0.762 | 0.762 | 0.863 | 0.834 | 0.648 |
| **Logistic Regression** | 0.785 | 0.754 | 0.667 | 0.667 | 0.667 | 0.800 | 0.673 | 0.508 |
| **Naive Bayes** | 0.677 | 0.612 | 0.429 | 0.500 | 0.462 | 0.646 | 0.576 | 0.234 |

## 4. Train / Val / Test accuracy (generalization)

| Model | Train | Val | Test | Train→Test gap |
| --- | --- | --- | --- | --- |
| Ensemble (best) | 0.996 | 0.877 | 0.862 | 13.4 pts |
| ResNet50 | 0.996 | 0.908 | 0.846 | 15.0 pts |
| ConvNeXt-Tiny | 0.999 | 0.877 | 0.815 | 18.4 pts |
| EfficientNet-B0 | 0.976 | 0.815 | 0.769 | 20.7 pts |
| ResNet18 | 0.999 | 0.892 | 0.831 | 16.9 pts |
| CNN | 0.983 | 0.892 | 0.815 | 16.8 pts |
| SVM (RBF) | 0.959 | 0.831 | 0.846 | 11.3 pts |
| Logistic Regression | 0.923 | 0.815 | 0.785 | 13.9 pts |
| Naive Bayes | 0.836 | 0.738 | 0.677 | 15.9 pts |

## 5. Deployed operating modes (Ensemble = ResNet50 + ConvNeXt-Tiny)

| Mode | Test acc | Unsafe recall | Use when |
| --- | --- | --- | --- |
| Balanced | 86.2% | 90.5% | max overall accuracy |
| High-recall (default) | 69.2% | 100.0% | safety-leaning |
| Max-recall (zero-miss) | 41.5% | 100.0% | never miss an unsafe |

## 6. Headline

**Ensemble — TEST accuracy 86.2%, unsafe-recall 90.5%, ROC-AUC 0.949.** The added unsafe data raised AUC past the old 0.929 ceiling and lifted the zero-miss (100%-recall) operating point to ~42% accuracy (was ~26%).

## 7. Figures & tables

See `paper_assets/figures/` (fig01–fig12) and `paper_assets/tables/`. Full per-metric data: `model/outputs_final/metrics_full.json`.
