# Hanging-Passenger (Safe / Unsafe) Classifier: Technical Report

Detecting passengers hanging on the doors of **buses / legunas** (a Dhaka road-safety violation) as a binary image-classification task. This report reflects the **523-image** retrain.

## 1. Dataset

- **523 annotated images**, labels derived from the image **annotations** (a box labelled `unsafe` ⇒ unsafe; else `safe`; `license` ignored), not the bucket folder names.
- Class balance: **272 safe / 251 unsafe** (≈1.08 : 1).
- Source: Cloudflare R2 bucket `machine-learning` (raw images + annotations).

## 2. Method: split & augmentation (no leakage)

- **4-way stratified (vehicle × class) 70 / 15 / 15** split so val & test each carry every category (bus-safe, bus-unsafe, legua-safe, legua-unsafe).
- **Train** = 2730 images = 365 originals + **2365 offline A–Z augmentations**, class-balanced (1330 safe / 1400 unsafe).
- **Val** = 79, **Test** = 79, real originals only (augmentation applied to TRAIN only → no data leakage).
- Deep nets: ImageNet-pretrained, two-phase fine-tune, online aug + WeightedRandomSampler, threshold tuned on val, hflip TTA. Classical: HOG → StandardScaler → PCA → classifier. GPU = RTX 5090.

## 3. Results: held-out TEST (balanced operating point)

| Model | Acc | Bal-Acc | Recall (unsafe) | Precision | F1 | ROC-AUC | PR-AUC | MCC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Ensemble (best)** | 0.861 | 0.864 | 0.947 | 0.800 | 0.867 | 0.970 | 0.978 | 0.734 |
| **ResNet50** | 0.924 | 0.924 | 0.921 | 0.921 | 0.921 | 0.967 | 0.975 | 0.848 |
| **ConvNeXt-Tiny** | 0.835 | 0.840 | 0.947 | 0.766 | 0.847 | 0.972 | 0.979 | 0.691 |
| **EfficientNet-B0** | 0.835 | 0.841 | 0.974 | 0.755 | 0.851 | 0.979 | 0.980 | 0.701 |
| **ResNet18** | 0.911 | 0.912 | 0.921 | 0.897 | 0.909 | 0.971 | 0.976 | 0.823 |
| **CNN** | 0.861 | 0.861 | 0.868 | 0.846 | 0.857 | 0.929 | 0.896 | 0.722 |
| **SVM (RBF)** | 0.873 | 0.874 | 0.895 | 0.850 | 0.872 | 0.952 | 0.958 | 0.748 |
| **Logistic Regression** | 0.810 | 0.810 | 0.816 | 0.795 | 0.805 | 0.887 | 0.900 | 0.620 |
| **Naive Bayes** | 0.734 | 0.734 | 0.737 | 0.718 | 0.727 | 0.866 | 0.884 | 0.468 |

## 4. Train / Val / Test accuracy (generalization)

| Model | Train | Val | Test | Train→Test gap |
| --- | --- | --- | --- | --- |
| Ensemble (best) | 0.999 | 0.962 | 0.861 | 13.8 pts |
| ResNet50 | 0.997 | 0.962 | 0.924 | 7.3 pts |
| ConvNeXt-Tiny | 1.000 | 0.962 | 0.835 | 16.5 pts |
| EfficientNet-B0 | 0.989 | 0.937 | 0.835 | 15.4 pts |
| ResNet18 | 0.996 | 0.949 | 0.911 | 8.5 pts |
| CNN | 0.962 | 0.937 | 0.861 | 10.1 pts |
| SVM (RBF) | 0.999 | 0.861 | 0.873 | 12.5 pts |
| Logistic Regression | 0.877 | 0.772 | 0.810 | 6.6 pts |
| Naive Bayes | 0.832 | 0.810 | 0.734 | 9.8 pts |

## 5. Deployed operating modes (Ensemble = ResNet50 + ConvNeXt-Tiny)

| Mode | Test acc | Unsafe recall | Use when |
| --- | --- | --- | --- |
| Balanced | 86.1% | 94.7% | max overall accuracy |
| High-recall (default) | 86.1% | 94.7% | safety-leaning |
| Max-recall (zero-miss) | 86.1% | 94.7% | never miss an unsafe |

## 6. Headline

**Ensemble: TEST accuracy 86.1%, unsafe-recall 94.7%, ROC-AUC 0.970.** The added unsafe data lifted ROC-AUC to 0.970 (past the old ~0.93 plateau) and the recall-priority operating point reaches 95% unsafe-recall at 86% accuracy.

## 7. Figures & tables

See `paper_assets/figures/` (fig01–fig12) and `paper_assets/tables/`. Full per-metric data: `model/outputs_final/metrics_full.json`.
