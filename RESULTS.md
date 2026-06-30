# Safe / Unsafe (Hanging-Passenger) Classifier — Retrained Results

Binary image classification: passengers hanging on **bus / leguna** doors (`unsafe`) vs not (`safe`). Dhaka road-safety task.

## Dataset
- Source: Cloudflare R2 bucket 'machine-learning' (raw + annotations)
- Labels: annotations (unsafe if any 'unsafe' box, else safe)
- **Total labeled images: 432** (originals: safe 292 / unsafe 140)
- Split strategy: 4-way stratified (vehicle x class) 70/15/15; augment TRAIN only -> no leakage
- **Train**: 1600 images = 302 originals + **1298 offline augmentations** (safe 816 / unsafe 784, balanced)
- **Val**: 65 (safe 44 / unsafe 21) — real originals only
- **Test**: 65 (safe 44 / unsafe 21) — untouched holdout, real originals only
- Val & Test are 4-way stratified so each contains every category (bus-safe, bus-unsafe, legua-safe, legua-unsafe).
- Augmentation is applied to **TRAIN only** → no data leakage.

## 1. Main results — BALANCED operating point (max accuracy)

| Model | Train acc | Val acc | Test acc | Test recall (unsafe) | Test prec (unsafe) | Test F1 (unsafe) | Test ROC-AUC | Test PR-AUC | Test MCC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Ensemble (best)** | 99.6% | 87.7% | 86.2% | 90.5% | 73.1% | 0.809 | 0.949 | 0.919 | 0.712 |
| **ResNet50** | 99.6% | 90.8% | 84.6% | 71.4% | 78.9% | 0.750 | 0.943 | 0.914 | 0.641 |
| **ConvNeXt-Tiny** | 99.9% | 87.7% | 81.5% | 85.7% | 66.7% | 0.750 | 0.944 | 0.915 | 0.619 |
| **EfficientNet-B0** | 97.6% | 81.5% | 76.9% | 85.7% | 60.0% | 0.706 | 0.909 | 0.887 | 0.548 |
| **ResNet18** | 99.9% | 89.2% | 83.1% | 76.2% | 72.7% | 0.744 | 0.892 | 0.877 | 0.618 |
| **CNN** | 98.3% | 89.2% | 81.5% | 66.7% | 73.7% | 0.700 | 0.866 | 0.778 | 0.569 |
| **SVM (RBF)** | 95.9% | 83.1% | 84.6% | 76.2% | 76.2% | 0.762 | 0.863 | 0.834 | 0.648 |
| **Logistic Regression** | 92.3% | 81.5% | 78.5% | 66.7% | 66.7% | 0.667 | 0.800 | 0.673 | 0.508 |
| **Naive Bayes** | 83.6% | 73.8% | 67.7% | 42.9% | 50.0% | 0.462 | 0.646 | 0.576 | 0.234 |

## 2. HIGH-RECALL operating point (catch >=95% of unsafe — safety default)

| Model | Test acc | Test recall (unsafe) | Test precision (unsafe) | Test specificity |
| --- | --- | --- | --- | --- |
| **Ensemble (best)** | 69.2% | 100.0% | 51.2% | 54.5% |
| **ResNet50** | 72.3% | 95.2% | 54.1% | 61.4% |
| **ConvNeXt-Tiny** | 66.2% | 100.0% | 48.8% | 50.0% |
| **EfficientNet-B0** | 58.5% | 100.0% | 43.8% | 38.6% |
| **ResNet18** | 64.6% | 85.7% | 47.4% | 54.5% |
| **CNN** | 61.5% | 90.5% | 45.2% | 47.7% |
| **SVM (RBF)** | 63.1% | 90.5% | 46.3% | 50.0% |
| **Logistic Regression** | 52.3% | 90.5% | 39.6% | 34.1% |
| **Naive Bayes** | 46.2% | 81.0% | 35.4% | 29.5% |

## 3. MAX-RECALL / zero-miss operating point (100% unsafe recall)

| Model | Test acc | Test recall (unsafe) | Test precision (unsafe) |
| --- | --- | --- | --- |
| **Ensemble (best)** | 41.5% | 100.0% | 35.6% |
| **ResNet50** | 33.8% | 100.0% | 32.8% |
| **ConvNeXt-Tiny** | 56.9% | 100.0% | 42.9% |
| **EfficientNet-B0** | 32.3% | 100.0% | 32.3% |
| **ResNet18** | 47.7% | 100.0% | 38.2% |
| **CNN** | 52.3% | 100.0% | 40.4% |
| **SVM (RBF)** | 63.1% | 90.5% | 46.3% |
| **Logistic Regression** | 44.6% | 95.2% | 36.4% |
| **Naive Bayes** | 43.1% | 85.7% | 34.6% |

## 4. Operating thresholds per model (tuned on validation)

| Model | balanced | high_recall | max_recall | deployed default |
| --- | --- | --- | --- | --- |
| Ensemble (best) | 0.175 | 0.035 | 0.013 | high_recall |
| ResNet50 | 0.445 | 0.040 | 0.005 | high_recall |
| ConvNeXt-Tiny | 0.135 | 0.030 | 0.021 | high_recall |
| EfficientNet-B0 | 0.165 | 0.040 | 0.001 | high_recall |
| ResNet18 | 0.320 | 0.025 | 0.011 | high_recall |
| CNN | 0.585 | 0.110 | 0.060 | high_recall |
| SVM (RBF) | 0.380 | 0.035 | 0.035 | high_recall |
| Logistic Regression | 0.500 | 0.020 | 0.010 | high_recall |
| Naive Bayes | 0.830 | 0.270 | 0.162 | high_recall |

## 5. Train vs Test (overfitting view, balanced point)

| Model | Train acc | Test acc | Gap |
| --- | --- | --- | --- |
| Ensemble (best) | 99.6% | 86.2% | 13.4 pts |
| ResNet50 | 99.6% | 84.6% | 15.0 pts |
| ConvNeXt-Tiny | 99.9% | 81.5% | 18.4 pts |
| EfficientNet-B0 | 97.6% | 76.9% | 20.7 pts |
| ResNet18 | 99.9% | 83.1% | 16.9 pts |
| CNN | 98.3% | 81.5% | 16.8 pts |
| SVM (RBF) | 95.9% | 84.6% | 11.3 pts |
| Logistic Regression | 92.3% | 78.5% | 13.9 pts |
| Naive Bayes | 83.6% | 67.7% | 15.9 pts |

## Figures (`outputs_final/figures/`)
- `roc_test.png`
- `pr_test.png`
- `bars_test.png`
- `train_vs_test.png`
- `cm_ensemble.png`
- `cm_resnet50.png`
- `cm_convnext-tiny.png`
- `cm_efficientnet-b0.png`
- `cm_resnet18.png`
- `cm_cnn.png`
- `cm_svm.png`
- `cm_logistic.png`
- `cm_naive.png`

Full per-metric data (21 metrics x 3 splits x 3 operating points x 9 models): `outputs_final/metrics_full.json`.