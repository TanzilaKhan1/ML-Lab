# Safe / Unsafe (Hanging-Passenger) Classifier: Retrained Results

Binary image classification: passengers hanging on **bus / leguna** doors (`unsafe`) vs not (`safe`). Dhaka road-safety task.

## Dataset
- Source: Cloudflare R2 bucket 'machine-learning' (raw + annotations)
- Labels: annotations (unsafe if any 'unsafe' box, else safe)
- **Total labeled images: 523** (originals: safe 272 / unsafe 251)
- Split strategy: 4-way stratified (vehicle x class) 70/15/15; augment TRAIN only -> no leakage
- **Train**: 2730 images = 365 originals + **2365 offline augmentations** (safe 1330 / unsafe 1400, balanced)
- **Val**: 79 (safe 41 / unsafe 38), real originals only
- **Test**: 79 (safe 41 / unsafe 38), untouched holdout, real originals only
- Val & Test are 4-way stratified so each contains every category (bus-safe, bus-unsafe, legua-safe, legua-unsafe).
- Augmentation is applied to **TRAIN only** → no data leakage.

## 1. Main results: BALANCED operating point (max accuracy)

| Model | Train acc | Val acc | Test acc | Test recall (unsafe) | Test prec (unsafe) | Test F1 (unsafe) | Test ROC-AUC | Test PR-AUC | Test MCC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Ensemble (best)** | 99.9% | 96.2% | 86.1% | 94.7% | 80.0% | 0.867 | 0.970 | 0.978 | 0.734 |
| **ResNet50** | 99.7% | 96.2% | 92.4% | 92.1% | 92.1% | 0.921 | 0.967 | 0.975 | 0.848 |
| **ConvNeXt-Tiny** | 100.0% | 96.2% | 83.5% | 94.7% | 76.6% | 0.847 | 0.972 | 0.979 | 0.691 |
| **EfficientNet-B0** | 98.9% | 93.7% | 83.5% | 97.4% | 75.5% | 0.851 | 0.979 | 0.980 | 0.701 |
| **ResNet18** | 99.6% | 94.9% | 91.1% | 92.1% | 89.7% | 0.909 | 0.971 | 0.976 | 0.823 |
| **CNN** | 96.2% | 93.7% | 86.1% | 86.8% | 84.6% | 0.857 | 0.929 | 0.896 | 0.722 |
| **SVM (RBF)** | 99.9% | 86.1% | 87.3% | 89.5% | 85.0% | 0.872 | 0.952 | 0.958 | 0.748 |
| **Logistic Regression** | 87.7% | 77.2% | 81.0% | 81.6% | 79.5% | 0.805 | 0.887 | 0.900 | 0.620 |
| **Naive Bayes** | 83.2% | 81.0% | 73.4% | 73.7% | 71.8% | 0.727 | 0.866 | 0.884 | 0.468 |

## 2. HIGH-RECALL operating point (catch >=95% of unsafe, safety default)

| Model | Test acc | Test recall (unsafe) | Test precision (unsafe) | Test specificity |
| --- | --- | --- | --- | --- |
| **Ensemble (best)** | 86.1% | 94.7% | 80.0% | 78.0% |
| **ResNet50** | 83.5% | 92.1% | 77.8% | 75.6% |
| **ConvNeXt-Tiny** | 88.6% | 94.7% | 83.7% | 82.9% |
| **EfficientNet-B0** | 86.1% | 97.4% | 78.7% | 75.6% |
| **ResNet18** | 81.0% | 94.7% | 73.5% | 68.3% |
| **CNN** | 83.5% | 92.1% | 77.8% | 75.6% |
| **SVM (RBF)** | 68.4% | 100.0% | 60.3% | 39.0% |
| **Logistic Regression** | 62.0% | 97.4% | 56.1% | 29.3% |
| **Naive Bayes** | 57.0% | 100.0% | 52.8% | 17.1% |

## 3. MAX-RECALL / zero-miss operating point (100% unsafe recall)

| Model | Test acc | Test recall (unsafe) | Test precision (unsafe) |
| --- | --- | --- | --- |
| **Ensemble (best)** | 86.1% | 94.7% | 80.0% |
| **ResNet50** | 81.0% | 97.4% | 72.5% |
| **ConvNeXt-Tiny** | 88.6% | 94.7% | 83.7% |
| **EfficientNet-B0** | 86.1% | 97.4% | 78.7% |
| **ResNet18** | 59.5% | 100.0% | 54.3% |
| **CNN** | 83.5% | 97.4% | 75.5% |
| **SVM (RBF)** | 70.9% | 100.0% | 62.3% |
| **Logistic Regression** | 50.6% | 100.0% | 49.4% |
| **Naive Bayes** | 53.2% | 100.0% | 50.7% |

## 4. Operating thresholds per model (tuned on validation)

| Model | balanced | high_recall | max_recall | deployed default |
| --- | --- | --- | --- | --- |
| Ensemble (best) | 0.170 | 0.175 | 0.175 | high_recall |
| ResNet50 | 0.480 | 0.110 | 0.074 | high_recall |
| ConvNeXt-Tiny | 0.170 | 0.235 | 0.237 | high_recall |
| EfficientNet-B0 | 0.185 | 0.245 | 0.248 | high_recall |
| ResNet18 | 0.200 | 0.030 | 0.001 | high_recall |
| CNN | 0.425 | 0.215 | 0.136 | high_recall |
| SVM (RBF) | 0.425 | 0.015 | 0.020 | high_recall |
| Logistic Regression | 0.465 | 0.050 | 0.002 | high_recall |
| Naive Bayes | 0.615 | 0.025 | 0.008 | high_recall |

## 5. Train vs Test (overfitting view, balanced point)

| Model | Train acc | Test acc | Gap |
| --- | --- | --- | --- |
| Ensemble (best) | 99.9% | 86.1% | 13.8 pts |
| ResNet50 | 99.7% | 92.4% | 7.3 pts |
| ConvNeXt-Tiny | 100.0% | 83.5% | 16.5 pts |
| EfficientNet-B0 | 98.9% | 83.5% | 15.4 pts |
| ResNet18 | 99.6% | 91.1% | 8.5 pts |
| CNN | 96.2% | 86.1% | 10.1 pts |
| SVM (RBF) | 99.9% | 87.3% | 12.5 pts |
| Logistic Regression | 87.7% | 81.0% | 6.6 pts |
| Naive Bayes | 83.2% | 73.4% | 9.8 pts |

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