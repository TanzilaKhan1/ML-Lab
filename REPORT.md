# Safety Classification Report
## Bus & Legua Door-Hanging Detection

**Date:** 2026-04-22  
**Author:** CREVIOS  
**Task:** Binary image classification — detect whether passengers are hanging on bus/legua doors (unsafe behaviour)

---

## 1. Problem Statement

Passengers hanging on the doors of buses and legunas is a dangerous and common safety violation on Dhaka roads. This project builds a machine learning pipeline to automatically classify images as:

| Label | Class | Meaning |
|---|---|---|
| `0` | **Positive (safe)** | Passengers are NOT hanging on the door |
| `1` | **Negative (unsafe)** | Passengers ARE hanging on the door |

**Primary metric: Recall on the unsafe class (label=1).** Missing an unsafe passenger is worse than a false alarm.

---

## 2. Dataset

### 2.1 Raw Images
Images were collected across two vehicle types and uploaded to Cloudflare R2 object storage, annotated using a custom web-based annotation tool.

| Source | Positive (safe) | Negative (unsafe) | Total |
|---|---|---|---|
| Bus | 8 | 33 | 41 |
| Legua | 11 | 18 | 29 |
| **Total** | **19** | **51** | **70** |

### 2.2 Augmentation
To address the small dataset size, each image was augmented using 32 distinct transformations including:

- Horizontal flip
- Rotations (±10°, ±15°, ±20°)
- Brightness adjustments (×0.7, ×1.3)
- Contrast adjustments (×0.7, ×1.3)
- Saturation adjustments (×0.6, ×1.4)
- Sharpening and Gaussian blur
- Gaussian noise
- Five crop-and-resize variants (centre, top-left, top-right, bottom-left, bottom-right)
- Combinations of the above (flip + brightness, rotation + contrast, etc.)

**After augmentation:**

| Class | Original | Augmented | Total |
|---|---|---|---|
| Positive (safe) | 44 | 506 | **550** |
| Negative (unsafe) | 26 | 524 | **550** |
| **Total** | **70** | **1,030** | **1,100** |

### 2.3 Train / Val / Test Split

A **group-aware split** was used to prevent data leakage. All augmented copies derived from the same source image are assigned to the same partition — so the model is evaluated only on images from source photos it has never seen.

| Partition | Samples | Positive | Negative | Source Groups |
|---|---|---|---|---|
| Train | 759 | 377 | 382 | 48 |
| Validation | 171 | 87 | 84 | 11 |
| Test | 170 | 86 | 84 | 11 |

> **Why group-split matters:** A naive random split achieves a misleading ~95% accuracy because augmented versions of the same photo appear in both train and test. With the group-split, honest test accuracy drops to 75–86%, reflecting true generalisation to unseen source images.

---

## 3. Feature Extraction

**Method:** Histogram of Oriented Gradients (HOG)

| Parameter | Value |
|---|---|
| Image resize | 128 × 128 px |
| Colour mode | RGB |
| Orientations | 9 |
| Pixels per cell | 16 × 16 |
| Cells per block | 2 × 2 |
| Feature vector length | **1,764** |

HOG captures edge direction patterns that distinguish hanging postures from normal standing postures. Features were normalised with `StandardScaler` (zero mean, unit variance) fitted on the training set only (and re-fitted per fold during cross-validation to prevent distribution leakage).

---

## 4. Models

Three classical machine learning classifiers were trained and compared. Each model is wrapped in an sklearn `Pipeline` with `StandardScaler` so the scaler is fitted independently per CV fold — no leakage.

### 4.1 Logistic Regression
- Solver: `lbfgs`, max iterations: 2,000
- Regularisation: C = 1.0

### 4.2 Gaussian Naive Bayes
- Assumes feature-conditional independence (Gaussian distribution per feature)
- No hyperparameters tuned

### 4.3 SVM — RBF Kernel
- Kernel: Radial Basis Function (RBF)
- C = 10, gamma = `scale`
- Probability calibration enabled (`probability=True`)

---

## 5. Results

### 5.1 Summary

| Model | CV Acc | CV Unsafe Rec | Train Acc | Val Acc | Test Acc | Test Unsafe Rec | Test AUC | Test MCC |
|---|---|---|---|---|---|---|---|---|
| **SVM (RBF kernel)** | 76.55% ± 6.63% | 67.04% ± 7.93% | 100.00% | 91.23% | **86.47%** | 75.00% | **0.9360** | **0.7477** |
| Naive Bayes | 70.75% ± 6.67% | 63.89% ± 5.44% | 85.77% | 91.81% | 78.24% | 72.62% | 0.7634 | 0.5673 |
| Logistic Regression | 62.85% ± 7.71% | 56.85% ± 13.08% | 100.00% | 65.50% | 75.88% | 69.05% | 0.8149 | 0.5212 |

![Summary Chart](results/summary.png)

---

### 5.2 Logistic Regression

**Test Accuracy: 75.88% | Unsafe Recall: 69.05% | AUC: 0.8149 | AP: 0.8642 | MCC: 0.5212**

| | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Positive (safe) | 0.732 | 0.826 | 0.776 | 86 |
| Negative (unsafe) | 0.795 | 0.690 | 0.739 | 84 |
| **Accuracy** | | | **0.759** | **170** |
| Macro avg | 0.763 | 0.758 | 0.757 | 170 |

**Confusion Matrix — Test Set**

![LR Confusion Matrix](results/Logistic_Regression_test_cm.png)

**ROC & Precision-Recall Curves**

![LR Curves](results/Logistic_Regression_curves.png)

**Learning Curve**

![LR Learning Curve](results/Logistic_Regression_learning_curve.png)

---

### 5.3 Naive Bayes

**Test Accuracy: 78.24% | Unsafe Recall: 72.62% | AUC: 0.7634 | AP: 0.7586 | MCC: 0.5673**

| | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Positive (safe) | 0.758 | 0.837 | 0.796 | 86 |
| Negative (unsafe) | 0.813 | 0.726 | 0.767 | 84 |
| **Accuracy** | | | **0.782** | **170** |
| Macro avg | 0.786 | 0.782 | 0.781 | 170 |

**Confusion Matrix — Test Set**

![NB Confusion Matrix](results/Naive_Bayes_test_cm.png)

**ROC & Precision-Recall Curves**

![NB Curves](results/Naive_Bayes_curves.png)

**Learning Curve**

![NB Learning Curve](results/Naive_Bayes_learning_curve.png)

---

### 5.4 SVM (RBF Kernel)

**Test Accuracy: 86.47% | Unsafe Recall: 75.00% | AUC: 0.9360 | AP: 0.9467 | MCC: 0.7477**

| | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Positive (safe) | 0.800 | 0.977 | 0.880 | 86 |
| Negative (unsafe) | 0.969 | 0.750 | 0.846 | 84 |
| **Accuracy** | | | **0.865** | **170** |
| Macro avg | 0.885 | 0.863 | 0.863 | 170 |

**Confusion Matrix — Test Set**

![SVM Confusion Matrix](results/SVM_RBF_kernel_test_cm.png)

**ROC & Precision-Recall Curves**

![SVM Curves](results/SVM_RBF_kernel_curves.png)

**Learning Curve**

![SVM Learning Curve](results/SVM_RBF_kernel_learning_curve.png)

---

## 6. Analysis

### 6.1 Overfitting (Train vs Test Gap)

Both Logistic Regression and SVM reach **100% training accuracy** — they memorise the 759 training samples. This is expected when a high-dimensional feature space (1,764 HOG features) has more dimensions than the dataset has unique source images. Regularisation hyperparameter search (GridSearchCV) was not performed; the current C=1.0 (LR) and C=10 (SVM) are defaults. Despite memorising the training set, SVM still generalises to 86.47% test accuracy because the RBF kernel finds a compact support structure that doesn't overfit the test distribution as badly as the accuracy drop might suggest.

### 6.2 Why SVM Wins

SVM with an RBF kernel finds a non-linear decision boundary in the 1,764-dimensional HOG feature space. The kernel implicitly maps features into a higher-dimensional space where the two classes become more separable. The AUC of **0.9360** and AP of **0.9467** indicate strong discriminative ability — it correctly ranks a randomly chosen unsafe image above a safe one 93.6% of the time.

### 6.3 Why Naive Bayes Underperforms on AUC

Naive Bayes achieves the second-best test accuracy (78.24%) but the lowest AUC (0.7634) and lowest AP (0.7586). HOG features are spatially correlated — pixels in adjacent cells share edge information. Naive Bayes assumes feature independence, which is violated here, leading to poorly calibrated probability estimates even when point predictions are reasonable.

### 6.4 Logistic Regression Trade-offs

Logistic Regression sits between the two in terms of test accuracy. It models a global linear boundary in HOG space which is not expressive enough to capture the complex posture differences between hanging and non-hanging passengers. Notably, its unsafe recall on validation (78.57%) is higher than SVM's at the point-prediction level, but SVM's AUC (0.9360 vs 0.8149) shows SVM has superior probability ranking.

### 6.5 Val vs Test Gap

The validation accuracy is noticeably higher than test accuracy for SVM (91.23% → 86.47%) and Naive Bayes (91.81% → 78.24%). With only 70 unique source images, each partition contains just 11 source groups (~110–170 augmented samples). Individual partition accuracy estimates carry high variance — a single atypical source image in the test set can shift accuracy by several percentage points.

### 6.6 Cross-Validation Variance

The 3-fold group CV shows ±6–8% standard deviation across folds (accuracy) and ±5–13% on unsafe recall, again reflecting the limited diversity of 70 unique source images. CV scores (63–77%) are more conservative than held-out test scores (76–86%), because each CV fold trains on fewer source images than the full training set.

---

## 7. Limitations

| Limitation | Impact |
|---|---|
| **70 unique source images** | High variance in all estimates; confidence intervals are wide |
| **Augmentation ≠ real diversity** | Augmented images share the same scene/lighting/angle as the original; the model has not seen genuinely new environments |
| **HOG only** | No colour, texture, or temporal information used |
| **No spatial localisation** | The classifier scores the whole image; it cannot locate the hanging passenger |
| **No hyperparameter search** | C and gamma for LR/SVM are defaults; GridSearchCV could improve test recall |
| **Static classifiers** | LR, NB, SVM cannot learn hierarchical representations the way deep CNNs can |

---

## 8. Recommendations

1. **Collect more source images** — at least 200–300 diverse source images to reduce variance and improve generalisation
2. **Hyperparameter tuning** — run `GridSearchCV` over C ∈ {0.01, 0.1, 1, 10, 100} for LR and SVM to reduce the train→test accuracy gap caused by memorisation
3. **Use a CNN** (e.g., MobileNetV2 fine-tuned) — even a lightweight network will significantly outperform HOG + SVM given enough data
4. **Add object detection** — localise passengers first (YOLO annotations already exist in this project), then classify the detected region
5. **Harder augmentation** — include weather simulation (rain, haze), motion blur, and nighttime conditions to cover Dhaka road scenarios
6. **Deploy SVM as the baseline** — at 86.47% test accuracy, AUC 0.9360, and AP 0.9467, SVM is the strongest classical model and suitable as a reference baseline

---

## 9. File Structure

```
ML-Lab/
├── pipeline.py                  # Full ML pipeline
├── augment.py                   # Image augmentation script
├── download_annotated.py        # Download + draw annotations from R2
├── REPORT.md                    # This report
├── Final_Dataset/
│   └── raw_images/
│       ├── positive/            # 550 images (safe)
│       └── negative/            # 550 images (unsafe)
└── results/
    ├── summary.png
    ├── *_test_cm.png            # Confusion matrices
    ├── *_curves.png             # ROC + Precision-Recall curves
    └── *_learning_curve.png     # Learning curves
```

---

## 10. Environment

```
Python       3.12
scikit-learn 1.8.0
scikit-image 0.26.0
Pillow       12.2.0
numpy        2.4.4
matplotlib   3.10.8
Manager      Poetry 2.1.1
```
