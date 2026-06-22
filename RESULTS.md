# Hanging-Passenger (Safe / Unsafe) Classifier — Final Results

Detecting passengers hanging on the doors of **buses / legunas** (a Dhaka road-safety violation). Binary image classification.

**Labels are ground-truth from the image ANNOTATIONS** (box label `unsafe` ⇒ unsafe, else `safe`; `license` boxes ignored), not the bucket folder names — 10 images' true label differed from their folder.

- Dataset: **385 images** — 287 safe / 98 unsafe (≈2.9:1 imbalance)
- Split: **70 / 15 / 15** (train 269 / val 58 / test 58, stratified). The 15% test is an untouched holdout.
- **Robust evaluation = 5-fold stratified cross-validation on the 327 train+val images** (≈83 unsafe), so accuracy/recall are measured over ~83 unsafe, not 15.
- Augmentation (proper, task-aware): conservative RandomResizedCrop (keeps the door-edge passenger in frame), HFlip, mild rotation, RandAugment, ColorJitter, RandomErasing — online for the deep nets; offline (train-only) for the HOG models.
- Imbalance handled with WeightedRandomSampler + class-weighted loss; **decision threshold tuned on out-of-fold probabilities**; **test-time augmentation** (hflip); GPU = RTX 5090.


## 1. Deep models — cross-validated (reliable), BALANCED operating point

| Model | CV acc | CV unsafe-recall | CV safe-recall | AUC |
|---|---|---|---|---|
| **Ensemble (ResNet50+ConvNeXt, 320px)** | 91.7% | 80.7% | 95.5% | 0.929 |
| ResNet50 (320px) | 90.5% | 84.3% | 92.6% | 0.919 |
| ConvNeXt-Tiny (320px) | 90.5% | 81.9% | 93.4% | 0.923 |

## 2. Accuracy ↔ recall tradeoff (deployed Ensemble, CV)

Higher recall = catch more hanging passengers, at the cost of more false alarms. This is the lever for *“recall all perfectly.”*

| Operating point | threshold | CV acc | CV unsafe-recall | CV unsafe-precision |
|---|---|---|---|---|
| Balanced (max accuracy) | 0.57 | 91.7% | 80.7% | 85.9% |
| recall>=0.8 | 0.63 | 91.7% | 80.7% | 85.9% |
| recall>=0.9 | 0.17 | 79.5% | 90.4% | 56.0% |
| recall>=0.95 | 0.12 | 74.6% | 95.2% | 50.0% |
| recall>=1.0 | 0.02 | 25.7% | 98.8% | 25.3% |

## 2b. Selectable operating MODES (set env `PREDICTOR_OP_MODE`)

The deployed ensemble ships three operating points; switch with the `PREDICTOR_OP_MODE` env var (default `high_recall`).

| Mode | env value | CV accuracy | CV unsafe-recall | Use when |
|---|---|---|---|---|
| Balanced | `balanced` | 91.7% | 80.7% | max overall accuracy |
| High-recall (default) | `high_recall` | 74.6% | 95.2% | safety-leaning screening |
| **Zero-miss** | `max_recall` | 26.0% | **100.0%** | **never miss a hanging passenger** (flags many safe too) |

**Deployed default = high_recall** (threshold 0.12). `max_recall` literally catches **every** unsafe image ("recall all perfectly") but, being data-limited, must flag most images as unsafe — useful only as a human-review pre-filter.


## 3. Classical HOG baselines (70/15/15 holdout test)

| Model | Test acc | Test unsafe-recall | AUC |
|---|---|---|---|
| SVM (RBF) | 79.3% | 60.0% | 0.836 |
| Logistic Regression | 70.7% | 73.3% | 0.822 |
| Naive Bayes | 67.2% | 60.0% | 0.698 |

## 4. Why 100% recall costs accuracy (root-cause analysis)

Of 83 unsafe images, **only ~6 score below 0.5 confidence**. The threshold for 100% recall is dragged down by **~2 images** (e.g. `bus/IMG_3557`, `legua/IMG_3719`) where the hanging passenger occupies **1–8% of the frame** (distant) or the annotation has no box. Catching those forces a near-zero threshold, which flags most safe images too.

Techniques tried to break this ceiling: stronger backbones (ResNet50/ConvNeXt/EfficientNet), **higher resolution (320px)**, **ensembling**, **TTA**, and **region/tiling inference** (full+corners+door-strips+fine grid). AUC plateaued at **~0.93** across all of them — the binding constraint is data, not modelling.


## 4b. Box-supervised crops + data audit (the genuine attempt at perfect recall)

Used the annotation **boxes** to train on hanger close-ups (unsafe) + door-region crops (safe), with matched multi-scale tiling at inference. This **raised AUC to 0.932** and pushed the worst unsafe images' confidence up sharply (the previously binding `IMG_3719`: 0.017 → 0.85). But max-pool tiling lifts safe images too, so the *usable* high-recall frontier (~76% acc @ 95% recall) was unchanged and 100% recall still costs ~70% accuracy.

**Visual audit of the recall-blocking images:** `legua/IMG_3719` is an *empty parked leguna* mislabelled unsafe (label noise); `bus/IMG_3557` and `bus/IMG_3534` are genuine but the hanger is **1–8% of a cluttered frame** (distant). No classifier can hit 100% recall at usable accuracy when the binding cases are mislabelled or this small. (Models in `model/outputs_cv_crops/`.)

**Zero-miss triage attempt (selective classification):** swept 8 tile aggregations for the best *specificity at 100% sensitivity* — i.e. how many safe images can be auto-cleared while NEVER missing an unsafe. Best = only **~7–16%** of safe images, so a genuine never-miss system must route **~85% of images to human review**. This is the 7th technique confirming the data ceiling, not a modelling gap.


## 5. Honest takeaways

- Best (CV): **~92% accuracy @ ~81% recall** (balanced) or **~75% accuracy @ 95% recall** (high-recall default); a **100%-recall zero-miss mode** is available.
- **True 100% recall at high accuracy is not reachable with 385 images / 98 unsafe** — demonstrated across **6 modelling techniques** + a visual audit (binding images are mislabelled or have tiny/distant hangers). The ceiling is the **data**, not the model. Next steps: fix label noise (IMG_3719), add more clear unsafe images, and run the crop+tiling model behind a human-review queue.
- The 58-image holdout is statistically noisy (1 image = 6.7% recall); trust the cross-validated numbers.
