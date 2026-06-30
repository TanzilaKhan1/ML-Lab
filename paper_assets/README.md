# Paper Assets — Hanging-Passenger (Safe/Unsafe) Classifier

Regenerated from the **432-image** retrain (annotation-derived labels; train-only
A–Z augmentation → 1600 train images; honest 65-image val & test holdouts).
Theme: warm publication palette. All numbers come from `model/outputs_final/metrics_full.json`.

**Headline:** Ensemble (ResNet50 + ConvNeXt-Tiny, TTA) — TEST accuracy
86.2%, unsafe-recall 90.5%, ROC-AUC 0.949.

## Figures (`figures/`)
| File | What |
|---|---|
| fig01_dataset_overview.png | class balance, vehicle×class, split composition, augmentation growth |
| fig02_model_comparison.png | accuracy / unsafe-recall / F1 / AUC across all 9 models |
| fig03_roc_test.png | ROC curves (test) |
| fig04_pr_curves.png | precision–recall curves (test) |
| fig05_confusion_matrices.png | confusion matrix grid (test, balanced) |
| fig06_tradeoff.png | accuracy↔recall trade-off with the 3 operating modes |
| fig07_train_val_test_error.png | **train / val / test error** per model |
| fig08_generalization_gap.png | train vs test accuracy (overfitting gap) |
| fig09_metric_heatmap.png | full metric heatmap (test) |
| fig10_results_table.png | rendered results table |
| fig11_operating_modes.png | ensemble balanced / high-recall / max-recall |
| fig12_per_class_recall.png | safe vs unsafe recall per model |

## Tables (`tables/`)
- `table_models_test.{md,csv}` — all test metrics
- `table_train_val_test.{md,csv}` — train/val/test accuracy & error
- `table_operating_modes.md` — ensemble operating modes
- `table_dataset.md` — dataset composition
