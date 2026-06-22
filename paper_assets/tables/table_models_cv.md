# Cross-validated metrics (5-fold, n=327) — deep models, balanced threshold

| model | cv_acc | cv_unsafe_recall | cv_unsafe_prec | cv_f1 | cv_auc | cv_ap | cv_mcc |
|---|---|---|---|---|---|---|---|
| ResNet50 | 0.905 | 0.843 | 0.795 | 0.819 | 0.919 | 0.879 | 0.755 |
| ConvNeXt-Tiny | 0.905 | 0.819 | 0.810 | 0.814 | 0.923 | 0.874 | 0.751 |
| EfficientNet-B0 | 0.865 | 0.783 | 0.714 | 0.747 | 0.884 | 0.820 | 0.657 |
| Ensemble (best) | 0.917 | 0.807 | 0.859 | 0.832 | 0.929 | 0.883 | 0.778 |