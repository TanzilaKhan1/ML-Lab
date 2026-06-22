# Model comparison — held-out test set (n=58), balanced operating point

| model | family | device | params | test_acc | test_unsafe_recall | test_unsafe_prec | test_f1 | test_auc | test_ap | test_mcc |
|---|---|---|---|---|---|---|---|---|---|---|
| Logistic Regression | Classical | CPU | — | 0.707 | 0.733 | 0.458 | 0.564 | 0.822 | 0.745 | 0.383 |
| Naive Bayes | Classical | CPU | — | 0.672 | 0.600 | 0.409 | 0.486 | 0.698 | 0.578 | 0.269 |
| SVM (RBF) | Classical | CPU | — | 0.793 | 0.667 | 0.588 | 0.625 | 0.836 | 0.762 | 0.485 |
| CNN | CNN-scratch | GPU | 1,206,370 | 0.914 | 0.667 | 1.000 | 0.800 | 0.822 | 0.823 | 0.773 |
| ResNet18 | Transfer | GPU | 11,177,538 | 0.862 | 0.667 | 0.769 | 0.714 | 0.826 | 0.812 | 0.627 |
| EfficientNet-B0 | Transfer | GPU | 4,010,110 | 0.828 | 0.733 | 0.647 | 0.688 | 0.864 | 0.819 | 0.571 |
| ResNet50 | Transfer | GPU | 23,512,130 | 0.828 | 0.800 | 0.632 | 0.706 | 0.885 | 0.847 | 0.595 |
| ConvNeXt-Tiny | Transfer | GPU | 27,821,666 | 0.862 | 0.667 | 0.769 | 0.714 | 0.907 | 0.837 | 0.627 |
| Ensemble (best) | Ensemble | GPU | — | 0.845 | 0.667 | 0.714 | 0.690 | 0.901 | 0.837 | 0.587 |