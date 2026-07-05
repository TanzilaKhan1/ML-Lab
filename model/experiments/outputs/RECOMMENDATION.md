# Recommendation

- **Best PR-AUC overall:** `convnext_tiny:svm+KMeansSMOTE·emb` (PR-AUC 0.9934, recall 0.9737, acc 0.8987, AUC 0.9923).
- **Best accuracy overall:** `convnext_tiny:logreg+BorderlineSMOTE·emb` (acc 0.962, recall 0.9474, PR-AUC 0.9856, AUC 0.982).
- **Best safety (0.6·recall+0.4·PR-AUC):** `convnext_tiny:svm+KMeansSMOTE·emb` (recall 0.9737, PR-AUC 0.9934, acc 0.8987).
- **Lowest variance (gap, recall≥0.80):** `resnet50:logreg+SMOTEENN·emb` (gap +0.002, test_err 0.1266, recall 0.8947).
- **Best embedding-sampling:** `convnext_tiny:svm+KMeansSMOTE·emb` (PR-AUC 0.9934, recall 0.9737, acc 0.8987, AUC 0.9923).
- **Best classical:** `svm+RandomOver` (PR-AUC 0.933, recall 0.8947, acc 0.8101).
  - best **convnext_tiny**: `linear_then_finetune` (recall 0.9737, PR-AUC 0.9871, AUC 0.9833, acc 0.8481).
  - best **resnet18**: `linear_then_finetune` (recall 0.9474, PR-AUC 0.9721, AUC 0.9679, acc 0.8228).
  - best **resnet50**: `focal` (recall 0.9474, PR-AUC 0.9852, AUC 0.9833, acc 0.8987).
