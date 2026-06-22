# Ensemble operating modes (CV, set via PREDICTOR_OP_MODE)

| mode | threshold | cv_accuracy | cv_unsafe_recall | cv_unsafe_precision |
|---|---|---|---|---|
| balanced | 0.570 | 0.917 | 0.807 | 0.859 |
| high_recall | 0.125 | 0.746 | 0.952 | 0.500 |
| max_recall (zero-miss) | 0.017 | 0.260 | 1.000 | 0.255 |