# Label audit - high-confidence model/label disagreements

Cross-validated calibrated ensemble disagrees with the current annotation (severity >= 0.70).
Review these to fix label noise - the main blocker for high recall.

- `current=safe, model=unsafe` (prob near 1.0): likely a MISSED unsafe (add to unsafe class)
- `current=unsafe, model=safe` (prob near 0): likely a spurious unsafe label (e.g. IMG_3719)

| severity | image | current | model_says | unsafe_prob |
|---|---|---|---|---|
| 0.989 | bus/negative/IMG_3305 | safe | unsafe | 0.989 |
| 0.986 | legua/negative/IMG_4211 | safe | unsafe | 0.986 |
| 0.983 | legua/positive/IMG_3719 | unsafe | safe | 0.017 |
| 0.961 | bus/negative/IMG_3939 | safe | unsafe | 0.961 |
| 0.949 | bus/positive/IMG_3305 | unsafe | safe | 0.051 |
| 0.945 | bus/positive/IMG_3671 | unsafe | safe | 0.055 |
| 0.933 | bus/positive/IMG_3557 | unsafe | safe | 0.067 |
| 0.885 | bus/negative/IMG_3555 | safe | unsafe | 0.885 |
| 0.872 | legua/positive/IMG_3873 | unsafe | safe | 0.128 |
| 0.867 | bus/positive/IMG_3534 | unsafe | safe | 0.133 |
| 0.846 | bus/positive/IMG_3778 | unsafe | safe | 0.154 |
| 0.839 | legua/negative/IMG_3966 | safe | unsafe | 0.839 |
| 0.836 | bus/positive/IMG_3306 | unsafe | safe | 0.164 |
| 0.835 | legua/positive/IMG_4210 | unsafe | safe | 0.165 |
| 0.79 | bus/negative/IMG_3537 | safe | unsafe | 0.79 |
| 0.785 | bus/positive/IMG_3937 | unsafe | safe | 0.215 |
| 0.78 | bus/positive/IMG_3298 | unsafe | safe | 0.22 |
| 0.759 | bus/negative/IMG_3539 | safe | unsafe | 0.759 |
| 0.754 | legua/positive/IMG_3816 | unsafe | safe | 0.246 |
| 0.752 | bus/negative/IMG_3540 | safe | unsafe | 0.752 |
| 0.746 | bus/positive/IMG_3910 | unsafe | safe | 0.254 |
| 0.735 | bus/negative/IMG_3488 | safe | unsafe | 0.735 |
| 0.708 | bus/negative/IMG_3316 | safe | unsafe | 0.708 |