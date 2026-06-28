# Error Analysis — Human-Level Performance & Bias/Variance Decision

**Task:** Binary image classification — *safe* vs *unsafe (door-hanging) travel* on Dhaka buses and legunas.
**Covers assignment bullets 3 and 4:** (3) deciding human-level performance (HLP); (4) bias vs variance — which to reduce first, which method first, and why.

> **Provenance / no-hallucination note.** Numbers are taken from the **latest** measured sources, with each one marked:
> `[TABLES]` = `paper_assets/tables/` — **the authoritative latest results** (held-out test + 5-fold CV, balanced operating point);
> `[TRAIN]` = `model/metrics.json` — the **only** file that reports *training* error (measured on original un-augmented images at argmax threshold via `eval_train_error.py`);
> `[SHEET]` = `misclassified_6sheets.xlsx`; `[COCO]` = `coco_export.json` (annotations, used to verify dataset & label noise);
> `[RESULTS]` = `RESULTS.md` / `LABEL_AUDIT.md`; `[LIT]` = external papers (searched live).
> **Stale source excluded:** `REPORT.md` (2026-04-22, old 70-image classical-only pipeline).
> **One honest caveat (read §4.4):** held-out errors come from `[TABLES]` but train errors only exist in `[TRAIN]`. For CNN / EfficientNet-B0 / ConvNeXt these two files come from *different model runs*, so their train↔held-out pairing is cross-run; for all other models the two agree.

---

## 1. The framework and its equations (verified, not from memory)

Standard bias/variance decomposition (Andrew Ng, *Machine Learning Yearning*) `[LIT]`:

```
Bayes (optimal / irreducible) error   ≈  human-level error          (proxy)
avoidable bias   =  training error       −  human-level error
variance         =  dev (held-out) error −  training error
total bias       =  Bayes (unavoidable) error  +  avoidable bias
```

Decision rule (Ng): **avoidable bias > variance → reduce bias first; else → reduce variance first.**

- **Bias-limited (underfit):** large avoidable bias — can't fit training data to the human bar.
- **Variance-limited (overfit / data-limited):** large variance — fits training, generalises poorly; fix = more/cleaner data or more regularisation, *not* a bigger model.
- **Well-balanced:** both gaps small.

Error = 1 − accuracy at the balanced (argmax) threshold; train error on **un-augmented** images (augmented train error hides variance — `[LIT]` arXiv:2105.13343).

---

## 2. What we are given (latest)

| Fact | Value | Source |
|---|---|---|
| Dataset | 385 images — **287 safe / 98 unsafe** (≈2.9:1) | `[COCO]` verifies exactly 287 safe + 98 unsafe (+1 unlabeled image dropped) |
| Split | 70/15/15 → 269 train / 58 val / 58 test (~15 unsafe per holdout) | `[TRAIN]` |
| Majority baseline | 74.5% accuracy ("always safe", 0 violations caught) → track **unsafe-recall** | `[TRAIN]` |
| Reliable dev metric | **5-fold CV** on 327 images (~83 unsafe); 58-image test is noisy (1 img ≈ 6.7%) | `[TABLES]`/`[RESULTS]` |
| Contamination | ResNet50/ConvNeXt/EfficientNet retrained on **train+val** → their *val* is not held-out; use **CV** (reliable) or test for them | `[TRAIN]` |
| Modelling ceiling | AUC plateaus at **~0.93** across 6+ techniques incl. box-crops/localization | `[RESULTS]` |

---

## 3. Human-Level Performance (assignment bullet 3)

Decided the two required ways — group perception and SOTA literature — then take the stricter.

### 3.1 Perception-based
Rule is **vehicle-conditional**: leguna → *any* outside rider is unsafe; bus → unsafe only when *>1* hangs at the door. Difficulty concentrates in foot-on-step "inside vs outside boundary" ambiguity (`IMG_4212`, `IMG_3747`), tiny/distant hangers at 1–8% of a cluttered frame (`IMG_3557`, `IMG_3534`, `IMG_3652`), night, and the bus ">1" rule `[SHEET]`/`[RESULTS]`. Project's stated perception numbers: quick glance ~8–10% error; careful expert (can zoom) **~3–4%** error `[RESULTS]`.

### 3.2 SOTA / literature-based (searched live)
- Controlled-condition machine ceilings (subject visible): bus boarding/alighting counting **>98%** [LIT, MDPI Sensors 2026]; boarding **posture F1 >96%** [LIT, MDPI Appl. Sci. 2025]; commercial door counting **up to 98%** [LIT] → competent observer on clear images ≈ **0.97–0.98**.
- Real-world clutter collapses naive systems to **72–75%** [LIT, same study].
- Annotation agreement: **≥0.90 objective**, **0.70–0.85 moderately subjective** [LIT, Claru]; safety-risk boundary "**inherently subjective**" [LIT, PaSBench-Video].

### 3.3 Decision — HLP error = 4% (accuracy ≈ 96%)

| Regime | Human agreement | Justification |
|---|---|---|
| Clear images | ~0.97–0.98 | matches controlled posture/counting SOTA 96–98% [LIT] |
| Borderline subset (foot-on-step, occluded, night, tiny/distant, two-door, ">1") | ~0.80–0.85 | "boundary inherently subjective" [LIT] |

**→ HLP error = 4% (Bayes proxy).** Use the stricter expert value so avoidable bias is not overstated.

**Why not 0% — independently confirmed by `[COCO]`:** `IMG_3719` carries an **`unsafe`** annotation but is an empty parked leguna → genuine mislabel; folder-vs-box conflicts exist (`IMG_3873` neg-folder/unsafe-box, `IMG_4211` pos-folder/safe-box); one image has no safe/unsafe box at all; `IMG_3484` & `IMG_3774` carry **both** safe and unsafe boxes (irreducible ambiguity). With label noise present, neither human nor model reaches 0%.

---

## 4. Bias vs variance — measured, latest tables (assignment bullet 4)

### 4.1 Decomposition (held-out from `[TABLES]`, train from `[TRAIN]`, HLP = 4%)

Held-out = **5-fold CV** for the four deep models the CV table covers (reliable, n=327); **held-out test** for the other five (n=58). `avoidable bias = train − 4%`; `variance = held-out − train`.

| Model | Train err `[TRAIN]` | Held-out err `[TABLES]` | Held-out basis | Avoidable bias | Variance | Regime |
|---|---|---|---|---|---|---|
| **Ensemble (best)** | pending | **8.3%** (acc .917) | CV | — | ~+8% | variance (train pending) |
| ResNet50 | 0.0% | 9.5% (acc .905) | CV | −4.0% | **+9.5%** | variance-limited |
| ConvNeXt-Tiny | 0.4% | 9.5% (acc .905) | CV | −3.6% | **+9.1%** | variance-limited |
| EfficientNet-B0 | 1.9% | 13.5% (acc .865) | CV | −2.1% | **+11.6%** | variance-limited |
| ResNet18 | 0.4% | 13.8% (acc .862) | test | −3.6% | **+13.4%** | variance-limited |
| CNN (scratch) | 7.8% | 8.6% (acc .914) | test | +3.8% | +0.8% | **well-balanced** |
| SVM (RBF, HOG) | 1.9% | 20.7% (acc .793) | test | −2.1% | **+18.8%** | variance-limited |
| Logistic Reg (HOG) | 7.4% | 29.3% (acc .707) | test | +3.4% | **+21.9%** | variance-limited |
| Naive Bayes (HOG) | 10.0% | 32.8% (acc .672) | test | +6.0% | **+22.8%** | variance-limited |

### 4.2 Which model is well-balanced / variance-limited / bias-limited

- **Bias-limited: none.** No model's avoidable bias exceeds its variance by a meaningful margin. The largest avoidable bias is Naive Bayes (+6.0%), dwarfed by its +22.8% variance.
- **Well-balanced: only the from-scratch CNN** — variance +0.8%, avoidable bias +3.8% (both small; the slight gap is bias, not variance). *(With the latest test number 0.914, CNN's train 7.8% ≈ held-out 8.6% → it does not overfit; it under-fits very slightly at 128px.)*
- **Variance-limited: the other eight** — every transfer net, the ensemble, and **all three classical HOG models**. The classical models **overfit** (e.g. SVM 1.9% train → 20.7% test), the opposite of the "HOG underfits" intuition.
- **Best performer to deploy: the Ensemble** — CV acc **0.917**, AUC **0.929**, recall 0.807 `[TABLES]`; 0 errors unique to it in the misclassified pool `[SHEET]`. ("Best performer" ≠ "balanced regime": its members are ~0% train / 8.3% CV → variance-limited.)

> Note: ResNet18's regime depends on which held-out you read — it looks *balanced* against val (3.5% in `[TRAIN]`) but *variance-limited* against the latest test (13.8% in `[TABLES]`). Per the instruction to use the latest tables, it is **variance-limited**. CNN is the only model balanced under either choice.

### 4.3 Decision: reduce VARIANCE first (not bias)

Applying Ng's rule across the table: for **8 of 9** models `variance ≫ avoidable bias`, and avoidable bias is ≤0 or small-positive. → **variance reduction is the priority.** The lone exception (CNN) is a small bias gap on a non-deployed scratch model.

Three independent confirmations:
1. **The decomposition:** +9% to +23% variance vs ~0% avoidable bias across the zoo `[TABLES]`/`[TRAIN]`.
2. **AUC plateau ~0.93** across 6+ architectures incl. the box-crop/localization attempt (0.929→0.932 only) `[RESULTS]` — extra capacity / the detector idea doesn't help → not a bias problem.
3. **Binding errors are data:** of 83 unsafe images only ~6 score <0.5; the recall ceiling is set by ~2 mislabelled/tiny-target images `[RESULTS]`, and `[COCO]` confirms the label noise directly.

### 4.4 Which specific method first, and why (all variance/data-centric — no paid labellers)

1. **Fix label noise first.** Correct `IMG_3719`; review the 23 `LABEL_AUDIT.md` conflicts; resolve the folder-vs-box and dual-box cases `[COCO]`; de-duplicate the 3 clashing filenames `[SHEET]`. *Why first:* biggest cheapest win, and bias/variance numbers are untrustworthy while ~10–15% of flagged rows are mislabelled. Relabelling existing images is allowed under "no data labellers."
2. **Collect more clear UNSAFE images** (only 98). Highest-return fix for a variance-limited model; targets the FN-heavy safety metric. Prioritise the repeating sheet clusters `[SHEET]`: open-door / person-near-opening FPs, empty-vehicle FPs, small/distant hanger FNs, night/low-light.
3. **Keep current regularisation** — RandAugment, RandomErasing, weighted sampler, class-weighted loss, OOF threshold tuning, TTA `[RESULTS]`. **Do not just scale model size** (proven to plateau).

**Deprioritised:** bigger backbones / longer training. Detect-then-count (a real bias lever for the two-door/counting cases) is **Phase 2** only — its crop/tiling prototype already returned marginal gain `[RESULTS]`.

**Cross-run caveat (the one honesty flag):** held-out values for CNN/EfficientNet-B0/ConvNeXt come from the **latest paper run** `[TABLES]`, but their *train* errors come from an **earlier run** `[TRAIN]` (the paper's CNN test = 0.914, while `metrics.json`/`all_results.json` CNN = 0.879 — two distinct runs). So those three rows pair train and held-out across runs. ResNet50, ResNet18, and the three classical models are internally consistent across both files. The direction of the verdict (variance-limited) holds under either run.

---

## 5. Misclassification structure (supporting evidence, from the 6 sheets)

Pool = 83 unique images / 86 rows = union of any of 9 models wrong. All `[SHEET]`.

| Model | Errors | FP (safe→unsafe) | FN (unsafe→safe) | Only-this-model-wrong |
|---|---|---|---|---|
| Logistic Regression | 52 | 39 | 13 | 33 |
| CNN (scratch) | 28 | 8 | 20 | 18 |
| SVM | 25 | 16 | 9 | 5 |
| EfficientNet-B0 | 15 | 11 | 4 | 3 |
| ResNet18 | 12 | 6 | 6 | 1 |
| ResNet50 | 12 | 7 | 5 | 1 |
| ConvNeXt-Tiny | 11 | 6 | 5 | 0 |
| Ensemble | 11 | 6 | 5 | 0 |

≥8-of-9 fail (structural / borderline-Bayes): `IMG_3484` (two-door, dual-box), `IMG_3652` (tiny side-door hanger), `IMG_3774` (distant double-decker, dual-box). The 33 LogReg + 18 CNN single-model errors are per-model noise. Every per-row fix the members proposed (open-door / empty-vehicle hard negatives, door crops, night augmentation) maps to the **variance/data** remedy in §4.4.

---

## 6. Assignment housekeeping still open

1. **Ensemble train error** — the one `pending` cell; run `eval_train_error.py` to fill it.
2. **Reconcile the two runs** — re-export `metrics.json` from the same checkpoints behind `[TABLES]` so train and held-out match for CNN/EffNetB0/ConvNeXt.
3. **Per-member paragraph** (bullet 2): each member synthesises their sheet's reasons + ways-forward.
4. **HLP empirical check** (optional): blind-label the 50-image sample (`eval_train_error.py --hlp-sample 50`) and report measured inter-annotator disagreement to replace the 0.04 estimate.

---

### Source index
- `[TABLES]` `paper_assets/tables/` — `table_models_test.csv`/`.md` (held-out test, n=58), `table_models_cv.md` (5-fold CV deep models), `table_operating_modes.md` — **authoritative latest**
- `[TRAIN]` `model/metrics.json` (≡ `predictor/model/metrics.json`) — only source of training error
- `[SHEET]` `misclassified_6sheets.xlsx` (taif, amio, asif, tanzila, tazkia, walid)
- `[COCO]` `coco_export.json` — 287 safe / 98 unsafe verified; label-noise confirmation
- `[RESULTS]` `RESULTS.md`, `LABEL_AUDIT.md`
- `[LIT]`:
  - Bias/variance equations — Andrew Ng, *Machine Learning Yearning*: https://home-wordpress.deeplearning.ai/wp-content/uploads/2022/03/andrew-ng-machine-learning-yearning.pdf
  - Un-augmented eval basis, arXiv:2105.13343: https://arxiv.org/abs/2105.13343
  - Bus boarding/alighting YOLO — MDPI Sensors 2026: https://www.mdpi.com/1424-8220/26/5/1418
  - Boarding posture detection — MDPI Appl. Sci. 2025: https://www.mdpi.com/2076-3417/15/10/5367
  - Anomalous behavior on buses — Sci. Reports 2025: https://www.nature.com/articles/s41598-025-85962-8
  - Inter-annotator agreement — Claru: https://claru.ai/glossary/inter-annotator-agreement
  - PaSBench-Video — arXiv 2606.02443: https://arxiv.org/html/2606.02443
