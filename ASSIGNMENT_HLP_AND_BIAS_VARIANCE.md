# Error Analysis — Human-Level Performance & Bias/Variance Decision

**Task:** Binary image classification — *safe* vs *unsafe (door-hanging) travel* on Dhaka buses and legunas.
**Covers assignment bullets 3 and 4:** (3) deciding human-level performance (HLP); (4) bias vs variance — which to reduce first, which method first, and why.

> **Provenance / no-hallucination note.** Numbers come from the **432-image retrain (2026-06-30)**, each marked:
> `[FULL]` = `model/outputs_final/metrics_full.json` — **authoritative**: per-model train / val / test at the balanced operating point (21 metrics × 3 splits);
> `[TABLES]` = `paper_assets/tables/` — the same numbers as committed tables/figures;
> `[SHEET]` = `misclassified_6sheets.xlsx` (6-person review, val+test);
> `[COCO]` = `coco_export.json` (433 images, annotations);
> `[RESULTS]` = `RESULTS.md` / `LABEL_AUDIT.md`; `[LIT]` = external papers.
> **Single run:** train, val and test all come from the *same* checkpoints, so every train↔held-out pairing below is internally consistent (no cross-run caveat).

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

Error = 1 − accuracy at the balanced threshold. Train error is measured on the **un-augmented** original train images (augmented train error hides variance — `[LIT]` arXiv:2105.13343); val and test are real originals (no leakage).

---

## 2. What we are given (latest, 432-image retrain)

| Fact | Value | Source |
|---|---|---|
| Dataset | **432 images — 292 safe / 140 unsafe (≈2.1:1)** | `[COCO]`/`[RESULTS]` |
| Split | 70/15/15 stratified by vehicle×class → **302 train (→1600 after train-only A–Z aug) / 65 val / 65 test** (~21 unsafe per holdout) | `[FULL]` |
| Majority baseline | **67.6%** accuracy ("always safe", 0 violations caught) → track **unsafe-recall** | `[FULL]` |
| Held-out metric | **Val (65)** = dev (used for threshold tuning + model selection, slightly optimistic); **Test (65)** = untouched cross-check (1 img ≈ 1.5%) | `[FULL]` |
| Contamination | none — every model trains on **train only**; val/test are clean holdouts | `[FULL]` |
| Modelling ceiling | **broken** — adding unsafe data lifted AUC from the old ~0.93 plateau to **0.949** (Ensemble) | `[RESULTS]` |

---

## 3. Human-Level Performance (assignment bullet 3)

Decided the two required ways — group perception and SOTA literature — then take the stricter.

### 3.1 Perception-based
Rule is **vehicle-conditional**: leguna → *any* outside rider is unsafe; bus → unsafe only when *>1* hangs at the door. Difficulty concentrates in foot-on-step "inside vs outside boundary" ambiguity, tiny/distant hangers at 1–8% of a cluttered frame (`IMG_3557`, `IMG_3534`), night, and the bus ">1" rule `[SHEET]`/`[RESULTS]`. Project's perception numbers: quick glance ~8–10% error; careful expert (can zoom) **~3–4%** error.

### 3.2 SOTA / literature-based
- Controlled-condition machine ceilings (subject visible): bus boarding/alighting counting **>98%** [LIT, MDPI Sensors 2026]; boarding **posture F1 >96%** [LIT, MDPI Appl. Sci. 2025]; commercial door counting **up to 98%** [LIT] → competent observer on clear images ≈ **0.97–0.98**.
- Real-world clutter collapses naive systems to **72–75%** [LIT].
- Annotation agreement: **≥0.90 objective**, **0.70–0.85 moderately subjective** [LIT, Claru]; safety-risk boundary "**inherently subjective**" [LIT, PaSBench-Video].

### 3.3 Decision — HLP error = 4% (accuracy ≈ 96%)

| Regime | Human agreement | Justification |
|---|---|---|
| Clear images | ~0.97–0.98 | matches controlled posture/counting SOTA 96–98% [LIT] |
| Borderline subset (foot-on-step, occluded, night, tiny/distant, two-door, ">1") | ~0.80–0.85 | "boundary inherently subjective" [LIT] |

**→ HLP error = 4% (Bayes proxy).** Use the stricter expert value so avoidable bias is not overstated.

**Why not 0%:** the relabelled 432-set is now clean — `LABEL_AUDIT.md` reports **0 folder/annotation conflicts** (the old `IMG_3719`-type mislabels were fixed). What remains is *irreducible*: tiny/distant hangers (1–8% of the frame), genuinely ambiguous leaning-vs-hanging poses, and ~0.70–0.85 inter-annotator agreement on subjective safety labels `[LIT]`. So neither human nor model reaches 0%.

---

## 4. Bias vs variance — measured, latest tables (assignment bullet 4)

### 4.1 Decomposition (train/val/test from `[FULL]`, HLP = 4%, balanced threshold)

`avoidable bias = train − 4%`; `variance = val (dev) − train`; the Test column is the untouched cross-check.

| Model | Train err | Val err (dev) | Test err | Test AUC | Avoidable bias | Variance | Regime |
|---|---|---|---|---|---|---|---|
| **Ensemble (best)** | 0.4% | 12.3% | 13.8% | **0.949** | −3.6% | **+11.9%** | variance-limited |
| ResNet50 | 0.4% | 9.2% | 15.4% | 0.943 | −3.6% | **+8.8%** | variance-limited |
| ConvNeXt-Tiny | 0.1% | 12.3% | 18.5% | 0.944 | −3.9% | **+12.2%** | variance-limited |
| EfficientNet-B0 | 2.4% | 18.5% | 23.1% | 0.909 | −1.6% | **+16.1%** | variance-limited |
| ResNet18 | 0.1% | 10.8% | 16.9% | 0.892 | −3.9% | **+10.7%** | variance-limited |
| CNN (scratch) | 1.7% | 10.8% | 18.5% | 0.866 | −2.3% | **+9.1%** | variance-limited |
| SVM (RBF, HOG) | 4.1% | 16.9% | 15.4% | 0.863 | +0.1% | **+12.8%** | variance-limited |
| Logistic Reg (HOG) | 7.7% | 18.5% | 21.5% | 0.800 | +3.7% | **+10.8%** | variance-limited |
| Naive Bayes (HOG) | 16.4% | 26.2% | 32.3% | 0.646 | **+12.4%** | +9.8% | bias-leaning (underfits) |

### 4.2 Which model is well-balanced / variance-limited / bias-limited

- **Variance-limited: 8 of 9** — every deep net, the ensemble, and SVM/LogReg show variance ≫ (small/≤0) avoidable bias. The deep nets fit train ~0% but sit 9–18% on val/test.
- **Bias-leaning: only Naive Bayes** — avoidable bias +12.4% slightly exceeds its variance; the weakest model genuinely *underfits* (it is not deployed).
- **Well-balanced: none** this round — the from-scratch CNN now overfits too (1.7% train → 18.5% test), unlike the previous smaller-data run.
- **Best performer to deploy: the Ensemble** — Test acc **86.2%**, unsafe-recall **90.5%**, **AUC 0.949** `[FULL]`; fewest false negatives (2) of any model `[SHEET]`. ("Best performer" ≠ "balanced regime": ~0% train / 12–14% held-out → variance-limited.)

### 4.3 Decision: reduce VARIANCE first (not bias)

Applying Ng's rule: for **8 of 9** models `variance ≳ avoidable bias`, with avoidable bias ≤0 or small. → **variance reduction is the priority.** The lone exception (Naive Bayes) is a non-deployed weak baseline.

Three independent confirmations:
1. **The decomposition:** +9% to +16% variance vs ~0% avoidable bias across the deployable zoo `[FULL]`.
2. **Confirmed by intervention:** adding 42 unsafe images (98→140) **broke the old ~0.93 AUC plateau** (→0.949) and raised the zero-miss operating point from ~26% to ~42–57% accuracy `[RESULTS]` — direct evidence the binding constraint was **data**, not model capacity.
3. **Binding errors are data:** the remaining misses are tiny/distant hangers in cluttered frames `[SHEET]`, not a fitting failure.

### 4.4 Which specific method first, and why (all variance/data-centric)

1. **Collect more clear UNSAFE images.** Still the highest-return fix for a variance-limited model and the FN-heavy safety metric. The jump 98→140 already paid off (AUC +0.02, recall ceiling broken); more variety (night, two-door, tiny/distant) keeps closing the train→test gap.
2. **Keep / strengthen regularisation** — the train-only A–Z augmentation (→1600 imgs), WeightedRandomSampler, class-weighted loss, val threshold tuning, hflip TTA `[RESULTS]`. **Do not just scale model size** (the old plateau showed capacity isn't the limit).
3. **Maintain label hygiene** — the set is currently clean (`LABEL_AUDIT.md` = 0 conflicts); re-audit after each new collection round.

**Deprioritised:** bigger backbones / longer training. Detect-then-count (a real bias lever for the two-door/counting cases) is **Phase 2** — its crop/tiling prototype returned only marginal gain `[RESULTS]`.

---

## 5. Misclassification structure (supporting evidence, 6 sheets)

Per-model errors on the held-out **test (n=65)**, balanced threshold. All `[SHEET]`/`[FULL]`.

| Model | Errors | FP (safe→unsafe) | FN (unsafe→safe) |
|---|---|---|---|
| Naive Bayes | 21 | 9 | 12 |
| EfficientNet-B0 | 15 | 12 | 3 |
| Logistic Regression | 14 | 7 | 7 |
| ConvNeXt-Tiny | 12 | 9 | 3 |
| CNN (scratch) | 12 | 5 | 7 |
| ResNet18 | 11 | 6 | 5 |
| ResNet50 | 10 | 4 | 6 |
| SVM | 10 | 5 | 5 |
| **Ensemble** | **9** | 7 | **2** |

The Ensemble makes the fewest safety-critical misses (2 FN). Remaining FNs are tiny/distant hangers; FPs are open-door / person-near-opening frames — both map to the **data/variance** remedy in §4.4.

---

## 6. Assignment housekeeping

1. **Train errors** — now measured for all 9 models `[FULL]` (no `pending` cells, no cross-run pairing).
2. **Single run** — train, val and test all come from the same 432-image retrain, so the decomposition is internally consistent.
3. **Per-member paragraph** (bullet 2): each member synthesises their sheet (`misclassified_6sheets.xlsx`, 6 person-sheets) reasons + ways-forward.
4. **HLP empirical check** (optional): blind-label a sample and report measured inter-annotator disagreement to replace the 0.04 estimate.

---

### Source index
- `[FULL]` `model/outputs_final/metrics_full.json` — authoritative train/val/test × balanced/high-recall/max-recall, all 9 models
- `[TABLES]` `paper_assets/tables/` — `table_models_test.{csv,md}`, `table_train_val_test.{csv,md}`, `table_operating_modes.md`, `table_dataset.md`
- `[SHEET]` `misclassified_6sheets.xlsx` (asif, tanzila, taif, amio, tazkia, walid) — val+test review
- `[COCO]` `coco_export.json` — 433 images, 292 safe / 140 unsafe (image-level)
- `[RESULTS]` `RESULTS.md`, `REPORT.md`, `LABEL_AUDIT.md` (0 conflicts)
- `[LIT]`:
  - Bias/variance equations — Andrew Ng, *Machine Learning Yearning*: https://home-wordpress.deeplearning.ai/wp-content/uploads/2022/03/andrew-ng-machine-learning-yearning.pdf
  - Un-augmented eval basis, arXiv:2105.13343: https://arxiv.org/abs/2105.13343
  - Bus boarding/alighting YOLO — MDPI Sensors 2026: https://www.mdpi.com/1424-8220/26/5/1418
  - Boarding posture detection — MDPI Appl. Sci. 2025: https://www.mdpi.com/2076-3417/15/10/5367
  - Anomalous behavior on buses — Sci. Reports 2025: https://www.nature.com/articles/s41598-025-85962-8
  - Inter-annotator agreement — Claru: https://claru.ai/glossary/inter-annotator-agreement
  - PaSBench-Video — arXiv 2606.02443: https://arxiv.org/html/2606.02443
