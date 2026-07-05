"""Classical (HOG) imbalance experiments — samplers + balanced ensembles.

For each technique: imblearn Pipeline so resampling touches TRAIN only (no leakage);
fit on original imbalanced train (204/98), evaluate on real val/test. SMOTE/ADASYN
are valid here because they operate on the 1764-d HOG feature vectors (not pixels).

Writes outputs/results_classical.json
"""
from __future__ import annotations
import json, time, warnings
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline as SkPipeline

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import RandomOverSampler, SMOTE, BorderlineSMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek, SMOTEENN
from imblearn.ensemble import (BalancedBaggingClassifier, EasyEnsembleClassifier,
                               BalancedRandomForestClassifier)

import common as C

warnings.filterwarnings("ignore")
SEED = C.SEED


def clf(kind, balanced=False):
    cw = "balanced" if balanced else None
    if kind == "svm":
        return SVC(C=1.0, kernel="rbf", gamma="scale", probability=True,
                   class_weight=cw, random_state=SEED)
    return LogisticRegression(C=1.0, max_iter=5000, class_weight=cw, random_state=SEED)


def samplers():
    return {
        "none": None,
        "RandomOver": RandomOverSampler(random_state=SEED),
        "RandomUnder": RandomUnderSampler(random_state=SEED),
        "SMOTE": SMOTE(random_state=SEED, k_neighbors=5),
        "BorderlineSMOTE": BorderlineSMOTE(random_state=SEED, k_neighbors=5),
        "ADASYN": ADASYN(random_state=SEED, n_neighbors=5),
        "SMOTETomek": SMOTETomek(random_state=SEED),
        "SMOTEENN": SMOTEENN(random_state=SEED),
    }


def main():
    sp = C.load_splits(originals_only=True)
    print("split sizes (originals):", {k: (len(v[0]), int(v[1].sum())) for k, v in sp.items()},
          "(n, n_unsafe)")
    print("extracting HOG ...")
    X = {k: C.hog_features(sp[k][0]) for k in ("train", "val", "test")}
    y = {k: sp[k][1] for k in ("train", "val", "test")}

    results = []

    def evaluate(name, family, model):
        try:
            model.fit(X["train"], y["train"])
            probs = {k: (y[k], model.predict_proba(X[k])[:, 1]) for k in ("train", "val", "test")}
        except Exception as e:
            # samplers tuned for heavy imbalance can fail on near-balanced data — skip gracefully
            print(f"  {name:28s} SKIP ({type(e).__name__}: {e})")
            return
        rec = C.summarize(name, family, probs)
        results.append(rec)
        te = rec["test"]
        print(f"  {name:28s} TEST acc {te['accuracy']:.3f} rec {te['recall_unsafe']:.3f} "
              f"PR-AUC {te['pr_auc']} | val-var {rec['variance_val']:+.3f}")

    # 1) samplers x {svm, logreg}
    for ckind in ("svm", "logreg"):
        for sname, samp in samplers().items():
            steps = []
            if samp is not None:
                steps.append(("sampler", samp))
            steps += [("scaler", StandardScaler()),
                      ("pca", PCA(n_components=0.95, random_state=SEED)),
                      ("clf", clf(ckind))]
            evaluate(f"{ckind}+{sname}", f"classical/{ckind}", ImbPipeline(steps))
        # cost-sensitive (class_weight) baseline
        evaluate(f"{ckind}+class_weight", f"classical/{ckind}",
                 SkPipeline([("scaler", StandardScaler()),
                             ("pca", PCA(n_components=0.95, random_state=SEED)),
                             ("clf", clf(ckind, balanced=True))]))

    # 2) balanced ensembles (internal undersampling)
    base = [("scaler", StandardScaler()), ("pca", PCA(n_components=0.95, random_state=SEED))]
    evaluate("BalancedRandomForest", "classical/ensemble",
             SkPipeline(base + [("clf", BalancedRandomForestClassifier(
                 n_estimators=300, random_state=SEED, n_jobs=-1))]))
    evaluate("EasyEnsemble", "classical/ensemble",
             SkPipeline(base + [("clf", EasyEnsembleClassifier(
                 n_estimators=20, random_state=SEED, n_jobs=-1))]))
    evaluate("BalancedBagging", "classical/ensemble",
             SkPipeline(base + [("clf", BalancedBaggingClassifier(
                 n_estimators=30, random_state=SEED, n_jobs=-1))]))

    (C.OUT / "results_classical.json").write_text(json.dumps(results, indent=2))
    print(f"\nwrote {C.OUT/'results_classical.json'} ({len(results)} runs)")


if __name__ == "__main__":
    main()
