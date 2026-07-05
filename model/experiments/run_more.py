"""Phase-2 testing on the validated winner (ConvNeXt-Tiny frozen embeddings):
  1) smote_variants benchmark leaders (polynom-fit-SMOTE, ProWSyn, SMOTE-IPF, G-SMOTE, Lee)
  2) embedding-TTA (avg of center + hflip views)
  3) repeated-seed CV (error bars) on the top recipe
  4) multi-backbone soft-vote ensemble (convnext + resnet50 + efficientnet)

All 5-fold OOF (leak-free) on TRAIN+VAL (367), final check on TEST (65).
Run:  TORCH_HOME=<proj>/.torch_cache python run_more.py
Writes outputs/results_more.json
"""
from __future__ import annotations
import json, warnings
import numpy as np
import torch, torch.nn as nn
from torchvision import models as tvm, transforms
from PIL import Image, ImageFile
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE, SVMSMOTE, ADASYN
import common as C

warnings.filterwarnings("ignore")
ImageFile.LOAD_TRUNCATED_IMAGES = True
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
IMG, SEED = 224, C.SEED


def embedder(bb):
    if bb == "resnet50":
        m = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2); m.fc = nn.Identity()
    elif bb == "convnext_tiny":
        m = tvm.convnext_tiny(weights=tvm.ConvNeXt_Tiny_Weights.IMAGENET1K_V1); m.classifier[2] = nn.Identity()
    elif bb == "efficientnet_b0":
        m = tvm.efficientnet_b0(weights=tvm.EfficientNet_B0_Weights.IMAGENET1K_V1); m.classifier = nn.Identity()
    return m.eval().to(DEV)


def _tf():
    return transforms.Compose([transforms.Resize(int(IMG * 1.15)), transforms.CenterCrop(IMG),
                               transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


@torch.no_grad()
def extract(model, paths, tta=False):
    tf = _tf(); out = []
    for i in range(0, len(paths), 32):
        ims = [tf(Image.open(p).convert("RGB")) for p in paths[i:i + 32]]
        b = torch.stack(ims).to(DEV)
        e = model(b)
        if tta:
            e = (e + model(torch.flip(b, dims=[3]))) / 2
        out.append(e.cpu().numpy())
    return np.concatenate(out).astype(np.float32)


def svm():
    return SVC(kernel="rbf", gamma="scale", probability=True, random_state=SEED)


def svm_bal():
    return SVC(kernel="rbf", gamma="scale", probability=True, class_weight="balanced", random_state=SEED)


def cv_oof(X, y, resample, make_clf, seed=SEED):
    """5-fold OOF probs. `resample(Xtr,ytr)->(Xr,yr)` or None (identity)."""
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    oof = np.zeros(len(y))
    for tr, va in skf.split(X, y):
        Xtr, ytr = X[tr], y[tr]
        if resample is not None:
            try:
                Xtr, ytr = resample(Xtr, ytr)
            except Exception:
                pass
        sc = StandardScaler().fit(Xtr)
        clf = make_clf().fit(sc.transform(Xtr), ytr)
        oof[va] = clf.predict_proba(sc.transform(X[va]))[:, 1]
    return oof


def full_test(X, y, Xte, resample, make_clf):
    Xtr, ytr = X, y
    if resample is not None:
        try:
            Xtr, ytr = resample(X, y)
        except Exception:
            pass
    sc = StandardScaler().fit(Xtr)
    clf = make_clf().fit(sc.transform(Xtr), ytr)
    return clf.predict_proba(sc.transform(Xte))[:, 1]


def record(results, name, X, y, Xte, yte, resample, make_clf):
    oof = cv_oof(X, y, resample, make_clf)
    thr = C.tune_threshold(oof, y, "balanced")
    cv = C.metric_block(y, oof, thr)
    te = C.metric_block(yte, full_test(X, y, Xte, resample, make_clf), thr)
    results.append({"name": name, "cv": cv, "test": te})
    print(f"  {name:32s} CV AUC {cv['roc_auc']:.3f} PR {cv['pr_auc']:.3f} rec {cv['recall_unsafe']:.3f} "
          f"acc {cv['accuracy']:.3f} | TEST AUC {te['roc_auc']:.3f} rec {te['recall_unsafe']:.3f}")


def main():
    sp = C.load_splits(originals_only=True)
    tvp = sp["train"][0] + sp["val"][0]; ytv = np.concatenate([sp["train"][1], sp["val"][1]])
    tep, yte = sp["test"][0], sp["test"][1]
    results = []

    print("extracting ConvNeXt embeddings (plain + TTA) ...")
    cx = embedder("convnext_tiny")
    Xcx, Xcx_te = extract(cx, tvp), extract(cx, tep)
    Xcx_tta, Xcx_te_tta = extract(cx, tvp, tta=True), extract(cx, tep, tta=True)
    del cx; torch.cuda.empty_cache()

    # ---- 1) smote_variants benchmark leaders on ConvNeXt emb (SVM) ----
    print("\n# 1) smote_variants leaders (ConvNeXt emb + SVM)")
    try:
        import smote_variants as sv
        import logging; logging.getLogger("smote_variants").setLevel(logging.ERROR)
        leaders = {"polynom_fit_SMOTE_star": sv.polynom_fit_SMOTE_star, "ProWSyn": sv.ProWSyn,
                   "SMOTE_IPF": sv.SMOTE_IPF, "G_SMOTE": sv.G_SMOTE, "Lee": sv.Lee, "SMOBD": sv.SMOBD}
        for nm, cls in leaders.items():
            def rs(Xa, ya, _cls=cls):
                return _cls(random_state=SEED).sample(Xa, ya)
            try:
                record(results, f"convnext:svm+{nm}", Xcx, ytv, Xcx_te, yte, rs, svm)
            except Exception as e:
                print(f"  {nm:32s} SKIP ({type(e).__name__}: {e})")
    except Exception as e:
        print("  smote_variants unavailable:", e)
    # imblearn references for comparison
    for nm, S in [("SVMSMOTE", SVMSMOTE), ("SMOTE", SMOTE), ("ADASYN", ADASYN)]:
        record(results, f"convnext:svm+{nm}", Xcx, ytv, Xcx_te, yte,
               (lambda Xa, ya, _S=S: _S(random_state=SEED).fit_resample(Xa, ya)), svm)
    record(results, "convnext:svm+class_weight", Xcx, ytv, Xcx_te, yte, None, svm_bal)

    # ---- 2) embedding-TTA on the top recipe ----
    print("\n# 2) embedding-TTA (center+hflip) vs plain")
    record(results, "convnext:svm+SVMSMOTE (noTTA)", Xcx, ytv, Xcx_te, yte,
           (lambda Xa, ya: SVMSMOTE(random_state=SEED).fit_resample(Xa, ya)), svm)
    record(results, "convnext:svm+SVMSMOTE (TTA)", Xcx_tta, ytv, Xcx_te_tta, yte,
           (lambda Xa, ya: SVMSMOTE(random_state=SEED).fit_resample(Xa, ya)), svm)

    # ---- 3) repeated-seed CV (error bars) on the top recipe ----
    print("\n# 3) repeated-seed CV (convnext:svm+SVMSMOTE)")
    aucs, prs, recs = [], [], []
    for s in (42, 7, 123, 2024, 1):
        oof = cv_oof(Xcx, ytv, lambda Xa, ya: SVMSMOTE(random_state=SEED).fit_resample(Xa, ya), svm, seed=s)
        thr = C.tune_threshold(oof, ytv, "balanced"); m = C.metric_block(ytv, oof, thr)
        aucs.append(m["roc_auc"]); prs.append(m["pr_auc"]); recs.append(m["recall_unsafe"])
    rep = {"name": "REPEATED-CV convnext:svm+SVMSMOTE",
           "cv_auc_mean": round(float(np.mean(aucs)), 4), "cv_auc_std": round(float(np.std(aucs)), 4),
           "cv_prauc_mean": round(float(np.mean(prs)), 4), "cv_prauc_std": round(float(np.std(prs)), 4),
           "cv_recall_mean": round(float(np.mean(recs)), 4), "cv_recall_std": round(float(np.std(recs)), 4),
           "seeds": [42, 7, 123, 2024, 1]}
    results.append(rep)
    print(f"  AUC {rep['cv_auc_mean']}±{rep['cv_auc_std']} | PR-AUC {rep['cv_prauc_mean']}±{rep['cv_prauc_std']} "
          f"| recall {rep['cv_recall_mean']}±{rep['cv_recall_std']}")

    # ---- 4) multi-backbone soft-vote ensemble ----
    print("\n# 4) soft-vote ensemble (convnext + resnet50 + efficientnet, svm+class_weight)")
    embs = {"convnext_tiny": (Xcx, Xcx_te)}
    for bb in ("resnet50", "efficientnet_b0"):
        m = embedder(bb); embs[bb] = (extract(m, tvp), extract(m, tep)); del m; torch.cuda.empty_cache()
    oof_stack, test_stack = [], []
    for bb, (Xtv, Xte) in embs.items():
        oof_stack.append(cv_oof(Xtv, ytv, None, svm_bal))
        test_stack.append(full_test(Xtv, ytv, Xte, None, svm_bal))
    oof_ens = np.mean(oof_stack, axis=0); test_ens = np.mean(test_stack, axis=0)
    thr = C.tune_threshold(oof_ens, ytv, "balanced")
    cv = C.metric_block(ytv, oof_ens, thr); te = C.metric_block(yte, test_ens, thr)
    results.append({"name": "ENSEMBLE softvote(3 backbones) svm+class_weight", "cv": cv, "test": te})
    print(f"  ENSEMBLE CV AUC {cv['roc_auc']:.3f} PR {cv['pr_auc']:.3f} rec {cv['recall_unsafe']:.3f} "
          f"acc {cv['accuracy']:.3f} | TEST AUC {te['roc_auc']:.3f} rec {te['recall_unsafe']:.3f} acc {te['accuracy']:.3f}")

    (C.OUT / "results_more.json").write_text(json.dumps(results, indent=2))
    print(f"\nwrote {C.OUT/'results_more.json'}")


if __name__ == "__main__":
    main()
