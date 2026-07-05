"""Cross-validated embedding-space sampling — the rigorous validation.

Kills the noisy n=65 single-split caveat: 5-fold stratified CV over TRAIN+VAL
(367 imgs, 119 unsafe) with the sampler INSIDE each fold (cross_val_predict ->
no leakage). The OOF probabilities give reliable AUC/PR-AUC and an honestly-tuned
threshold, which is then checked once on the untouched TEST set.

Feature sets: frozen embeddings of resnet50 / convnext_tiny / efficientnet_b0,
plus their CONCATENATION. Samplers: none, SMOTE, SVMSMOTE, ADASYN, SMOTEENN,
class_weight. Classifiers: SVM, LogReg.

Run:  TORCH_HOME=<proj>/.torch_cache python run_cv_embed.py
Writes outputs/results_cv_embed.json
"""
from __future__ import annotations
import json, warnings
import numpy as np
import torch, torch.nn as nn
from torchvision import models as tvm, transforms
from PIL import Image, ImageFile

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline as SkPipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE, SVMSMOTE, ADASYN
from imblearn.combine import SMOTEENN

import common as C

warnings.filterwarnings("ignore")
ImageFile.LOAD_TRUNCATED_IMAGES = True
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
IMG, SEED = 224, C.SEED
BACKBONES = ["resnet50", "convnext_tiny", "efficientnet_b0"]


def embedder(bb):
    if bb == "resnet50":
        m = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2); m.fc = nn.Identity()
    elif bb == "convnext_tiny":
        m = tvm.convnext_tiny(weights=tvm.ConvNeXt_Tiny_Weights.IMAGENET1K_V1); m.classifier[2] = nn.Identity()
    elif bb == "efficientnet_b0":
        m = tvm.efficientnet_b0(weights=tvm.EfficientNet_B0_Weights.IMAGENET1K_V1); m.classifier = nn.Identity()
    return m.eval().to(DEV)


@torch.no_grad()
def extract(model, paths):
    tf = transforms.Compose([transforms.Resize(int(IMG * 1.15)), transforms.CenterCrop(IMG),
                             transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    out = []
    for i in range(0, len(paths), 32):
        b = torch.stack([tf(Image.open(p).convert("RGB")) for p in paths[i:i + 32]]).to(DEV)
        out.append(model(b).cpu().numpy())
    return np.concatenate(out).astype(np.float32)


def clf(kind, balanced=False):
    cw = "balanced" if balanced else None
    return (SVC(kernel="rbf", gamma="scale", probability=True, class_weight=cw, random_state=SEED)
            if kind == "svm" else
            LogisticRegression(max_iter=5000, class_weight=cw, random_state=SEED))


def samplers():
    return {"none": None, "SMOTE": SMOTE(random_state=SEED), "SVMSMOTE": SVMSMOTE(random_state=SEED),
            "ADASYN": ADASYN(random_state=SEED), "SMOTEENN": SMOTEENN(random_state=SEED)}


def main():
    sp = C.load_splits(originals_only=True)
    Xtv_paths = sp["train"][0] + sp["val"][0]
    ytv = np.concatenate([sp["train"][1], sp["val"][1]])
    Xte_paths, yte = sp["test"][0], sp["test"][1]
    print(f"CV pool (train+val): {len(ytv)} (unsafe {int(ytv.sum())}) | test {len(yte)} (unsafe {int(yte.sum())})")

    emb_tv, emb_te = {}, {}
    for bb in BACKBONES:
        print(f"  extracting {bb} ...")
        m = embedder(bb); emb_tv[bb] = extract(m, Xtv_paths); emb_te[bb] = extract(m, Xte_paths)
        del m; torch.cuda.empty_cache()
    emb_tv["concat"] = np.concatenate([emb_tv[b] for b in BACKBONES], axis=1)
    emb_te["concat"] = np.concatenate([emb_te[b] for b in BACKBONES], axis=1)
    feature_sets = BACKBONES + ["concat"]
    print("  feature dims:", {f: emb_tv[f].shape[1] for f in feature_sets})

    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    results = []

    def run(fs, ckind, sname, samp):
        Xtv, Xte = emb_tv[fs], emb_te[fs]
        balanced = (sname == "class_weight")
        steps = ([("sampler", samp)] if samp is not None else []) + \
                [("scaler", StandardScaler()), ("clf", clf(ckind, balanced))]
        pipe = ImbPipeline(steps) if samp is not None else SkPipeline(steps)
        try:
            oof = cross_val_predict(pipe, Xtv, ytv, cv=skf, method="predict_proba")[:, 1]
        except Exception as e:
            # samplers tuned for heavy imbalance can fail on near-balanced data — skip gracefully
            print(f"  {fs}:{ckind}+{sname:20s} SKIP ({type(e).__name__}: {e})")
            return
        thr = C.tune_threshold(oof, ytv, "balanced")
        cv = C.metric_block(ytv, oof, thr)            # OOF (reliable) metrics
        pipe.fit(Xtv, ytv)
        te = C.metric_block(yte, pipe.predict_proba(Xte)[:, 1], thr)
        rec = {"name": f"{fs}:{ckind}+{sname}", "feature_set": fs, "clf": ckind, "sampler": sname,
               "threshold": round(thr, 4), "cv": cv, "test": te}
        results.append(rec)
        print(f"  {rec['name']:34s} CV: AUC {cv['roc_auc']:.3f} PR {cv['pr_auc']:.3f} "
              f"rec {cv['recall_unsafe']:.3f} acc {cv['accuracy']:.3f} | TEST AUC {te['roc_auc']:.3f} "
              f"rec {te['recall_unsafe']:.3f} acc {te['accuracy']:.3f}")

    for fs in feature_sets:
        for ckind in ("svm", "logreg"):
            for sname, samp in samplers().items():
                run(fs, ckind, sname, samp)
            run(fs, ckind, "class_weight", None)

    results.sort(key=lambda r: -r["cv"]["pr_auc"])
    (C.OUT / "results_cv_embed.json").write_text(json.dumps(results, indent=2))
    print(f"\nwrote {C.OUT/'results_cv_embed.json'} ({len(results)} configs)")
    print("\n=== TOP 8 by CV PR-AUC ===")
    for r in results[:8]:
        cv, te = r["cv"], r["test"]
        print(f"  {r['name']:34s} CVrec {cv['recall_unsafe']:.3f} CVprauc {cv['pr_auc']:.3f} "
              f"CVauc {cv['roc_auc']:.3f} | TESTrec {te['recall_unsafe']:.3f} TESTauc {te['roc_auc']:.3f}")


if __name__ == "__main__":
    main()
