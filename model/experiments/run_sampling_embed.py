"""Feature-space (embedding) sampling — the research-correct way to apply
SMOTE-family oversampling to images for deep models (DeepSMOTE-style).

Extract FROZEN pretrained-backbone embeddings for train/val/test, then run the
full imblearn sampler matrix on the embeddings via a leak-free Pipeline (sampler
touches TRAIN only). This tests whether SMOTE/ADASYN/etc. help on deep features
(vs the naive image-level random over/under in run_deep.py).

Usage:  TORCH_HOME=<proj>/.torch_cache python run_sampling_embed.py [backbone]
        backbone in {convnext_tiny, resnet50, resnet18}; default convnext_tiny
Writes outputs/results_embed_<backbone>.json
"""
from __future__ import annotations
import json, sys, warnings
import numpy as np
import torch
import torch.nn as nn
from torchvision import models as tvm, transforms
from PIL import Image, ImageFile

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.pipeline import Pipeline as SkPipeline
from imblearn.over_sampling import (RandomOverSampler, SMOTE, BorderlineSMOTE,
                                    SVMSMOTE, KMeansSMOTE, ADASYN)
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek, SMOTEENN

import common as C

warnings.filterwarnings("ignore")
ImageFile.LOAD_TRUNCATED_IMAGES = True
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
IMG = 224
BACKBONE = sys.argv[1] if len(sys.argv) > 1 else "convnext_tiny"
SEED = C.SEED


def embedder(backbone):
    if backbone == "convnext_tiny":
        m = tvm.convnext_tiny(weights=tvm.ConvNeXt_Tiny_Weights.IMAGENET1K_V1); m.classifier[2] = nn.Identity()
    elif backbone == "resnet50":
        m = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2); m.fc = nn.Identity()
    elif backbone == "resnet18":
        m = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1); m.fc = nn.Identity()
    else:
        raise ValueError(backbone)
    return m.eval().to(DEV)


def eval_tf():
    return transforms.Compose([transforms.Resize(int(IMG * 1.15)), transforms.CenterCrop(IMG),
                               transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


@torch.no_grad()
def extract(model, paths):
    tf = eval_tf(); out = []
    for i in range(0, len(paths), 32):
        batch = torch.stack([tf(Image.open(p).convert("RGB")) for p in paths[i:i + 32]]).to(DEV)
        out.append(model(batch).cpu().numpy())
    return np.concatenate(out).astype(np.float32)


def samplers():
    return {
        "none": None,
        "RandomOver": RandomOverSampler(random_state=SEED),
        "RandomUnder": RandomUnderSampler(random_state=SEED),
        "SMOTE": SMOTE(random_state=SEED, k_neighbors=5),
        "BorderlineSMOTE": BorderlineSMOTE(random_state=SEED, k_neighbors=5),
        "SVMSMOTE": SVMSMOTE(random_state=SEED, k_neighbors=5),
        "KMeansSMOTE": KMeansSMOTE(random_state=SEED, k_neighbors=4,
                                   cluster_balance_threshold=0.05),
        "ADASYN": ADASYN(random_state=SEED, n_neighbors=5),
        "SMOTETomek": SMOTETomek(random_state=SEED),
        "SMOTEENN": SMOTEENN(random_state=SEED),
    }


def clf(kind, balanced=False):
    cw = "balanced" if balanced else None
    if kind == "svm":
        return SVC(C=1.0, kernel="rbf", gamma="scale", probability=True, class_weight=cw, random_state=SEED)
    return LogisticRegression(C=1.0, max_iter=5000, class_weight=cw, random_state=SEED)


def main():
    sp = C.load_splits(originals_only=True)
    print(f"backbone {BACKBONE} | extracting embeddings on {DEV} ...")
    model = embedder(BACKBONE)
    X = {k: extract(model, sp[k][0]) for k in ("train", "val", "test")}
    y = {k: sp[k][1] for k in ("train", "val", "test")}
    print(f"embedding dim: {X['train'].shape[1]} | train {X['train'].shape[0]} (unsafe {int(y['train'].sum())})")

    results = []

    def evaluate(name, model_):
        try:
            model_.fit(X["train"], y["train"])
        except Exception as e:
            print(f"  {name:28s} SKIP ({type(e).__name__})"); return
        probs = {k: (y[k], model_.predict_proba(X[k])[:, 1]) for k in ("train", "val", "test")}
        rec = C.summarize(name, f"embed/{BACKBONE}", probs)
        results.append(rec); te = rec["test"]
        print(f"  {name:28s} TEST acc {te['accuracy']:.3f} rec {te['recall_unsafe']:.3f} "
              f"PR-AUC {te['pr_auc']} AUC {te['roc_auc']} gap {rec['gap_train_test']:+.3f}")

    for ckind in ("logreg", "svm"):
        for sname, samp in samplers().items():
            steps = ([("sampler", samp)] if samp is not None else []) + \
                    [("scaler", StandardScaler()), ("clf", clf(ckind))]
            evaluate(f"{ckind}+{sname}", ImbPipeline(steps))
        evaluate(f"{ckind}+class_weight",
                 SkPipeline([("scaler", StandardScaler()), ("clf", clf(ckind, balanced=True))]))

    out = C.OUT / f"results_embed_{BACKBONE}.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out} ({len(results)} runs)")


if __name__ == "__main__":
    main()
