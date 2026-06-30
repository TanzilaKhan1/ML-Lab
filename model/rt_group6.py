"""Build the 6-person error-analysis workbook from VAL + TEST inference.

- Uses the already-computed val+test predictions of all 9 models
  (outputs_final/probs_cache.npz; prediction = P(unsafe) >= that model's
  tuned BALANCED threshold).
- Splits the 130 val+test images evenly across 6 people (round-robin over a
  category-sorted list so everyone gets a balanced mix of val/test x
  vehicle x class).
- One sheet per person, IDENTICAL humanized columns, an embedded thumbnail per
  row, wrong-model cells shaded red / correct green, and a blank Reasoning column.

Writes: repo/misclassified_6sheets.xlsx
"""
from __future__ import annotations
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image as PILImage
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill, Border, Side
from openpyxl.utils import get_column_letter
from openpyxl.drawing.image import Image as XLImage

import du

REPO = du.ROOT.parent
OUTF = du.ROOT / "outputs_final"
THUMBS = du.ROOT / "outputs_final" / "_thumbs"
MET = json.loads((OUTF / "metrics_full.json").read_text())["models"]
Z = np.load(OUTF / "probs_cache.npz")
META = json.loads((OUTF / "probs_cache_meta.json").read_text())

MEMBERS = ["asif", "tanzila", "taif", "amio", "tazkia", "walid"]
# (column header, display model name, npz key)
MODELS = [("Ensemble", "Ensemble (best)", "ensemble"), ("ResNet50", "ResNet50", "resnet50"),
          ("ConvNeXt-Tiny", "ConvNeXt-Tiny", "convnext"), ("EfficientNet-B0", "EfficientNet-B0", "efficientnet"),
          ("ResNet18", "ResNet18", "resnet18"), ("CNN", "CNN", "cnn"),
          ("SVM", "SVM (RBF)", "svm"), ("Logistic Regression", "Logistic Regression", "logreg"),
          ("Naive Bayes", "Naive Bayes", "nb")]
VEH = {"bus": "Bus", "legua": "Leguna"}
SPLIT_NAME = {"val": "Validation", "test": "Test"}


def main():
    THUMBS.mkdir(parents=True, exist_ok=True)
    thr = {h: MET[full]["operating_thresholds"]["threshold_balanced"] for h, full, _ in MODELS}

    # assemble every val+test image as a record
    recs = []
    for sp in ("val", "test"):
        paths = META["splits"][sp]["paths"]; y = META["splits"][sp]["y"]
        for i, p in enumerate(paths):
            pth = Path(p)
            stem = pth.stem
            true = "Unsafe" if y[i] == 1 else "Safe"
            preds, wrong = {}, []
            for h, full, key in MODELS:
                pu = float(Z[f"p_{key}_{sp}"][i])
                pred = "Unsafe" if pu >= thr[h] else "Safe"
                preds[h] = pred
                if pred != true:
                    wrong.append(h)
            recs.append({"stem": stem, "vehicle": VEH.get(pth.parent.parent.name, pth.parent.parent.name),
                         "split": SPLIT_NAME[sp], "true": true, "preds": preds,
                         "wrong": wrong, "path": p,
                         "imgnum": int("".join(ch for ch in stem if ch.isdigit()) or 0)})

    # balanced round-robin split across 6 people
    recs.sort(key=lambda r: (r["split"], r["vehicle"], r["true"], r["imgnum"]))
    groups = {m: [] for m in MEMBERS}
    for i, r in enumerate(recs):
        groups[MEMBERS[i % len(MEMBERS)]].append(r)

    # styles
    hdr_fill = PatternFill("solid", fgColor="7B1E1E"); hdr_font = Font(bold=True, color="FFFFFF", size=11)
    wrong_fill = PatternFill("solid", fgColor="F6C6C0"); right_fill = PatternFill("solid", fgColor="CFE8CF")
    safe_font = Font(color="1B5E20", bold=True); unsafe_font = Font(color="8C1D0B", bold=True)
    center = Alignment(horizontal="center", vertical="center", wrap_text=True)
    left = Alignment(horizontal="left", vertical="center", wrap_text=True)
    thin = Side(style="thin", color="E0CDB4"); border = Border(thin, thin, thin, thin)

    cols = ["Photo", "Image", "Vehicle", "Dataset", "True Label"] + [h for h, _, _ in MODELS] + \
           ["Models That Got It Wrong", "# Wrong", "Reasoning / Notes"]

    wb = Workbook(); wb.remove(wb.active)
    for name in MEMBERS:
        ws = wb.create_sheet(name.capitalize())
        ws.append(cols)
        for c in range(1, len(cols) + 1):
            cell = ws.cell(row=1, column=c); cell.fill = hdr_fill; cell.font = hdr_font
            cell.alignment = center; cell.border = border
        ws.row_dimensions[1].height = 30
        for r in groups[name]:
            row = ["", f"{r['stem']}.png", r["vehicle"], r["split"], r["true"]] + \
                  [r["preds"][h] for h, _, _ in MODELS] + \
                  [", ".join(r["wrong"]) if r["wrong"] else "— all correct —", len(r["wrong"]), ""]
            ws.append(row)
            ridx = ws.max_row
            ws.row_dimensions[ridx].height = 70
            # embed thumbnail
            try:
                tp = THUMBS / f"{r['split']}_{r['vehicle']}_{r['stem']}.png"
                if not tp.exists():
                    im = PILImage.open(r["path"]).convert("RGB"); im.thumbnail((130, 130))
                    im.save(tp)
                xim = XLImage(str(tp)); xim.width, xim.height = 92, 92
                ws.add_image(xim, f"A{ridx}")
            except Exception as e:
                ws.cell(row=ridx, column=1, value="(no image)")
            # formatting + coloring
            for c in range(1, len(cols) + 1):
                cell = ws.cell(row=ridx, column=c); cell.border = border
                cell.alignment = left if c == len(cols) else center
            tl = ws.cell(row=ridx, column=5); tl.font = unsafe_font if r["true"] == "Unsafe" else safe_font
            for j, (h, _, _) in enumerate(MODELS):
                cell = ws.cell(row=ridx, column=6 + j)
                pred = r["preds"][h]
                cell.fill = wrong_fill if pred != r["true"] else right_fill
                cell.font = unsafe_font if pred == "Unsafe" else safe_font
        # widths + freeze header + autofilter
        widths = [15, 15, 10, 12, 11] + [13] * len(MODELS) + [26, 9, 40]
        for c, wdt in enumerate(widths, 1):
            ws.column_dimensions[get_column_letter(c)].width = wdt
        ws.freeze_panes = "B2"
        ws.auto_filter.ref = f"B1:{get_column_letter(len(cols))}1"

    out = REPO / "misclassified_6sheets.xlsx"
    wb.save(out)

    print(f"wrote {out.name}  ({len(wb.sheetnames)} person-sheets, identical columns)")
    print(f"pool = {len(recs)} val+test images  ·  inference = balanced threshold per model")
    print("\nper-person assignment:")
    for name in MEMBERS:
        g = groups[name]
        c = Counter(r["true"] for r in g); s = Counter(r["split"] for r in g)
        anyerr = sum(1 for r in g if r["wrong"])
        print(f"  {name:8s}: {len(g):2d} imgs | Safe {c['Safe']:2d} Unsafe {c['Unsafe']:2d} | "
              f"Val {s['Validation']:2d} Test {s['Test']:2d} | rows with >=1 wrong: {anyerr}")


if __name__ == "__main__":
    main()
