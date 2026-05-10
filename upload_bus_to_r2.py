"""
One-shot uploader: convert Dataset/{bus,leguna}/{positive,negative}/* to PNG
(<1 MB each) and upload to
  s3://machine-learning/raw/bus/{positive,negative}/<base>.png
  s3://machine-learning/raw/legua/{positive,negative}/<base>.png

Note: local source folder is `leguna/` but the R2 dataset prefix is `legua/`
(matches the existing UPLOAD_FOLDERS list in annotator/lib/storage.ts).

Reads R2 credentials from annotator/.env.local. Does not modify any storage
code or any object that already exists in the bucket apart from the new keys
this script writes.
"""

import io
import os
import sys
from pathlib import Path

import boto3
from PIL import Image
import pillow_heif

pillow_heif.register_heif_opener()

ROOT      = Path(__file__).parent
SRC_DIR   = ROOT / "Dataset"
ENV_FILE  = ROOT / "annotator" / ".env.local"

# (local subdir under ml/, R2 dataset name)
DATASETS = (
    ("bus",    "bus"),
    ("leguna", "legua"),
)

MAX_BYTES = 1_048_576           # strict 1 MB
SUBFOLDERS = ("positive", "negative")
EXT_OK = {".heic", ".heif", ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def load_env(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip()
    return env


def encode_png_under_limit(img: Image.Image, limit: int) -> bytes:
    """Encode as PNG, downscaling by 0.85x until size < limit. RGB to drop alpha."""
    img = img.convert("RGB")
    cur = img
    while True:
        buf = io.BytesIO()
        cur.save(buf, format="PNG", optimize=True)
        data = buf.getvalue()
        if len(data) < limit:
            return data
        new_w = max(1, int(cur.width * 0.85))
        new_h = max(1, int(cur.height * 0.85))
        if new_w == cur.width and new_h == cur.height:
            return data       # cannot shrink further
        if new_w < 64 or new_h < 64:
            return data       # don't go absurdly small
        cur = cur.resize((new_w, new_h), Image.LANCZOS)


def main() -> int:
    if not ENV_FILE.exists():
        print(f"ERROR: env file not found: {ENV_FILE}", file=sys.stderr)
        return 1
    env = load_env(ENV_FILE)
    required = ["R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_ENDPOINT", "R2_BUCKET"]
    missing = [k for k in required if not env.get(k)]
    if missing:
        print(f"ERROR: missing env vars in {ENV_FILE}: {missing}", file=sys.stderr)
        return 1

    bucket   = env["R2_BUCKET"]
    prefix   = env.get("R2_PREFIX_RAW", "raw")
    endpoint = env["R2_ENDPOINT"]

    print(f"R2 bucket   : {bucket}")
    print(f"R2 endpoint : {endpoint}")
    print(f"Raw prefix  : {prefix}")
    print(f"Source dir  : {SRC_DIR}")
    print()

    s3 = boto3.client(
        "s3",
        region_name="auto",
        endpoint_url=endpoint,
        aws_access_key_id=env["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=env["R2_SECRET_ACCESS_KEY"],
    )

    counters = {"uploaded": 0, "skipped": 0, "failed": 0}

    for local_name, dataset in DATASETS:
        for sub in SUBFOLDERS:
            folder = SRC_DIR / local_name / sub
            if not folder.is_dir():
                print(f"WARN: folder missing: {folder}")
                continue
            files = sorted(p for p in folder.iterdir()
                           if p.is_file() and p.suffix.lower() in EXT_OK)
            print(f"[{dataset}/{sub}] {len(files)} candidate files (from {local_name}/{sub})")

            for path in files:
                base = path.stem               # IMG_3477
                key  = f"{prefix}/{dataset}/{sub}/{base}.png"

                try:
                    img = Image.open(path)
                    # Force decode now so HEIC/EXIF errors surface here
                    img.load()
                except Exception as e:
                    print(f"  FAIL  decode {path.name}: {e}")
                    counters["failed"] += 1
                    continue

                try:
                    data = encode_png_under_limit(img, MAX_BYTES)
                except Exception as e:
                    print(f"  FAIL  encode {path.name}: {e}")
                    counters["failed"] += 1
                    continue

                if len(data) >= MAX_BYTES:
                    print(f"  WARN  {path.name}: still {len(data)} bytes after downscale")

                try:
                    s3.put_object(
                        Bucket=bucket,
                        Key=key,
                        Body=data,
                        ContentType="image/png",
                    )
                except Exception as e:
                    print(f"  FAIL  upload {key}: {e}")
                    counters["failed"] += 1
                    continue

                print(f"  OK    {path.name:24s} -> {key}  ({len(data)/1024:.0f} KB)")
                counters["uploaded"] += 1

    print()
    print(f"Uploaded : {counters['uploaded']}")
    print(f"Failed   : {counters['failed']}")
    return 0 if counters["failed"] == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
