"""
Upload pre-converted PNGs from Dataset_png/{bus,legua}/{positive,negative}/*.png
to R2 at:
  s3://<R2_BUCKET>/<R2_PREFIX_RAW>/{bus,legua}/{positive,negative}/<base>.png

Run convert_dataset_to_png.py FIRST. Reads R2 credentials from
annotator/.env.local. Idempotent at object level (overwrites same key with
same content; Content-Length matches).

Run:
    poetry run python upload_bus_to_r2.py
or background:
    nohup poetry run python upload_bus_to_r2.py > upload.log 2>&1 &
"""

import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import boto3
from botocore.config import Config

ROOT     = Path(__file__).parent
SRC_DIR  = ROOT / "Dataset_png"
ENV_FILE = ROOT / "annotator" / ".env.local"

DATASETS   = ("bus", "legua")
SUBFOLDERS = ("positive", "negative")
WORKERS    = 8
MAX_BYTES  = 1_048_576           # strict 1 MB — must match convert step


def load_env(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip()
    return env


def make_s3(env: dict[str, str]):
    # Single retry layer: boto3's adaptive mode already handles transient
    # 5xx/429 with sane backoff. We add an outer app-level retry for the
    # specific SSL/connection-reset failures we observed against R2 that
    # boto3 does not retry on its own. To avoid 8x4=32-attempt amplification
    # on a real outage, the boto3 layer is set to a small max_attempts.
    cfg = Config(
        retries={"max_attempts": 3, "mode": "adaptive"},
        connect_timeout=15,
        read_timeout=120,
        max_pool_connections=WORKERS,
    )
    return boto3.client(
        "s3",
        region_name="auto",
        endpoint_url=env["R2_ENDPOINT"],
        aws_access_key_id=env["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=env["R2_SECRET_ACCESS_KEY"],
        config=cfg,
    )


def collect_jobs(prefix: str) -> list[tuple[Path, str]]:
    jobs: list[tuple[Path, str]] = []
    for dataset in DATASETS:
        for sub in SUBFOLDERS:
            folder = SRC_DIR / dataset / sub
            if not folder.is_dir():
                print(f"WARN: missing folder: {folder}")
                continue
            for p in sorted(folder.iterdir()):
                if p.suffix.lower() != ".png":
                    continue
                key = f"{prefix}/{dataset}/{sub}/{p.name}"
                jobs.append((p, key))
    return jobs


def upload_one(s3, bucket: str, path: Path, key: str) -> tuple[str, str, int]:
    # Guard the local read so a deleted/permission-flipped file produces a
    # per-file FAIL row instead of crashing the worker thread (which would
    # surface as fut.result() raising and terminating the entire upload).
    try:
        data = path.read_bytes()
    except OSError as e:
        return ("fail", f"{key}: read error: {e}", 0)
    if len(data) > MAX_BYTES:
        return ("fail", f"{key}: local file {len(data)} > {MAX_BYTES} (re-run convert)", 0)
    last_err: Exception | None = None
    for attempt in range(1, 5):
        try:
            s3.put_object(Bucket=bucket, Key=key, Body=data, ContentType="image/png")
            tag = "" if attempt == 1 else f" (attempt {attempt})"
            return ("ok", f"{path.name} -> {key}{tag}", len(data))
        except Exception as e:
            last_err = e
            # Log every retry so the failure mode (creds vs throttling vs
            # network) is debuggable from the run log, not just the final one.
            if attempt < 4:
                print(f"  RETRY {key} attempt {attempt}: {e}", file=sys.stderr)
            # Exponential backoff with jitter — the jitter prevents all 8
            # worker threads from retrying in lockstep on a transient outage.
            base = 0.5 * (2 ** (attempt - 1))         # 0.5, 1, 2, 4s
            time.sleep(base + random.uniform(0, 0.5))
    return ("fail", f"{key}: {last_err}", 0)


def main() -> int:
    if not ENV_FILE.exists():
        print(f"ERROR: env file not found: {ENV_FILE}", file=sys.stderr)
        return 1
    if not SRC_DIR.is_dir():
        print(f"ERROR: source dir not found: {SRC_DIR}. "
              f"Run convert_dataset_to_png.py first.", file=sys.stderr)
        return 1

    env = load_env(ENV_FILE)
    required = ["R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_ENDPOINT", "R2_BUCKET"]
    missing = [k for k in required if not env.get(k)]
    if missing:
        print(f"ERROR: missing env vars in {ENV_FILE}: {missing}", file=sys.stderr)
        return 1

    bucket = env["R2_BUCKET"]
    prefix = env.get("R2_PREFIX_RAW", "raw")

    print(f"R2 bucket   : {bucket}")
    print(f"R2 endpoint : {env['R2_ENDPOINT']}")
    print(f"Raw prefix  : {prefix}")
    print(f"Source dir  : {SRC_DIR}")
    print()

    s3   = make_s3(env)
    jobs = collect_jobs(prefix)
    print(f"Files to upload: {len(jobs)}")
    print()

    counters = {"ok": 0, "fail": 0}
    total_bytes = 0

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = [ex.submit(upload_one, s3, bucket, p, k) for p, k in jobs]
        for fut in as_completed(futs):
            status, msg, n = fut.result()
            counters[status] += 1
            if status == "ok":
                total_bytes += n
                print(f"  OK    {msg}  ({n/1024:.0f} KB)")
            else:
                print(f"  FAIL  {msg}")

    print()
    print(f"Uploaded : {counters['ok']}  ({total_bytes/1024/1024:.1f} MB)")
    print(f"Failed   : {counters['fail']}")
    return 0 if counters["fail"] == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
