"""
Convert Dataset/{bus,leguna}/{positive,negative}/* to PNG (<1 MB) into
Dataset_png/{bus,legua}/{positive,negative}/<base>.png.

Local source uses `leguna/`; output uses the R2 dataset name `legua/`.
Idempotent: skips an output that already exists and is newer than its source.
Parallel: multiprocessing pool over CPU cores.

Run:
    poetry run python convert_dataset_to_png.py
or background:
    nohup poetry run python convert_dataset_to_png.py > convert.log 2>&1 &
"""

import io
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import oxipng
from PIL import Image
import pillow_heif

pillow_heif.register_heif_opener()

def _pick_quantize_method() -> Image.Quantize:
    """LIBIMAGEQUANT enum may exist but lib not compiled in. Probe at runtime."""
    probe = Image.new("RGB", (4, 4), (255, 0, 0))
    for name in ("LIBIMAGEQUANT", "FASTOCTREE", "MEDIANCUT"):
        m = getattr(Image.Quantize, name, None)
        if m is None:
            continue
        try:
            probe.quantize(colors=8, method=m)
            return m
        except Exception:
            continue
    return Image.Quantize.MEDIANCUT


QUANTIZE_METHOD = _pick_quantize_method()

ROOT     = Path(__file__).parent
SRC_DIR  = ROOT / "Dataset"
OUT_DIR  = ROOT / "Dataset_png"

# (local subdir, R2 dataset name)
DATASETS = (
    ("bus",    "bus"),
    ("leguna", "legua"),
)

MAX_BYTES    = 1_048_576
SUBFOLDERS   = ("positive", "negative")
EXT_OK       = {".heic", ".heif", ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
MAX_LONG_SIDE = 2048                # pre-cap longest side; ML rarely needs more
LOSSLESS_PIXEL_LIMIT = 600_000      # only try lossless if image small (~775x775)
MIN_DIM      = 384
COLOR_LADDER = (256, 128, 64)
DOWNSCALE    = 0.85
OXIPNG_LEVEL = 2                    # default; ~95% of L6 compression in <1/10 time


def _fit_long_side(img: Image.Image, max_side: int) -> Image.Image:
    w, h = img.size
    long_side = max(w, h)
    if long_side <= max_side:
        return img
    scale = max_side / long_side
    return img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)


def _oxipng_bytes(data: bytes) -> bytes:
    """Run oxipng. On failure log to stderr (workers run in child processes;
    a shared counter would require IPC). A misconfigured oxipng surfaces as
    repeated WARN lines instead of silently shipping ~3x larger files."""
    try:
        return oxipng.optimize_from_memory(
            data,
            level=OXIPNG_LEVEL,
            strip=oxipng.StripChunks.safe(),
        )
    except Exception as e:
        print(f"WARN: oxipng failed for {len(data)}B input: {e}", file=sys.stderr)
        return data


def _encode_lossless(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True, compress_level=9)
    return _oxipng_bytes(buf.getvalue())


def _encode_palette(img: Image.Image, colors: int) -> bytes:
    pal = img.quantize(colors=colors, method=QUANTIZE_METHOD)
    buf = io.BytesIO()
    pal.save(buf, format="PNG", optimize=True, compress_level=9)
    return _oxipng_bytes(buf.getvalue())


def encode_png_under_limit(img: Image.Image, limit: int) -> bytes:
    """
    PNG <= limit bytes. Strategy:
      0) pre-cap longest side to MAX_LONG_SIDE
      1) only try lossless if pixel count small enough to plausibly fit
      2) palette quantize 256/128/64 colors
      3) downscale 0.85x, repeat (2)
    Raise ValueError if cannot fit before MIN_DIM.
    """
    cur = _fit_long_side(img.convert("RGB"), MAX_LONG_SIDE)
    best = b""

    if cur.width * cur.height <= LOSSLESS_PIXEL_LIMIT:
        data = _encode_lossless(cur)
        if len(data) <= limit:
            return data
        best = data

    while True:
        for colors in COLOR_LADDER:
            data = _encode_palette(cur, colors)
            if not best or len(data) < len(best):
                best = data
            if len(data) <= limit:
                return data

        new_w = max(1, int(cur.width * DOWNSCALE))
        new_h = max(1, int(cur.height * DOWNSCALE))
        if new_w == cur.width and new_h == cur.height:
            raise ValueError(f"cannot shrink further; best {len(best)} > {limit}")
        if new_w < MIN_DIM or new_h < MIN_DIM:
            raise ValueError(
                f"would drop below {MIN_DIM}px ({new_w}x{new_h}); best {len(best)} > {limit}"
            )
        cur = cur.resize((new_w, new_h), Image.LANCZOS)


def convert_one(src: Path, dst: Path) -> tuple[str, str, int]:
    """Returns (status, message, bytes). status in {ok, skip, fail}.

    Skip rule: dst exists, is non-empty, AND has mtime >= src. The size>0
    check guards against fresh-clone mtime resets (every file would otherwise
    be re-converted) by accepting a previously-completed write as definitive.
    """
    src_rel = src
    try:
        if dst.exists():
            dst_stat = dst.stat()
            if dst_stat.st_size > 0 and dst_stat.st_mtime >= src.stat().st_mtime:
                return ("skip", f"{src.name} -> {dst.name}", dst_stat.st_size)
        img = Image.open(src)
        img.load()
        # encode_png_under_limit guarantees <= MAX_BYTES or raises ValueError;
        # the outer except converts that to a fail entry so partial writes
        # never reach disk (atomic .tmp rename below).
        data = encode_png_under_limit(img, MAX_BYTES)
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_suffix(dst.suffix + ".tmp")
        tmp.write_bytes(data)
        os.replace(tmp, dst)
        return ("ok", f"{src.name} -> {dst.name}", len(data))
    except Exception as e:
        # Include the relative path so two same-named files in different
        # categories (e.g. bus/positive/IMG_3477 vs legua/negative/IMG_3477)
        # are debuggable from the log.
        try:
            src_rel = src.relative_to(SRC_DIR)
        except ValueError:
            pass
        return ("fail", f"{src_rel}: {e}", 0)


def collect_jobs() -> list[tuple[Path, Path]]:
    jobs: list[tuple[Path, Path]] = []
    for local_name, dataset in DATASETS:
        for sub in SUBFOLDERS:
            folder = SRC_DIR / local_name / sub
            if not folder.is_dir():
                print(f"WARN: missing folder: {folder}")
                continue
            for p in sorted(folder.iterdir()):
                if not p.is_file() or p.suffix.lower() not in EXT_OK:
                    continue
                dst = OUT_DIR / dataset / sub / f"{p.stem}.png"
                jobs.append((p, dst))
    return jobs


def main() -> int:
    jobs = collect_jobs()
    print(f"Source : {SRC_DIR}")
    print(f"Output : {OUT_DIR}")
    print(f"Files  : {len(jobs)}")
    print()

    workers = max(1, (os.cpu_count() or 2))
    counters = {"ok": 0, "skip": 0, "fail": 0}
    total_bytes = 0

    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(convert_one, src, dst): (src, dst) for src, dst in jobs}
        for fut in as_completed(futs):
            status, msg, n = fut.result()
            counters[status] += 1
            if status == "ok":
                total_bytes += n
                print(f"  OK    {msg}  ({n/1024:.0f} KB)")
            elif status == "skip":
                print(f"  SKIP  {msg}")
            else:
                print(f"  FAIL  {msg}")

    print()
    print(f"OK     : {counters['ok']}  ({total_bytes/1024/1024:.1f} MB written)")
    print(f"Skip   : {counters['skip']}")
    print(f"Fail   : {counters['fail']}")
    return 0 if counters["fail"] == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
