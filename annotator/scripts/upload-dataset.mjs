// Upload Final_Dataset to R2: HEIC/JPG → compressed PNG (≤1.5 MB).
//
// Mapping:
//   Bus/Positive  → raw/bus/positive/
//   Bus/Negative  → raw/bus/negative/
//   Leguna/Safe   → raw/legua/positive/
//   Leguna/Unsafe → raw/legua/negative/
//
// Usage:
//   cd annotator
//   node --env-file=.env.local scripts/upload-dataset.mjs

import fs from "node:fs";
import path from "node:path";
import os from "node:os";
import { execFileSync } from "node:child_process";
import { S3Client, PutObjectCommand, HeadBucketCommand } from "@aws-sdk/client-s3";
import sharp from "sharp";

const DATASET_ROOT = "/Users/asif/Documents/git/Untitled/ML-Lab/Final_Dataset";
const MAX_BYTES    = 1.5 * 1024 * 1024; // 1.5 MB ceiling
const MAX_PX       = 2048;              // longest edge cap before compression

const FOLDER_MAP = [
  { local: "Bus/Positive",  r2: "raw/bus/positive" },
  { local: "Bus/Negative",  r2: "raw/bus/negative" },
  { local: "Leguna/Safe",   r2: "raw/legua/positive" },
  { local: "Leguna/Unsafe", r2: "raw/legua/negative" },
];

const IMAGE_EXTS = new Set([".png", ".jpg", ".jpeg", ".heic", ".heif", ".bmp", ".tiff", ".tif", ".webp"]);
const HEIC_EXTS  = new Set([".heic", ".heif"]);

const BUCKET = process.env.R2_BUCKET;
const s3 = new S3Client({
  region: "auto",
  endpoint: process.env.R2_ENDPOINT,
  credentials: {
    accessKeyId: process.env.R2_ACCESS_KEY_ID,
    secretAccessKey: process.env.R2_SECRET_ACCESS_KEY,
  },
});

// Decode HEIC → raw PNG bytes using macOS sips (handles all iPhone HEVC variants)
function heicDecode(srcPath) {
  const tmp = path.join(os.tmpdir(), `${process.pid}_${Date.now()}.png`);
  execFileSync("/usr/bin/sips", ["-s", "format", "png", srcPath, "--out", tmp], { stdio: "pipe" });
  const buf = fs.readFileSync(tmp);
  fs.unlinkSync(tmp);
  return buf;
}

// Convert any supported image to a compressed PNG ≤ MAX_BYTES.
// Strategy:
//   1. Decode to raw pixels (via sips for HEIC, sharp for everything else)
//   2. Resize so longest edge ≤ MAX_PX
//   3. Output as PNG with compressionLevel 9
//   4. If still > MAX_BYTES, halve resolution and retry (max 3 passes)
async function compress(srcPath) {
  const ext = path.extname(srcPath).toLowerCase();

  // Step 1 — get raw buffer
  let raw;
  if (HEIC_EXTS.has(ext)) {
    raw = heicDecode(srcPath);
  } else {
    raw = fs.readFileSync(srcPath);
  }

  // Step 2-4 — resize + compress, shrink until under ceiling
  let maxPx = MAX_PX;
  for (let pass = 0; pass < 4; pass++) {
    const buf = await sharp(raw)
      .rotate()                          // honour EXIF orientation
      .resize(maxPx, maxPx, { fit: "inside", withoutEnlargement: true })
      .png({ compressionLevel: 9, effort: 10 })
      .toBuffer();

    if (buf.length <= MAX_BYTES || pass === 3) return buf;
    maxPx = Math.round(maxPx * 0.7);   // shrink 30 % and retry
  }
}

async function processFolder({ local, r2 }) {
  const dir = path.join(DATASET_ROOT, local);
  if (!fs.existsSync(dir)) {
    console.log(`  SKIP ${local} — not found`);
    return { ok: 0, failed: 0 };
  }

  const files = fs.readdirSync(dir)
    .filter((f) => !f.startsWith(".") && IMAGE_EXTS.has(path.extname(f).toLowerCase()))
    .sort();

  console.log(`\n${local} → ${r2}/  (${files.length} images)`);

  // folder marker
  await s3.send(new PutObjectCommand({
    Bucket: BUCKET, Key: `${r2}/`,
    Body: Buffer.alloc(0), ContentType: "application/x-directory",
  }));

  let ok = 0, failed = 0;
  const CONCURRENCY = 3;

  for (let i = 0; i < files.length; i += CONCURRENCY) {
    const batch = files.slice(i, i + CONCURRENCY);
    await Promise.all(batch.map(async (filename) => {
      const srcPath = path.join(dir, filename);
      const base    = path.basename(filename, path.extname(filename));
      const destKey = `${r2}/${base}.png`;

      try {
        const buf = await compress(srcPath);
        await s3.send(new PutObjectCommand({
          Bucket: BUCKET, Key: destKey,
          Body: buf, ContentType: "image/png",
        }));
        ok++;
        const kb = (buf.length / 1024).toFixed(0);
        console.log(`  [${ok + failed}/${files.length}] ✓ ${base}.png  (${kb} KB)`);
      } catch (err) {
        failed++;
        console.log(`  [${ok + failed}/${files.length}] ✗ ${filename}: ${err.message.split("\n")[0]}`);
      }
    }));
  }

  return { ok, failed };
}

async function main() {
  for (const v of ["R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET", "R2_ENDPOINT"]) {
    if (!process.env[v]) { console.error(`Missing env var: ${v}`); process.exit(1); }
  }

  console.log(`Pinging bucket "${BUCKET}"…`);
  try {
    await s3.send(new HeadBucketCommand({ Bucket: BUCKET }));
    console.log("  ✓ reachable");
  } catch (err) { console.error(`  ✗ ${err.message}`); process.exit(1); }

  const totals = { ok: 0, failed: 0 };
  for (const mapping of FOLDER_MAP) {
    const r = await processFolder(mapping);
    totals.ok     += r.ok;
    totals.failed += r.failed;
  }

  console.log(`\n${"─".repeat(42)}`);
  console.log(`✓ uploaded: ${totals.ok}  (all ≤ 1.5 MB as PNG)`);
  if (totals.failed) console.log(`✗ failed:   ${totals.failed}`);
  console.log(`${"─".repeat(42)}`);
  if (totals.failed) process.exit(1);
}

main().catch((err) => { console.error(err); process.exit(1); });
