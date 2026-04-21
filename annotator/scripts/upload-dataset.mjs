// Upload Final_Dataset to R2 with HEIC/JPG → PNG conversion.
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

const DATASET_ROOT = "/Users/asif/Downloads/Final_Dataset";

const FOLDER_MAP = [
  { local: "Bus/Positive",  r2: "raw/bus/positive" },
  { local: "Bus/Negative",  r2: "raw/bus/negative" },
  { local: "Leguna/Safe",   r2: "raw/legua/positive" },
  { local: "Leguna/Unsafe", r2: "raw/legua/negative" },
];

const IMAGE_EXTS    = new Set([".png", ".jpg", ".jpeg", ".heic", ".heif", ".bmp", ".tiff", ".tif", ".webp"]);
const HEIC_EXTS     = new Set([".heic", ".heif"]);
const SHARP_EXTS    = new Set([".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"]);

const BUCKET = process.env.R2_BUCKET;
const s3 = new S3Client({
  region: "auto",
  endpoint: process.env.R2_ENDPOINT,
  credentials: {
    accessKeyId: process.env.R2_ACCESS_KEY_ID,
    secretAccessKey: process.env.R2_SECRET_ACCESS_KEY,
  },
});

// Convert HEIC → PNG using macOS sips (handles all iPhone HEVC variants)
function heicToPng(srcPath) {
  const tmp = path.join(os.tmpdir(), `${Date.now()}_${path.basename(srcPath, path.extname(srcPath))}.png`);
  execFileSync("/usr/bin/sips", ["-s", "format", "png", srcPath, "--out", tmp], { stdio: "pipe" });
  const buf = fs.readFileSync(tmp);
  fs.unlinkSync(tmp);
  return buf;
}

async function toPng(srcPath) {
  const ext = path.extname(srcPath).toLowerCase();
  if (HEIC_EXTS.has(ext)) {
    return heicToPng(srcPath);
  }
  if (SHARP_EXTS.has(ext)) {
    return Buffer.from(await sharp(srcPath).png({ compressionLevel: 8 }).toBuffer());
  }
  // Already PNG
  return fs.readFileSync(srcPath);
}

async function processFolder({ local, r2 }) {
  const dir = path.join(DATASET_ROOT, local);
  if (!fs.existsSync(dir)) {
    console.log(`  SKIP ${local} — directory not found`);
    return { ok: 0, failed: 0 };
  }

  const files = fs.readdirSync(dir)
    .filter((f) => !f.startsWith(".") && IMAGE_EXTS.has(path.extname(f).toLowerCase()))
    .sort();

  console.log(`\n${local} → ${r2}/  (${files.length} images)`);

  // Folder marker
  await s3.send(new PutObjectCommand({
    Bucket: BUCKET, Key: `${r2}/`,
    Body: Buffer.alloc(0), ContentType: "application/x-directory",
  }));

  let ok = 0, failed = 0;
  const CONCURRENCY = 3; // sips is CPU-heavy; keep concurrency low

  for (let i = 0; i < files.length; i += CONCURRENCY) {
    const batch = files.slice(i, i + CONCURRENCY);
    await Promise.all(batch.map(async (filename) => {
      const srcPath = path.join(dir, filename);
      const base    = path.basename(filename, path.extname(filename));
      const destKey = `${r2}/${base}.png`;
      const idx     = i + batch.indexOf(filename) + 1;

      try {
        const buf = await toPng(srcPath);
        await s3.send(new PutObjectCommand({
          Bucket: BUCKET, Key: destKey,
          Body: buf, ContentType: "image/png",
        }));
        ok++;
        console.log(`  [${ok + failed}/${files.length}] ✓ ${base}.png  (${(buf.length / 1024).toFixed(0)} KB)`);
      } catch (err) {
        failed++;
        console.log(`  [${ok + failed}/${files.length}] ✗ ${filename}: ${err.message.split("\n")[0]}`);
      }
    }));
  }

  return { ok, failed };
}

async function main() {
  for (const name of ["R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET", "R2_ENDPOINT"]) {
    if (!process.env[name]) { console.error(`Missing env var: ${name}`); process.exit(1); }
  }

  console.log(`Pinging bucket "${BUCKET}"…`);
  try {
    await s3.send(new HeadBucketCommand({ Bucket: BUCKET }));
    console.log("  ✓ reachable");
  } catch (err) {
    console.error(`  ✗ ${err.message}`); process.exit(1);
  }

  const totals = { ok: 0, failed: 0 };
  for (const mapping of FOLDER_MAP) {
    const r = await processFolder(mapping);
    totals.ok     += r.ok;
    totals.failed += r.failed;
  }

  console.log(`\n${"─".repeat(40)}`);
  console.log(`✓ uploaded: ${totals.ok}`);
  if (totals.failed) console.log(`✗ failed:   ${totals.failed}`);
  console.log(`${"─".repeat(40)}`);
  if (totals.failed) process.exit(1);
}

main().catch((err) => { console.error(err); process.exit(1); });
