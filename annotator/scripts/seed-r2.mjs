// One-time seed: create folder markers and upload the sample images + existing
// annotations from disk into R2. Safe to re-run — PUT is idempotent.
//
// Usage:
//   cd annotator
//   node --env-file=.env.local scripts/seed-r2.mjs
//
// Requires env vars: R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY,
// R2_BUCKET, R2_ENDPOINT.

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { S3Client, PutObjectCommand, HeadBucketCommand } from "@aws-sdk/client-s3";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(__dirname, "..");

const required = ["R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET", "R2_ENDPOINT"];
for (const name of required) {
  if (!process.env[name]) {
    console.error(`Missing env var: ${name}. Run with: node --env-file=.env.local scripts/seed-r2.mjs`);
    process.exit(1);
  }
}

const BUCKET = process.env.R2_BUCKET;
const PREFIX_RAW = process.env.R2_PREFIX_RAW || "raw";
const PREFIX_ANN = process.env.R2_PREFIX_ANNOTATIONS || "annotations";
const PREFIX_EXP = process.env.R2_PREFIX_EXPORTS || "exports";

const s3 = new S3Client({
  region: "auto",
  endpoint: process.env.R2_ENDPOINT,
  credentials: {
    accessKeyId: process.env.R2_ACCESS_KEY_ID,
    secretAccessKey: process.env.R2_SECRET_ACCESS_KEY,
  },
});

const MIME = {
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".gif": "image/gif",
  ".webp": "image/webp",
  ".bmp": "image/bmp",
  ".tiff": "image/tiff",
  ".svg": "image/svg+xml",
  ".json": "application/json",
  ".txt": "text/plain",
};

async function put(Key, Body, ContentType) {
  await s3.send(new PutObjectCommand({ Bucket: BUCKET, Key, Body, ContentType }));
  console.log(`  → PUT ${Key}  (${Body.length} bytes, ${ContentType || "n/a"})`);
}

async function main() {
  console.log(`Pinging bucket "${BUCKET}"…`);
  try {
    await s3.send(new HeadBucketCommand({ Bucket: BUCKET }));
    console.log("  ✓ bucket reachable");
  } catch (err) {
    console.error(`  ✗ cannot reach bucket: ${err.message}`);
    process.exit(1);
  }

  console.log("\nCreating folder markers…");
  for (const prefix of [PREFIX_RAW, PREFIX_ANN, PREFIX_EXP]) {
    await put(`${prefix}/`, Buffer.alloc(0), "application/x-directory");
  }

  console.log("\nUploading raw images from public/raw/ …");
  const rawDir = path.join(root, "public", "raw");
  if (fs.existsSync(rawDir)) {
    for (const f of fs.readdirSync(rawDir).sort()) {
      const full = path.join(rawDir, f);
      const stat = fs.statSync(full);
      if (!stat.isFile()) continue;
      const ext = path.extname(f).toLowerCase();
      if (!MIME[ext] || ext === ".json" || ext === ".txt") continue;
      const body = fs.readFileSync(full);
      await put(`${PREFIX_RAW}/${f}`, body, MIME[ext]);
    }
  } else {
    console.log("  (no public/raw/ directory — skipping)");
  }

  console.log("\nUploading annotation JSONs from annotations/ …");
  const annDir = path.join(root, "annotations");
  if (fs.existsSync(annDir)) {
    for (const f of fs.readdirSync(annDir).sort()) {
      if (!f.endsWith(".json")) continue;
      const body = fs.readFileSync(path.join(annDir, f));
      await put(`${PREFIX_ANN}/${f}`, body, "application/json");
    }
  } else {
    console.log("  (no annotations/ directory — skipping)");
  }

  console.log("\nUploading existing exports from exports/ …");
  const expDir = path.join(root, "exports");
  if (fs.existsSync(expDir)) {
    for (const f of fs.readdirSync(expDir).sort()) {
      const full = path.join(expDir, f);
      if (!fs.statSync(full).isFile()) continue;
      const ext = path.extname(f).toLowerCase();
      const body = fs.readFileSync(full);
      await put(`${PREFIX_EXP}/${f}`, body, MIME[ext] || "application/octet-stream");
    }
  } else {
    console.log("  (no exports/ directory — skipping)");
  }

  console.log("\n✓ Done.");
}

main().catch((err) => {
  console.error("Seed failed:", err);
  process.exit(1);
});
