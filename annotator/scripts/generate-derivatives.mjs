// Regenerate thumb (256px) + preview (1280px) WebP derivatives in R2 for
// every raw image already uploaded. Run this after a bulk upload so the
// Render app never has to decode full-resolution images on demand — it
// just serves these pre-built derivatives (see app/api/derived route).
//
// Mirrors the SIZES / key layout in app/api/derived/[size]/[...filename]/route.ts
// exactly, so the app's cache-hit path recognizes what this script writes.
//
// Usage:
//   cd annotator
//   node --env-file=.env.local scripts/generate-derivatives.mjs            # regenerate all
//   node --env-file=.env.local scripts/generate-derivatives.mjs --dry-run  # list only

import { S3Client, ListObjectsV2Command, GetObjectCommand, PutObjectCommand } from "@aws-sdk/client-s3";
import { NodeHttpHandler } from "@smithy/node-http-handler";
import sharp from "sharp";

const DRY_RUN     = process.argv.includes("--dry-run");
const CONCURRENCY = 4;

const BUCKET         = process.env.R2_BUCKET;
const PREFIX_RAW     = process.env.R2_PREFIX_RAW || "raw";
const PREFIX_DERIVED = process.env.R2_PREFIX_DERIVED || "derived";

const TRASH_PREFIX = "_trash";
const IMAGE_EXTS = new Set([".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff", ".tif", ".svg"]);

// Must match SIZES in app/api/derived/[size]/[...filename]/route.ts
const SIZES = {
  thumb:   { side: 256,  quality: 70 },
  preview: { side: 1280, quality: 82 },
};

sharp.cache({ memory: 100, files: 0 });

const s3 = new S3Client({
  region: "auto",
  endpoint: process.env.R2_ENDPOINT,
  credentials: {
    accessKeyId: process.env.R2_ACCESS_KEY_ID,
    secretAccessKey: process.env.R2_SECRET_ACCESS_KEY,
  },
  requestHandler: new NodeHttpHandler({
    connectionTimeout: 10_000,
    requestTimeout: 30_000,
  }),
  maxAttempts: 3,
});

function extname(key) {
  const i = key.lastIndexOf(".");
  return i === -1 ? "" : key.slice(i).toLowerCase();
}

function joinKey(...parts) {
  return parts.map((p) => p.replace(/^\/+|\/+$/g, "")).filter(Boolean).join("/");
}

function derivedKey(size, rawFilename) {
  const base = rawFilename.replace(/\.[^.]+$/, "");
  return joinKey(PREFIX_DERIVED, size, base + ".webp");
}

async function listAllKeys(prefix) {
  const keys = [];
  let token;
  const listPrefix = prefix.endsWith("/") ? prefix : prefix + "/";
  do {
    const out = await s3.send(new ListObjectsV2Command({
      Bucket: BUCKET, Prefix: listPrefix, ContinuationToken: token,
    }));
    for (const obj of out.Contents || []) if (obj.Key) keys.push(obj.Key);
    token = out.IsTruncated ? out.NextContinuationToken : undefined;
  } while (token);
  return keys;
}

async function getBuffer(key) {
  const out = await s3.send(new GetObjectCommand({ Bucket: BUCKET, Key: key }));
  const bytes = await out.Body.transformToByteArray();
  return Buffer.from(bytes);
}

async function processImage(rawFilename) {
  const rawKeyFull = joinKey(PREFIX_RAW, rawFilename);
  const buf = await getBuffer(rawKeyFull);

  const meta = await sharp(buf, { failOn: "none" }).rotate().metadata();
  const origW = meta.width || 0;
  const origH = meta.height || 0;

  for (const [size, cfg] of Object.entries(SIZES)) {
    const dKey = derivedKey(size, rawFilename);
    if (DRY_RUN) {
      console.log(`  [dry] ${rawFilename} -> ${dKey}  (${origW}x${origH} src)`);
      continue;
    }
    const webp = await sharp(buf, { failOn: "none" })
      .rotate()
      .resize({ width: cfg.side, height: cfg.side, fit: "inside", withoutEnlargement: true })
      .webp({ quality: cfg.quality, effort: 4 })
      .toBuffer();
    await s3.send(new PutObjectCommand({
      Bucket: BUCKET, Key: dKey, Body: webp, ContentType: "image/webp",
      Metadata: { "original-width": String(origW), "original-height": String(origH) },
    }));
    console.log(`  OK  ${dKey}  (${(webp.length / 1024).toFixed(0)} KB)`);
  }
}

async function main() {
  for (const v of ["R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET", "R2_ENDPOINT"]) {
    if (!process.env[v]) { console.error(`Missing env var: ${v}`); process.exit(1); }
  }

  console.log(`Bucket   : ${BUCKET}`);
  console.log(`Raw      : ${PREFIX_RAW}/`);
  console.log(`Derived  : ${PREFIX_DERIVED}/{thumb,preview}/`);
  console.log(`Mode     : ${DRY_RUN ? "DRY RUN (no writes)" : "REGENERATE"}`);

  const keys = await listAllKeys(PREFIX_RAW);
  const imageKeys = keys
    .filter((k) => IMAGE_EXTS.has(extname(k)))
    .filter((k) => !k.includes(`/${TRASH_PREFIX}/`) && !k.startsWith(`${PREFIX_RAW}/${TRASH_PREFIX}/`))
    .sort();

  const rawPrefix = PREFIX_RAW + "/";
  const filenames = imageKeys.map((k) => (k.startsWith(rawPrefix) ? k.slice(rawPrefix.length) : k));

  console.log(`\nFound ${filenames.length} raw images.\n`);

  let done = 0, failed = 0;
  for (let i = 0; i < filenames.length; i += CONCURRENCY) {
    const batch = filenames.slice(i, i + CONCURRENCY);
    await Promise.all(batch.map(async (f) => {
      try {
        await processImage(f);
        done++;
      } catch (err) {
        failed++;
        console.log(`  FAIL ${f}: ${String(err.message).split("\n")[0]}`);
      }
    }));
  }

  console.log(`\n${"─".repeat(48)}`);
  console.log(`${DRY_RUN ? "would process" : "processed"}: ${done}/${filenames.length}`);
  if (failed) console.log(`failed: ${failed}`);
  console.log(`${"─".repeat(48)}`);
  if (failed) process.exit(1);
}

main().catch((err) => { console.error(err); process.exit(1); });
