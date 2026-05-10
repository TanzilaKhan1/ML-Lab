import { NextResponse } from "next/server";
import zlib from "node:zlib";
import sharp from "sharp";
import {
  listKeys,
  getObjectBuffer,
  PREFIX_ANN,
  PREFIX_RAW,
} from "@/lib/r2";
import { exportCOCO, buildYoloLabels, loadAllAnnotated } from "@/lib/storage";

export const dynamic = "force-dynamic";
export const maxDuration = 120;

// ---------------------------------------------------------------------------
// CRC-32 (IEEE 802.3)
// ---------------------------------------------------------------------------
const CRC_TABLE = (() => {
  const t = new Uint32Array(256);
  for (let i = 0; i < 256; i++) {
    let c = i;
    for (let j = 0; j < 8; j++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    t[i] = c;
  }
  return t;
})();
function crc32(buf: Buffer): number {
  let c = 0xffffffff;
  for (let i = 0; i < buf.length; i++) c = CRC_TABLE[(c ^ buf[i]) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}

// ---------------------------------------------------------------------------
// Minimal in-memory ZIP builder
// ---------------------------------------------------------------------------
interface ZipEntry { name: string; data: Buffer }

function buildZip(entries: ZipEntry[]): Buffer {
  const localParts: Buffer[] = [];
  const cdParts: Buffer[] = [];
  let offset = 0;

  for (const { name, data } of entries) {
    const nameBytes = Buffer.from(name, "utf8");
    const compressed = zlib.deflateRawSync(data, { level: 6 });
    const useStore = compressed.length >= data.length;
    const body = useStore ? data : compressed;
    const method = useStore ? 0 : 8;
    const crc = crc32(data);

    const lh = Buffer.alloc(30 + nameBytes.length);
    lh.writeUInt32LE(0x04034b50, 0);
    lh.writeUInt16LE(20, 4);
    lh.writeUInt16LE(0, 6);
    lh.writeUInt16LE(method, 8);
    lh.writeUInt16LE(0, 10);
    lh.writeUInt16LE(0, 12);
    lh.writeUInt32LE(crc, 14);
    lh.writeUInt32LE(body.length, 18);
    lh.writeUInt32LE(data.length, 22);
    lh.writeUInt16LE(nameBytes.length, 26);
    lh.writeUInt16LE(0, 28);
    nameBytes.copy(lh, 30);

    localParts.push(lh, body);

    const cd = Buffer.alloc(46 + nameBytes.length);
    cd.writeUInt32LE(0x02014b50, 0);
    cd.writeUInt16LE(0x0314, 4);
    cd.writeUInt16LE(20, 6);
    cd.writeUInt16LE(0, 8);
    cd.writeUInt16LE(method, 10);
    cd.writeUInt16LE(0, 12);
    cd.writeUInt16LE(0, 14);
    cd.writeUInt32LE(crc, 16);
    cd.writeUInt32LE(body.length, 20);
    cd.writeUInt32LE(data.length, 24);
    cd.writeUInt16LE(nameBytes.length, 28);
    cd.writeUInt16LE(0, 30);
    cd.writeUInt16LE(0, 32);
    cd.writeUInt16LE(0, 34);
    cd.writeUInt16LE(0, 36);
    cd.writeUInt32LE(0, 38);
    cd.writeUInt32LE(offset, 42);
    nameBytes.copy(cd, 46);

    cdParts.push(cd);
    offset += lh.length + body.length;
  }

  const cdBuf = Buffer.concat(cdParts);
  const eocd = Buffer.alloc(22);
  eocd.writeUInt32LE(0x06054b50, 0);
  eocd.writeUInt16LE(0, 4);
  eocd.writeUInt16LE(0, 6);
  eocd.writeUInt16LE(entries.length, 8);
  eocd.writeUInt16LE(entries.length, 10);
  eocd.writeUInt32LE(cdBuf.length, 12);
  eocd.writeUInt32LE(offset, 16);
  eocd.writeUInt16LE(0, 20);

  return Buffer.concat([...localParts, cdBuf, eocd]);
}

// ---------------------------------------------------------------------------
// Route
// ---------------------------------------------------------------------------
export async function GET() {
  const rawPrefix = PREFIX_RAW();
  const annPrefix = PREFIX_ANN();
  const IMAGE_EXTS = new Set([".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff", ".svg"]);

  // Exclude soft-deleted items (anything under <prefix>/_trash/).
  const isUnderTrash = (k: string, base: string) => {
    const p = base + "/";
    const rel = k.startsWith(p) ? k.slice(p.length) : k;
    return rel === "_trash" || rel.startsWith("_trash/");
  };

  // List all raw image keys (excluding empty folder-marker objects)
  const rawKeys = (await listKeys(rawPrefix)).filter((k) => {
    if (isUnderTrash(k, rawPrefix)) return false;
    const ext = k.slice(k.lastIndexOf(".")).toLowerCase();
    return IMAGE_EXTS.has(ext);
  });

  // List all annotation JSON keys
  const annKeys = (await listKeys(annPrefix)).filter((k) => k.endsWith(".json") && !isUnderTrash(k, annPrefix));

  // Build annotation map: relative filename → annotation data
  type AnnData = {
    status?: string;
    annotations?: unknown[];
    imageWidth?: number;
    imageHeight?: number;
  };
  const annMap = new Map<string, AnnData>();
  await Promise.all(
    annKeys.map(async (k) => {
      const buf = await getObjectBuffer(k);
      if (!buf) return;
      try {
        const data = JSON.parse(buf.body.toString("utf8")) as AnnData;
        // Derive the relative filename from the annotation key
        const prefix = annPrefix + "/";
        const rel = k.startsWith(prefix) ? k.slice(prefix.length) : k;
        // rel is like "bus/positive/img.json" → filename is "bus/positive/img.png"
        annMap.set(rel, data);
      } catch { /* skip malformed */ }
    }),
  );

  const entries: ZipEntry[] = [];

  // 1. Manifest: status of every image
  const rawPrefix2 = rawPrefix + "/";
  const manifest = await Promise.all(
    rawKeys.map(async (k) => {
      const relPath = k.startsWith(rawPrefix2) ? k.slice(rawPrefix2.length) : k;
      const base = relPath.replace(/\.[^.]+$/, "");
      const annKey = base + ".json";
      const ann = annMap.get(annKey);
      return {
        path: relPath,
        status: ann?.status ?? "unannotated",
        annotationCount: (ann?.annotations ?? []).length,
        imageWidth: ann?.imageWidth ?? null,
        imageHeight: ann?.imageHeight ?? null,
      };
    }),
  );
  entries.push({
    name: "manifest.json",
    data: Buffer.from(JSON.stringify(manifest, null, 2), "utf8"),
  });

  // 2. Individual annotation JSONs
  await Promise.all(
    annKeys.map(async (k) => {
      const buf = await getObjectBuffer(k);
      if (!buf) return;
      const prefix = annPrefix + "/";
      const rel = k.startsWith(prefix) ? k.slice(prefix.length) : k;
      entries.push({ name: `annotations/${rel}`, data: buf.body });
    }),
  );

  // 3. Images as PNG — fetch in batches of 10 to avoid memory spikes
  const BATCH = 10;
  for (let i = 0; i < rawKeys.length; i += BATCH) {
    const batch = rawKeys.slice(i, i + BATCH);
    const results = await Promise.all(
      batch.map(async (k) => {
        const buf = await getObjectBuffer(k);
        if (!buf) return null;
        const relPath = k.startsWith(rawPrefix2) ? k.slice(rawPrefix2.length) : k;
        const ext = relPath.slice(relPath.lastIndexOf(".")).toLowerCase();
        let imgBuf = buf.body;
        let outName = relPath;
        // Convert to PNG if not already
        if (ext !== ".png") {
          try {
            imgBuf = Buffer.from(await sharp(imgBuf).png({ compressionLevel: 8 }).toBuffer());
            outName = relPath.slice(0, relPath.length - ext.length) + ".png";
          } catch { /* include as-is on conversion failure */ }
        }
        return { name: `images/${outName}`, data: imgBuf };
      }),
    );
    for (const r of results) {
      if (r) entries.push(r);
    }
  }

  // Track which optional sections shipped so the README reflects reality
  // and partial-failure footprints are visible to the recipient.
  let cocoIncluded = false;
  let yoloIncluded = false;

  // 4. COCO export (also writes to R2 as a side effect).
  try {
    const coco = await exportCOCO();
    entries.push({
      name: "coco/coco_export.json",
      data: Buffer.from(JSON.stringify(coco, null, 2), "utf8"),
    });
    cocoIncluded = true;
  } catch (err) {
    console.error("[zip] COCO export failed", err);
    entries.push({
      name: "coco/ERROR.txt",
      data: Buffer.from(
        `COCO export failed at ${new Date().toISOString()}.\n` +
          `Reason: ${err instanceof Error ? err.message : String(err)}\n`,
        "utf8",
      ),
    });
  }

  // 5. YOLO bundle: per-image .txt labels + classes.txt + dataset.yaml.
  // Image and label paths are kept parallel under images/ and yolo/labels/
  // so the zip is drop-in usable by Ultralytics tooling.
  // Built into a local array first; appended only if the entire bundle
  // succeeded so we never ship a half-built yolo tree (some labels but no
  // classes.txt / dataset.yaml).
  try {
    const annotated = await loadAllAnnotated();
    const { files: yoloFiles, classes } = buildYoloLabels(annotated);
    const yoloEntries: ZipEntry[] = [];
    for (const [filename, { text }] of yoloFiles) {
      const labelName = filename.replace(/\.[^.]+$/, ".txt");
      yoloEntries.push({
        name: `yolo/labels/${labelName}`,
        data: Buffer.from(text, "utf8"),
      });
    }
    yoloEntries.push({
      name: "yolo/classes.txt",
      data: Buffer.from(classes.join("\n"), "utf8"),
    });
    const yaml = [
      "# Ultralytics-compatible dataset descriptor.",
      "# Adjust `path` to the absolute root after extracting this zip.",
      "# `train` and `val` point to the same set — split downstream as needed.",
      "path: .",
      "train: images",
      "val: images",
      `nc: ${classes.length}`,
      `names: [${classes.map((c) => JSON.stringify(c)).join(", ")}]`,
      "",
    ].join("\n");
    yoloEntries.push({ name: "yolo/dataset.yaml", data: Buffer.from(yaml, "utf8") });
    entries.push(...yoloEntries);
    yoloIncluded = true;
  } catch (err) {
    console.error("[zip] YOLO bundle failed", err);
    entries.push({
      name: "yolo/ERROR.txt",
      data: Buffer.from(
        `YOLO bundle failed at ${new Date().toISOString()}.\n` +
          `Reason: ${err instanceof Error ? err.message : String(err)}\n`,
        "utf8",
      ),
    });
  }

  // 6. README.md so the recipient knows the layout.
  // Image extension is preserved when the source is already PNG; non-PNG
  // sources are converted to PNG by the loop above (see step 3).
  const readme = [
    "# Annotation Export",
    "",
    `Generated: ${new Date().toISOString()}`,
    "",
    "## Layout",
    "",
    "- `manifest.json` — per-image status & annotation count",
    "- `annotations/<folder>/<name>.json` — full per-image annotation source",
    "- `images/<folder>/<name>.png` — every raw image (re-encoded to PNG when source was not PNG)",
    cocoIncluded
      ? "- `coco/coco_export.json` — COCO-format dataset"
      : "- `coco/ERROR.txt` — COCO export failed; see file for details",
    yoloIncluded
      ? "- `yolo/labels/<folder>/<name>.txt` — YOLO label file per image"
      : "- `yolo/ERROR.txt` — YOLO bundle failed; see file for details",
    ...(yoloIncluded
      ? [
          "- `yolo/classes.txt` — class index → name (line N == class N)",
          "- `yolo/dataset.yaml` — Ultralytics dataset descriptor (train and val refer to the same `images/` set)",
        ]
      : []),
    "",
  ].join("\n");
  entries.push({ name: "README.md", data: Buffer.from(readme, "utf8") });

  if (entries.length === 0) {
    return NextResponse.json({ error: "No data found" }, { status: 404 });
  }

  const zip = buildZip(entries);
  const date = new Date().toISOString().slice(0, 10);

  return new NextResponse(new Uint8Array(zip), {
    status: 200,
    headers: {
      "Content-Type": "application/zip",
      "Content-Disposition": `attachment; filename="annotations_${date}.zip"`,
      "Content-Length": String(zip.length),
    },
  });
}
