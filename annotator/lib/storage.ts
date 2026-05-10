import "server-only";
import path from "path";
import {
  PREFIX_ANN,
  PREFIX_EXP,
  PREFIX_RAW,
  copyObject,
  deleteObject,
  getObjectBuffer,
  getObjectText,
  joinKey,
  listKeys,
  objectExists,
  putObject,
  deleteObjects,
} from "./r2";

export const UPLOAD_FOLDERS = [
  "bus/positive",
  "bus/negative",
  "legua/positive",
  "legua/negative",
] as const;
export type UploadFolder = (typeof UPLOAD_FOLDERS)[number];

/** Soft-deleted images live under this prefix (relative to PREFIX_RAW / PREFIX_ANN). */
const TRASH_PREFIX = "_trash";

function isTrashed(filename: string): boolean {
  return filename === TRASH_PREFIX || filename.startsWith(TRASH_PREFIX + "/");
}

// ---------------------------------------------------------------------------
// Typed errors. Routes use instanceof checks to map to HTTP status codes;
// keeps status mapping decoupled from message wording.
// ---------------------------------------------------------------------------
export class InvalidFilenameError extends Error {
  constructor(filename: string, reason: string) {
    super(`Invalid filename "${filename}": ${reason}`);
    this.name = "InvalidFilenameError";
  }
}
export class NotFoundError extends Error {
  constructor(filename: string) {
    super(`Source image not found: ${filename}`);
    this.name = "NotFoundError";
  }
}
export class AlreadyExistsError extends Error {
  constructor(filename: string) {
    super(`Destination already exists: ${filename}`);
    this.name = "AlreadyExistsError";
  }
}
export class StorageStepError extends Error {
  readonly step: string;
  readonly cause: unknown;
  constructor(step: string, context: Record<string, string>, cause: unknown) {
    const ctx = Object.entries(context).map(([k, v]) => `${k}=${v}`).join(" ");
    super(`Storage step "${step}" failed (${ctx}): ${cause instanceof Error ? cause.message : String(cause)}`);
    this.name = "StorageStepError";
    this.step = step;
    this.cause = cause;
  }
}

/** Run an awaitable, wrapping any throw in a StorageStepError that names the
 *  failed step and includes key context for operator diagnosis. */
async function step<T>(name: string, context: Record<string, string>, fn: () => Promise<T>): Promise<T> {
  try {
    return await fn();
  } catch (err) {
    console.error(`[storage] step="${name}" failed`, { ...context, err });
    throw new StorageStepError(name, context, err);
  }
}

/**
 * Validate that `filename` is a well-formed ACTIVE (non-trashed) raw key.
 * Catches malformed inputs before they reach R2, so we never copy/delete a key
 * the rest of the codebase doesn't recognize.
 */
function validateActiveFilename(filename: string): void {
  if (!filename) throw new InvalidFilenameError(filename, "empty");
  if (filename.startsWith("/")) throw new InvalidFilenameError(filename, "leading slash");
  if (filename.includes("..")) throw new InvalidFilenameError(filename, "path traversal");
  if (isTrashed(filename)) throw new InvalidFilenameError(filename, "is trashed");
  const matches = (UPLOAD_FOLDERS as readonly string[]).some((f) => filename.startsWith(f + "/"));
  if (!matches) {
    throw new InvalidFilenameError(filename, `must start with one of: ${UPLOAD_FOLDERS.join(", ")}`);
  }
  const basename = filename.split("/").pop();
  if (!basename) throw new InvalidFilenameError(filename, "missing basename");
}
import type { ImageAnnotation, ImageInfo, ImageStatus, ProjectStats } from "./types";

const IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff", ".svg"];

const IMAGE_MIME: Record<string, string> = {
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".png": "image/png",
  ".gif": "image/gif",
  ".bmp": "image/bmp",
  ".webp": "image/webp",
  ".tiff": "image/tiff",
  ".svg": "image/svg+xml",
};

function rawKey(filename: string): string {
  return joinKey(PREFIX_RAW(), filename);
}

function annKey(filename: string): string {
  // Strip the image extension, add .json
  const base = filename.replace(/\.[^.]+$/, "");
  return joinKey(PREFIX_ANN(), base + ".json");
}

function expKey(filename: string): string {
  return joinKey(PREFIX_EXP(), filename);
}

/** Parse JSON safely. Returns null on malformed input (see bug audit note). */
function safeJsonParse<T>(s: string | null): T | null {
  if (s == null) return null;
  try {
    return JSON.parse(s) as T;
  } catch {
    return null;
  }
}

function isImageKey(key: string): boolean {
  return IMAGE_EXTS.includes(path.extname(key).toLowerCase());
}

/** Extract the filename portion of a raw/ key. */
function rawFilename(key: string): string {
  const prefix = PREFIX_RAW() + "/";
  return key.startsWith(prefix) ? key.slice(prefix.length) : key;
}

async function loadAnnotation(filename: string): Promise<ImageAnnotation | null> {
  const text = await getObjectText(annKey(filename));
  return safeJsonParse<ImageAnnotation>(text);
}

// ---------------------------------------------------------------------------
// Manifest — single file that tracks {status, count} for every image so the
// sidebar/stats don't have to fetch N annotation JSONs on every page load.
//
// Consistency: writes are read-modify-write; concurrent edits to DIFFERENT
// images can race (last-write-wins on the manifest row). The authoritative
// per-image annotation JSON is always written FIRST, so the only user-visible
// effect of a race is a momentarily stale row in the sidebar, which self-heals
// on the next save for that image. An admin "Rebuild manifest" action forces
// a full rescan.
// ---------------------------------------------------------------------------
interface ManifestEntry { status: ImageStatus; count: number }
interface Manifest { version: 1; updatedAt: string; images: Record<string, ManifestEntry> }

const MANIFEST_KEY = () => joinKey(PREFIX_ANN(), "_manifest.json");

async function readManifest(): Promise<Manifest | null> {
  const text = await getObjectText(MANIFEST_KEY());
  const m = safeJsonParse<Manifest>(text);
  if (!m || m.version !== 1 || !m.images) return null;
  return m;
}

async function writeManifest(m: Manifest): Promise<void> {
  m.updatedAt = new Date().toISOString();
  await putObject(MANIFEST_KEY(), JSON.stringify(m), "application/json");
}

async function upsertManifestEntry(filename: string, entry: ManifestEntry): Promise<void> {
  const m = (await readManifest()) ?? { version: 1, updatedAt: "", images: {} };
  m.images[filename] = entry;
  await writeManifest(m);
}

/** No-op if entry is absent. Race: read-modify-write, last writer wins. */
async function deleteManifestEntry(filename: string): Promise<void> {
  const m = await readManifest();
  if (!m) {
    console.warn("[storage] deleteManifestEntry: no manifest", { filename });
    return;
  }
  if (!(filename in m.images)) return;
  delete m.images[filename];
  await writeManifest(m);
}

/**
 * Rename one entry; preserves status/count. Read-modify-write — last writer
 * wins on concurrent calls. If the source entry is missing (stale manifest),
 * insert a default row for `newName` so the moved image stays visible in the
 * sidebar instead of silently disappearing.
 */
async function renameManifestEntry(oldName: string, newName: string): Promise<void> {
  const m = (await readManifest()) ?? { version: 1, updatedAt: "", images: {} };
  const entry = m.images[oldName];
  delete m.images[oldName];
  if (entry) {
    m.images[newName] = entry;
  } else {
    console.warn("[storage] renameManifestEntry: source entry missing, inserting default", { oldName, newName });
    m.images[newName] = { status: "unannotated", count: 0 };
  }
  await writeManifest(m);
}

/** Full rescan — expensive but authoritative. Called on first access if no
 *  manifest exists, and manually via the admin UI after bulk uploads. */
export async function rebuildManifest(): Promise<Manifest> {
  const keys = await listKeys(PREFIX_RAW());
  const imageKeys = keys.filter(isImageKey).filter((k) => !isTrashed(rawFilename(k))).sort();

  const entries = await Promise.all(imageKeys.map(async (k) => {
    const filename = rawFilename(k);
    const data = await loadAnnotation(filename);
    const entry: ManifestEntry = {
      status: (data?.status || "unannotated") as ImageStatus,
      count: (data?.annotations || []).length,
    };
    return [filename, entry] as const;
  }));

  const m: Manifest = {
    version: 1,
    updatedAt: new Date().toISOString(),
    images: Object.fromEntries(entries),
  };
  await writeManifest(m);
  return m;
}

export async function getImages(filter?: ImageStatus | "all"): Promise<ImageInfo[]> {
  // 1 LIST + 1 GET (manifest) regardless of image count.
  // Falls back to full rescan on first call / after admin reset.
  const [keys, manifest] = await Promise.all([listKeys(PREFIX_RAW()), readManifest()]);
  const imageKeys = keys.filter(isImageKey).filter((k) => !isTrashed(rawFilename(k))).sort();
  const m = manifest ?? (await rebuildManifest());

  const results: ImageInfo[] = imageKeys.map((k) => {
    const filename = rawFilename(k);
    const entry = m.images[filename];
    return {
      filename,
      status: (entry?.status ?? "unannotated") as ImageStatus,
      annotationCount: entry?.count ?? 0,
    };
  });

  if (filter && filter !== "all") {
    return results.filter((i) => i.status === filter);
  }
  return results;
}

export async function getAnnotation(filename: string): Promise<ImageAnnotation> {
  const data = await loadAnnotation(filename);
  if (data) return data;
  return {
    filename,
    annotations: [],
    labels: [],
    status: "unannotated",
    reviewComment: "",
    history: [],
  };
}

export async function saveAnnotation(data: ImageAnnotation): Promise<void> {
  data.lastModified = new Date().toISOString();
  // Annotation file first (source of truth), manifest after (fast-path cache).
  await putObject(annKey(data.filename), JSON.stringify(data, null, 2), "application/json");
  await upsertManifestEntry(data.filename, {
    status: (data.status || "annotated") as ImageStatus,
    count: (data.annotations || []).length,
  });
}

export async function saveUploadedImage(filename: string, buffer: Buffer): Promise<void> {
  const mime = IMAGE_MIME[path.extname(filename).toLowerCase()] || "application/octet-stream";
  await putObject(rawKey(filename), buffer, mime);
}

/** Fetch a raw image for proxying to the browser. Returns null if missing. */
export async function getRawImage(filename: string): Promise<{ body: Buffer; contentType: string } | null> {
  const res = await getObjectBuffer(rawKey(filename));
  if (!res) return null;
  const ext = path.extname(filename).toLowerCase();
  // Prefer stored content type; fall back to extension lookup.
  const contentType = res.contentType || IMAGE_MIME[ext] || "application/octet-stream";
  return { body: res.body, contentType };
}

export async function updateStatus(
  filename: string,
  status: ImageStatus,
  comment?: string,
): Promise<void> {
  const existing = await loadAnnotation(filename);
  const data: ImageAnnotation = existing ?? {
    filename,
    annotations: [],
    labels: [],
    status: "unannotated",
    reviewComment: "",
    history: [],
  };
  data.status = status;
  if (comment) {
    data.reviewComment = comment;
    data.history = data.history || [];
    data.history.push({ action: status, comment, timestamp: new Date().toISOString() });
  }
  data.lastModified = new Date().toISOString();
  await putObject(annKey(filename), JSON.stringify(data, null, 2), "application/json");
  await upsertManifestEntry(filename, { status, count: (data.annotations || []).length });
}

export async function getStats(): Promise<ProjectStats> {
  const [keys, manifest] = await Promise.all([listKeys(PREFIX_RAW()), readManifest()]);
  const imageKeys = keys.filter(isImageKey).filter((k) => !isTrashed(rawFilename(k)));
  const m = manifest ?? (await rebuildManifest());

  const stats: ProjectStats = {
    total: imageKeys.length,
    unannotated: 0,
    annotated: 0,
    accepted: 0,
    rejected: 0,
  };

  for (const k of imageKeys) {
    const filename = rawFilename(k);
    const s = (m.images[filename]?.status ?? "unannotated") as ImageStatus;
    stats[s] = (stats[s] || 0) + 1;
  }
  return stats;
}

export async function getAllLabels(): Promise<string[]> {
  const keys = await listKeys(PREFIX_ANN());
  const annPrefix = PREFIX_ANN() + "/";
  const jsonKeys = keys.filter((k) => {
    if (!k.endsWith(".json")) return false;
    const rel = k.startsWith(annPrefix) ? k.slice(annPrefix.length) : k;
    return !isTrashed(rel);
  });
  const labelSet = new Set<string>();

  const bodies = await Promise.all(jsonKeys.map((k) => getObjectText(k)));
  for (const text of bodies) {
    const data = safeJsonParse<ImageAnnotation>(text);
    if (!data) continue;
    for (const ann of data.annotations || []) {
      if (ann.label) labelSet.add(ann.label);
    }
  }
  return [...labelSet].sort();
}

/**
 * Tight axis-aligned bounding box of a bbox that may be rotated.
 * The rect is positioned at (x, y) (top-left) with size (w, h) and rotated
 * by `angle` degrees clockwise around (x, y).
 */
function rotatedBboxAabb(x: number, y: number, w: number, h: number, angle: number): [number, number, number, number] {
  if (!angle) return [x, y, w, h];
  const rad = (angle * Math.PI) / 180;
  const cos = Math.cos(rad), sin = Math.sin(rad);
  const corners = [
    [x, y],
    [x + w * cos,         y + w * sin],
    [x + w * cos - h * sin, y + w * sin + h * cos],
    [x - h * sin,         y + h * cos],
  ];
  const xs = corners.map((c) => c[0]);
  const ys = corners.map((c) => c[1]);
  const minX = Math.min(...xs), minY = Math.min(...ys);
  const maxX = Math.max(...xs), maxY = Math.max(...ys);
  return [minX, minY, maxX - minX, maxY - minY];
}

/**
 * Derive an axis-aligned [x, y, w, h] bounding box suitable for YOLO export.
 * Returns null if the shape has no meaningful 2D extent (e.g. keypoint).
 */
function deriveYoloBox(ann: import("./types").Annotation): [number, number, number, number] | null {
  if ((ann.type === "bbox" || ann.type === "ellipse") && ann.x != null && ann.y != null && ann.width && ann.height) {
    // Rotated bbox → use tight AABB of rotated corners so the YOLO rect
    // actually covers the visible shape.
    return rotatedBboxAabb(ann.x, ann.y, ann.width, ann.height, ann.angle || 0);
  }
  if ((ann.type === "polygon" || ann.type === "polyline") && ann.points && ann.points.length > 0) {
    const xs = ann.points.map((p) => p[0]);
    const ys = ann.points.map((p) => p[1]);
    const minX = Math.min(...xs);
    const minY = Math.min(...ys);
    const maxX = Math.max(...xs);
    const maxY = Math.max(...ys);
    const w = maxX - minX;
    const h = maxY - minY;
    if (w <= 0 || h <= 0) return null;
    return [minX, minY, w, h];
  }
  return null;
}

async function loadAllAnnotated(): Promise<{ filename: string; data: ImageAnnotation }[]> {
  const keys = await listKeys(PREFIX_RAW());
  const imageKeys = keys.filter(isImageKey).filter((k) => !isTrashed(rawFilename(k))).sort();
  const results: { filename: string; data: ImageAnnotation }[] = [];
  const texts = await Promise.all(
    imageKeys.map((k) => getObjectText(annKey(rawFilename(k)))),
  );
  for (let i = 0; i < imageKeys.length; i++) {
    const data = safeJsonParse<ImageAnnotation>(texts[i]);
    if (!data) continue;
    results.push({ filename: rawFilename(imageKeys[i]), data });
  }
  return results;
}

export async function exportCOCO() {
  const annotated = await loadAllAnnotated();

  const coco: Record<string, unknown> = {
    info: { description: "Image Annotation Export", date_created: new Date().toISOString(), version: "1.0" },
    licenses: [],
    images: [] as Record<string, unknown>[],
    annotations: [] as Record<string, unknown>[],
    categories: [] as Record<string, unknown>[],
  };

  const categoryMap: Record<string, number> = {};
  let catId = 1;
  let annId = 1;

  annotated.forEach(({ filename, data }, idx) => {
    if (data.status === "rejected") return;

    const imageId = idx + 1;
    (coco.images as Record<string, unknown>[]).push({
      id: imageId,
      file_name: filename,
      width: data.imageWidth || 0,
      height: data.imageHeight || 0,
    });

    for (const ann of data.annotations || []) {
      if (!categoryMap[ann.label]) {
        categoryMap[ann.label] = catId;
        (coco.categories as Record<string, unknown>[]).push({
          id: catId,
          name: ann.label,
          supercategory: "none",
        });
        catId++;
      }

      const cocoAnn: Record<string, unknown> = {
        id: annId++,
        image_id: imageId,
        category_id: categoryMap[ann.label],
        iscrowd: 0,
      };

      if (ann.attributes && Object.keys(ann.attributes).length > 0) {
        cocoAnn.attributes = ann.attributes;
      }

      if (ann.type === "bbox") {
        const angle = ann.angle || 0;
        if (angle) {
          // Tight AABB covering the rotated rect; original params kept in attrs
          // so consumers that understand rotation can reconstruct exactly.
          const [bx, by, bw, bh] = rotatedBboxAabb(ann.x || 0, ann.y || 0, ann.width || 0, ann.height || 0, angle);
          cocoAnn.bbox = [bx, by, bw, bh];
          cocoAnn.area = (ann.width || 0) * (ann.height || 0); // un-rotated rect area
          cocoAnn.attributes = {
            ...(cocoAnn.attributes as object | undefined),
            rotation: angle,
            original_bbox: [ann.x, ann.y, ann.width, ann.height],
          };
        } else {
          cocoAnn.bbox = [ann.x, ann.y, ann.width, ann.height];
          cocoAnn.area = (ann.width || 0) * (ann.height || 0);
        }
      } else if (ann.type === "ellipse") {
        // COCO has no native ellipse; store bbox and mark shape in attributes.
        cocoAnn.bbox = [ann.x, ann.y, ann.width, ann.height];
        cocoAnn.area = Math.PI * ((ann.width || 0) / 2) * ((ann.height || 0) / 2);
        cocoAnn.attributes = { ...(cocoAnn.attributes as object | undefined), shape: "ellipse" };
      } else if (ann.type === "polygon" && ann.points) {
        const flat = ann.points.flat();
        cocoAnn.segmentation = [flat];
        const xs = ann.points.map((p: number[]) => p[0]);
        const ys = ann.points.map((p: number[]) => p[1]);
        const minX = Math.min(...xs);
        const minY = Math.min(...ys);
        const maxX = Math.max(...xs);
        const maxY = Math.max(...ys);
        cocoAnn.bbox = [minX, minY, maxX - minX, maxY - minY];
        let area = 0;
        for (let i = 0; i < ann.points.length; i++) {
          const j = (i + 1) % ann.points.length;
          area += ann.points[i][0] * ann.points[j][1];
          area -= ann.points[j][0] * ann.points[i][1];
        }
        cocoAnn.area = Math.abs(area / 2);
      } else if (ann.type === "polyline" && ann.points) {
        // COCO has no native polyline; encode points and a shape marker.
        const xs = ann.points.map((p: number[]) => p[0]);
        const ys = ann.points.map((p: number[]) => p[1]);
        const minX = Math.min(...xs);
        const minY = Math.min(...ys);
        const maxX = Math.max(...xs);
        const maxY = Math.max(...ys);
        cocoAnn.bbox = [minX, minY, maxX - minX, maxY - minY];
        cocoAnn.area = 0;
        cocoAnn.segmentation = [ann.points.flat()];
        cocoAnn.attributes = { ...(cocoAnn.attributes as object | undefined), shape: "polyline" };
      } else if (ann.type === "keypoint") {
        cocoAnn.keypoints = [ann.x, ann.y, 2];
        cocoAnn.num_keypoints = 1;
        cocoAnn.bbox = [ann.x, ann.y, 0, 0];
        cocoAnn.area = 0;
      }

      (coco.annotations as Record<string, unknown>[]).push(cocoAnn);
    }
  });

  await putObject(expKey("coco_export.json"), JSON.stringify(coco, null, 2), "application/json");
  return coco;
}

/** Create the four canonical folder markers in R2 so they appear in dashboards. */
export async function initFolders(): Promise<string[]> {
  const markers = UPLOAD_FOLDERS.map((f) => joinKey(PREFIX_RAW(), f) + "/");
  await Promise.all(markers.map((k) => putObject(k, "", "application/x-directory")));
  return [...UPLOAD_FOLDERS];
}

/**
 * Delete all raw images and annotation JSONs from R2.
 * Export files under PREFIX_EXP are left intact.
 */
export async function clearAllData(): Promise<{ raw: number; annotations: number }> {
  const [rawKeys, annKeys] = await Promise.all([
    listKeys(PREFIX_RAW()),
    listKeys(PREFIX_ANN()),
  ]);
  const [rawCount, annCount] = await Promise.all([
    deleteObjects(rawKeys),
    deleteObjects(annKeys),
  ]);
  return { raw: rawCount, annotations: annCount };
}

export async function exportYOLO() {
  const annotated = await loadAllAnnotated();

  const categoryMap: Record<string, number> = {};
  let catId = 0;
  const results: { file: string; annotations: number }[] = [];

  for (const { filename, data } of annotated) {
    if (data.status === "rejected") continue;

    const imgW = data.imageWidth || 1;
    const imgH = data.imageHeight || 1;
    const lines: string[] = [];

    for (const ann of data.annotations || []) {
      if (!(ann.label in categoryMap)) {
        categoryMap[ann.label] = catId++;
      }
      // YOLO only consumes axis-aligned boxes. Include bbox + ellipse (both
      // carry x/y/width/height) and derive a tight box for polygon/polyline.
      const yoloBox = deriveYoloBox(ann);
      if (yoloBox) {
        const [x, y, w, h] = yoloBox;
        const cx = (x + w / 2) / imgW;
        const cy = (y + h / 2) / imgH;
        lines.push(
          `${categoryMap[ann.label]} ${cx.toFixed(6)} ${cy.toFixed(6)} ${(w / imgW).toFixed(6)} ${(h / imgH).toFixed(6)}`,
        );
      }
    }

    if (lines.length > 0) {
      const yoloKey = expKey(filename.replace(/\.[^.]+$/, ".txt"));
      await putObject(yoloKey, lines.join("\n"), "text/plain");
      results.push({ file: filename, annotations: lines.length });
    }
  }

  const classes = Object.entries(categoryMap)
    .sort((a, b) => a[1] - b[1])
    .map((e) => e[0]);
  await putObject(expKey("classes.txt"), classes.join("\n"), "text/plain");
  return { format: "yolo", files: results.length, classes };
}

/**
 * Move an image (and its paired annotation JSON, if any) to a different
 * canonical folder. The basename is preserved.
 *
 * NOT atomic: copy/delete/manifest are independent R2 ops. On partial failure,
 * the StorageStepError names the failed step and we attempt to roll back the
 * dest copy if the source delete fails. Operator-visible via console.error.
 *
 * Returns the new filename. Throws InvalidFilenameError, NotFoundError,
 * AlreadyExistsError, or StorageStepError.
 */
export async function moveImage(filename: string, destFolder: UploadFolder): Promise<string> {
  if (!(UPLOAD_FOLDERS as readonly string[]).includes(destFolder)) {
    throw new InvalidFilenameError(destFolder, "not a valid destination folder");
  }
  validateActiveFilename(filename);
  const basename = filename.split("/").pop()!;

  const newFilename = `${destFolder}/${basename}`;
  if (newFilename === filename) return filename;

  const srcRaw = rawKey(filename);
  const dstRaw = rawKey(newFilename);
  if (await objectExists(dstRaw)) throw new AlreadyExistsError(newFilename);
  if (!(await objectExists(srcRaw))) throw new NotFoundError(filename);

  await step("move:copy-raw", { srcRaw, dstRaw }, () => copyObject(srcRaw, dstRaw));
  try {
    await step("move:delete-raw", { srcRaw }, () => deleteObject(srcRaw));
  } catch (err) {
    // Rollback dest so we don't leave a duplicate.
    await deleteObject(dstRaw).catch((rbErr) =>
      console.error("[storage] move:rollback-dest-raw failed", { dstRaw, rbErr }),
    );
    throw err;
  }

  const srcAnn = annKey(filename);
  const dstAnn = annKey(newFilename);
  if (await objectExists(srcAnn)) {
    await step("move:copy-ann", { srcAnn, dstAnn }, () => copyObject(srcAnn, dstAnn));
    await step("move:delete-ann", { srcAnn }, () => deleteObject(srcAnn));
  }

  await step("move:rename-manifest", { filename, newFilename }, () =>
    renameManifestEntry(filename, newFilename),
  );
  return newFilename;
}

/**
 * Soft-delete an image: move it (and its annotation) under _trash/ so it
 * disappears from the UI but can be recovered out-of-band by an operator.
 *
 * Always appends a timestamp to the trashed name so concurrent deletes can't
 * race the image and annotation onto different timestamps (which would break
 * the pairing). Returns the trashed key.
 */
export async function softDeleteImage(filename: string): Promise<string> {
  validateActiveFilename(filename);
  const srcRaw = rawKey(filename);
  if (!(await objectExists(srcRaw))) throw new NotFoundError(filename);

  const ext = path.extname(filename);
  const stem = filename.slice(0, filename.length - ext.length);
  const trashedName = `${TRASH_PREFIX}/${stem}.${Date.now()}${ext}`;
  const dstRaw = rawKey(trashedName);

  await step("trash:copy-raw", { srcRaw, dstRaw }, () => copyObject(srcRaw, dstRaw));
  try {
    await step("trash:delete-raw", { srcRaw }, () => deleteObject(srcRaw));
  } catch (err) {
    await deleteObject(dstRaw).catch((rbErr) =>
      console.error("[storage] trash:rollback-dest-raw failed", { dstRaw, rbErr }),
    );
    throw err;
  }

  const srcAnn = annKey(filename);
  if (await objectExists(srcAnn)) {
    const dstAnn = annKey(trashedName);
    await step("trash:copy-ann", { srcAnn, dstAnn }, () => copyObject(srcAnn, dstAnn));
    await step("trash:delete-ann", { srcAnn }, () => deleteObject(srcAnn));
  }

  await step("trash:delete-manifest", { filename }, () => deleteManifestEntry(filename));
  return trashedName;
}
