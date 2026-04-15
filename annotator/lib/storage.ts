import "server-only";
import path from "path";
import {
  PREFIX_ANN,
  PREFIX_EXP,
  PREFIX_RAW,
  getObjectBuffer,
  getObjectText,
  joinKey,
  listKeys,
  putObject,
} from "./r2";
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

export async function getImages(filter?: ImageStatus | "all"): Promise<ImageInfo[]> {
  const keys = await listKeys(PREFIX_RAW());
  const imageKeys = keys.filter(isImageKey).sort();

  // Fetch annotations in parallel. A malformed annotation JSON is treated as
  // "unannotated" rather than crashing the whole listing.
  const results = await Promise.all(
    imageKeys.map(async (k): Promise<ImageInfo> => {
      const filename = rawFilename(k);
      const data = await loadAnnotation(filename);
      if (!data) return { filename, status: "unannotated", annotationCount: 0 };
      return {
        filename,
        status: (data.status || "annotated") as ImageStatus,
        annotationCount: (data.annotations || []).length,
      };
    }),
  );

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
  await putObject(annKey(data.filename), JSON.stringify(data, null, 2), "application/json");
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
}

export async function getStats(): Promise<ProjectStats> {
  const keys = await listKeys(PREFIX_RAW());
  const imageKeys = keys.filter(isImageKey);

  const stats: ProjectStats = {
    total: imageKeys.length,
    unannotated: 0,
    annotated: 0,
    accepted: 0,
    rejected: 0,
  };

  const statuses = await Promise.all(
    imageKeys.map(async (k) => {
      const filename = rawFilename(k);
      const data = await loadAnnotation(filename);
      if (!data) return "unannotated" as ImageStatus;
      return (data.status || "annotated") as ImageStatus;
    }),
  );

  for (const s of statuses) {
    stats[s] = (stats[s] || 0) + 1;
  }
  return stats;
}

export async function getAllLabels(): Promise<string[]> {
  const keys = await listKeys(PREFIX_ANN());
  const jsonKeys = keys.filter((k) => k.endsWith(".json"));
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
 * Derive an axis-aligned [x, y, w, h] bounding box suitable for YOLO export.
 * Returns null if the shape has no meaningful 2D extent (e.g. keypoint).
 */
function deriveYoloBox(ann: import("./types").Annotation): [number, number, number, number] | null {
  if ((ann.type === "bbox" || ann.type === "ellipse") && ann.x != null && ann.y != null && ann.width && ann.height) {
    return [ann.x, ann.y, ann.width, ann.height];
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
  const imageKeys = keys.filter(isImageKey).sort();
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
        cocoAnn.bbox = [ann.x, ann.y, ann.width, ann.height];
        cocoAnn.area = (ann.width || 0) * (ann.height || 0);
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
