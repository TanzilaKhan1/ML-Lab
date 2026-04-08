import fs from "fs";
import path from "path";
import type { ImageAnnotation, ImageInfo, ImageStatus, ProjectStats } from "./types";

const RAW_DIR = path.join(process.cwd(), "public", "raw");
const ANNOTATIONS_DIR = path.join(process.cwd(), "annotations");
const EXPORTS_DIR = path.join(process.cwd(), "exports");

const IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff", ".svg"];

function ensureDirs() {
  for (const dir of [RAW_DIR, ANNOTATIONS_DIR, EXPORTS_DIR]) {
    if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
  }
}

function annotPath(filename: string) {
  return path.join(ANNOTATIONS_DIR, filename.replace(/\.[^.]+$/, ".json"));
}

export function getImages(filter?: ImageStatus | "all"): ImageInfo[] {
  ensureDirs();
  const files = fs.readdirSync(RAW_DIR)
    .filter((f) => IMAGE_EXTS.includes(path.extname(f).toLowerCase()))
    .sort();

  const images: ImageInfo[] = files.map((f) => {
    const ap = annotPath(f);
    let status: ImageStatus = "unannotated";
    let annotationCount = 0;
    if (fs.existsSync(ap)) {
      const data = JSON.parse(fs.readFileSync(ap, "utf8"));
      status = data.status || "annotated";
      annotationCount = (data.annotations || []).length;
    }
    return { filename: f, status, annotationCount };
  });

  if (filter && filter !== "all") {
    return images.filter((i) => i.status === filter);
  }
  return images;
}

export function getAnnotation(filename: string): ImageAnnotation {
  ensureDirs();
  const ap = annotPath(filename);
  if (fs.existsSync(ap)) {
    return JSON.parse(fs.readFileSync(ap, "utf8"));
  }
  return {
    filename,
    annotations: [],
    labels: [],
    status: "unannotated",
    reviewComment: "",
    history: [],
  };
}

export function saveAnnotation(data: ImageAnnotation): void {
  ensureDirs();
  data.lastModified = new Date().toISOString();
  fs.writeFileSync(annotPath(data.filename), JSON.stringify(data, null, 2));
}

export function saveUploadedImage(filename: string, buffer: Buffer): void {
  ensureDirs();
  fs.writeFileSync(path.join(RAW_DIR, filename), buffer);
}

export function updateStatus(filename: string, status: ImageStatus, comment?: string): void {
  ensureDirs();
  const ap = annotPath(filename);
  let data: ImageAnnotation;
  if (fs.existsSync(ap)) {
    data = JSON.parse(fs.readFileSync(ap, "utf8"));
  } else {
    data = { filename, annotations: [], labels: [], status: "unannotated", reviewComment: "", history: [] };
  }
  data.status = status;
  if (comment) {
    data.reviewComment = comment;
    data.history = data.history || [];
    data.history.push({ action: status, comment, timestamp: new Date().toISOString() });
  }
  data.lastModified = new Date().toISOString();
  fs.writeFileSync(ap, JSON.stringify(data, null, 2));
}

export function getStats(): ProjectStats {
  ensureDirs();
  const files = fs.readdirSync(RAW_DIR)
    .filter((f) => IMAGE_EXTS.includes(path.extname(f).toLowerCase()));

  const stats: ProjectStats = { total: files.length, unannotated: 0, annotated: 0, accepted: 0, rejected: 0 };
  for (const f of files) {
    const ap = annotPath(f);
    if (fs.existsSync(ap)) {
      const data = JSON.parse(fs.readFileSync(ap, "utf8"));
      const s = (data.status || "annotated") as ImageStatus;
      stats[s] = (stats[s] || 0) + 1;
    } else {
      stats.unannotated++;
    }
  }
  return stats;
}

export function getAllLabels(): string[] {
  ensureDirs();
  const labelSet = new Set<string>();
  if (!fs.existsSync(ANNOTATIONS_DIR)) return [];
  const annotFiles = fs.readdirSync(ANNOTATIONS_DIR).filter((f) => f.endsWith(".json"));
  for (const f of annotFiles) {
    const data = JSON.parse(fs.readFileSync(path.join(ANNOTATIONS_DIR, f), "utf8"));
    for (const ann of data.annotations || []) {
      if (ann.label) labelSet.add(ann.label);
    }
  }
  return [...labelSet].sort();
}

export function exportCOCO() {
  ensureDirs();
  const files = fs.readdirSync(RAW_DIR)
    .filter((f) => IMAGE_EXTS.includes(path.extname(f).toLowerCase()));

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

  files.forEach((f, idx) => {
    const ap = annotPath(f);
    if (!fs.existsSync(ap)) return;
    const data: ImageAnnotation = JSON.parse(fs.readFileSync(ap, "utf8"));
    if (data.status === "rejected") return;

    const imageId = idx + 1;
    (coco.images as Record<string, unknown>[]).push({
      id: imageId, file_name: f, width: data.imageWidth || 0, height: data.imageHeight || 0,
    });

    for (const ann of data.annotations || []) {
      if (!categoryMap[ann.label]) {
        categoryMap[ann.label] = catId;
        (coco.categories as Record<string, unknown>[]).push({ id: catId, name: ann.label, supercategory: "none" });
        catId++;
      }

      const cocoAnn: Record<string, unknown> = {
        id: annId++, image_id: imageId, category_id: categoryMap[ann.label],
        iscrowd: 0,
      };

      if (ann.type === "bbox") {
        cocoAnn.bbox = [ann.x, ann.y, ann.width, ann.height];
        cocoAnn.area = (ann.width || 0) * (ann.height || 0);
      } else if (ann.type === "polygon" && ann.points) {
        const flat = ann.points.flat();
        cocoAnn.segmentation = [flat];
        // Compute bounding box from polygon points
        const xs = ann.points.map((p: number[]) => p[0]);
        const ys = ann.points.map((p: number[]) => p[1]);
        const minX = Math.min(...xs), minY = Math.min(...ys);
        const maxX = Math.max(...xs), maxY = Math.max(...ys);
        cocoAnn.bbox = [minX, minY, maxX - minX, maxY - minY];
        // Shoelace formula for polygon area
        let area = 0;
        for (let i = 0; i < ann.points.length; i++) {
          const j = (i + 1) % ann.points.length;
          area += ann.points[i][0] * ann.points[j][1];
          area -= ann.points[j][0] * ann.points[i][1];
        }
        cocoAnn.area = Math.abs(area / 2);
      } else if (ann.type === "keypoint") {
        cocoAnn.keypoints = [ann.x, ann.y, 2];
        cocoAnn.num_keypoints = 1;
        cocoAnn.bbox = [ann.x, ann.y, 0, 0];
        cocoAnn.area = 0;
      }

      (coco.annotations as Record<string, unknown>[]).push(cocoAnn);
    }
  });

  const exportPath = path.join(EXPORTS_DIR, "coco_export.json");
  fs.writeFileSync(exportPath, JSON.stringify(coco, null, 2));
  return coco;
}

export function exportYOLO() {
  ensureDirs();
  const files = fs.readdirSync(RAW_DIR)
    .filter((f) => IMAGE_EXTS.includes(path.extname(f).toLowerCase()));

  const categoryMap: Record<string, number> = {};
  let catId = 0;
  const results: { file: string; annotations: number }[] = [];

  for (const f of files) {
    const ap = annotPath(f);
    if (!fs.existsSync(ap)) continue;
    const data: ImageAnnotation = JSON.parse(fs.readFileSync(ap, "utf8"));
    if (data.status === "rejected") continue;

    const imgW = data.imageWidth || 1;
    const imgH = data.imageHeight || 1;
    const lines: string[] = [];

    for (const ann of data.annotations || []) {
      if (!categoryMap.hasOwnProperty(ann.label)) {
        categoryMap[ann.label] = catId++;
      }
      if (ann.type === "bbox" && ann.x != null && ann.y != null && ann.width && ann.height) {
        const cx = (ann.x + ann.width / 2) / imgW;
        const cy = (ann.y + ann.height / 2) / imgH;
        const w = ann.width / imgW;
        const h = ann.height / imgH;
        lines.push(`${categoryMap[ann.label]} ${cx.toFixed(6)} ${cy.toFixed(6)} ${w.toFixed(6)} ${h.toFixed(6)}`);
      }
    }

    if (lines.length > 0) {
      const yoloPath = path.join(EXPORTS_DIR, f.replace(/\.[^.]+$/, ".txt"));
      fs.writeFileSync(yoloPath, lines.join("\n"));
      results.push({ file: f, annotations: lines.length });
    }
  }

  const classes = Object.entries(categoryMap).sort((a, b) => a[1] - b[1]).map((e) => e[0]);
  fs.writeFileSync(path.join(EXPORTS_DIR, "classes.txt"), classes.join("\n"));
  return { format: "yolo", files: results.length, classes };
}
