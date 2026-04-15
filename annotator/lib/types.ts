export type AnnotationType = "bbox" | "polygon" | "polyline" | "ellipse" | "keypoint";
export type ImageStatus = "unannotated" | "annotated" | "accepted" | "rejected";

export interface Annotation {
  id: string;
  type: AnnotationType;
  label: string;
  color: string;
  // bbox / ellipse (x,y is top-left of the bounding rect; width/height span it)
  x?: number;
  y?: number;
  width?: number;
  height?: number;
  // polygon (closed) / polyline (open)
  points?: number[][];
  // keypoint uses x, y from above
  attributes?: Record<string, string>;
  hidden?: boolean;
  locked?: boolean;
}

export interface HistoryEntry {
  action: string;
  comment: string;
  timestamp: string;
}

export interface ImageAnnotation {
  filename: string;
  annotations: Annotation[];
  labels: LabelDef[];
  status: ImageStatus;
  reviewComment: string;
  history: HistoryEntry[];
  imageWidth?: number;
  imageHeight?: number;
  lastModified?: string;
}

export interface ImageInfo {
  filename: string;
  status: ImageStatus;
  annotationCount: number;
}

export interface LabelDef {
  name: string;
  color: string;
}

export interface ProjectStats {
  total: number;
  unannotated: number;
  annotated: number;
  accepted: number;
  rejected: number;
}
