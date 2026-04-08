export type AnnotationType = "bbox" | "polygon" | "keypoint";
export type ImageStatus = "unannotated" | "annotated" | "accepted" | "rejected";

export interface Annotation {
  id: string;
  type: AnnotationType;
  label: string;
  color: string;
  // bbox
  x?: number;
  y?: number;
  width?: number;
  height?: number;
  // polygon
  points?: number[][];
  // keypoint
  // uses x, y from above
  attributes?: Record<string, string>;
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
