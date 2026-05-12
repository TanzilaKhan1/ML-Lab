"use client";

import { useEffect, useRef, useState, useCallback, memo } from "react";
import type {
  Annotation,
  ImageAnnotation,
  ImageInfo,
  ImageStatus,
  LabelDef,
  ProjectStats,
} from "@/lib/types";
import { v4 as uuidv4 } from "uuid";
import { toast } from "sonner";
import {
  MousePointer2,
  Square,
  Hexagon,
  Spline,
  Circle as CircleIcon,
  Crosshair,
  Hand,
  ZoomIn,
  ZoomOut,
  Maximize2,
  Undo2,
  Redo2,
  Trash2,
  Eye,
  EyeOff,
  ChevronLeft,
  ChevronRight,
  Settings,
  Keyboard,
  FileDown,
  ImageIcon,
  Upload,
  MoreVertical,
  FolderInput,
  PanelRight,
} from "lucide-react";

import { Button } from "@/components/ui/button";
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
  SheetTrigger,
} from "@/components/ui/sheet";
import { Input } from "@/components/ui/input";
import { Slider } from "@/components/ui/slider";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarHeader,
  SidebarInset,
  SidebarProvider,
  SidebarTrigger,
} from "@/components/ui/sidebar";
import { Checkbox } from "@/components/ui/checkbox";
import { Label } from "@/components/ui/label";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
} from "@/components/ui/dialog";
import { cn } from "@/lib/utils";
import { ShortcutsDialog } from "@/components/panels/shortcuts-dialog";
import { AdjustmentsPanel } from "@/components/panels/adjustments-panel";
import { LabelsPanel } from "@/components/panels/labels-panel";
import { AnnotationsList } from "@/components/panels/annotations-list";
import { ReviewPanel } from "@/components/panels/review-panel";
import { PropertiesPanel } from "@/components/panels/properties-panel";

/* eslint-disable @typescript-eslint/no-explicit-any */

type Tool = "select" | "bbox" | "polygon" | "polyline" | "ellipse" | "keypoint" | "pan";

/** Encode each path segment without encoding the separating slashes. */
function encodeFilePath(filename: string): string {
  return filename.split("/").map(encodeURIComponent).join("/");
}

const UPLOAD_FOLDERS = ["bus/positive", "bus/negative", "legua/positive", "legua/negative"] as const;
type UploadFolder = (typeof UPLOAD_FOLDERS)[number];

/** Undo/redo snapshots now cover annotations AND labels, so label add/delete
 *  and color changes round-trip through undo. */
type Snapshot = { annotations: Annotation[]; labels: LabelDef[] };

/** Skip keyboard shortcuts when the user is editing text or a modal/select is
 *  open. Input/Textarea/contenteditable + Radix Dialog + Radix Select/Menu
 *  all register as non-shortcut contexts. */
function isEditingField(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false;
  if (target instanceof HTMLInputElement || target instanceof HTMLTextAreaElement) return true;
  if (target.isContentEditable) return true;
  if (target.closest('[role="textbox"], [role="combobox"], [role="listbox"], [role="menu"]')) return true;
  return false;
}

function anyModalOpen(): boolean {
  if (typeof document === "undefined") return false;
  return document.querySelector('[role="dialog"][data-state="open"]') !== null;
}

const DEFAULT_LABELS: LabelDef[] = [
  { name: "safe",    color: "#82E0AA" },
  { name: "unsafe",  color: "#FF6B6B" },
  { name: "license", color: "#DDA0DD" },
  { name: "person",  color: "#4ECDC4" },
  { name: "text",    color: "#FFEAA7" },
];

const LABEL_COLORS = [
  "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7",
  "#DDA0DD", "#98D8C8", "#F7DC6F", "#BB8FCE", "#85C1E9",
  "#F0B27A", "#82E0AA", "#F1948A", "#AED6F1", "#D2B4DE",
];

const TOOLS: { id: Tool; label: string; shortcut: string; icon: React.ComponentType<{ className?: string }> }[] = [
  { id: "select", label: "Select", shortcut: "V", icon: MousePointer2 },
  { id: "bbox", label: "Box", shortcut: "B", icon: Square },
  { id: "polygon", label: "Polygon", shortcut: "P", icon: Hexagon },
  { id: "polyline", label: "Polyline", shortcut: "L", icon: Spline },
  { id: "ellipse", label: "Ellipse", shortcut: "E", icon: CircleIcon },
  { id: "keypoint", label: "Keypoint", shortcut: "K", icon: Crosshair },
  { id: "pan", label: "Pan", shortcut: "G", icon: Hand },
];

/**
 * Lazy thumbnail — fetches the full raw image from R2 only when scrolled into
 * view. Browser cache + Cache-Control make repeat navigations cheap.
 */
const ImageThumb = memo(function ImageThumb({ filename }: { filename: string }) {
  const ref = useRef<HTMLDivElement>(null);
  const [visible, setVisible] = useState(false);
  const [loaded, setLoaded] = useState(false);

  useEffect(() => {
    const el = ref.current;
    if (!el || visible) return;
    const io = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setVisible(true);
          io.disconnect();
        }
      },
      { root: el.closest(".sidebar-images-scroll") || null, rootMargin: "200px" },
    );
    io.observe(el);
    return () => io.disconnect();
  }, [visible]);

  return (
    <div
      ref={ref}
      className="w-10 h-10 rounded-md bg-background border border-border shrink-0 overflow-hidden flex items-center justify-center relative"
    >
      {visible ? (
        <>
          {!loaded && <span className="absolute inset-0 skeleton-shimmer" />}
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={`/api/derived/thumb/${encodeFilePath(filename)}`}
            alt=""
            className={cn(
              "w-full h-full object-cover transition-opacity duration-300",
              loaded ? "opacity-100" : "opacity-0",
            )}
            loading="lazy"
            decoding="async"
            draggable={false}
            onLoad={() => setLoaded(true)}
            onError={() => setLoaded(true)}
          />
        </>
      ) : (
        <ImageIcon className="size-4 text-muted-foreground/50" />
      )}
    </div>
  );
});

const statusDot: Record<ImageStatus, string> = {
  unannotated: "bg-muted-foreground",
  annotated: "bg-primary",
  accepted: "bg-emerald-400",
  rejected: "bg-red-400",
};

export default function AnnotatorPage() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const fabricRef = useRef<any>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  const [images, setImages] = useState<ImageInfo[]>([]);
  const [currentImage, setCurrentImage] = useState<string | null>(null);
  const [annotations, setAnnotations] = useState<Annotation[]>([]);
  const [labels, setLabels] = useState<LabelDef[]>(DEFAULT_LABELS);
  const [activeLabel, setActiveLabel] = useState<string>(DEFAULT_LABELS[0].name);
  const [activeTool, setActiveTool] = useState<Tool>("select");
  const [stats, setStats] = useState<ProjectStats>({ total: 0, unannotated: 0, annotated: 0, accepted: 0, rejected: 0 });
  const [filter, setFilter] = useState<ImageStatus | "all">("all");
  const [selectedAnnotation, setSelectedAnnotation] = useState<string | null>(null);
  const [imageStatus, setImageStatus] = useState<ImageStatus>("unannotated");
  const [reviewComment, setReviewComment] = useState("");
  const [reviewHistory, setReviewHistory] = useState<{ action: string; comment: string; timestamp: string }[]>([]);
  const [annotationsVisible, setAnnotationsVisible] = useState(true);
  const [zoomLevel, setZoomLevel] = useState(100);
  const [cursorPos, setCursorPos] = useState({ x: 0, y: 0 });
  const [newLabelName, setNewLabelName] = useState("");
  const [newLabelColor, setNewLabelColor] = useState(LABEL_COLORS[5]);
  const [showExportModal, setShowExportModal] = useState(false);
  const [showShortcutsModal, setShowShortcutsModal] = useState(false);
  const [showSettingsModal, setShowSettingsModal] = useState(false);
  const [mobileInspectorOpen, setMobileInspectorOpen] = useState(false);
  const [exportResult, setExportResult] = useState<string | null>(null);
  const [saveIndicator, setSaveIndicator] = useState("Auto-save on");
  const [brightness, setBrightness] = useState(100);
  const [contrast, setContrast] = useState(100);
  const [opacity, setOpacity] = useState(0);
  const [autoAdvance, setAutoAdvance] = useState(false);
  const [isDragOver, setIsDragOver] = useState(false);
  const [uploadFolder, setUploadFolder] = useState<UploadFolder>("bus/positive");
  const [showClearConfirm, setShowClearConfirm] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");
  const [labelFilter, setLabelFilter] = useState<string | null>(null);
  const [clipboard, setClipboard] = useState<Annotation | null>(null);
  const [attrDraftKey, setAttrDraftKey] = useState("");
  const [attrDraftVal, setAttrDraftVal] = useState("");
  const [pendingLabelDelete, setPendingLabelDelete] = useState<{ name: string; count: number } | null>(null);
  const [imageDimState, setImageDimState] = useState({ width: 0, height: 0 });
  const [isLoading, setIsLoading] = useState(false);

  const undoStack = useRef<Snapshot[]>([]);
  const redoStack = useRef<Snapshot[]>([]);
  const polygonPoints = useRef<number[][]>([]);
  const polygonLines = useRef<any[]>([]);
  const polygonDots = useRef<any[]>([]);
  const polygonCloseLine = useRef<any>(null);
  const polygonFirstDot = useRef<any>(null);
  const isDrawing = useRef(false);
  const drawStart = useRef<{ x: number; y: number } | null>(null);
  const tempRect = useRef<any>(null);
  const imageDims = useRef<{ width: number; height: number }>({ width: 0, height: 0 });
  const autoSaveTimerRef = useRef<NodeJS.Timeout | null>(null);
  const autoSaveAbortRef = useRef<AbortController | null>(null);
  // Per-image-load abort: torn down on stale so previous loads stop
  // consuming bandwidth/decoding when the user switches images quickly.
  const loadAbortRef = useRef<AbortController | null>(null);
  // O(1) label lookup for object:modified — populated by renderAnnotations,
  // keyed by annotationId, value is the FabricText label object.
  const labelMap = useRef<Map<string, any>>(new Map());
  const isPanning = useRef(false);
  const panStart = useRef<{ x: number; y: number } | null>(null);
  const spaceHeld = useRef(false);
  const prevToolRef = useRef<Tool>("select");

  const activeToolRef = useRef(activeTool);
  const activeLabelRef = useRef(activeLabel);
  const labelsRef = useRef(labels);
  const annotationsRef = useRef(annotations);
  const currentImageRef = useRef(currentImage);
  const opacityRef = useRef(opacity);
  const autoAdvanceRef = useRef(autoAdvance);
  const imagesRef = useRef(images);
  const autoAdvancePending = useRef(false);
  const reviewCommentRef = useRef(reviewComment);
  const reviewHistoryRef = useRef(reviewHistory);
  const imageStatusRef = useRef(imageStatus);

  useEffect(() => { activeToolRef.current = activeTool; }, [activeTool]);
  useEffect(() => { activeLabelRef.current = activeLabel; }, [activeLabel]);
  useEffect(() => { labelsRef.current = labels; }, [labels]);
  useEffect(() => { annotationsRef.current = annotations; }, [annotations]);
  useEffect(() => { currentImageRef.current = currentImage; }, [currentImage]);
  useEffect(() => { autoAdvanceRef.current = autoAdvance; }, [autoAdvance]);
  useEffect(() => { imagesRef.current = images; }, [images]);
  useEffect(() => { opacityRef.current = opacity; }, [opacity]);
  useEffect(() => { reviewCommentRef.current = reviewComment; }, [reviewComment]);
  useEffect(() => { reviewHistoryRef.current = reviewHistory; }, [reviewHistory]);
  useEffect(() => { imageStatusRef.current = imageStatus; }, [imageStatus]);

  // Flush the current image's state immediately when the tab is closed or hidden.
  // Uses keepalive so the request survives page unload.
  // All reads go through refs so the effect never needs to re-register.
  useEffect(() => {
    const flush = () => {
      const filename = currentImageRef.current;
      if (!filename) return;
      if (autoSaveTimerRef.current) {
        clearTimeout(autoSaveTimerRef.current);
        autoSaveTimerRef.current = null;
      }
      const s = imageStatusRef.current;
      fetch(`/api/annotations/${encodeFilePath(filename)}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        keepalive: true,
        body: JSON.stringify({
          filename,
          annotations: annotationsRef.current,
          labels: labelsRef.current,
          status: s === "accepted" || s === "rejected" ? s : annotationsRef.current.length > 0 ? "annotated" : "unannotated",
          reviewComment: reviewCommentRef.current,
          history: reviewHistoryRef.current,
          imageWidth: imageDims.current.width,
          imageHeight: imageDims.current.height,
        }),
      }).catch(() => {});
    };
    const onHide = () => { if (document.visibilityState === "hidden") flush(); };
    window.addEventListener("beforeunload", flush);
    document.addEventListener("visibilitychange", onHide);
    return () => {
      window.removeEventListener("beforeunload", flush);
      document.removeEventListener("visibilitychange", onHide);
    };
  }, []); // refs only — intentionally empty deps

  const fetchImages = useCallback(async () => {
    const res = await fetch(`/api/images?filter=${filter}`);
    setImages(await res.json());
  }, [filter]);

  const fetchStats = useCallback(async () => {
    const res = await fetch("/api/stats");
    setStats(await res.json());
  }, []);

  useEffect(() => { fetchImages(); fetchStats(); }, [fetchImages, fetchStats]);

  // Brightness/contrast applied via CSS filter on the canvas DOM layer, so they
  // affect the displayed image but not the exported annotation pixel values.
  useEffect(() => {
    const container = containerRef.current;
    if (container) {
      const canvasEl = container.querySelector(".canvas-container") as HTMLElement;
      if (canvasEl) {
        canvasEl.style.filter = `brightness(${brightness}%) contrast(${contrast}%)`;
      }
    }
  }, [brightness, contrast]);

  const getLabelColor = useCallback((name: string) => {
    const label = labelsRef.current.find((l) => l.name === name);
    return label?.color || "#FF6B6B";
  }, []);

  const renderAnnotations = useCallback(async (anns: Annotation[], canvas?: any) => {
    const c = canvas || fabricRef.current;
    if (!c) return;
    const fabric = await import("fabric");

    const toRemove = c.getObjects().filter((o: any) => !o.isBackgroundImage);
    for (const obj of toRemove) c.remove(obj);
    labelMap.current.clear();

    const opacityHex = Math.round((opacityRef.current / 100) * 255).toString(16).padStart(2, "0");

    for (const ann of anns) {
      if (ann.hidden) continue;
      const interactive = activeToolRef.current === "select" && !ann.locked;

      if (ann.type === "bbox") {
        const rect = new fabric.Rect({
          left: ann.x, top: ann.y, width: ann.width, height: ann.height,
          angle: ann.angle || 0,
          fill: ann.color + opacityHex, stroke: ann.color, strokeWidth: 2,
          strokeUniform: true,
          selectable: interactive, evented: interactive,
          cornerColor: "#fff", cornerStrokeColor: ann.color, cornerSize: 8,
          transparentCorners: false, borderColor: ann.color,
          lockRotation: false,
          // Static (non-selected/non-interactive) shapes get raster-cached so
          // the canvas only re-rasterizes on modify, not every frame.
          objectCaching: !interactive,
          noScaleCache: false,
        });
        // Make the rotation handle clearly visible: a larger circle rendered
        // 44px above the top-center of the bounding box.
        if (interactive) {
          (rect as any).controls = {
            ...(rect as any).controls,
            mtr: new (fabric as any).Control({
              x: 0,
              y: -0.5,
              offsetY: -44,
              withConnection: true,
              actionName: "rotate",
              cursorStyle: "crosshair",
              actionHandler: (fabric as any).controlsUtils.rotationWithSnapping,
              render: (ctx: CanvasRenderingContext2D, left: number, top: number) => {
                const size = 18;
                ctx.save();
                ctx.translate(left, top);
                ctx.beginPath();
                ctx.arc(0, 0, size / 2, 0, Math.PI * 2);
                ctx.fillStyle = "#fff";
                ctx.fill();
                ctx.strokeStyle = ann.color;
                ctx.lineWidth = 2;
                ctx.stroke();
                // draw a small rotation arrow arc
                ctx.beginPath();
                ctx.arc(0, 0, 4, -Math.PI * 0.75, Math.PI * 0.25);
                ctx.strokeStyle = ann.color;
                ctx.lineWidth = 1.5;
                ctx.stroke();
                // arrowhead
                ctx.beginPath();
                ctx.moveTo(3, -3);
                ctx.lineTo(5, -1);
                ctx.lineTo(2, 0);
                ctx.closePath();
                ctx.fillStyle = ann.color;
                ctx.fill();
                ctx.restore();
              },
            }),
          };
        }
        (rect as any).annotationId = ann.id;
        (rect as any).annotationType = "bbox";
        // Place the label at local (2, -20) relative to the rect's top-left
        // origin, then rotate by the same angle so it hugs the top edge.
        const angle = ann.angle || 0;
        const rad = (angle * Math.PI) / 180;
        const lx = 2, ly = -20;
        const textLeft = (ann.x || 0) + lx * Math.cos(rad) - ly * Math.sin(rad);
        const textTop  = (ann.y || 0) + lx * Math.sin(rad) + ly * Math.cos(rad);
        const text = new fabric.FabricText(ann.label, {
          left: textLeft, top: textTop, angle,
          fontSize: 12, fill: "#fff", backgroundColor: ann.color + "DD",
          padding: 3, selectable: false, evented: false,
          fontFamily: "system-ui, -apple-system, Segoe UI, Helvetica, Arial, sans-serif",
        });
        (text as any).isLabel = true;
        (text as any).annotationId = ann.id;
        labelMap.current.set(ann.id, text);
        c.add(rect); c.add(text);
      } else if (ann.type === "ellipse") {
        const rx = (ann.width || 0) / 2;
        const ry = (ann.height || 0) / 2;
        const ellipse = new fabric.Ellipse({
          left: ann.x, top: ann.y, rx, ry,
          fill: ann.color + opacityHex, stroke: ann.color, strokeWidth: 2,
          strokeUniform: true,
          selectable: interactive, evented: interactive,
          cornerColor: "#fff", cornerStrokeColor: ann.color, cornerSize: 8,
          transparentCorners: false, borderColor: ann.color,
          lockRotation: true,
          objectCaching: !interactive,
          noScaleCache: false,
        });
        (ellipse as any).annotationId = ann.id;
        (ellipse as any).annotationType = "ellipse";
        const text = new fabric.FabricText(ann.label, {
          left: (ann.x || 0) + 2, top: (ann.y || 0) - 20,
          fontSize: 12, fill: "#fff", backgroundColor: ann.color + "DD",
          padding: 3, selectable: false, evented: false,
          fontFamily: "system-ui, -apple-system, Segoe UI, Helvetica, Arial, sans-serif",
        });
        (text as any).isLabel = true;
        (text as any).annotationId = ann.id;
        labelMap.current.set(ann.id, text);
        c.add(ellipse); c.add(text);
      } else if (ann.type === "polygon" && ann.points) {
        const points = ann.points.map((p: number[]) => new fabric.Point(p[0], p[1]));
        const polygon = new fabric.Polygon(points, {
          fill: ann.color + opacityHex, stroke: ann.color, strokeWidth: 2,
          strokeUniform: true,
          selectable: interactive, evented: interactive,
          lockRotation: true,
          // Static polygons cache fine; interactive ones disable cache so
          // per-vertex drag mutations stay pixel-accurate.
          objectCaching: !interactive,
          noScaleCache: false,
          cornerColor: "#fff", cornerStrokeColor: ann.color, cornerSize: 9,
          transparentCorners: false, borderColor: ann.color,
          cornerStyle: "circle",
        });
        (polygon as any).annotationId = ann.id;
        (polygon as any).annotationType = "polygon";
        if (interactive) {
          polygon.controls = (fabric as any).controlsUtils.createPolyControls(polygon);
        }
        const bounds = polygon.getBoundingRect();
        const text = new fabric.FabricText(ann.label, {
          left: bounds.left + 2, top: bounds.top - 20,
          fontSize: 12, fill: "#fff", backgroundColor: ann.color + "DD",
          padding: 3, selectable: false, evented: false,
          fontFamily: "system-ui, -apple-system, Segoe UI, Helvetica, Arial, sans-serif",
        });
        (text as any).isLabel = true;
        (text as any).annotationId = ann.id;
        labelMap.current.set(ann.id, text);
        c.add(polygon); c.add(text);
      } else if (ann.type === "polyline" && ann.points) {
        const points = ann.points.map((p: number[]) => new fabric.Point(p[0], p[1]));
        const polyline = new fabric.Polyline(points, {
          fill: "transparent", stroke: ann.color, strokeWidth: 3,
          strokeUniform: true,
          selectable: interactive, evented: interactive,
          lockRotation: true,
          objectCaching: !interactive,
          noScaleCache: false,
          cornerColor: "#fff", cornerStrokeColor: ann.color, cornerSize: 9,
          transparentCorners: false, borderColor: ann.color,
          cornerStyle: "circle",
        });
        (polyline as any).annotationId = ann.id;
        (polyline as any).annotationType = "polyline";
        if (interactive) {
          polyline.controls = (fabric as any).controlsUtils.createPolyControls(polyline);
        }
        const bounds = polyline.getBoundingRect();
        const text = new fabric.FabricText(ann.label, {
          left: bounds.left + 2, top: bounds.top - 20,
          fontSize: 12, fill: "#fff", backgroundColor: ann.color + "DD",
          padding: 3, selectable: false, evented: false,
          fontFamily: "system-ui, -apple-system, Segoe UI, Helvetica, Arial, sans-serif",
        });
        (text as any).isLabel = true;
        (text as any).annotationId = ann.id;
        labelMap.current.set(ann.id, text);
        c.add(polyline); c.add(text);
      } else if (ann.type === "keypoint") {
        const outerCircle = new fabric.Circle({
          left: (ann.x || 0) - 10, top: (ann.y || 0) - 10, radius: 10,
          fill: "transparent", stroke: ann.color, strokeWidth: 2,
          strokeUniform: true,
          selectable: false, evented: false,
        });
        const circle = new fabric.Circle({
          left: (ann.x || 0) - 4, top: (ann.y || 0) - 4, radius: 4,
          fill: ann.color, stroke: "#fff", strokeWidth: 1,
          strokeUniform: true,
          selectable: interactive, evented: interactive,
        });
        (circle as any).annotationId = ann.id;
        (circle as any).annotationType = "keypoint";
        const center = new fabric.Circle({
          left: (ann.x || 0) - 1, top: (ann.y || 0) - 1, radius: 1,
          fill: "#fff", strokeWidth: 0,
          selectable: false, evented: false,
        });
        // Decoration only; not a label, has no annotationId. Skipped by the
        // `isLabel && annotationId` guard in object:modified's label-finder.
        (center as any).isDecoration = true;
        const text = new fabric.FabricText(ann.label, {
          left: (ann.x || 0) + 14, top: (ann.y || 0) - 8,
          fontSize: 11, fill: "#fff", backgroundColor: ann.color + "DD",
          padding: 2, selectable: false, evented: false,
          fontFamily: "system-ui, -apple-system, Segoe UI, Helvetica, Arial, sans-serif",
        });
        (text as any).isLabel = true;
        (text as any).annotationId = ann.id;
        labelMap.current.set(ann.id, text);
        c.add(outerCircle); c.add(circle); c.add(center); c.add(text);
      }
    }
    c.requestRenderAll();
  }, []);

  /**
   * Live-update fills when the opacity slider moves, without tearing down and
   * rebuilding every Fabric object. Cheap even with hundreds of annotations.
   */
  useEffect(() => {
    const c = fabricRef.current;
    if (!c) return;
    const opacityHex = Math.round((opacity / 100) * 255).toString(16).padStart(2, "0");
    const annMap = new Map(annotationsRef.current.map((a) => [a.id, a]));
    c.getObjects().forEach((obj: any) => {
      if (!obj.annotationId) return;
      const ann = annMap.get(obj.annotationId);
      if (!ann) return;
      if (ann.type === "bbox" || ann.type === "ellipse" || ann.type === "polygon") {
        obj.set("fill", ann.color + opacityHex);
      }
    });
    c.requestRenderAll();
  }, [opacity]);

  // fast = 150 ms debounce for structural edits (draw, delete, resize, status);
  // default 600 ms is for keystroke-driven updates (review comment, attributes).
  const scheduleAutoSave = useCallback((anns: Annotation[], fast: boolean = false) => {
    if (autoSaveTimerRef.current) clearTimeout(autoSaveTimerRef.current);
    const targetFilename = currentImageRef.current;
    if (!targetFilename) return;
    const width = imageDims.current.width;
    const height = imageDims.current.height;
    setSaveIndicator("Saving...");
    autoSaveTimerRef.current = setTimeout(async () => {
      autoSaveAbortRef.current?.abort();
      const ac = new AbortController();
      autoSaveAbortRef.current = ac;
      try {
        // Preserve existing image status if it's already accepted/rejected;
        // autosave shouldn't silently revert a reviewed image to "annotated".
        const currentStatus = imageStatusRef.current;
        const nextStatus: ImageStatus =
          currentStatus === "accepted" || currentStatus === "rejected"
            ? currentStatus
            : anns.length > 0 ? "annotated" : "unannotated";
        await fetch(`/api/annotations/${encodeFilePath(targetFilename)}`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            filename: targetFilename,
            annotations: anns,
            labels: labelsRef.current,
            status: nextStatus,
            reviewComment: reviewCommentRef.current,
            history: reviewHistoryRef.current,
            imageWidth: width,
            imageHeight: height,
          }),
          signal: ac.signal,
        });
        setSaveIndicator("Saved");
        setTimeout(() => setSaveIndicator("Auto-save on"), 1500);
        fetchImages(); fetchStats();
      } catch (err) {
        if ((err as { name?: string })?.name !== "AbortError") {
          setSaveIndicator("Save failed");
          toast.error("Save failed", { description: String(err) });
        }
      }
    }, fast ? 150 : 600);
  }, [fetchImages, fetchStats]);

  const snapshot = useCallback((): Snapshot => ({
    annotations: JSON.parse(JSON.stringify(annotationsRef.current)),
    labels: JSON.parse(JSON.stringify(labelsRef.current)),
  }), []);

  const applySnapshot = useCallback((s: Snapshot) => {
    setAnnotations(s.annotations); annotationsRef.current = s.annotations;
    setLabels(s.labels); labelsRef.current = s.labels;
    renderAnnotations(s.annotations); scheduleAutoSave(s.annotations, true);
  }, [renderAnnotations, scheduleAutoSave]);

  const pushUndo = useCallback(() => {
    undoStack.current.push(snapshot());
    redoStack.current = [];
  }, [snapshot]);

  const undo = useCallback(() => {
    if (undoStack.current.length === 0) return;
    redoStack.current.push(snapshot());
    applySnapshot(undoStack.current.pop()!);
  }, [snapshot, applySnapshot]);

  const redo = useCallback(() => {
    if (redoStack.current.length === 0) return;
    undoStack.current.push(snapshot());
    applySnapshot(redoStack.current.pop()!);
  }, [snapshot, applySnapshot]);

  const deleteSelected = useCallback(() => {
    if (!selectedAnnotation) return;
    pushUndo();
    const newAnns = annotationsRef.current.filter((a) => a.id !== selectedAnnotation);
    setAnnotations(newAnns); annotationsRef.current = newAnns;
    setSelectedAnnotation(null);
    renderAnnotations(newAnns); scheduleAutoSave(newAnns, true);
  }, [selectedAnnotation, pushUndo, renderAnnotations, scheduleAutoSave]);

  const copySelected = useCallback(() => {
    const ann = annotationsRef.current.find((a) => a.id === selectedAnnotation);
    if (!ann) return;
    setClipboard(JSON.parse(JSON.stringify(ann)));
    toast.success("Copied annotation");
  }, [selectedAnnotation]);

  const pasteClipboard = useCallback(() => {
    if (!clipboard) return;
    pushUndo();
    const offset = 12;
    const cloned: Annotation = JSON.parse(JSON.stringify(clipboard));
    cloned.id = uuidv4();
    if (cloned.points) {
      cloned.points = cloned.points.map(([x, y]) => [x + offset, y + offset]);
    } else {
      if (cloned.x != null) cloned.x += offset;
      if (cloned.y != null) cloned.y += offset;
    }
    const newAnns = [...annotationsRef.current, cloned];
    setAnnotations(newAnns); annotationsRef.current = newAnns;
    setSelectedAnnotation(cloned.id);
    renderAnnotations(newAnns); scheduleAutoSave(newAnns, true);
  }, [clipboard, pushUndo, renderAnnotations, scheduleAutoSave]);

  const duplicateSelected = useCallback(() => {
    const ann = annotationsRef.current.find((a) => a.id === selectedAnnotation);
    if (!ann) return;
    pushUndo();
    const offset = 12;
    const cloned: Annotation = JSON.parse(JSON.stringify(ann));
    cloned.id = uuidv4();
    if (cloned.points) {
      cloned.points = cloned.points.map(([x, y]) => [x + offset, y + offset]);
    } else {
      if (cloned.x != null) cloned.x += offset;
      if (cloned.y != null) cloned.y += offset;
    }
    const newAnns = [...annotationsRef.current, cloned];
    setAnnotations(newAnns); annotationsRef.current = newAnns;
    setSelectedAnnotation(cloned.id);
    renderAnnotations(newAnns); scheduleAutoSave(newAnns, true);
  }, [selectedAnnotation, pushUndo, renderAnnotations, scheduleAutoSave]);

  const deleteAnnotation = useCallback((id: string) => {
    pushUndo();
    const newAnns = annotationsRef.current.filter((a) => a.id !== id);
    setAnnotations(newAnns); annotationsRef.current = newAnns;
    if (selectedAnnotation === id) setSelectedAnnotation(null);
    renderAnnotations(newAnns); scheduleAutoSave(newAnns, true);
  }, [pushUndo, selectedAnnotation, renderAnnotations, scheduleAutoSave]);

  const toggleAnnotationFlag = useCallback(
    (id: string, key: "hidden" | "locked") => {
      const newAnns = annotationsRef.current.map((a) =>
        a.id === id ? { ...a, [key]: !a[key] } : a,
      );
      setAnnotations(newAnns); annotationsRef.current = newAnns;
      if (key === "hidden" && selectedAnnotation === id) setSelectedAnnotation(null);
      renderAnnotations(newAnns); scheduleAutoSave(newAnns, true);
    },
    [renderAnnotations, scheduleAutoSave, selectedAnnotation],
  );

  const setAnnotationAttribute = useCallback(
    (id: string, key: string, value: string | null) => {
      const newAnns = annotationsRef.current.map((a) => {
        if (a.id !== id) return a;
        const attrs = { ...(a.attributes || {}) };
        if (value == null) delete attrs[key];
        else attrs[key] = value;
        return { ...a, attributes: attrs };
      });
      setAnnotations(newAnns); annotationsRef.current = newAnns;
      scheduleAutoSave(newAnns);
    },
    [scheduleAutoSave],
  );

  const setAnnotationAngle = useCallback(
    (id: string, angle: number) => {
      pushUndo();
      const normalized = ((angle % 360) + 360) % 360;
      const newAnns = annotationsRef.current.map((a) =>
        a.id === id ? { ...a, angle: normalized } : a,
      );
      setAnnotations(newAnns); annotationsRef.current = newAnns;
      renderAnnotations(newAnns); scheduleAutoSave(newAnns, true);
    },
    [pushUndo, renderAnnotations, scheduleAutoSave],
  );

  /**
   * Fit the background image in the canvas using the viewport transform.
   * Image is always placed at (0,0) with scale 1 (so scene coords == pixel coords).
   * Uses canvas.getWidth/Height so we measure what Fabric actually renders to,
   * not the container (which can desync during layout/resize).
   */
  const centerAndFitImage = useCallback(() => {
    const c = fabricRef.current;
    const container = containerRef.current;
    if (!c || !container) return;
    const imgW = imageDims.current.width;
    const imgH = imageDims.current.height;
    if (!imgW || !imgH) return;
    // Keep canvas dimensions in sync with container before measuring — otherwise
    // a stale canvas size produces the wrong fit and the image lands off-center.
    const cw = container.clientWidth;
    const ch = container.clientHeight;
    if (c.getWidth() !== cw || c.getHeight() !== ch) {
      c.setDimensions({ width: cw, height: ch });
    }
    const scale = Math.min(cw / imgW, ch / imgH) * 0.95;
    const tx = (cw - imgW * scale) / 2;
    const ty = (ch - imgH * scale) / 2;
    c.setViewportTransform([scale, 0, 0, scale, tx, ty]);
    setZoomLevel(Math.round(scale * 100));
  }, []);

  const zoomIn = useCallback(() => {
    const c = fabricRef.current; if (!c) return;
    const container = containerRef.current; if (!container) return;
    const zoom = Math.min(c.getZoom() * 1.2, 10);
    import("fabric").then(({ Point }) => {
      c.zoomToPoint(new Point(container.clientWidth / 2, container.clientHeight / 2), zoom);
      setZoomLevel(Math.round(zoom * 100));
    });
  }, []);

  const zoomOut = useCallback(() => {
    const c = fabricRef.current; if (!c) return;
    const container = containerRef.current; if (!container) return;
    const zoom = Math.max(c.getZoom() * 0.8, 0.05);
    import("fabric").then(({ Point }) => {
      c.zoomToPoint(new Point(container.clientWidth / 2, container.clientHeight / 2), zoom);
      setZoomLevel(Math.round(zoom * 100));
    });
  }, []);

  const fitToView = useCallback(() => { centerAndFitImage(); }, [centerAndFitImage]);

  const toggleAnnotations = useCallback(() => {
    const c = fabricRef.current; if (!c) return;
    const visible = !annotationsVisible;
    setAnnotationsVisible(visible);
    c.getObjects().forEach((obj: any) => {
      if (!obj.isBackgroundImage) obj.set("visible", visible);
    });
    c.requestRenderAll();
  }, [annotationsVisible]);

  const setStatus = useCallback(async (status: ImageStatus) => {
    if (!currentImage) return;
    const comment = reviewComment || `Status set to ${status}`;
    try {
      const res = await fetch(`/api/annotations/${encodeFilePath(currentImage)}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ status, comment }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
    } catch (err) {
      toast.error(`Failed to ${status === "accepted" ? "accept" : status === "rejected" ? "reject" : "update"}`, {
        description: String(err),
      });
      return;
    }
    setImageStatus(status);
    setReviewHistory((prev) => [...prev, { action: status, comment, timestamp: new Date().toISOString() }]);
    fetchImages(); fetchStats();
    setReviewComment("");
    if (status === "accepted") toast.success("Accepted");
    else if (status === "rejected") toast("Rejected", { description: comment.length > 60 ? comment.slice(0, 57) + "…" : comment });
    if (autoAdvanceRef.current && (status === "accepted" || status === "rejected")) {
      autoAdvancePending.current = true;
    }
  }, [currentImage, reviewComment, fetchImages, fetchStats]);

  const saveNow = useCallback(async () => {
    if (!currentImage) return;
    setSaveIndicator("Saving...");
    await fetch(`/api/annotations/${encodeFilePath(currentImage)}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        filename: currentImage, annotations, labels,
        status: annotations.length > 0 ? (imageStatus === "unannotated" ? "annotated" : imageStatus) : "unannotated",
        reviewComment, history: reviewHistory,
        imageWidth: imageDims.current.width, imageHeight: imageDims.current.height,
      }),
    });
    setSaveIndicator("Saved");
    setTimeout(() => setSaveIndicator("Auto-save on"), 1500);
    fetchImages(); fetchStats();
  }, [currentImage, annotations, labels, imageStatus, reviewComment, reviewHistory, fetchImages, fetchStats]);

  // Monotonic load counter. Rapid sidebar clicks race two loadImage calls in
  // parallel; we discard results from stale calls so the latest click wins.
  const loadGen = useRef(0);

  const loadImage = useCallback(async (filename: string) => {
    const myGen = ++loadGen.current;
    const stale = () => loadGen.current !== myGen;

    if (autoSaveTimerRef.current) {
      clearTimeout(autoSaveTimerRef.current);
      autoSaveTimerRef.current = null;
    }
    autoSaveAbortRef.current?.abort();

    // Flush the previous image's full state (annotations + labels + review).
    // Previously we guarded on annotations.length > 0, which lost review
    // comments on images that only had a comment and no shapes.
    const prevFilename = currentImageRef.current;
    if (prevFilename && prevFilename !== filename) {
      try {
        await fetch(`/api/annotations/${encodeFilePath(prevFilename)}`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            filename: prevFilename,
            annotations: annotationsRef.current,
            labels: labelsRef.current,
            status: (() => {
              const s = imageStatusRef.current;
              return s === "accepted" || s === "rejected"
                ? s
                : annotationsRef.current.length > 0 ? "annotated" : "unannotated";
            })(),
            reviewComment: reviewCommentRef.current,
            history: reviewHistoryRef.current,
            imageWidth: imageDims.current.width,
            imageHeight: imageDims.current.height,
          }),
        });
      } catch {
        toast.error("Previous image failed to save", {
          description: "Your changes may be lost. Check your connection.",
        });
      }
    }
    if (stale()) return;

    setIsLoading(true);
    setCurrentImage(filename);
    currentImageRef.current = filename;

    let data: ImageAnnotation;
    try {
      const res = await fetch(`/api/annotations/${encodeFilePath(filename)}`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      data = await res.json();
    } catch (err) {
      if (stale()) return;
      setIsLoading(false);
      toast.error("Failed to load annotations", { description: String(err) });
      return;
    }
    if (stale()) return;

    setAnnotations(data.annotations || []);
    annotationsRef.current = data.annotations || [];
    setImageStatus(data.status || "unannotated");
    setReviewComment(data.reviewComment || "");
    setReviewHistory(data.history || []);
    if (data.labels && data.labels.length > 0) {
      setLabels(data.labels); labelsRef.current = data.labels;
    }
    undoStack.current = []; redoStack.current = [];
    setSelectedAnnotation(null);

    const canvas = fabricRef.current;
    if (!canvas) { setIsLoading(false); return; }

    canvas.clear();
    labelMap.current.clear();
    canvas.setViewportTransform([1, 0, 0, 1, 0, 0]);

    // Cancel any previous in-flight image fetches so rapid arrow-key navigation
    // doesn't pile up bandwidth + decoded pixel buffers in memory.
    loadAbortRef.current?.abort();
    const ac = new AbortController();
    loadAbortRef.current = ac;

    // Progressive load:
    //   1) fetch the small preview WebP and decode off-thread, paint it scaled
    //      UP to true raw dims so the user sees something while raw is fetching
    //   2) in parallel, fetch the full raw and atomically swap it in when ready
    // Trade: extra preview bandwidth for lower time-to-first-paint. Both fetches
    // run concurrently; the raw is downloaded regardless. Scene coords are
    // always in TRUE raw-pixel space: preview is rendered with scaleX/Y =
    // trueDim/previewDim, so any annotation drawn during the preview phase is
    // already in raw coords; the swap is a no-op for coords.
    const fabric = await import("fabric");

    /** Fetch + decode an image off the main thread via HTMLImageElement.decode().
     *  Per HTML spec, decode() runs on the browser's image-decoding worker
     *  rather than blocking the main thread on first paint.
     *  Returns headers (for derived dims) and the decoded element. */
    const fetchAndDecode = async (
      url: string,
      signal: AbortSignal,
    ): Promise<{ el: HTMLImageElement; w: number; h: number; headers: Headers } | null> => {
      const res = await fetch(url, { signal });
      if (!res.ok) {
        console.warn("[loadImage] fetch non-2xx", { url, status: res.status });
        return null;
      }
      const blob = await res.blob();
      const blobUrl = URL.createObjectURL(blob);
      try {
        const el = new Image();
        el.crossOrigin = "anonymous";
        el.decoding = "async";
        el.src = blobUrl;
        await el.decode();
        return { el, w: el.naturalWidth, h: el.naturalHeight, headers: res.headers };
      } finally {
        URL.revokeObjectURL(blobUrl);
      }
    };

    const previewUrl = `/api/derived/preview/${encodeFilePath(filename)}`;
    const rawUrl = `/api/raw/${encodeFilePath(filename)}`;

    // Kick off raw fetch in parallel; consume it after preview paints.
    const rawPromise = fetchAndDecode(rawUrl, ac.signal).catch((err) => {
      if (err?.name === "AbortError") return null;
      console.error("[loadImage] raw decode failed", { filename, err });
      return null;
    });

    let trueW = data.imageWidth || 0;
    let trueH = data.imageHeight || 0;
    let previewImg: any = null;
    let lastError: unknown = null;
    let annotationsRendered = false;

    try {
      const dec = await fetchAndDecode(previewUrl, ac.signal);
      if (dec) {
        const hw = Number(dec.headers.get("X-Original-Width") || 0);
        const hh = Number(dec.headers.get("X-Original-Height") || 0);
        if (hw && hh) { trueW = hw; trueH = hh; }
        if (!trueW || !trueH) { trueW = dec.w; trueH = dec.h; }
        previewImg = new fabric.FabricImage(dec.el);
        previewImg.set({
          left: 0, top: 0,
          scaleX: trueW / dec.w, scaleY: trueH / dec.h,
          selectable: false, evented: false, hoverCursor: "default",
        });
        (previewImg as any).isBackgroundImage = true;
        (previewImg as any).isPreview = true;
      }
    } catch (err) {
      if ((err as { name?: string })?.name !== "AbortError") {
        lastError = err;
        console.warn("[loadImage] preview unavailable, will wait for raw", err);
      }
    }
    if (stale() || ac.signal.aborted) return;

    if (previewImg) {
      canvas.add(previewImg);
      canvas.sendObjectToBack(previewImg);
      imageDims.current = { width: trueW, height: trueH };
      setImageDimState({ width: trueW, height: trueH });
      renderAnnotations(data.annotations || [], canvas);
      annotationsRendered = true;
      requestAnimationFrame(() => {
        if (stale()) return;
        centerAndFitImage();
        canvas.requestRenderAll();
      });
    }

    // Wait for the full raw, then atomically swap it for the preview.
    const rawDecoded = await rawPromise;
    if (stale() || ac.signal.aborted) return;

    if (!rawDecoded) {
      if (!previewImg) {
        setIsLoading(false);
        toast.error("Failed to load image", {
          description: lastError ? `${filename}: ${String(lastError)}` : filename,
        });
        return;
      }
      // Raw failed but preview rendered — warn so the user knows they're
      // looking at a downscaled image and may want to retry before drawing
      // pixel-precise annotations.
      toast.warning("Showing preview", {
        description: "Full-resolution image failed to load. Reload the page to retry.",
      });
    } else {
      let fullImg: any;
      try {
        fullImg = new fabric.FabricImage(rawDecoded.el);
        fullImg.set({
          left: 0, top: 0, scaleX: 1, scaleY: 1,
          selectable: false, evented: false, hoverCursor: "default",
        });
        (fullImg as any).isBackgroundImage = true;
        // ADD-then-REMOVE order: if FabricImage construction or canvas.add
        // throws, the preview stays on screen instead of leaving the user
        // with a blank background. After add succeeds, remove the preview.
        canvas.add(fullImg);
        canvas.sendObjectToBack(fullImg);
        if (previewImg) {
          canvas.remove(previewImg);
          previewImg = null;
        }
        // Authoritative dims come from the raw element.
        trueW = rawDecoded.w;
        trueH = rawDecoded.h;
        imageDims.current = { width: trueW, height: trueH };
        setImageDimState({ width: trueW, height: trueH });
      } catch (err) {
        console.error("[loadImage] raw swap failed; keeping preview", err);
        toast.warning("Showing preview", {
          description: "Failed to upgrade to full resolution.",
        });
      }
    }

    // First-load path (no preview was ever painted) — render now.
    if (!annotationsRendered && rawDecoded) {
      renderAnnotations(data.annotations || [], canvas);
    }
    // renderOnAddRemove is false — every code path that mutates the canvas
    // MUST call requestRenderAll() or the change will not be visible.
    canvas.requestRenderAll();
    requestAnimationFrame(() => {
      if (stale()) return;
      centerAndFitImage();
      canvas.requestRenderAll();
    });
    setTimeout(() => { if (!stale()) centerAndFitImage(); }, 80);
    setBrightness(100); setContrast(100);
    setIsLoading(false);
  }, [renderAnnotations, centerAndFitImage]);

  // Init fabric once.
  useEffect(() => {
    if (!canvasRef.current || fabricRef.current) return;

    const init = async () => {
      const fabric = await import("fabric");
      const container = containerRef.current;
      if (!container || !canvasRef.current) return;

      const canvas = new fabric.Canvas(canvasRef.current, {
        width: container.clientWidth, height: container.clientHeight,
        selection: false, preserveObjectStacking: true,
        defaultCursor: "crosshair",
        // Sharp lines on retina; smooth upscaled image bg.
        enableRetinaScaling: true,
        imageSmoothingEnabled: true,
        // Manual control of when to paint — we batch with requestRenderAll().
        renderOnAddRemove: false,
      });
      fabricRef.current = canvas;

      const finalizePolyShape = (tool: "polygon" | "polyline") => {
        const minPts = tool === "polyline" ? 2 : 3;
        if (polygonPoints.current.length < minPts) return false;
        undoStack.current.push({
          annotations: JSON.parse(JSON.stringify(annotationsRef.current)),
          labels: JSON.parse(JSON.stringify(labelsRef.current)),
        });
        redoStack.current = [];
        const ann: Annotation = {
          id: uuidv4(),
          type: tool,
          label: activeLabelRef.current,
          color: getLabelColor(activeLabelRef.current),
          points: [...polygonPoints.current],
        };
        for (const l of polygonLines.current) canvas.remove(l);
        for (const d of polygonDots.current) canvas.remove(d);
        if (polygonCloseLine.current) canvas.remove(polygonCloseLine.current);
        polygonPoints.current = [];
        polygonLines.current = [];
        polygonDots.current = [];
        polygonCloseLine.current = null;
        polygonFirstDot.current = null;
        const newAnns = [...annotationsRef.current, ann];
        setAnnotations(newAnns); annotationsRef.current = newAnns;
        renderAnnotations(newAnns, canvas);
        scheduleAutoSave(newAnns, true);
        return true;
      };
      (canvas as any).__finalizePolyShape = finalizePolyShape;

      canvas.on("mouse:down", (opt: any) => {
        const tool = activeToolRef.current;
        const e = opt.e as MouseEvent;

        if (e.button === 1 || tool === "pan" || spaceHeld.current) {
          isPanning.current = true;
          panStart.current = { x: e.clientX, y: e.clientY };
          canvas.defaultCursor = "grabbing";
          canvas.setCursor("grabbing");
          return;
        }

        if (tool === "select") return;

        const pointer = canvas.getScenePoint(opt.e);

        if (tool === "bbox") {
          isDrawing.current = true;
          drawStart.current = { x: pointer.x, y: pointer.y };
          const rect = new fabric.Rect({
            left: pointer.x, top: pointer.y, width: 0, height: 0,
            fill: "transparent", stroke: getLabelColor(activeLabelRef.current),
            strokeWidth: 2, strokeDashArray: [6, 4], strokeUniform: true,
            selectable: false, evented: false,
          });
          tempRect.current = rect; canvas.add(rect);
        } else if (tool === "ellipse") {
          isDrawing.current = true;
          drawStart.current = { x: pointer.x, y: pointer.y };
          const ellipse = new fabric.Ellipse({
            left: pointer.x, top: pointer.y, rx: 0, ry: 0,
            fill: "transparent", stroke: getLabelColor(activeLabelRef.current),
            strokeWidth: 2, strokeDashArray: [6, 4], strokeUniform: true,
            selectable: false, evented: false,
          });
          tempRect.current = ellipse; canvas.add(ellipse);
        } else if (tool === "polyline") {
          polygonPoints.current.push([pointer.x, pointer.y]);
          const isFirst = polygonPoints.current.length === 1;
          const dot = new fabric.Circle({
            left: pointer.x - (isFirst ? 5 : 3), top: pointer.y - (isFirst ? 5 : 3),
            radius: isFirst ? 5 : 3,
            fill: getLabelColor(activeLabelRef.current), stroke: "#fff", strokeWidth: 1,
            strokeUniform: true,
            selectable: false, evented: false,
          });
          canvas.add(dot); polygonDots.current.push(dot);
          if (isFirst) polygonFirstDot.current = dot;
          if (polygonPoints.current.length > 1) {
            const pts = polygonPoints.current;
            const prev = pts[pts.length - 2]; const curr = pts[pts.length - 1];
            const line = new fabric.Line([prev[0], prev[1], curr[0], curr[1]], {
              stroke: getLabelColor(activeLabelRef.current), strokeWidth: 2,
              strokeUniform: true,
              selectable: false, evented: false,
            });
            canvas.add(line); polygonLines.current.push(line);
          }
          canvas.requestRenderAll();
        } else if (tool === "polygon") {
          const pts = polygonPoints.current;
          if (pts.length >= 3) {
            const first = pts[0];
            const dx = pointer.x - first[0];
            const dy = pointer.y - first[1];
            const threshold = 10 / Math.max(canvas.getZoom(), 0.1);
            if (dx * dx + dy * dy < threshold * threshold) {
              finalizePolyShape("polygon");
              return;
            }
          }
          polygonPoints.current.push([pointer.x, pointer.y]);
          const isFirst = polygonPoints.current.length === 1;
          const dot = new fabric.Circle({
            left: pointer.x - (isFirst ? 6 : 4), top: pointer.y - (isFirst ? 6 : 4),
            radius: isFirst ? 6 : 4,
            fill: getLabelColor(activeLabelRef.current),
            stroke: "#fff", strokeWidth: isFirst ? 2 : 1,
            strokeUniform: true,
            selectable: false, evented: false,
          });
          canvas.add(dot); polygonDots.current.push(dot);
          if (isFirst) polygonFirstDot.current = dot;
          if (polygonPoints.current.length > 1) {
            const allPts = polygonPoints.current;
            const prev = allPts[allPts.length - 2]; const curr = allPts[allPts.length - 1];
            const line = new fabric.Line([prev[0], prev[1], curr[0], curr[1]], {
              stroke: getLabelColor(activeLabelRef.current), strokeWidth: 2,
              strokeDashArray: [4, 4], strokeUniform: true,
              selectable: false, evented: false,
            });
            canvas.add(line); polygonLines.current.push(line);
          }
          canvas.requestRenderAll();
        } else if (tool === "keypoint") {
          undoStack.current.push({
            annotations: JSON.parse(JSON.stringify(annotationsRef.current)),
            labels: JSON.parse(JSON.stringify(labelsRef.current)),
          });
          redoStack.current = [];
          const ann: Annotation = {
            id: uuidv4(), type: "keypoint", label: activeLabelRef.current,
            color: getLabelColor(activeLabelRef.current), x: pointer.x, y: pointer.y,
          };
          const newAnns = [...annotationsRef.current, ann];
          setAnnotations(newAnns); annotationsRef.current = newAnns;
          renderAnnotations(newAnns, canvas);
          scheduleAutoSave(newAnns, true);
        }
      });

      canvas.on("mouse:move", (opt: any) => {
        const e = opt.e as MouseEvent;

        if (isPanning.current && panStart.current) {
          const vpt = canvas.viewportTransform!;
          vpt[4] += e.clientX - panStart.current.x;
          vpt[5] += e.clientY - panStart.current.y;
          panStart.current = { x: e.clientX, y: e.clientY };
          canvas.requestRenderAll();
          return;
        }

        const pointer = canvas.getScenePoint(opt.e);
        setCursorPos({ x: Math.round(pointer.x), y: Math.round(pointer.y) });

        if (isDrawing.current && tempRect.current && drawStart.current) {
          const x = Math.min(drawStart.current.x, pointer.x);
          const y = Math.min(drawStart.current.y, pointer.y);
          const w = Math.abs(pointer.x - drawStart.current.x);
          const h = Math.abs(pointer.y - drawStart.current.y);
          if (activeToolRef.current === "ellipse") {
            tempRect.current.set({ left: x, top: y, rx: w / 2, ry: h / 2 });
          } else {
            tempRect.current.set({ left: x, top: y, width: w, height: h });
          }
          canvas.requestRenderAll();
        }

        const activeTool = activeToolRef.current;
        if ((activeTool === "polygon" || activeTool === "polyline") && polygonPoints.current.length >= 1) {
          const pts = polygonPoints.current;
          const last = pts[pts.length - 1];
          const first = pts[0];
          if (!polygonCloseLine.current) {
            const line = new fabric.Line([last[0], last[1], pointer.x, pointer.y], {
              stroke: getLabelColor(activeLabelRef.current), strokeWidth: 1,
              strokeDashArray: [2, 4], strokeUniform: true,
              selectable: false, evented: false,
            });
            polygonCloseLine.current = line;
            canvas.add(line);
          } else {
            polygonCloseLine.current.set({ x1: last[0], y1: last[1], x2: pointer.x, y2: pointer.y });
          }
          if (activeTool === "polygon" && pts.length >= 3 && polygonFirstDot.current) {
            const dx = pointer.x - first[0];
            const dy = pointer.y - first[1];
            const threshold = 10 / Math.max(canvas.getZoom(), 0.1);
            const near = dx * dx + dy * dy < threshold * threshold;
            polygonFirstDot.current.set({
              radius: near ? 8 : 6,
              left: first[0] - (near ? 8 : 6),
              top: first[1] - (near ? 8 : 6),
              strokeWidth: near ? 3 : 2,
            });
          }
          canvas.requestRenderAll();
        }
      });

      canvas.on("mouse:up", () => {
        if (isPanning.current) {
          isPanning.current = false;
          panStart.current = null;
          const tool = activeToolRef.current;
          canvas.defaultCursor = tool === "select" ? "default" : tool === "pan" ? "grab" : "crosshair";
          canvas.setCursor(canvas.defaultCursor);
          return;
        }

        if (isDrawing.current && tempRect.current && drawStart.current) {
          const shape = tempRect.current;
          const tool = activeToolRef.current;
          const isEllipse = tool === "ellipse";
          const w = isEllipse ? (shape.rx || 0) * 2 : shape.width || 0;
          const h = isEllipse ? (shape.ry || 0) * 2 : shape.height || 0;
          canvas.remove(shape); tempRect.current = null;
          isDrawing.current = false; drawStart.current = null;
          if (w > 5 && h > 5) {
            undoStack.current.push({
              annotations: JSON.parse(JSON.stringify(annotationsRef.current)),
              labels: JSON.parse(JSON.stringify(labelsRef.current)),
            });
            redoStack.current = [];
            const ann: Annotation = {
              id: uuidv4(),
              type: isEllipse ? "ellipse" : "bbox",
              label: activeLabelRef.current,
              color: getLabelColor(activeLabelRef.current),
              x: shape.left || 0, y: shape.top || 0, width: w, height: h,
            };
            const newAnns = [...annotationsRef.current, ann];
            setAnnotations(newAnns); annotationsRef.current = newAnns;
            renderAnnotations(newAnns, canvas);
            scheduleAutoSave(newAnns, true);
          }
        }
      });

      canvas.on("mouse:dblclick", () => {
        const tool = activeToolRef.current;
        if (tool === "polygon" || tool === "polyline") {
          finalizePolyShape(tool);
        }
      });

      canvas.on("mouse:wheel", (opt: any) => {
        const delta = opt.e.deltaY;
        let zoom = canvas.getZoom();
        zoom *= 0.999 ** delta;
        zoom = Math.min(Math.max(0.05, zoom), 10);
        canvas.zoomToPoint(new fabric.Point(opt.e.offsetX, opt.e.offsetY), zoom);
        setZoomLevel(Math.round(zoom * 100));
        opt.e.preventDefault(); opt.e.stopPropagation();
      });

      canvas.on("selection:created", (e: any) => {
        const obj = e.selected?.[0];
        if (obj?.annotationId) setSelectedAnnotation(obj.annotationId);
      });
      canvas.on("selection:updated", (e: any) => {
        const obj = e.selected?.[0];
        if (obj?.annotationId) setSelectedAnnotation(obj.annotationId);
      });
      canvas.on("selection:cleared", () => setSelectedAnnotation(null));

      canvas.on("object:modified", (e: any) => {
        const obj = e.target;
        if (!obj?.annotationId) return;
        const id = obj.annotationId;
        const type = obj.annotationType;

        const effWidth = obj.width * obj.scaleX;
        const effHeight = obj.height * obj.scaleY;

        const newAnns = annotationsRef.current.map((a) => {
          if (a.id !== id) return a;
          if (type === "bbox") {
            // Normalise angle to [0, 360) so it never grows unbounded.
            const angle = ((obj.angle || 0) % 360 + 360) % 360;
            return { ...a, x: obj.left, y: obj.top, width: effWidth, height: effHeight, angle };
          }
          if (type === "ellipse") {
            const rx = (obj.rx || 0) * obj.scaleX;
            const ry = (obj.ry || 0) * obj.scaleY;
            return { ...a, x: obj.left, y: obj.top, width: rx * 2, height: ry * 2 };
          }
          if (type === "keypoint") {
            return { ...a, x: (obj.left || 0) + 4, y: (obj.top || 0) + 4 };
          }
          if ((type === "polygon" || type === "polyline") && a.points) {
            const m = obj.calcTransformMatrix() as number[];
            const ox = obj.pathOffset?.x || 0;
            const oy = obj.pathOffset?.y || 0;
            const pts: number[][] = obj.points.map((p: { x: number; y: number }) => {
              const lx = p.x - ox;
              const ly = p.y - oy;
              return [m[0] * lx + m[2] * ly + m[4], m[1] * lx + m[3] * ly + m[5]];
            });
            return { ...a, points: pts };
          }
          return a;
        });

        if (type === "bbox") {
          obj.set({ scaleX: 1, scaleY: 1, width: effWidth, height: effHeight });
        } else if (type === "ellipse") {
          obj.set({ scaleX: 1, scaleY: 1, rx: (obj.rx || 0) * obj.scaleX, ry: (obj.ry || 0) * obj.scaleY });
        }

        // Reposition the label inline instead of rebuilding ALL fabric objects.
        // Full rebuild used to fire on every drag frame — O(N) per modify event,
        // 60Hz during drag, was the dominant CPU cost on large annotation sets.
        // Map lookup keeps this O(1) regardless of how many annotations exist.
        const label = labelMap.current.get(id);
        if (label) {
          if (type === "bbox") {
            const angle = ((obj.angle || 0) % 360 + 360) % 360;
            const rad = (angle * Math.PI) / 180;
            const lx = 2, ly = -20;
            label.set({
              left: (obj.left || 0) + lx * Math.cos(rad) - ly * Math.sin(rad),
              top:  (obj.top  || 0) + lx * Math.sin(rad) + ly * Math.cos(rad),
              angle,
            });
          } else if (type === "ellipse") {
            label.set({ left: (obj.left || 0) + 2, top: (obj.top || 0) - 20 });
          } else if (type === "polygon" || type === "polyline") {
            const b = obj.getBoundingRect();
            label.set({ left: b.left + 2, top: b.top - 20 });
          } else if (type === "keypoint") {
            label.set({ left: (obj.left || 0) + 14, top: (obj.top || 0) - 8 });
          }
          label.setCoords();
        }

        setAnnotations(newAnns); annotationsRef.current = newAnns;
        scheduleAutoSave(newAnns, true);
        canvas.requestRenderAll();
      });

      // --- RESIZE: ResizeObserver tracks layout changes (sidebar collapse,
      // browser zoom, devtools open, etc.) and keeps the canvas locked to the
      // container. Simple window resize events miss these.
      let pendingFit = 0;
      const scheduleFit = () => {
        cancelAnimationFrame(pendingFit);
        pendingFit = requestAnimationFrame(() => {
          const cw = container.clientWidth;
          const ch = container.clientHeight;
          if (canvas.getWidth() !== cw || canvas.getHeight() !== ch) {
            canvas.setDimensions({ width: cw, height: ch });
          }
          centerAndFitImage();
          canvas.requestRenderAll();
        });
      };
      const ro = new ResizeObserver(() => scheduleFit());
      ro.observe(container);
      window.addEventListener("resize", scheduleFit);

      const canvasEl = container.querySelector("canvas");
      const preventMiddle = (e: MouseEvent) => { if (e.button === 1) e.preventDefault(); };
      canvasEl?.addEventListener("mousedown", preventMiddle);

      return () => {
        ro.disconnect();
        cancelAnimationFrame(pendingFit);
        window.removeEventListener("resize", scheduleFit);
        canvasEl?.removeEventListener("mousedown", preventMiddle);
      };
    };

    init();

    // Capture refs at effect-init time so cleanup uses the same identities
    // (React lint guards against ref churn between mount and unmount).
    const labels = labelMap.current;
    return () => {
      loadAbortRef.current?.abort();
      labels.clear();
      const c = fabricRef.current;
      if (c) { c.dispose(); fabricRef.current = null; }
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    const c = fabricRef.current;
    if (!c) return;
    const cursorMap: Record<Tool, string> = {
      select: "default", bbox: "crosshair", polygon: "crosshair", polyline: "crosshair",
      ellipse: "crosshair", keypoint: "crosshair", pan: "grab",
    };
    c.defaultCursor = cursorMap[activeTool] || "default";
    c.selection = activeTool === "select";

    c.getObjects().forEach((obj: any) => {
      if (!obj.isBackgroundImage && !obj.isLabel && obj.annotationId) {
        const interactive = activeTool === "select";
        obj.set("selectable", interactive);
        obj.set("evented", interactive);
        // Keep objectCaching in sync with interactive state. Otherwise a
        // shape created in the now-stale state would either be raster-cached
        // while interactive (producing blurry drag previews) or uncached
        // while static (wasting CPU per frame).
        obj.set("objectCaching", !interactive);
        obj.dirty = true;
      }
    });
    c.discardActiveObject();
    c.requestRenderAll();
  }, [activeTool]);

  const goToImage = useCallback((direction: "next" | "prev") => {
    if (images.length === 0) return;
    const idx = images.findIndex((i) => i.filename === currentImage);
    const newIdx = direction === "next" ? (idx < images.length - 1 ? idx + 1 : 0) : (idx > 0 ? idx - 1 : images.length - 1);
    loadImage(images[newIdx].filename);
  }, [images, currentImage, loadImage]);

  useEffect(() => {
    if (autoAdvancePending.current) {
      autoAdvancePending.current = false;
      goToImage("next");
    }
  }, [imageStatus, goToImage]);

  const addLabel = useCallback(() => {
    const trimmed = newLabelName.trim();
    if (!trimmed) return;
    if (labels.find((l) => l.name === trimmed)) {
      toast.error("Label already exists");
      return;
    }
    pushUndo();
    const newLabels = [...labels, { name: trimmed, color: newLabelColor }];
    setLabels(newLabels); labelsRef.current = newLabels;
    setActiveLabel(trimmed); activeLabelRef.current = trimmed;
    setNewLabelName(""); setNewLabelColor(LABEL_COLORS[newLabels.length % LABEL_COLORS.length]);
  }, [newLabelName, newLabelColor, labels, pushUndo]);

  /**
   * Label delete with orphan protection. If any annotation uses this label,
   * show a confirmation dialog; otherwise remove immediately.
   */
  const requestRemoveLabel = useCallback((name: string) => {
    const count = annotationsRef.current.filter((a) => a.label === name).length;
    if (count === 0) {
      pushUndo();
      const newLabels = labelsRef.current.filter((l) => l.name !== name);
      setLabels(newLabels); labelsRef.current = newLabels;
      if (activeLabelRef.current === name && newLabels.length > 0) {
        setActiveLabel(newLabels[0].name);
        activeLabelRef.current = newLabels[0].name;
      }
      return;
    }
    setPendingLabelDelete({ name, count });
  }, [pushUndo]);

  const confirmDeleteLabelAndAnnotations = useCallback(() => {
    if (!pendingLabelDelete) return;
    const name = pendingLabelDelete.name;
    const newLabels = labelsRef.current.filter((l) => l.name !== name);
    const newAnns = annotationsRef.current.filter((a) => a.label !== name);
    pushUndo();
    setLabels(newLabels); labelsRef.current = newLabels;
    setAnnotations(newAnns); annotationsRef.current = newAnns;
    if (activeLabelRef.current === name && newLabels.length > 0) {
      setActiveLabel(newLabels[0].name);
      activeLabelRef.current = newLabels[0].name;
    }
    renderAnnotations(newAnns); scheduleAutoSave(newAnns, true);
    setPendingLabelDelete(null);
    toast.success(`Deleted label and ${pendingLabelDelete.count} annotation${pendingLabelDelete.count === 1 ? "" : "s"}`);
  }, [pendingLabelDelete, pushUndo, renderAnnotations, scheduleAutoSave]);

  const cancelPolygon = useCallback(() => {
    const c = fabricRef.current; if (!c) return;
    for (const l of polygonLines.current) c.remove(l);
    for (const d of polygonDots.current) c.remove(d);
    if (polygonCloseLine.current) c.remove(polygonCloseLine.current);
    polygonPoints.current = [];
    polygonLines.current = [];
    polygonDots.current = [];
    polygonCloseLine.current = null;
    polygonFirstDot.current = null;
    c.requestRenderAll();
  }, []);

  const uploadFiles = useCallback(async (files: File[], folder: UploadFolder) => {
    if (files.length === 0) return;
    const tId = toast.loading(`Uploading ${files.length} image${files.length === 1 ? "" : "s"} → ${folder}…`);
    const formData = new FormData();
    files.forEach((f) => formData.append("files", f));
    formData.append("folder", folder);
    try {
      const res = await fetch("/api/upload", { method: "POST", body: formData });
      const data = await res.json().catch(() => ({}));
      if (!res.ok && (data.uploaded?.length ?? 0) === 0) {
        throw new Error(data.rejected?.[0]?.reason || `HTTP ${res.status}`);
      }
      const count = data.uploaded?.length ?? 0;
      const rejectedCount = data.rejected?.length ?? 0;
      if (count > 0) {
        toast.success(`Uploaded ${count} image${count === 1 ? "" : "s"} to ${folder}`, {
          id: tId,
          description: rejectedCount > 0 ? `${rejectedCount} skipped` : undefined,
        });
      }
      for (const r of data.rejected || []) {
        toast.error(`Skipped ${r.name}`, { description: r.reason });
      }
      fetchImages(); fetchStats();
    } catch (err) {
      toast.error("Upload failed", { id: tId, description: String(err) });
    }
  }, [fetchImages, fetchStats]);

  const handleDrop = useCallback(async (e: React.DragEvent) => {
    e.preventDefault(); setIsDragOver(false);
    const files = Array.from(e.dataTransfer.files).filter(
      (f) => f.type.startsWith("image/") || /\.(heic|heif)$/i.test(f.name),
    );
    if (files.length === 0) {
      toast.error("No valid images in drop", { description: "Only image files are accepted." });
      return;
    }
    await uploadFiles(files, uploadFolder);
  }, [uploadFiles, uploadFolder]);

  const doExportCOCO = useCallback(async () => {
    const tId = toast.loading("Exporting COCO…");
    try {
      const res = await fetch("/api/export/coco");
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a"); a.href = url; a.download = "coco_export.json"; a.click();
      URL.revokeObjectURL(url);
      const msg = `${data.images?.length || 0} images, ${data.annotations?.length || 0} annotations`;
      setExportResult(`COCO exported — ${msg}.`);
      toast.success("COCO exported", { id: tId, description: msg });
    } catch (err) {
      toast.error("COCO export failed", { id: tId, description: String(err) });
    }
  }, []);

  const doExportYOLO = useCallback(async () => {
    const tId = toast.loading("Exporting YOLO…");
    try {
      const res = await fetch("/api/export/yolo");
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      const msg = `${data.files} files, ${data.classes?.length || 0} classes`;
      setExportResult(`YOLO exported — ${msg}. Files written to R2 /exports.`);
      toast.success("YOLO exported", { id: tId, description: msg });
    } catch (err) {
      toast.error("YOLO export failed", { id: tId, description: String(err) });
    }
  }, []);

  const doExportZip = useCallback(async () => {
    const tId = toast.loading("Building ZIP…");
    try {
      const res = await fetch("/api/export/zip");
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const blob = await res.blob();
      const cd = res.headers.get("Content-Disposition") || "";
      const match = cd.match(/filename="([^"]+)"/);
      const filename = match ? match[1] : "annotations.zip";
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a"); a.href = url; a.download = filename; a.click();
      URL.revokeObjectURL(url);
      const kb = Math.round(blob.size / 1024);
      setExportResult(`ZIP downloaded — ${kb} KB, includes annotation JSONs, COCO, and YOLO.`);
      toast.success("ZIP downloaded", { id: tId, description: `${kb} KB` });
    } catch (err) {
      toast.error("ZIP export failed", { id: tId, description: String(err) });
    }
  }, []);

  const doInitFolders = useCallback(async () => {
    const tId = toast.loading("Creating folder structure in R2…");
    try {
      const res = await fetch("/api/admin", { method: "POST" });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      toast.success("Folders created", { id: tId, description: (data.folders as string[]).join(", ") });
    } catch (err) {
      toast.error("Init failed", { id: tId, description: String(err) });
    }
  }, []);

  const doRebuildManifest = useCallback(async () => {
    const tId = toast.loading("Rebuilding manifest…");
    try {
      const res = await fetch("/api/admin", { method: "PUT" });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      toast.success("Manifest rebuilt", { id: tId, description: `${data.images} images indexed` });
      fetchImages(); fetchStats();
    } catch (err) {
      toast.error("Rebuild failed", { id: tId, description: String(err) });
    }
  }, [fetchImages, fetchStats]);

  const doClearAll = useCallback(async () => {
    const tId = toast.loading("Deleting all images and annotations…");
    setShowClearConfirm(false);
    try {
      const res = await fetch("/api/admin", { method: "DELETE" });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      toast.success("Cleared", { id: tId, description: `${data.raw} images, ${data.annotations} annotation files deleted` });
      setCurrentImage(null); currentImageRef.current = null;
      setAnnotations([]); annotationsRef.current = [];
      setImageStatus("unannotated");
      fetchImages(); fetchStats();
    } catch (err) {
      toast.error("Clear failed", { id: tId, description: String(err) });
    }
  }, [fetchImages, fetchStats]);

  // Cancel any pending autosave for `filename` AND clear current selection if
  // it matches. Must run BEFORE issuing a move/delete so a debounced autosave
  // (≤600ms) can't recreate the annotation at the OLD path after the server
  // has already moved it.
  const cancelInFlightForFilename = useCallback((filename: string) => {
    if (currentImageRef.current !== filename) return;
    if (autoSaveTimerRef.current) {
      clearTimeout(autoSaveTimerRef.current);
      autoSaveTimerRef.current = null;
    }
    autoSaveAbortRef.current?.abort();
    setCurrentImage(null);
    currentImageRef.current = null;
  }, []);

  // Parse JSON body defensively — proxies may return HTML on 5xx.
  const safeReadJson = async (res: Response): Promise<{ error?: string; filename?: string; trashedKey?: string }> => {
    const ct = res.headers.get("content-type") || "";
    if (!ct.includes("json")) return { error: `HTTP ${res.status}` };
    try { return await res.json(); } catch { return { error: `HTTP ${res.status} (non-JSON body)` }; }
  };

  const moveImageRequest = useCallback(async (filename: string, destFolder: UploadFolder) => {
    cancelInFlightForFilename(filename);
    const tId = toast.loading(`Moving to ${destFolder}…`);
    try {
      const res = await fetch("/api/images/move", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ filename, destFolder }),
      });
      const data = await safeReadJson(res);
      if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
      toast.success("Moved", { id: tId, description: data.filename });
    } catch (err) {
      toast.error("Move failed", { id: tId, description: String(err instanceof Error ? err.message : err) });
    } finally {
      // Always re-sync — server may have partially succeeded.
      fetchImages(); fetchStats();
    }
  }, [cancelInFlightForFilename, fetchImages, fetchStats]);

  const deleteImageRequest = useCallback(async (filename: string) => {
    cancelInFlightForFilename(filename);
    const tId = toast.loading("Deleting…");
    try {
      const res = await fetch(`/api/images/${encodeFilePath(filename)}`, { method: "DELETE" });
      const data = await safeReadJson(res);
      if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
      toast.success("Deleted", { id: tId, description: data.trashedKey || filename });
    } catch (err) {
      toast.error("Delete failed", { id: tId, description: String(err instanceof Error ? err.message : err) });
    } finally {
      fetchImages(); fetchStats();
    }
  }, [cancelInFlightForFilename, fetchImages, fetchStats]);

  useEffect(() => {
    const down = (e: KeyboardEvent) => {
      if (e.key === " " && !spaceHeld.current && !isEditingField(e.target) && !anyModalOpen()) {
        e.preventDefault();
        spaceHeld.current = true;
        prevToolRef.current = activeToolRef.current;
        const c = fabricRef.current;
        if (c) { c.defaultCursor = "grab"; c.setCursor("grab"); }
      }
    };
    const up = (e: KeyboardEvent) => {
      if (e.key === " ") {
        spaceHeld.current = false;
        const c = fabricRef.current;
        if (c) {
          const cursorMap: Record<Tool, string> = { select: "default", bbox: "crosshair", polygon: "crosshair", polyline: "crosshair", ellipse: "crosshair", keypoint: "crosshair", pan: "grab" };
          c.defaultCursor = cursorMap[activeToolRef.current] || "default";
          c.setCursor(c.defaultCursor);
        }
      }
    };
    window.addEventListener("keydown", down);
    window.addEventListener("keyup", up);
    return () => { window.removeEventListener("keydown", down); window.removeEventListener("keyup", up); };
  }, []);

  useEffect(() => {
    const handleKey = (e: KeyboardEvent) => {
      if (isEditingField(e.target)) return;
      // A modal is open → let the dialog own the keystroke; skip app shortcuts.
      if (anyModalOpen()) return;
      if (e.key === " ") return;

      if (e.ctrlKey || e.metaKey) {
        if (e.key === "z" && !e.shiftKey) { e.preventDefault(); undo(); }
        if (e.key === "z" && e.shiftKey) { e.preventDefault(); redo(); }
        if (e.key === "Z") { e.preventDefault(); redo(); }
        if (e.key === "s") { e.preventDefault(); saveNow(); }
        if (e.key === "c" || e.key === "C") { e.preventDefault(); copySelected(); }
        if (e.key === "v" || e.key === "V") { e.preventDefault(); pasteClipboard(); }
        if (e.key === "d" || e.key === "D") { e.preventDefault(); duplicateSelected(); }
        if (e.key === "a") { e.preventDefault(); }
        return;
      }

      switch (e.key) {
        case "v": case "V": setActiveTool("select"); break;
        case "b": case "B": setActiveTool("bbox"); break;
        case "p": case "P": setActiveTool("polygon"); break;
        case "l": case "L": setActiveTool("polyline"); break;
        case "e": case "E": setActiveTool("ellipse"); break;
        case "k": case "K": setActiveTool("keypoint"); break;
        case "g": case "G": setActiveTool("pan"); break;
        case "Delete": case "Backspace": {
          const tool = activeToolRef.current;
          const c = fabricRef.current;
          if (c && (tool === "polygon" || tool === "polyline") && polygonPoints.current.length > 0) {
            e.preventDefault();
            polygonPoints.current.pop();
            const lastLine = polygonLines.current.pop();
            if (lastLine) c.remove(lastLine);
            const lastDot = polygonDots.current.pop();
            if (lastDot) c.remove(lastDot);
            if (polygonPoints.current.length === 0) {
              if (polygonCloseLine.current) { c.remove(polygonCloseLine.current); polygonCloseLine.current = null; }
              polygonFirstDot.current = null;
            }
            c.requestRenderAll();
          } else {
            deleteSelected();
          }
          break;
        }
        case "n": e.preventDefault(); goToImage("next"); break;
        case "N": e.preventDefault(); goToImage("prev"); break;
        case "ArrowRight": e.preventDefault(); goToImage("next"); break;
        case "ArrowLeft": e.preventDefault(); goToImage("prev"); break;
        case "+": case "=": zoomIn(); break;
        case "-": zoomOut(); break;
        case "f": case "F": fitToView(); break;
        case "h": case "H": toggleAnnotations(); break;
        case "Escape": cancelPolygon(); setActiveTool("select"); break;
        case "Enter": {
          const c = fabricRef.current;
          const tool = activeToolRef.current;
          if (c && (tool === "polygon" || tool === "polyline") && polygonPoints.current.length > 0) {
            e.preventDefault();
            (c as any).__finalizePolyShape?.(tool);
          }
          break;
        }
        case "q": case "Q": if (currentImage) setStatus("accepted"); break;
        case "w": case "W": if (currentImage) setStatus("rejected"); break;
        default: {
          const num = parseInt(e.key);
          if (num >= 1 && num <= 9 && num <= labels.length) {
            setActiveLabel(labels[num - 1].name);
            activeLabelRef.current = labels[num - 1].name;
          }
        }
      }
    };
    window.addEventListener("keydown", handleKey);
    return () => window.removeEventListener("keydown", handleKey);
  }, [undo, redo, saveNow, deleteSelected, goToImage, zoomIn, zoomOut, fitToView, toggleAnnotations, cancelPolygon, setStatus, labels, currentImage, copySelected, pasteClipboard, duplicateSelected]);

  const progress = stats.total > 0 ? ((stats.annotated + stats.accepted + stats.rejected) / stats.total * 100) : 0;
  const currentIdx = images.findIndex((i) => i.filename === currentImage);

  const filteredImages = (() => {
    const q = searchQuery.trim().toLowerCase();
    return q ? images.filter((i) => i.filename.toLowerCase().includes(q)) : images;
  })();

  const inspectorPanels = (
    <>
      <LabelsPanel
        labels={labels}
        annotations={annotations}
        activeLabel={activeLabel}
        labelFilter={labelFilter}
        newLabelName={newLabelName}
        newLabelColor={newLabelColor}
        swatches={LABEL_COLORS}
        onSelectLabel={(name) => { setActiveLabel(name); activeLabelRef.current = name; }}
        onToggleFilter={(name) => setLabelFilter(labelFilter === name ? null : name)}
        onRequestRemove={requestRemoveLabel}
        onNameChange={setNewLabelName}
        onColorChange={setNewLabelColor}
        onAdd={addLabel}
      />
      {currentImage && (
        <AdjustmentsPanel
          brightness={brightness} setBrightness={setBrightness}
          contrast={contrast} setContrast={setContrast}
          opacity={opacity} setOpacity={setOpacity}
          onReset={() => {
            setBrightness(100); setContrast(100);
            setOpacity(0); opacityRef.current = 0;
          }}
        />
      )}
      <PropertiesPanel
        selectedAnnotation={selectedAnnotation}
        annotations={annotations}
        labels={labels}
        attrDraftKey={attrDraftKey}
        attrDraftVal={attrDraftVal}
        onAttrDraftKey={setAttrDraftKey}
        onAttrDraftVal={setAttrDraftVal}
        onChangeLabel={(id, newLabel) => {
          pushUndo();
          const newAnns = annotations.map((a) =>
            a.id === id ? { ...a, label: newLabel, color: getLabelColor(newLabel) } : a,
          );
          setAnnotations(newAnns); annotationsRef.current = newAnns;
          renderAnnotations(newAnns); scheduleAutoSave(newAnns, true);
        }}
        onSetAttribute={setAnnotationAttribute}
        onRotate={setAnnotationAngle}
        onCopy={copySelected}
        onDuplicate={duplicateSelected}
      />
      <AnnotationsList
        annotations={annotations}
        labelFilter={labelFilter}
        selectedAnnotation={selectedAnnotation}
        onSelect={setSelectedAnnotation}
        onToggleFlag={toggleAnnotationFlag}
        onDelete={deleteAnnotation}
        onClearFilter={() => setLabelFilter(null)}
      />
      <ReviewPanel
        currentImage={currentImage}
        imageStatus={imageStatus}
        reviewComment={reviewComment}
        reviewHistory={reviewHistory}
        onCommentChange={(v) => {
          setReviewComment(v);
          reviewCommentRef.current = v;
          scheduleAutoSave(annotationsRef.current);
        }}
        onStatus={setStatus}
      />
    </>
  );

  return (
    <SidebarProvider className="h-screen bg-background text-foreground">
      {/* Left: image list (collapsible — Cmd/Ctrl+B to toggle) */}
      <Sidebar collapsible="offcanvas">
        <SidebarHeader className="gap-2 p-3 border-b">
          <div className="flex items-center justify-between">
            <h3 className="text-[10px] uppercase tracking-wider text-muted-foreground font-semibold">Images</h3>
            <span className="text-[10px] text-muted-foreground tabular-nums">{filteredImages.length}</span>
          </div>
          {/* Upload folder selector */}
          <div className="space-y-1">
            <span className="text-[10px] text-muted-foreground">Upload to:</span>
            <div className="grid grid-cols-2 gap-1">
              {UPLOAD_FOLDERS.map((f) => (
                <button
                  key={f}
                  onClick={() => setUploadFolder(f)}
                  className={cn(
                    "px-1.5 py-1 text-[10px] rounded border leading-tight text-center transition-colors",
                    uploadFolder === f
                      ? "bg-primary text-primary-foreground border-primary"
                      : "hover:bg-accent border-border text-muted-foreground",
                  )}
                >
                  {f}
                </button>
              ))}
            </div>
          </div>
          <Input
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search filenames…"
            className="h-8 text-xs"
          />
          <Select value={filter} onValueChange={(v) => setFilter(v as ImageStatus | "all")}>
            <SelectTrigger size="sm" className="w-full text-xs">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All</SelectItem>
              <SelectItem value="unannotated">Unannotated</SelectItem>
              <SelectItem value="annotated">Annotated</SelectItem>
              <SelectItem value="accepted">Accepted</SelectItem>
              <SelectItem value="rejected">Rejected</SelectItem>
            </SelectContent>
          </Select>
        </SidebarHeader>
        <SidebarContent className="sidebar-images-scroll">
          <div className="p-1">
            {filteredImages.length === 0 ? (
              <div className="p-4 text-center text-muted-foreground text-xs">
                {images.length === 0 ? (
                  <>No images yet.<br /><span className="text-[11px]">Drop images onto the canvas to upload.</span></>
                ) : (
                  <>No matches for &ldquo;{searchQuery}&rdquo;</>
                )}
              </div>
            ) : (
              filteredImages.map((img) => {
                const folderOfImg = img.filename.split("/").slice(0, 2).join("/");
                const moveTargets = UPLOAD_FOLDERS.filter((f) => f !== folderOfImg);
                const isActive = currentImage === img.filename;
                return (
                  <div key={img.filename} className="relative group">
                    <button
                      onClick={() => loadImage(img.filename)}
                      className={cn(
                        "flex w-full items-center gap-2 px-2 py-1.5 pr-7 rounded-md text-xs transition-colors text-left",
                        isActive
                          ? "bg-primary text-primary-foreground"
                          : "hover:bg-accent hover:text-accent-foreground",
                      )}
                    >
                      <ImageThumb filename={img.filename} />
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-1.5">
                          <span className={cn("w-2 h-2 rounded-full shrink-0", statusDot[img.status])} />
                          {/* Show just the base filename, not the full path */}
                          <span className="truncate" title={img.filename}>{img.filename.split("/").pop()}</span>
                        </div>
                        <div className="flex items-center gap-1.5 mt-0.5">
                          {/* Folder badge — the first two path segments */}
                          {img.filename.includes("/") && (
                            <span className={cn("text-[9px] px-1 rounded font-mono shrink-0", isActive ? "bg-white/20" : "bg-muted text-muted-foreground")}>
                              {folderOfImg}
                            </span>
                          )}
                          <span className={cn("text-[10px] capitalize", isActive ? "text-primary-foreground/70" : "text-muted-foreground")}>{img.status}</span>
                          {img.annotationCount > 0 && (
                            <span className={cn("text-[10px] px-1 rounded font-semibold tabular-nums", isActive ? "bg-white/20" : "bg-muted text-muted-foreground")}>
                              {img.annotationCount}
                            </span>
                          )}
                        </div>
                      </div>
                    </button>
                    <DropdownMenu>
                      <DropdownMenuTrigger asChild>
                        <button
                          type="button"
                          aria-label={`Actions for ${img.filename}`}
                          onClick={(e) => e.stopPropagation()}
                          className={cn(
                            "absolute right-1 top-1/2 -translate-y-1/2 flex items-center justify-center h-6 w-6 rounded transition-opacity",
                            "opacity-0 group-hover:opacity-100 focus:opacity-100 data-[state=open]:opacity-100",
                            isActive ? "text-primary-foreground hover:bg-white/10" : "text-muted-foreground hover:bg-accent",
                          )}
                        >
                          <MoreVertical className="h-3.5 w-3.5" />
                        </button>
                      </DropdownMenuTrigger>
                      <DropdownMenuContent align="end" className="min-w-44">
                        <DropdownMenuLabel className="text-[10px] uppercase tracking-wider text-muted-foreground">
                          Move to
                        </DropdownMenuLabel>
                        {moveTargets.map((f) => (
                          <DropdownMenuItem
                            key={f}
                            onSelect={() => moveImageRequest(img.filename, f)}
                            className="text-xs"
                          >
                            <FolderInput className="h-3.5 w-3.5" />
                            <span className="font-mono">{f}</span>
                          </DropdownMenuItem>
                        ))}
                        <DropdownMenuSeparator />
                        <DropdownMenuItem
                          onSelect={() => deleteImageRequest(img.filename)}
                          variant="destructive"
                          className="text-xs"
                        >
                          <Trash2 className="h-3.5 w-3.5" />
                          Delete (move to trash)
                        </DropdownMenuItem>
                      </DropdownMenuContent>
                    </DropdownMenu>
                  </div>
                );
              })
            )}
          </div>
        </SidebarContent>
        <SidebarFooter className="p-3 border-t">
          <div className="h-1 bg-muted rounded-full overflow-hidden mb-1.5">
            <div className="h-full bg-gradient-to-r from-primary to-emerald-400 transition-all" style={{ width: `${progress}%` }} />
          </div>
          <span className="text-[11px] text-muted-foreground tabular-nums">
            {stats.annotated + stats.accepted + stats.rejected}/{stats.total} annotated
          </span>
        </SidebarFooter>
      </Sidebar>

      <SidebarInset className="min-w-0">
        {/* Top bar */}
        <header className="flex items-center justify-between gap-2 px-2 sm:px-4 h-12 bg-sidebar/95 supports-backdrop-filter:backdrop-blur-md border-b shrink-0 z-10 pt-safe">
          <div className="flex items-center gap-2 sm:gap-3 min-w-0">
            <SidebarTrigger />
            <h1 className="font-display text-base font-bold tracking-tight bg-gradient-to-r from-indigo-300 via-violet-300 to-fuchsia-300 bg-clip-text text-transparent shrink-0">
              Annotator
            </h1>
            <Separator orientation="vertical" className="h-5 hidden md:block" />
            <div className="hidden md:flex gap-1.5 overflow-hidden">
              <Badge variant="outline" className="tabular-nums">{stats.total} total</Badge>
              <Badge variant="secondary" className="tabular-nums">{stats.unannotated} todo</Badge>
              <Badge variant="default" className="tabular-nums hidden lg:inline-flex">{stats.annotated} done</Badge>
              <Badge className="bg-emerald-500/20 text-emerald-400 border-emerald-500/30 tabular-nums hidden lg:inline-flex">{stats.accepted} accepted</Badge>
              <Badge variant="destructive" className="tabular-nums hidden lg:inline-flex">{stats.rejected} rejected</Badge>
            </div>
          </div>
          <div className="flex items-center gap-1 sm:gap-2 shrink-0">
            <Button variant="ghost" size="sm" onClick={() => setShowSettingsModal(true)} className="hidden sm:inline-flex">
              <Settings /> <span className="hidden md:inline">Settings</span>
            </Button>
            <Button variant="ghost" size="sm" onClick={() => setShowExportModal(true)}>
              <FileDown /> <span className="hidden md:inline">Export</span>
            </Button>
            <Tooltip>
              <TooltipTrigger asChild>
                <Button variant="ghost" size="icon-sm" onClick={() => setShowShortcutsModal(true)} className="hidden sm:inline-flex">
                  <Keyboard />
                </Button>
              </TooltipTrigger>
              <TooltipContent>Keyboard shortcuts</TooltipContent>
            </Tooltip>
            {/* Mobile inspector trigger — opens right-side sheet < lg */}
            <Sheet open={mobileInspectorOpen} onOpenChange={setMobileInspectorOpen}>
              <Tooltip>
                <TooltipTrigger asChild>
                  <SheetTrigger asChild>
                    <Button variant="ghost" size="icon-sm" className="lg:hidden" aria-label="Open inspector">
                      <PanelRight />
                    </Button>
                  </SheetTrigger>
                </TooltipTrigger>
                <TooltipContent>Inspector</TooltipContent>
              </Tooltip>
              <SheetContent side="right" className="w-[88vw] sm:max-w-sm p-0 bg-sidebar border-l">
                <SheetHeader className="border-b p-3">
                  <SheetTitle>Inspector</SheetTitle>
                </SheetHeader>
                <div className="flex-1 overflow-y-auto">
                  {inspectorPanels}
                </div>
              </SheetContent>
            </Sheet>
          </div>
        </header>

        <div className="flex flex-1 min-h-0">
          {/* Center: toolbar + canvas */}
          <section className="flex-1 flex flex-col min-w-0">
          <div className="flex items-center gap-1 px-2 py-1.5 bg-sidebar border-b overflow-x-auto scrollbar-hide [&::-webkit-scrollbar]:hidden">
            <div className="flex gap-0.5 pr-2 border-r shrink-0">
              {TOOLS.map((t) => {
                const Icon = t.icon;
                return (
                  <Tooltip key={t.id}>
                    <TooltipTrigger asChild>
                      <Button
                        variant={activeTool === t.id ? "default" : "ghost"}
                        size="icon-sm"
                        onClick={() => setActiveTool(t.id)}
                      >
                        <Icon />
                      </Button>
                    </TooltipTrigger>
                    <TooltipContent>{t.label} ({t.shortcut})</TooltipContent>
                  </Tooltip>
                );
              })}
            </div>

            <div className="flex items-center gap-0.5 px-2 border-r shrink-0">
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button variant="ghost" size="icon-sm" onClick={zoomOut}><ZoomOut /></Button>
                </TooltipTrigger>
                <TooltipContent>Zoom out (-)</TooltipContent>
              </Tooltip>
              <span className="w-12 h-7 flex items-center justify-center text-[11px] text-muted-foreground font-mono tabular-nums">{zoomLevel}%</span>
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button variant="ghost" size="icon-sm" onClick={zoomIn}><ZoomIn /></Button>
                </TooltipTrigger>
                <TooltipContent>Zoom in (+)</TooltipContent>
              </Tooltip>
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button variant="ghost" size="icon-sm" onClick={fitToView}><Maximize2 /></Button>
                </TooltipTrigger>
                <TooltipContent>Fit to view (F)</TooltipContent>
              </Tooltip>
            </div>

            <div className="flex gap-0.5 px-2 border-r shrink-0">
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button variant="ghost" size="icon-sm" onClick={undo}><Undo2 /></Button>
                </TooltipTrigger>
                <TooltipContent>Undo (⌘Z)</TooltipContent>
              </Tooltip>
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button variant="ghost" size="icon-sm" onClick={redo}><Redo2 /></Button>
                </TooltipTrigger>
                <TooltipContent>Redo (⌘⇧Z)</TooltipContent>
              </Tooltip>
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button variant="ghost" size="icon-sm" onClick={deleteSelected} disabled={!selectedAnnotation}><Trash2 /></Button>
                </TooltipTrigger>
                <TooltipContent>Delete selected (Del)</TooltipContent>
              </Tooltip>
            </div>

            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  variant={annotationsVisible ? "ghost" : "destructive"}
                  size="icon-sm"
                  onClick={toggleAnnotations}
                >
                  {annotationsVisible ? <Eye /> : <EyeOff />}
                </Button>
              </TooltipTrigger>
              <TooltipContent>{annotationsVisible ? "Hide annotations" : "Show annotations"} (H)</TooltipContent>
            </Tooltip>

            <div className="flex items-center gap-1 ml-auto shrink-0">
              {currentImage && (
                <span className="text-[11px] text-muted-foreground mr-2 tabular-nums hidden sm:inline">{currentIdx + 1} / {images.length}</span>
              )}
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button variant="ghost" size="icon-sm" onClick={() => goToImage("prev")}><ChevronLeft /></Button>
                </TooltipTrigger>
                <TooltipContent>Previous image (←)</TooltipContent>
              </Tooltip>
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button variant="ghost" size="icon-sm" onClick={() => goToImage("next")}><ChevronRight /></Button>
                </TooltipTrigger>
                <TooltipContent>Next image (→)</TooltipContent>
              </Tooltip>
            </div>
          </div>

          <div
            ref={containerRef}
            className={cn("flex-1 relative overflow-hidden canvas-bg", isDragOver && "ring-2 ring-inset ring-primary")}
            onDragOver={(e) => { e.preventDefault(); setIsDragOver(true); }}
            onDragLeave={() => setIsDragOver(false)}
            onDrop={handleDrop}
          >
            <canvas ref={canvasRef} />
            {!currentImage && (
              <div className="absolute inset-0 flex flex-col items-center justify-center text-muted-foreground gap-4 px-6">
                <div className="relative size-16 rounded-2xl bg-card flex items-center justify-center border shadow-lg shadow-primary/5">
                  <span className="absolute inset-0 rounded-2xl bg-primary/10 blur-xl" aria-hidden />
                  <Upload className="size-7 text-primary relative" />
                </div>
                <div className="space-y-1 text-center max-w-md">
                  <h2 className="font-display text-xl text-foreground font-semibold tracking-tight">No image loaded</h2>
                  <p className="text-sm">
                    {images.length === 0
                      ? <>Drop images anywhere on this area, or click below.</>
                      : <>Pick an image from the sidebar, or drop new ones here.</>
                    }
                  </p>
                </div>
                <label className="cursor-pointer">
                  <input
                    type="file"
                    accept="image/*,.heic,.heif"
                    multiple
                    className="hidden"
                    onChange={async (ev) => {
                      const files = Array.from(ev.target.files || []);
                      await uploadFiles(files, uploadFolder);
                      ev.target.value = "";
                    }}
                  />
                  <span className="inline-flex items-center gap-2 h-10 px-5 rounded-lg bg-primary text-primary-foreground text-sm font-medium hover:bg-primary/85 active:translate-y-px transition shadow-md shadow-primary/20">
                    <Upload className="size-4" />
                    Upload to {uploadFolder}
                  </span>
                </label>
                <p className="text-[11px] text-muted-foreground">
                  Stored in R2 under <code className="bg-muted px-1.5 py-0.5 rounded font-mono">raw/{uploadFolder}/</code>
                </p>
              </div>
            )}
            {isLoading && currentImage && (
              <div className="absolute top-3 left-1/2 -translate-x-1/2 z-40 px-3 py-1.5 rounded-full bg-card/90 border supports-backdrop-filter:backdrop-blur-md text-xs text-foreground flex items-center gap-2 shadow-lg shadow-black/20">
                <span className="relative flex size-2">
                  <span className="absolute inset-0 rounded-full bg-primary animate-ping opacity-75" />
                  <span className="relative rounded-full size-2 bg-primary" />
                </span>
                <span className="truncate max-w-[60vw]">Loading {currentImage}…</span>
              </div>
            )}
            {isDragOver && (
              <div className="absolute inset-0 bg-primary/10 supports-backdrop-filter:backdrop-blur-sm flex items-center justify-center z-50 pointer-events-none animate-in fade-in duration-150">
                <div className="px-6 py-4 bg-card rounded-2xl border-2 border-dashed border-primary text-primary font-semibold flex items-center gap-2 shadow-xl shadow-primary/20">
                  <Upload className="size-5" />
                  Drop to upload
                </div>
              </div>
            )}
          </div>

          <div className="flex items-center gap-2 sm:gap-4 px-3 py-1 bg-sidebar border-t text-[11px] text-muted-foreground font-mono tabular-nums pb-safe overflow-x-auto [&::-webkit-scrollbar]:hidden">
            <span className="shrink-0">x: {cursorPos.x}, y: {cursorPos.y}</span>
            <span className="shrink-0">{zoomLevel}%</span>
            {currentImage && <>
              <span className="truncate max-w-40 font-sans hidden md:inline">{currentImage}</span>
              <span className="shrink-0 hidden sm:inline">{imageDimState.width}×{imageDimState.height}</span>
              <span className="shrink-0">{annotations.length} ann.</span>
            </>}
            <span className={cn("ml-auto shrink-0", saveIndicator === "Saved" && "text-emerald-400", saveIndicator === "Save failed" && "text-red-400")}>{saveIndicator}</span>
          </div>
        </section>

        {/* Right: labels / properties / annotations / review — hidden < lg (use Sheet) */}
        <aside className="hidden lg:flex w-80 shrink-0 bg-sidebar border-l flex-col overflow-hidden">
          <div className="flex-1 overflow-y-auto [&::-webkit-scrollbar]:w-1.5 [&::-webkit-scrollbar-track]:transparent [&::-webkit-scrollbar-thumb]:rounded-full [&::-webkit-scrollbar-thumb]:bg-border hover:[&::-webkit-scrollbar-thumb]:bg-muted-foreground/40">
            {inspectorPanels}
          </div>
        </aside>
        </div>
      </SidebarInset>

      <ShortcutsDialog open={showShortcutsModal} onOpenChange={setShowShortcutsModal} />


      {/* Export dialog */}
      <Dialog
        open={showExportModal}
        onOpenChange={(open) => { setShowExportModal(open); if (!open) setExportResult(null); }}
      >
        <DialogContent className="sm:max-w-[480px]">
          <DialogHeader>
            <DialogTitle>Export annotations</DialogTitle>
            <DialogDescription>Rejected images are excluded from all exports.</DialogDescription>
          </DialogHeader>
          <div className="space-y-3">
            <button
              onClick={doExportCOCO}
              className="w-full p-4 border rounded-lg text-left hover:border-primary hover:bg-accent transition group"
            >
              <h3 className="text-sm font-semibold group-hover:text-primary transition">COCO JSON</h3>
              <p className="text-xs text-muted-foreground mt-1">Standard format. Boxes, polygons, keypoints. Works with Detectron2, MMDetection, YOLOv5+.</p>
            </button>
            <button
              onClick={doExportYOLO}
              className="w-full p-4 border rounded-lg text-left hover:border-primary hover:bg-accent transition group"
            >
              <h3 className="text-sm font-semibold group-hover:text-primary transition">YOLO TXT</h3>
              <p className="text-xs text-muted-foreground mt-1">Normalized bounding boxes per image. Compatible with Ultralytics YOLOv5/v8. Written to R2 /exports.</p>
            </button>
            <button
              onClick={doExportZip}
              className="w-full p-4 border rounded-lg text-left hover:border-primary hover:bg-accent transition group border-dashed"
            >
              <h3 className="text-sm font-semibold group-hover:text-primary transition">Bulk ZIP Download</h3>
              <p className="text-xs text-muted-foreground mt-1">All annotation JSONs + COCO + YOLO in one ZIP file. Includes all annotated images.</p>
            </button>
            {exportResult && (
              <div className="p-3 bg-emerald-500/10 rounded-md text-xs text-emerald-400 font-mono border border-emerald-500/30">
                {exportResult}
              </div>
            )}
          </div>
        </DialogContent>
      </Dialog>

      {/* Settings dialog */}
      <Dialog open={showSettingsModal} onOpenChange={setShowSettingsModal}>
        <DialogContent className="sm:max-w-[420px]">
          <DialogHeader>
            <DialogTitle>Settings</DialogTitle>
          </DialogHeader>
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <Label htmlFor="auto-advance" className="text-sm">Auto-advance after accept/reject</Label>
              <Checkbox
                id="auto-advance"
                checked={autoAdvance}
                onCheckedChange={(v) => setAutoAdvance(Boolean(v))}
              />
            </div>
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <Label className="text-sm">Default annotation fill opacity</Label>
                <span className="text-xs text-muted-foreground tabular-nums">{opacity}%</span>
              </div>
              <Slider min={0} max={100} step={1} value={[opacity]} onValueChange={(v) => { setOpacity(v[0]); opacityRef.current = v[0]; }} />
            </div>

            <Separator />

            {/* Storage */}
            <div className="space-y-2">
              <Label className="text-sm font-semibold">R2 Storage</Label>
              <p className="text-xs text-muted-foreground">
                Folder structure: <code className="bg-muted px-1 py-0.5 rounded">raw/bus/positive</code>, <code className="bg-muted px-1 py-0.5 rounded">raw/bus/negative</code>, <code className="bg-muted px-1 py-0.5 rounded">raw/legua/positive</code>, <code className="bg-muted px-1 py-0.5 rounded">raw/legua/negative</code>
              </p>
              <Button variant="outline" size="sm" className="w-full" onClick={doInitFolders}>
                Create folder structure in R2
              </Button>
              <Button variant="outline" size="sm" className="w-full" onClick={doRebuildManifest}>
                Rebuild manifest (rescan all annotations)
              </Button>
            </div>

            <Separator />

            {/* Danger zone */}
            <div className="space-y-2 rounded-lg border border-destructive/30 p-3">
              <Label className="text-sm font-semibold text-destructive">Danger Zone</Label>
              <p className="text-xs text-muted-foreground">
                Permanently deletes all raw images and annotation files from R2. This cannot be undone.
              </p>
              <Button
                variant="destructive"
                size="sm"
                className="w-full"
                onClick={() => { setShowSettingsModal(false); setShowClearConfirm(true); }}
              >
                <Trash2 /> Delete all images &amp; annotations
              </Button>
            </div>
          </div>
        </DialogContent>
      </Dialog>

      {/* Confirm clear all */}
      <Dialog open={showClearConfirm} onOpenChange={setShowClearConfirm}>
        <DialogContent className="sm:max-w-[420px]">
          <DialogHeader>
            <DialogTitle>Delete everything?</DialogTitle>
            <DialogDescription>
              This will permanently delete <span className="font-semibold text-foreground">all {stats.total} images</span> and their annotations from R2.
              Exports in <code className="bg-muted px-1 py-0.5 rounded text-xs">exports/</code> are preserved.
              <br /><br />
              There is no undo.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter className="gap-2">
            <Button variant="ghost" onClick={() => setShowClearConfirm(false)}>Cancel</Button>
            <Button variant="destructive" onClick={doClearAll}>
              <Trash2 /> Yes, delete everything
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Confirm label delete (orphan protection) */}
      <Dialog
        open={pendingLabelDelete !== null}
        onOpenChange={(open) => { if (!open) setPendingLabelDelete(null); }}
      >
        <DialogContent className="sm:max-w-[420px]">
          <DialogHeader>
            <DialogTitle>Delete label &ldquo;{pendingLabelDelete?.name}&rdquo;?</DialogTitle>
            <DialogDescription>
              This label is used by <span className="font-semibold text-foreground">{pendingLabelDelete?.count}</span> annotation{pendingLabelDelete?.count === 1 ? "" : "s"}.
              Deleting the label will also delete all annotations that reference it.
              <br />
              <br />
              Prefer to keep the annotations? Cancel, then reassign them first.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter className="gap-2">
            <Button variant="ghost" onClick={() => setPendingLabelDelete(null)}>Cancel</Button>
            <Button variant="destructive" onClick={confirmDeleteLabelAndAnnotations}>
              <Trash2 /> Delete label &amp; annotations
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </SidebarProvider>
  );
}
