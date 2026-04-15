"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import type {
  Annotation,
  ImageAnnotation,
  ImageInfo,
  ImageStatus,
  LabelDef,
  ProjectStats,
} from "@/lib/types";
import { v4 as uuidv4 } from "uuid";

/**
 * Lazy thumbnail: renders a tiny <img> that loads from the R2 proxy only
 * when scrolled into view. The browser's own disk cache + Cache-Control on
 * /api/raw/ means re-navigating the sidebar is fast.
 */
function ImageThumb({ filename }: { filename: string }) {
  const ref = useRef<HTMLDivElement>(null);
  const [visible, setVisible] = useState(false);

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
      className="w-10 h-10 rounded bg-[#0a0b0f] border border-[#2a2d3e] shrink-0 overflow-hidden flex items-center justify-center"
    >
      {visible ? (
        // eslint-disable-next-line @next/next/no-img-element
        <img
          src={`/api/raw/${encodeURIComponent(filename)}`}
          alt=""
          className="w-full h-full object-cover"
          loading="lazy"
          draggable={false}
        />
      ) : null}
    </div>
  );
}

const DEFAULT_LABELS: LabelDef[] = [
  { name: "object", color: "#FF6B6B" },
  { name: "person", color: "#4ECDC4" },
  { name: "vehicle", color: "#45B7D1" },
  { name: "animal", color: "#96CEB4" },
  { name: "text", color: "#FFEAA7" },
];

const LABEL_COLORS = [
  "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7",
  "#DDA0DD", "#98D8C8", "#F7DC6F", "#BB8FCE", "#85C1E9",
  "#F0B27A", "#82E0AA", "#F1948A", "#AED6F1", "#D2B4DE",
];

type Tool = "select" | "bbox" | "polygon" | "polyline" | "ellipse" | "keypoint" | "pan";

/* eslint-disable @typescript-eslint/no-explicit-any */

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
  const [exportResult, setExportResult] = useState<string | null>(null);
  const [saveIndicator, setSaveIndicator] = useState("Auto-save ON");
  const [brightness, setBrightness] = useState(100);
  const [contrast, setContrast] = useState(100);
  const [opacity, setOpacity] = useState(70);
  const [autoAdvance, setAutoAdvance] = useState(false);
  const [isDragOver, setIsDragOver] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");
  const [labelFilter, setLabelFilter] = useState<string | null>(null);
  const [clipboard, setClipboard] = useState<Annotation | null>(null);
  const [attrDraftKey, setAttrDraftKey] = useState("");
  const [attrDraftVal, setAttrDraftVal] = useState("");

  const undoStack = useRef<Annotation[][]>([]);
  const redoStack = useRef<Annotation[][]>([]);
  const polygonPoints = useRef<number[][]>([]);
  const polygonLines = useRef<any[]>([]);
  const polygonDots = useRef<any[]>([]);
  const isDrawing = useRef(false);
  const drawStart = useRef<{ x: number; y: number } | null>(null);
  const tempRect = useRef<any>(null);
  const imageDims = useRef<{ width: number; height: number }>({ width: 0, height: 0 });
  const autoSaveTimerRef = useRef<NodeJS.Timeout | null>(null);
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

  useEffect(() => { activeToolRef.current = activeTool; }, [activeTool]);
  useEffect(() => { activeLabelRef.current = activeLabel; }, [activeLabel]);
  useEffect(() => { labelsRef.current = labels; }, [labels]);
  useEffect(() => { annotationsRef.current = annotations; }, [annotations]);
  useEffect(() => { currentImageRef.current = currentImage; }, [currentImage]);
  useEffect(() => { autoAdvanceRef.current = autoAdvance; }, [autoAdvance]);
  useEffect(() => { imagesRef.current = images; }, [images]);
  useEffect(() => { opacityRef.current = opacity; }, [opacity]);

  const fetchImages = useCallback(async () => {
    const res = await fetch(`/api/images?filter=${filter}`);
    setImages(await res.json());
  }, [filter]);

  const fetchStats = useCallback(async () => {
    const res = await fetch("/api/stats");
    setStats(await res.json());
  }, []);

  useEffect(() => { fetchImages(); fetchStats(); }, [fetchImages, fetchStats]);

  // Brightness/contrast filter on canvas container
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

    const opacityHex = Math.round((opacityRef.current / 100) * 255).toString(16).padStart(2, "0");

    for (const ann of anns) {
      if (ann.hidden) continue;
      const interactive = activeToolRef.current === "select" && !ann.locked;

      if (ann.type === "bbox") {
        const rect = new fabric.Rect({
          left: ann.x, top: ann.y, width: ann.width, height: ann.height,
          fill: ann.color + opacityHex, stroke: ann.color, strokeWidth: 2,
          selectable: interactive, evented: interactive,
          cornerColor: "#fff", cornerStrokeColor: ann.color, cornerSize: 8,
          transparentCorners: false, borderColor: ann.color,
          lockRotation: true,
        });
        (rect as any).annotationId = ann.id;
        (rect as any).annotationType = "bbox";
        const text = new fabric.FabricText(ann.label, {
          left: (ann.x || 0) + 2, top: (ann.y || 0) - 20,
          fontSize: 12, fill: "#fff", backgroundColor: ann.color + "DD",
          padding: 3, selectable: false, evented: false,
          fontFamily: "Inter, sans-serif",
        });
        (text as any).isLabel = true;
        c.add(rect); c.add(text);
      } else if (ann.type === "ellipse") {
        const rx = (ann.width || 0) / 2;
        const ry = (ann.height || 0) / 2;
        const ellipse = new fabric.Ellipse({
          left: ann.x, top: ann.y, rx, ry,
          fill: ann.color + opacityHex, stroke: ann.color, strokeWidth: 2,
          selectable: interactive, evented: interactive,
          cornerColor: "#fff", cornerStrokeColor: ann.color, cornerSize: 8,
          transparentCorners: false, borderColor: ann.color,
          lockRotation: true,
        });
        (ellipse as any).annotationId = ann.id;
        (ellipse as any).annotationType = "ellipse";
        const text = new fabric.FabricText(ann.label, {
          left: (ann.x || 0) + 2, top: (ann.y || 0) - 20,
          fontSize: 12, fill: "#fff", backgroundColor: ann.color + "DD",
          padding: 3, selectable: false, evented: false,
          fontFamily: "Inter, sans-serif",
        });
        (text as any).isLabel = true;
        c.add(ellipse); c.add(text);
      } else if (ann.type === "polygon" && ann.points) {
        const points = ann.points.map((p: number[]) => new fabric.Point(p[0], p[1]));
        const polygon = new fabric.Polygon(points, {
          fill: ann.color + opacityHex, stroke: ann.color, strokeWidth: 2,
          selectable: interactive, evented: interactive,
          lockRotation: true,
        });
        (polygon as any).annotationId = ann.id;
        (polygon as any).annotationType = "polygon";
        const bounds = polygon.getBoundingRect();
        const text = new fabric.FabricText(ann.label, {
          left: bounds.left + 2, top: bounds.top - 20,
          fontSize: 12, fill: "#fff", backgroundColor: ann.color + "DD",
          padding: 3, selectable: false, evented: false,
          fontFamily: "Inter, sans-serif",
        });
        (text as any).isLabel = true;
        c.add(polygon); c.add(text);
      } else if (ann.type === "polyline" && ann.points) {
        const points = ann.points.map((p: number[]) => new fabric.Point(p[0], p[1]));
        const polyline = new fabric.Polyline(points, {
          fill: "transparent", stroke: ann.color, strokeWidth: 3,
          selectable: interactive, evented: interactive,
          lockRotation: true,
        });
        (polyline as any).annotationId = ann.id;
        (polyline as any).annotationType = "polyline";
        const bounds = polyline.getBoundingRect();
        const text = new fabric.FabricText(ann.label, {
          left: bounds.left + 2, top: bounds.top - 20,
          fontSize: 12, fill: "#fff", backgroundColor: ann.color + "DD",
          padding: 3, selectable: false, evented: false,
          fontFamily: "Inter, sans-serif",
        });
        (text as any).isLabel = true;
        c.add(polyline); c.add(text);
      } else if (ann.type === "keypoint") {
        // Outer ring
        const outerCircle = new fabric.Circle({
          left: (ann.x || 0) - 10, top: (ann.y || 0) - 10, radius: 10,
          fill: "transparent", stroke: ann.color, strokeWidth: 2,
          selectable: false, evented: false,
        });
        // Inner dot
        const circle = new fabric.Circle({
          left: (ann.x || 0) - 5, top: (ann.y || 0) - 5, radius: 5,
          fill: ann.color, stroke: "#fff", strokeWidth: 2,
          selectable: interactive, evented: interactive,
        });
        (circle as any).annotationId = ann.id;
        (circle as any).annotationType = "keypoint";
        const text = new fabric.FabricText(ann.label, {
          left: (ann.x || 0) + 14, top: (ann.y || 0) - 8,
          fontSize: 11, fill: "#fff", backgroundColor: ann.color + "DD",
          padding: 2, selectable: false, evented: false,
          fontFamily: "Inter, sans-serif",
        });
        (text as any).isLabel = true;
        c.add(outerCircle); c.add(circle); c.add(text);
      }
    }
    c.renderAll();
  }, []);

  const scheduleAutoSave = useCallback((anns: Annotation[]) => {
    if (autoSaveTimerRef.current) clearTimeout(autoSaveTimerRef.current);
    // Capture filename and dims at schedule time. Reading them from refs inside
    // the timer would let a pending save write the old image's annotations to
    // whatever image is active when the timer fires.
    const targetFilename = currentImageRef.current;
    if (!targetFilename) return;
    const width = imageDims.current.width;
    const height = imageDims.current.height;
    setSaveIndicator("Saving...");
    autoSaveTimerRef.current = setTimeout(async () => {
      await fetch(`/api/annotations/${encodeURIComponent(targetFilename)}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          filename: targetFilename,
          annotations: anns,
          labels: labelsRef.current,
          status: anns.length > 0 ? "annotated" : "unannotated",
          imageWidth: width,
          imageHeight: height,
        }),
      });
      setSaveIndicator("Saved!");
      setTimeout(() => setSaveIndicator("Auto-save ON"), 1500);
      fetchImages(); fetchStats();
    }, 600);
  }, [fetchImages, fetchStats]);

  const pushUndo = useCallback(() => {
    undoStack.current.push(JSON.parse(JSON.stringify(annotationsRef.current)));
    redoStack.current = [];
  }, []);

  const undo = useCallback(() => {
    if (undoStack.current.length === 0) return;
    redoStack.current.push(JSON.parse(JSON.stringify(annotationsRef.current)));
    const prev = undoStack.current.pop()!;
    setAnnotations(prev); annotationsRef.current = prev;
    renderAnnotations(prev); scheduleAutoSave(prev);
  }, [renderAnnotations, scheduleAutoSave]);

  const redo = useCallback(() => {
    if (redoStack.current.length === 0) return;
    undoStack.current.push(JSON.parse(JSON.stringify(annotationsRef.current)));
    const next = redoStack.current.pop()!;
    setAnnotations(next); annotationsRef.current = next;
    renderAnnotations(next); scheduleAutoSave(next);
  }, [renderAnnotations, scheduleAutoSave]);

  const deleteSelected = useCallback(() => {
    if (!selectedAnnotation) return;
    pushUndo();
    const newAnns = annotationsRef.current.filter((a) => a.id !== selectedAnnotation);
    setAnnotations(newAnns); annotationsRef.current = newAnns;
    setSelectedAnnotation(null);
    renderAnnotations(newAnns); scheduleAutoSave(newAnns);
  }, [selectedAnnotation, pushUndo, renderAnnotations, scheduleAutoSave]);

  const copySelected = useCallback(() => {
    const ann = annotationsRef.current.find((a) => a.id === selectedAnnotation);
    if (!ann) return;
    // Deep clone so subsequent edits to the original don't mutate the clipboard.
    setClipboard(JSON.parse(JSON.stringify(ann)));
    setSaveIndicator("Copied");
    setTimeout(() => setSaveIndicator("Auto-save ON"), 1000);
  }, [selectedAnnotation]);

  /**
   * Paste the clipboard annotation into the current image with a small offset
   * so it's visually distinct from the source. Works across images because the
   * clipboard lives in component state.
   */
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
    renderAnnotations(newAnns); scheduleAutoSave(newAnns);
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
    renderAnnotations(newAnns); scheduleAutoSave(newAnns);
  }, [selectedAnnotation, pushUndo, renderAnnotations, scheduleAutoSave]);

  const deleteAnnotation = useCallback((id: string) => {
    pushUndo();
    const newAnns = annotationsRef.current.filter((a) => a.id !== id);
    setAnnotations(newAnns); annotationsRef.current = newAnns;
    if (selectedAnnotation === id) setSelectedAnnotation(null);
    renderAnnotations(newAnns); scheduleAutoSave(newAnns);
  }, [pushUndo, selectedAnnotation, renderAnnotations, scheduleAutoSave]);

  const toggleAnnotationFlag = useCallback(
    (id: string, key: "hidden" | "locked") => {
      const newAnns = annotationsRef.current.map((a) =>
        a.id === id ? { ...a, [key]: !a[key] } : a,
      );
      setAnnotations(newAnns); annotationsRef.current = newAnns;
      // If we just hid the selected annotation, drop the selection so the
      // properties panel doesn't point at something invisible.
      if (key === "hidden" && selectedAnnotation === id) setSelectedAnnotation(null);
      renderAnnotations(newAnns); scheduleAutoSave(newAnns);
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

  const zoomIn = useCallback(() => {
    const c = fabricRef.current; if (!c) return;
    const zoom = Math.min(c.getZoom() * 1.2, 10);
    c.setZoom(zoom); setZoomLevel(Math.round(zoom * 100));
  }, []);

  const zoomOut = useCallback(() => {
    const c = fabricRef.current; if (!c) return;
    const zoom = Math.max(c.getZoom() * 0.8, 0.1);
    c.setZoom(zoom); setZoomLevel(Math.round(zoom * 100));
  }, []);

  const fitToView = useCallback(() => {
    const c = fabricRef.current; if (!c) return;
    c.setZoom(1); c.setViewportTransform([1, 0, 0, 1, 0, 0]); setZoomLevel(100);
  }, []);

  const toggleAnnotations = useCallback(() => {
    const c = fabricRef.current; if (!c) return;
    const visible = !annotationsVisible;
    setAnnotationsVisible(visible);
    c.getObjects().forEach((obj: any) => {
      if (!obj.isBackgroundImage) obj.set("visible", visible);
    });
    c.renderAll();
  }, [annotationsVisible]);

  const setStatus = useCallback(async (status: ImageStatus) => {
    if (!currentImage) return;
    const comment = reviewComment || `Status set to ${status}`;
    await fetch(`/api/annotations/${encodeURIComponent(currentImage)}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ status, comment }),
    });
    setImageStatus(status);
    setReviewHistory((prev) => [...prev, { action: status, comment, timestamp: new Date().toISOString() }]);
    fetchImages(); fetchStats();
    setReviewComment("");

    // Auto-advance to next image after accept/reject (uses ref to avoid circular dep)
    if (autoAdvanceRef.current && (status === "accepted" || status === "rejected")) {
      autoAdvancePending.current = true;
    }
  }, [currentImage, reviewComment, fetchImages, fetchStats]);

  const saveNow = useCallback(async () => {
    if (!currentImage) return;
    setSaveIndicator("Saving...");
    await fetch(`/api/annotations/${encodeURIComponent(currentImage)}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        filename: currentImage, annotations, labels,
        status: annotations.length > 0 ? (imageStatus === "unannotated" ? "annotated" : imageStatus) : "unannotated",
        reviewComment, history: reviewHistory,
        imageWidth: imageDims.current.width, imageHeight: imageDims.current.height,
      }),
    });
    setSaveIndicator("Saved!");
    setTimeout(() => setSaveIndicator("Auto-save ON"), 1500);
    fetchImages(); fetchStats();
  }, [currentImage, annotations, labels, imageStatus, reviewComment, reviewHistory, fetchImages, fetchStats]);

  const loadImage = useCallback(async (filename: string) => {
    // Cancel any pending auto-save so it can't fire after we switch images.
    if (autoSaveTimerRef.current) {
      clearTimeout(autoSaveTimerRef.current);
      autoSaveTimerRef.current = null;
    }

    // Save current image before switching
    if (currentImageRef.current && annotationsRef.current.length > 0) {
      await fetch(`/api/annotations/${encodeURIComponent(currentImageRef.current)}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          filename: currentImageRef.current,
          annotations: annotationsRef.current,
          labels: labelsRef.current,
          status: "annotated",
          imageWidth: imageDims.current.width,
          imageHeight: imageDims.current.height,
        }),
      });
    }

    setCurrentImage(filename);
    currentImageRef.current = filename;

    const res = await fetch(`/api/annotations/${encodeURIComponent(filename)}`);
    const data: ImageAnnotation = await res.json();

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
    if (!canvas) return;

    canvas.clear();
    canvas.setViewportTransform([1, 0, 0, 1, 0, 0]);

    const fabric = await import("fabric");
    const img = await fabric.FabricImage.fromURL(`/api/raw/${encodeURIComponent(filename)}`, { crossOrigin: "anonymous" });
    imageDims.current = { width: img.width || 800, height: img.height || 600 };

    const container = containerRef.current;
    if (!container) return;
    const cw = container.clientWidth;
    const ch = container.clientHeight;
    const scale = Math.min(cw / (img.width || 800), ch / (img.height || 600)) * 0.9;

    img.set({
      left: (cw - (img.width || 800) * scale) / 2,
      top: (ch - (img.height || 600) * scale) / 2,
      scaleX: scale, scaleY: scale,
      selectable: false, evented: false, hoverCursor: "default",
    });
    (img as any).isBackgroundImage = true;
    canvas.add(img); canvas.sendObjectToBack(img);
    renderAnnotations(data.annotations || [], canvas);
    canvas.renderAll();
    setZoomLevel(100);
    setBrightness(100); setContrast(100);
  }, [renderAnnotations]);

  // Initialize Fabric canvas
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
      });
      fabricRef.current = canvas;

      // --- MOUSE DOWN ---
      canvas.on("mouse:down", (opt: any) => {
        const tool = activeToolRef.current;
        const e = opt.e as MouseEvent;

        // Middle mouse or pan tool = pan
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
            strokeWidth: 2, strokeDashArray: [6, 4], selectable: false, evented: false,
          });
          tempRect.current = rect; canvas.add(rect);
        } else if (tool === "ellipse") {
          isDrawing.current = true;
          drawStart.current = { x: pointer.x, y: pointer.y };
          const ellipse = new fabric.Ellipse({
            left: pointer.x, top: pointer.y, rx: 0, ry: 0,
            fill: "transparent", stroke: getLabelColor(activeLabelRef.current),
            strokeWidth: 2, strokeDashArray: [6, 4], selectable: false, evented: false,
          });
          // Reuse tempRect slot for the in-flight ellipse so mouse:move/up can touch it.
          tempRect.current = ellipse; canvas.add(ellipse);
        } else if (tool === "polyline") {
          // Same click-to-add-vertex UX as polygon; double-click finishes.
          polygonPoints.current.push([pointer.x, pointer.y]);
          const dot = new fabric.Circle({
            left: pointer.x - 3, top: pointer.y - 3, radius: 3,
            fill: getLabelColor(activeLabelRef.current), stroke: "#fff", strokeWidth: 1,
            selectable: false, evented: false,
          });
          canvas.add(dot); polygonDots.current.push(dot);
          if (polygonPoints.current.length > 1) {
            const pts = polygonPoints.current;
            const prev = pts[pts.length - 2]; const curr = pts[pts.length - 1];
            const line = new fabric.Line([prev[0], prev[1], curr[0], curr[1]], {
              stroke: getLabelColor(activeLabelRef.current), strokeWidth: 2,
              selectable: false, evented: false,
            });
            canvas.add(line); polygonLines.current.push(line);
          }
          canvas.renderAll();
        } else if (tool === "polygon") {
          polygonPoints.current.push([pointer.x, pointer.y]);
          const dot = new fabric.Circle({
            left: pointer.x - 4, top: pointer.y - 4, radius: 4,
            fill: getLabelColor(activeLabelRef.current), stroke: "#fff", strokeWidth: 1,
            selectable: false, evented: false,
          });
          canvas.add(dot); polygonDots.current.push(dot);
          if (polygonPoints.current.length > 1) {
            const pts = polygonPoints.current;
            const prev = pts[pts.length - 2]; const curr = pts[pts.length - 1];
            const line = new fabric.Line([prev[0], prev[1], curr[0], curr[1]], {
              stroke: getLabelColor(activeLabelRef.current), strokeWidth: 2,
              strokeDashArray: [4, 4], selectable: false, evented: false,
            });
            canvas.add(line); polygonLines.current.push(line);
          }
          canvas.renderAll();
        } else if (tool === "keypoint") {
          undoStack.current.push(JSON.parse(JSON.stringify(annotationsRef.current)));
          redoStack.current = [];
          const ann: Annotation = {
            id: uuidv4(), type: "keypoint", label: activeLabelRef.current,
            color: getLabelColor(activeLabelRef.current), x: pointer.x, y: pointer.y,
          };
          const newAnns = [...annotationsRef.current, ann];
          setAnnotations(newAnns); annotationsRef.current = newAnns;
          renderAnnotations(newAnns, canvas);
          scheduleAutoSave(newAnns);
        }
      });

      // --- MOUSE MOVE ---
      canvas.on("mouse:move", (opt: any) => {
        const e = opt.e as MouseEvent;

        // Panning
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
          canvas.renderAll();
        }
      });

      // --- MOUSE UP ---
      canvas.on("mouse:up", () => {
        // End panning
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
            undoStack.current.push(JSON.parse(JSON.stringify(annotationsRef.current)));
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
            scheduleAutoSave(newAnns);
          }
        }
      });

      // --- DOUBLE CLICK: finish polygon / polyline ---
      canvas.on("mouse:dblclick", () => {
        const tool = activeToolRef.current;
        const minPts = tool === "polyline" ? 2 : 3;
        if ((tool === "polygon" || tool === "polyline") && polygonPoints.current.length >= minPts) {
          undoStack.current.push(JSON.parse(JSON.stringify(annotationsRef.current)));
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
          polygonPoints.current = []; polygonLines.current = []; polygonDots.current = [];
          const newAnns = [...annotationsRef.current, ann];
          setAnnotations(newAnns); annotationsRef.current = newAnns;
          renderAnnotations(newAnns, canvas);
          scheduleAutoSave(newAnns);
        }
      });

      // --- MOUSE WHEEL: zoom ---
      canvas.on("mouse:wheel", (opt: any) => {
        const delta = opt.e.deltaY;
        let zoom = canvas.getZoom();
        zoom *= 0.999 ** delta;
        zoom = Math.min(Math.max(0.1, zoom), 10);
        canvas.zoomToPoint(new fabric.Point(opt.e.offsetX, opt.e.offsetY), zoom);
        setZoomLevel(Math.round(zoom * 100));
        opt.e.preventDefault(); opt.e.stopPropagation();
      });

      // --- SELECTION ---
      canvas.on("selection:created", (e: any) => {
        const obj = e.selected?.[0];
        if (obj?.annotationId) setSelectedAnnotation(obj.annotationId);
      });
      canvas.on("selection:updated", (e: any) => {
        const obj = e.selected?.[0];
        if (obj?.annotationId) setSelectedAnnotation(obj.annotationId);
      });
      canvas.on("selection:cleared", () => setSelectedAnnotation(null));

      // --- OBJECT MODIFIED: sync back to annotations ---
      canvas.on("object:modified", (e: any) => {
        const obj = e.target;
        if (!obj?.annotationId) return;
        const id = obj.annotationId;
        const type = obj.annotationType;

        // Read the effective size BEFORE we reset scale below.
        const effWidth = obj.width * obj.scaleX;
        const effHeight = obj.height * obj.scaleY;

        const newAnns = annotationsRef.current.map((a) => {
          if (a.id !== id) return a;
          if (type === "bbox") {
            return { ...a, x: obj.left, y: obj.top, width: effWidth, height: effHeight };
          }
          if (type === "ellipse") {
            // Fabric Ellipse exposes rx/ry; width/height in state mirror the
            // bounding rect so ellipses round-trip through save/load cleanly.
            const rx = (obj.rx || 0) * obj.scaleX;
            const ry = (obj.ry || 0) * obj.scaleY;
            return { ...a, x: obj.left, y: obj.top, width: rx * 2, height: ry * 2 };
          }
          if (type === "keypoint") {
            // The keypoint is rendered as a circle with radius 5 offset by -5,-5
            // so the stored point is the circle's center.
            return { ...a, x: (obj.left || 0) + 5, y: (obj.top || 0) + 5 };
          }
          if ((type === "polygon" || type === "polyline") && a.points) {
            // Fabric polygon/polyline store points in local coords relative to
            // pathOffset. Apply the transform matrix to recover scene coords.
            // Rotation is locked so the matrix is translate + scale only.
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

        // Reset transform so subsequent edits work from a clean baseline.
        if (type === "bbox") {
          obj.set({ scaleX: 1, scaleY: 1, width: effWidth, height: effHeight });
        } else if (type === "ellipse") {
          obj.set({ scaleX: 1, scaleY: 1, rx: (obj.rx || 0) * obj.scaleX, ry: (obj.ry || 0) * obj.scaleY });
        }

        setAnnotations(newAnns); annotationsRef.current = newAnns;
        scheduleAutoSave(newAnns);
        // Re-render so label positions follow the moved/resized shape and
        // polygon geometry is rebuilt from the updated points.
        setTimeout(() => renderAnnotations(newAnns, canvas), 50);
      });

      // --- RESIZE ---
      const handleResize = () => {
        if (container) {
          canvas.setDimensions({ width: container.clientWidth, height: container.clientHeight });
          canvas.renderAll();
        }
      };
      window.addEventListener("resize", handleResize);

      // Prevent middle-click auto-scroll
      const canvasEl = container.querySelector("canvas");
      const preventMiddle = (e: MouseEvent) => { if (e.button === 1) e.preventDefault(); };
      canvasEl?.addEventListener("mousedown", preventMiddle);

      return () => {
        window.removeEventListener("resize", handleResize);
        canvasEl?.removeEventListener("mousedown", preventMiddle);
      };
    };

    init();

    return () => {
      const c = fabricRef.current;
      if (c) { c.dispose(); fabricRef.current = null; }
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Update canvas cursor when tool changes
  useEffect(() => {
    const c = fabricRef.current;
    if (!c) return;
    const cursorMap: Record<Tool, string> = {
      select: "default", bbox: "crosshair", polygon: "crosshair", polyline: "crosshair",
      ellipse: "crosshair", keypoint: "crosshair", pan: "grab",
    };
    c.defaultCursor = cursorMap[activeTool] || "default";
    c.selection = activeTool === "select";

    // Update selectability of objects
    c.getObjects().forEach((obj: any) => {
      if (!obj.isBackgroundImage && !obj.isLabel && obj.annotationId) {
        obj.set("selectable", activeTool === "select");
        obj.set("evented", activeTool === "select");
      }
    });
    c.discardActiveObject();
    c.renderAll();
  }, [activeTool]);

  const goToImage = useCallback((direction: "next" | "prev") => {
    if (images.length === 0) return;
    const idx = images.findIndex((i) => i.filename === currentImage);
    const newIdx = direction === "next" ? (idx < images.length - 1 ? idx + 1 : 0) : (idx > 0 ? idx - 1 : images.length - 1);
    loadImage(images[newIdx].filename);
  }, [images, currentImage, loadImage]);

  // Auto-advance effect: triggered after status changes
  useEffect(() => {
    if (autoAdvancePending.current) {
      autoAdvancePending.current = false;
      goToImage("next");
    }
  }, [imageStatus, goToImage]);

  const addLabel = useCallback(() => {
    if (!newLabelName.trim() || labels.find((l) => l.name === newLabelName.trim())) return;
    const newLabels = [...labels, { name: newLabelName.trim(), color: newLabelColor }];
    setLabels(newLabels); labelsRef.current = newLabels;
    setActiveLabel(newLabelName.trim()); activeLabelRef.current = newLabelName.trim();
    setNewLabelName(""); setNewLabelColor(LABEL_COLORS[newLabels.length % LABEL_COLORS.length]);
  }, [newLabelName, newLabelColor, labels]);

  const removeLabel = useCallback((name: string) => {
    const newLabels = labels.filter((l) => l.name !== name);
    setLabels(newLabels); labelsRef.current = newLabels;
    if (activeLabel === name && newLabels.length > 0) setActiveLabel(newLabels[0].name);
  }, [labels, activeLabel]);

  const cancelPolygon = useCallback(() => {
    const c = fabricRef.current; if (!c) return;
    for (const l of polygonLines.current) c.remove(l);
    for (const d of polygonDots.current) c.remove(d);
    polygonPoints.current = []; polygonLines.current = []; polygonDots.current = [];
    c.renderAll();
  }, []);

  // Drag & drop upload
  const handleDrop = useCallback(async (e: React.DragEvent) => {
    e.preventDefault(); setIsDragOver(false);
    const files = Array.from(e.dataTransfer.files).filter((f) => f.type.startsWith("image/"));
    if (files.length === 0) return;
    const formData = new FormData();
    files.forEach((f) => formData.append("files", f));
    await fetch("/api/upload", { method: "POST", body: formData });
    fetchImages(); fetchStats();
  }, [fetchImages, fetchStats]);

  const doExportCOCO = useCallback(async () => {
    const res = await fetch("/api/export/coco");
    const data = await res.json();
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a"); a.href = url; a.download = "coco_export.json"; a.click();
    URL.revokeObjectURL(url);
    setExportResult(`Exported ${data.images?.length || 0} images, ${data.annotations?.length || 0} annotations in COCO format.`);
  }, []);

  const doExportYOLO = useCallback(async () => {
    const res = await fetch("/api/export/yolo");
    const data = await res.json();
    setExportResult(`Exported ${data.files} files, ${data.classes?.length || 0} classes in YOLO format. Files in /exports folder.`);
  }, []);

  // Keyboard shortcuts - space for pan
  useEffect(() => {
    const down = (e: KeyboardEvent) => {
      if (e.key === " " && !spaceHeld.current && !(e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement)) {
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

  // Main keyboard shortcuts
  useEffect(() => {
    const handleKey = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;
      if (e.key === " ") return; // handled separately for pan

      if (e.ctrlKey || e.metaKey) {
        if (e.key === "z" && !e.shiftKey) { e.preventDefault(); undo(); }
        if (e.key === "z" && e.shiftKey) { e.preventDefault(); redo(); }
        if (e.key === "Z") { e.preventDefault(); redo(); }
        if (e.key === "s") { e.preventDefault(); saveNow(); }
        if (e.key === "c" || e.key === "C") { e.preventDefault(); copySelected(); }
        if (e.key === "v" || e.key === "V") { e.preventDefault(); pasteClipboard(); }
        if (e.key === "d" || e.key === "D") { e.preventDefault(); duplicateSelected(); }
        if (e.key === "a") { e.preventDefault(); /* prevent select all */ }
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
        case "Delete": case "Backspace": deleteSelected(); break;
        case "n": e.preventDefault(); goToImage("next"); break;
        case "N": e.preventDefault(); goToImage("prev"); break;
        case "ArrowRight": e.preventDefault(); goToImage("next"); break;
        case "ArrowLeft": e.preventDefault(); goToImage("prev"); break;
        case "+": case "=": zoomIn(); break;
        case "-": zoomOut(); break;
        case "f": case "F": fitToView(); break;
        case "h": case "H": toggleAnnotations(); break;
        case "Escape": cancelPolygon(); setActiveTool("select"); break;
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

  const statusColor = (s: ImageStatus) => {
    const map: Record<ImageStatus, string> = { unannotated: "bg-zinc-500", annotated: "bg-indigo-500", accepted: "bg-emerald-400", rejected: "bg-red-400" };
    return map[s];
  };

  const statusBgColor = (s: ImageStatus) => {
    const map: Record<ImageStatus, string> = { accepted: "bg-emerald-500/10 text-emerald-400", rejected: "bg-red-500/10 text-red-400", annotated: "bg-indigo-500/10 text-indigo-400", unannotated: "bg-zinc-700/30 text-zinc-400" };
    return map[s];
  };

  const progress = stats.total > 0 ? ((stats.annotated + stats.accepted + stats.rejected) / stats.total * 100) : 0;
  const currentIdx = images.findIndex((i) => i.filename === currentImage);

  return (
    <div className="h-screen flex flex-col">
      {/* Top Bar */}
      <header className="flex items-center justify-between px-4 h-12 bg-[#161822] border-b border-[#2a2d3e] shrink-0 z-10">
        <div className="flex items-center gap-3">
          <h1 className="text-[15px] font-bold bg-gradient-to-r from-indigo-400 to-purple-400 bg-clip-text text-transparent">Image Annotator</h1>
          <div className="flex gap-2 ml-4">
            <span className="px-2.5 py-0.5 rounded-full text-[11px] font-semibold bg-[#1e2030] text-[#8688a0]">{stats.total} images</span>
            <span className="px-2.5 py-0.5 rounded-full text-[11px] font-semibold bg-zinc-500 text-white">{stats.unannotated} todo</span>
            <span className="px-2.5 py-0.5 rounded-full text-[11px] font-semibold bg-indigo-500 text-white">{stats.annotated} done</span>
            <span className="px-2.5 py-0.5 rounded-full text-[11px] font-semibold bg-emerald-400 text-black">{stats.accepted} accepted</span>
            <span className="px-2.5 py-0.5 rounded-full text-[11px] font-semibold bg-red-400 text-white">{stats.rejected} rejected</span>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <button onClick={() => setShowSettingsModal(true)} className="px-3 py-1.5 text-xs font-semibold border border-[#2a2d3e] rounded text-[#e0e0e6] hover:bg-[#252840] transition">Settings</button>
          <button onClick={() => setShowExportModal(true)} className="px-3 py-1.5 text-xs font-semibold border border-[#2a2d3e] rounded text-[#e0e0e6] hover:bg-[#252840] transition">Export</button>
          <button onClick={() => setShowShortcutsModal(true)} className="px-3 py-1.5 text-xs font-semibold border border-[#2a2d3e] rounded text-[#e0e0e6] hover:bg-[#252840] transition">?</button>
        </div>
      </header>

      <div className="flex flex-1 min-h-0">
        {/* Left Sidebar */}
        <aside className="w-60 min-w-60 bg-[#161822] border-r border-[#2a2d3e] flex flex-col">
          <div className="p-3 border-b border-[#2a2d3e] space-y-2">
            <h3 className="text-xs uppercase tracking-wider text-[#8688a0] font-semibold">Images</h3>
            <input
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search filenames…"
              className="w-full px-2 py-1.5 bg-[#1e2030] border border-[#2a2d3e] text-[#e0e0e6] rounded text-xs focus:border-indigo-500 focus:outline-none transition"
            />
            <select value={filter} onChange={(e) => setFilter(e.target.value as ImageStatus | "all")} className="w-full px-2 py-1.5 bg-[#1e2030] border border-[#2a2d3e] text-[#e0e0e6] rounded text-xs">
              <option value="all">All</option>
              <option value="unannotated">Unannotated</option>
              <option value="annotated">Annotated</option>
              <option value="accepted">Accepted</option>
              <option value="rejected">Rejected</option>
            </select>
          </div>
          <div className="flex-1 overflow-y-auto p-1 sidebar-images-scroll">
            {(() => {
              const q = searchQuery.trim().toLowerCase();
              const shown = q ? images.filter((i) => i.filename.toLowerCase().includes(q)) : images;
              if (shown.length === 0) {
                return images.length === 0 ? (
                  <div className="p-4 text-center text-[#8688a0] text-xs">No images found.<br />Drop images here to upload to R2 <code className="bg-[#1e2030] px-1 rounded">raw/</code></div>
                ) : (
                  <div className="p-4 text-center text-[#8688a0] text-xs">No images match &ldquo;{searchQuery}&rdquo;</div>
                );
              }
              return shown.map((img) => (
                <div key={img.filename} onClick={() => loadImage(img.filename)}
                  className={`flex items-center gap-2 px-2 py-1.5 rounded cursor-pointer text-xs transition-colors ${currentImage === img.filename ? "bg-indigo-500 text-white" : "hover:bg-[#252840]"}`}>
                  <ImageThumb filename={img.filename} />
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-1.5">
                      <div className={`w-2 h-2 rounded-full shrink-0 ${statusColor(img.status)}`} />
                      <span className="truncate" title={img.filename}>{img.filename}</span>
                    </div>
                    <div className="flex items-center gap-1.5 mt-0.5">
                      <span className={`text-[9px] capitalize ${currentImage === img.filename ? "text-white/70" : "text-[#8688a0]"}`}>{img.status}</span>
                      {img.annotationCount > 0 && (
                        <span className={`text-[9px] px-1 rounded-lg font-semibold ${currentImage === img.filename ? "bg-white/20 text-white" : "bg-[#1e2030] text-[#8688a0]"}`}>{img.annotationCount} ann</span>
                      )}
                    </div>
                  </div>
                </div>
              ));
            })()}
          </div>
          <div className="p-3 border-t border-[#2a2d3e]">
            <div className="h-1 bg-[#1e2030] rounded-full overflow-hidden mb-1.5">
              <div className="h-full bg-gradient-to-r from-indigo-500 to-emerald-400 transition-all duration-300" style={{ width: `${progress}%` }} />
            </div>
            <span className="text-[11px] text-[#8688a0]">{stats.annotated + stats.accepted + stats.rejected}/{stats.total} annotated</span>
          </div>
        </aside>

        {/* Center Canvas */}
        <main className="flex-1 flex flex-col min-w-0">
          {/* Toolbar */}
          <div className="flex items-center gap-1 px-3 py-1.5 bg-[#161822] border-b border-[#2a2d3e]">
            <div className="flex gap-0.5 pr-2 border-r border-[#2a2d3e]">
              {([["select", "V", "Select (V)"], ["bbox", "B", "Box (B)"], ["polygon", "P", "Polygon (P)"], ["polyline", "L", "Polyline (L)"], ["ellipse", "E", "Ellipse (E)"], ["keypoint", "K", "Keypoint (K)"], ["pan", "G", "Pan (G/Space)"]] as [Tool, string, string][]).map(([tool, key, title]) => (
                <button key={tool} onClick={() => setActiveTool(tool)}
                  className={`w-8 h-8 flex items-center justify-center rounded text-xs font-bold transition-colors ${activeTool === tool ? "bg-indigo-500 text-white" : "text-[#8688a0] hover:bg-[#252840] hover:text-[#e0e0e6]"}`}
                  title={title}>
                  {key}
                </button>
              ))}
            </div>
            <div className="flex gap-0.5 px-2 border-r border-[#2a2d3e]">
              <button onClick={zoomOut} className="w-8 h-8 flex items-center justify-center rounded text-[#8688a0] hover:bg-[#252840] hover:text-[#e0e0e6] transition text-sm font-bold" title="Zoom Out (-)">-</button>
              <span className="w-12 h-8 flex items-center justify-center text-[11px] text-[#8688a0] font-mono">{zoomLevel}%</span>
              <button onClick={zoomIn} className="w-8 h-8 flex items-center justify-center rounded text-[#8688a0] hover:bg-[#252840] hover:text-[#e0e0e6] transition text-sm font-bold" title="Zoom In (+)">+</button>
              <button onClick={fitToView} className="w-8 h-8 flex items-center justify-center rounded text-[#8688a0] hover:bg-[#252840] hover:text-[#e0e0e6] transition text-[10px] font-bold" title="Fit (F)">FIT</button>
            </div>
            <div className="flex gap-0.5 px-2 border-r border-[#2a2d3e]">
              <button onClick={undo} className="w-8 h-8 flex items-center justify-center rounded text-[#8688a0] hover:bg-[#252840] hover:text-[#e0e0e6] transition" title="Undo (Ctrl+Z)">
                <svg viewBox="0 0 24 24" width="14" height="14"><path d="M12.5 8c-2.65 0-5.05 1.04-6.83 2.73L2.5 7.5v9h9l-3.18-3.18C9.77 12.07 11.08 11.5 12.5 11.5c2.9 0 5.35 1.86 6.24 4.45l2.72-.84C20.2 11.36 16.7 8 12.5 8z" fill="currentColor"/></svg>
              </button>
              <button onClick={redo} className="w-8 h-8 flex items-center justify-center rounded text-[#8688a0] hover:bg-[#252840] hover:text-[#e0e0e6] transition" title="Redo (Ctrl+Shift+Z)">
                <svg viewBox="0 0 24 24" width="14" height="14"><path d="M11.5 8c2.65 0 5.05 1.04 6.83 2.73L21.5 7.5v9h-9l3.18-3.18C14.23 12.07 12.92 11.5 11.5 11.5c-2.9 0-5.35 1.86-6.24 4.45l-2.72-.84C3.8 11.36 7.3 8 11.5 8z" fill="currentColor"/></svg>
              </button>
              <button onClick={deleteSelected} className="w-8 h-8 flex items-center justify-center rounded text-[#8688a0] hover:bg-[#252840] hover:text-red-400 transition" title="Delete (Del)">
                <svg viewBox="0 0 24 24" width="14" height="14"><path d="M6 19c0 1.1.9 2 2 2h8c1.1 0 2-.9 2-2V7H6v12zM19 4h-3.5l-1-1h-5l-1 1H5v2h14V4z" fill="currentColor"/></svg>
              </button>
            </div>
            <button onClick={toggleAnnotations}
              className={`w-8 h-8 flex items-center justify-center rounded transition-colors ${annotationsVisible ? "text-[#8688a0] hover:bg-[#252840]" : "bg-red-400/20 text-red-400"}`} title="Toggle (H)">
              <svg viewBox="0 0 24 24" width="14" height="14"><path d="M12 4.5C7 4.5 2.73 7.61 1 12c1.73 4.39 6 7.5 11 7.5s9.27-3.11 11-7.5c-1.73-4.39-6-7.5-11-7.5zM12 17c-2.76 0-5-2.24-5-5s2.24-5 5-5 5 2.24 5 5-2.24 5-5 5zm0-8c-1.66 0-3 1.34-3 3s1.34 3 3 3 3-1.34 3-3-1.34-3-3-3z" fill="currentColor"/></svg>
            </button>
            {/* Image nav */}
            <div className="flex items-center gap-1 ml-auto">
              {currentImage && <span className="text-[11px] text-[#8688a0] mr-2">{currentIdx + 1} / {images.length}</span>}
              <button onClick={() => goToImage("prev")} className="px-2 h-8 flex items-center rounded text-xs text-[#8688a0] hover:bg-[#252840] hover:text-[#e0e0e6] transition" title="Previous (Arrow Left)">
                <svg viewBox="0 0 24 24" width="14" height="14"><path d="M15.41 7.41L14 6l-6 6 6 6 1.41-1.41L10.83 12z" fill="currentColor"/></svg>
              </button>
              <button onClick={() => goToImage("next")} className="px-2 h-8 flex items-center rounded text-xs text-[#8688a0] hover:bg-[#252840] hover:text-[#e0e0e6] transition" title="Next (Arrow Right)">
                <svg viewBox="0 0 24 24" width="14" height="14"><path d="M10 6L8.59 7.41 13.17 12l-4.58 4.59L10 18l6-6z" fill="currentColor"/></svg>
              </button>
            </div>
          </div>

          {/* Canvas */}
          <div ref={containerRef}
            className={`flex-1 relative overflow-hidden ${isDragOver ? "ring-2 ring-inset ring-indigo-500" : ""}`}
            style={{
              background: "#0a0b0f",
              backgroundImage: "linear-gradient(45deg, #111320 25%, transparent 25%), linear-gradient(-45deg, #111320 25%, transparent 25%), linear-gradient(45deg, transparent 75%, #111320 75%), linear-gradient(-45deg, transparent 75%, #111320 75%)",
              backgroundSize: "20px 20px", backgroundPosition: "0 0, 0 10px, 10px -10px, -10px 0",
            }}
            onDragOver={(e) => { e.preventDefault(); setIsDragOver(true); }}
            onDragLeave={() => setIsDragOver(false)}
            onDrop={handleDrop}
          >
            <canvas ref={canvasRef} />
            {!currentImage && (
              <div className="absolute inset-0 flex flex-col items-center justify-center text-[#8688a0] gap-3 pointer-events-none">
                <div className="w-16 h-16 rounded-2xl bg-[#1e2030] flex items-center justify-center">
                  <svg viewBox="0 0 24 24" width="32" height="32" className="text-indigo-400"><path d="M21 19V5c0-1.1-.9-2-2-2H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2zM8.5 13.5l2.5 3.01L14.5 12l4.5 6H5l3.5-4.5z" fill="currentColor"/></svg>
                </div>
                <h2 className="text-lg text-[#e0e0e6] font-semibold">No image loaded</h2>
                <p className="text-sm">Drop images here to upload to R2 <code className="bg-[#1e2030] px-2 py-0.5 rounded font-mono text-xs">raw/</code></p>
              </div>
            )}
            {isDragOver && (
              <div className="absolute inset-0 bg-indigo-500/10 flex items-center justify-center z-50 pointer-events-none">
                <div className="px-6 py-4 bg-[#161822] rounded-lg border-2 border-dashed border-indigo-500 text-indigo-400 font-semibold">
                  Drop images to upload
                </div>
              </div>
            )}
          </div>

          {/* Bottom bar */}
          <div className="flex items-center gap-4 px-3 py-1 bg-[#161822] border-t border-[#2a2d3e] text-[11px] text-[#8688a0] font-[family-name:var(--font-mono)]">
            <span>x: {cursorPos.x}, y: {cursorPos.y}</span>
            <span>{zoomLevel}%</span>
            {currentImage && <>
              <span className="truncate max-w-40">{currentImage}</span>
              <span>{imageDims.current.width}x{imageDims.current.height}</span>
              <span>{annotations.length} annotations</span>
            </>}
            <span className="ml-auto text-emerald-400">{saveIndicator}</span>
          </div>
        </main>

        {/* Right Sidebar */}
        <aside className="w-72 min-w-72 bg-[#161822] border-l border-[#2a2d3e] overflow-y-auto flex flex-col">
          {/* Labels */}
          <div className="border-b border-[#2a2d3e]">
            <div className="flex items-center justify-between px-3 py-2.5">
              <h3 className="text-xs uppercase tracking-wider text-[#8688a0] font-semibold">Labels</h3>
              <span className="text-[10px] text-[#8688a0]">Press 1-9</span>
            </div>
            <div className="px-3 pb-2 flex flex-wrap gap-1">
              {labels.map((l, idx) => {
                const count = annotations.filter((a) => a.label === l.name).length;
                return (
                  <button key={l.name}
                    onClick={() => { setActiveLabel(l.name); activeLabelRef.current = l.name; }}
                    onDoubleClick={() => setLabelFilter(labelFilter === l.name ? null : l.name)}
                    title={`Click to select, double-click to filter annotations by "${l.name}"`}
                    className={`group flex items-center gap-1 px-2.5 py-1 rounded-full text-[11px] font-semibold text-white transition-all ${activeLabel === l.name ? "ring-2 ring-white shadow-lg scale-105" : "hover:scale-105"} ${labelFilter === l.name ? "outline outline-2 outline-white/70" : ""}`}
                    style={{ backgroundColor: l.color }}>
                    <span className="text-[9px] px-1 rounded bg-black/30 font-mono">{idx + 1}</span>
                    {l.name}
                    {count > 0 && <span className="text-[9px] px-1 rounded bg-black/40 font-mono">{count}</span>}
                    <span onClick={(e) => { e.stopPropagation(); removeLabel(l.name); }}
                      className="ml-0.5 w-3.5 h-3.5 rounded-full flex items-center justify-center text-[9px] bg-black/30 hover:bg-red-500 cursor-pointer opacity-0 group-hover:opacity-100 transition-opacity">x</span>
                  </button>
                );
              })}
            </div>
            <div className="flex gap-1.5 px-3 pb-3">
              <input value={newLabelName} onChange={(e) => setNewLabelName(e.target.value)} onKeyDown={(e) => e.key === "Enter" && addLabel()}
                placeholder="New label..." className="flex-1 px-2 py-1.5 bg-[#1e2030] border border-[#2a2d3e] text-[#e0e0e6] rounded text-xs focus:border-indigo-500 focus:outline-none transition" />
              <input type="color" value={newLabelColor} onChange={(e) => setNewLabelColor(e.target.value)} className="w-8 h-8 border-none rounded cursor-pointer bg-transparent" />
              <button onClick={addLabel} className="px-2.5 py-1.5 bg-indigo-500 text-white rounded text-xs font-semibold hover:bg-indigo-400 transition">Add</button>
            </div>
          </div>

          {/* Image Adjustments */}
          {currentImage && (
            <div className="border-b border-[#2a2d3e] px-3 py-2.5">
              <h3 className="text-xs uppercase tracking-wider text-[#8688a0] font-semibold mb-2">Adjustments</h3>
              <div className="space-y-2">
                <div className="flex items-center gap-2 text-xs">
                  <span className="w-16 text-[#8688a0]">Bright</span>
                  <input type="range" min="20" max="200" value={brightness} onChange={(e) => setBrightness(+e.target.value)}
                    className="flex-1 h-1 accent-indigo-500" />
                  <span className="w-8 text-right text-[#8688a0] font-mono text-[10px]">{brightness}%</span>
                </div>
                <div className="flex items-center gap-2 text-xs">
                  <span className="w-16 text-[#8688a0]">Contrast</span>
                  <input type="range" min="20" max="200" value={contrast} onChange={(e) => setContrast(+e.target.value)}
                    className="flex-1 h-1 accent-indigo-500" />
                  <span className="w-8 text-right text-[#8688a0] font-mono text-[10px]">{contrast}%</span>
                </div>
                <div className="flex items-center gap-2 text-xs">
                  <span className="w-16 text-[#8688a0]">Fill</span>
                  <input type="range" min="0" max="100" value={opacity} onChange={(e) => { setOpacity(+e.target.value); opacityRef.current = +e.target.value; renderAnnotations(annotations); }}
                    className="flex-1 h-1 accent-indigo-500" />
                  <span className="w-8 text-right text-[#8688a0] font-mono text-[10px]">{opacity}%</span>
                </div>
                <button onClick={() => { setBrightness(100); setContrast(100); setOpacity(70); opacityRef.current = 70; renderAnnotations(annotations); }}
                  className="text-[10px] text-indigo-400 hover:text-indigo-300 transition">Reset adjustments</button>
              </div>
            </div>
          )}

          {/* Properties */}
          <div className="border-b border-[#2a2d3e]">
            <div className="px-3 py-2.5"><h3 className="text-xs uppercase tracking-wider text-[#8688a0] font-semibold">Properties</h3></div>
            <div className="px-3 pb-3">
              {selectedAnnotation ? (() => {
                const ann = annotations.find((a) => a.id === selectedAnnotation);
                if (!ann) return <p className="text-xs text-[#8688a0]">Not found</p>;
                const attrEntries = Object.entries(ann.attributes || {});
                return (
                  <div className="space-y-1.5 text-xs">
                    <div className="flex justify-between"><span className="text-[#8688a0]">Type</span><span className="font-semibold capitalize px-2 py-0.5 rounded text-[10px]" style={{ backgroundColor: ann.color + "33", color: ann.color }}>{ann.type}</span></div>
                    <div className="flex justify-between items-center"><span className="text-[#8688a0]">Label</span>
                      <select value={ann.label} onChange={(e) => {
                        pushUndo();
                        const newAnns = annotations.map((a) => a.id === ann.id ? { ...a, label: e.target.value, color: getLabelColor(e.target.value) } : a);
                        setAnnotations(newAnns); annotationsRef.current = newAnns;
                        renderAnnotations(newAnns); scheduleAutoSave(newAnns);
                      }} className="px-2 py-1 bg-[#1e2030] border border-[#2a2d3e] text-[#e0e0e6] rounded text-xs">
                        {labels.map((l) => <option key={l.name} value={l.name}>{l.name}</option>)}
                      </select>
                    </div>
                    {(ann.type === "bbox" || ann.type === "ellipse") && <>
                      <div className="flex justify-between"><span className="text-[#8688a0]">Position</span><span className="font-mono text-[10px]">{Math.round(ann.x || 0)}, {Math.round(ann.y || 0)}</span></div>
                      <div className="flex justify-between"><span className="text-[#8688a0]">Size</span><span className="font-mono text-[10px]">{Math.round(ann.width || 0)} x {Math.round(ann.height || 0)}</span></div>
                    </>}
                    {(ann.type === "polygon" || ann.type === "polyline") && <div className="flex justify-between"><span className="text-[#8688a0]">Vertices</span><span>{ann.points?.length || 0}</span></div>}
                    {ann.type === "keypoint" && <div className="flex justify-between"><span className="text-[#8688a0]">Position</span><span className="font-mono text-[10px]">{Math.round(ann.x || 0)}, {Math.round(ann.y || 0)}</span></div>}

                    {/* Attributes editor: arbitrary k/v pairs, persisted per-annotation */}
                    <div className="pt-2 border-t border-[#2a2d3e]">
                      <div className="flex items-center justify-between mb-1">
                        <span className="text-[10px] uppercase tracking-wider text-[#8688a0] font-semibold">Attributes</span>
                        <span className="text-[10px] text-[#8688a0]">{attrEntries.length}</span>
                      </div>
                      <div className="space-y-1">
                        {attrEntries.map(([k, v]) => (
                          <div key={k} className="flex items-center gap-1">
                            <span className="text-[10px] text-[#8688a0] font-mono w-16 truncate" title={k}>{k}</span>
                            <input value={v} onChange={(ev) => setAnnotationAttribute(ann.id, k, ev.target.value)}
                              className="flex-1 px-1.5 py-0.5 bg-[#1e2030] border border-[#2a2d3e] rounded text-[10px] text-[#e0e0e6] focus:border-indigo-500 focus:outline-none" />
                            <button onClick={() => setAnnotationAttribute(ann.id, k, null)}
                              className="text-red-400 hover:text-red-300 text-sm leading-none w-4" title="Remove">&times;</button>
                          </div>
                        ))}
                      </div>
                      <div className="flex gap-1 mt-1.5">
                        <input value={attrDraftKey} onChange={(ev) => setAttrDraftKey(ev.target.value)} placeholder="key"
                          className="flex-1 w-0 px-1.5 py-1 bg-[#1e2030] border border-[#2a2d3e] rounded text-[10px] text-[#e0e0e6] font-mono focus:border-indigo-500 focus:outline-none" />
                        <input value={attrDraftVal} onChange={(ev) => setAttrDraftVal(ev.target.value)} placeholder="value"
                          onKeyDown={(ev) => {
                            if (ev.key === "Enter" && attrDraftKey.trim()) {
                              setAnnotationAttribute(ann.id, attrDraftKey.trim(), attrDraftVal);
                              setAttrDraftKey(""); setAttrDraftVal("");
                            }
                          }}
                          className="flex-1 w-0 px-1.5 py-1 bg-[#1e2030] border border-[#2a2d3e] rounded text-[10px] text-[#e0e0e6] focus:border-indigo-500 focus:outline-none" />
                        <button onClick={() => {
                          if (!attrDraftKey.trim()) return;
                          setAnnotationAttribute(ann.id, attrDraftKey.trim(), attrDraftVal);
                          setAttrDraftKey(""); setAttrDraftVal("");
                        }} className="px-2 py-1 bg-indigo-500 text-white rounded text-[10px] font-semibold hover:bg-indigo-400 transition">Add</button>
                      </div>
                    </div>

                    {/* Copy / duplicate shortcuts for the selected annotation */}
                    <div className="flex gap-1.5 pt-2">
                      <button onClick={copySelected} className="flex-1 py-1 border border-[#2a2d3e] rounded text-[10px] font-semibold text-[#8688a0] hover:bg-[#252840] hover:text-[#e0e0e6] transition" title="Copy (Ctrl+C)">Copy</button>
                      <button onClick={duplicateSelected} className="flex-1 py-1 border border-[#2a2d3e] rounded text-[10px] font-semibold text-[#8688a0] hover:bg-[#252840] hover:text-[#e0e0e6] transition" title="Duplicate (Ctrl+D)">Duplicate</button>
                    </div>
                  </div>
                );
              })() : <p className="text-xs text-[#8688a0]">Select an annotation to edit</p>}
            </div>
          </div>

          {/* Annotations list */}
          <div className="border-b border-[#2a2d3e]">
            <div className="flex items-center justify-between px-3 py-2.5">
              <h3 className="text-xs uppercase tracking-wider text-[#8688a0] font-semibold">Annotations</h3>
              <div className="flex items-center gap-1.5">
                {labelFilter && (
                  <button onClick={() => setLabelFilter(null)}
                    className="text-[10px] px-1.5 py-0.5 rounded bg-indigo-500/20 text-indigo-300 hover:bg-indigo-500/30 transition"
                    title="Clear label filter">
                    {labelFilter} &times;
                  </button>
                )}
                <span className="text-[10px] px-2 py-0.5 rounded-lg bg-indigo-500 text-white font-semibold">{annotations.length}</span>
              </div>
            </div>
            <div className="max-h-52 overflow-y-auto">
              {annotations
                .filter((ann) => !labelFilter || ann.label === labelFilter)
                .map((ann) => (
                <div key={ann.id} onClick={() => setSelectedAnnotation(ann.id)}
                  className={`flex items-center gap-1.5 px-3 py-1.5 cursor-pointer text-xs transition-colors group ${selectedAnnotation === ann.id ? "bg-indigo-500 text-white" : "hover:bg-[#252840]"} ${ann.hidden ? "opacity-40" : ""}`}>
                  <span className="w-4 h-4 rounded text-[9px] font-bold flex items-center justify-center text-white shrink-0" style={{ backgroundColor: ann.color }}>
                    {ann.type[0].toUpperCase()}
                  </span>
                  <span className="flex-1 truncate">{ann.label}</span>
                  {ann.attributes && Object.keys(ann.attributes).length > 0 && (
                    <span className="text-[9px] px-1 rounded bg-[#1e2030] text-[#8688a0]" title="Has attributes">{Object.keys(ann.attributes).length}</span>
                  )}
                  <button onClick={(e) => { e.stopPropagation(); toggleAnnotationFlag(ann.id, "hidden"); }}
                    className={`w-4 h-4 flex items-center justify-center rounded hover:bg-white/10 transition ${ann.hidden ? "text-red-400 opacity-100" : "text-[#8688a0] opacity-0 group-hover:opacity-100"}`}
                    title={ann.hidden ? "Show" : "Hide"}>
                    <svg viewBox="0 0 24 24" width="11" height="11"><path d={ann.hidden ? "M12 7c2.76 0 5 2.24 5 5 0 .65-.13 1.26-.36 1.83l2.92 2.92c1.51-1.26 2.7-2.89 3.43-4.75-1.73-4.39-6-7.5-11-7.5-1.4 0-2.74.25-3.98.7l2.16 2.16C10.74 7.13 11.35 7 12 7zM2 4.27l2.28 2.28.46.46C3.08 8.3 1.78 10.02 1 12c1.73 4.39 6 7.5 11 7.5 1.55 0 3.03-.3 4.38-.84l.42.42L19.73 22 21 20.73 3.27 3 2 4.27zM7.53 9.8l1.55 1.55c-.05.21-.08.43-.08.65 0 1.66 1.34 3 3 3 .22 0 .44-.03.65-.08l1.55 1.55c-.67.33-1.41.53-2.2.53-2.76 0-5-2.24-5-5 0-.79.2-1.53.53-2.2z" : "M12 4.5C7 4.5 2.73 7.61 1 12c1.73 4.39 6 7.5 11 7.5s9.27-3.11 11-7.5c-1.73-4.39-6-7.5-11-7.5zM12 17c-2.76 0-5-2.24-5-5s2.24-5 5-5 5 2.24 5 5-2.24 5-5 5zm0-8c-1.66 0-3 1.34-3 3s1.34 3 3 3 3-1.34 3-3-1.34-3-3-3z"} fill="currentColor"/></svg>
                  </button>
                  <button onClick={(e) => { e.stopPropagation(); toggleAnnotationFlag(ann.id, "locked"); }}
                    className={`w-4 h-4 flex items-center justify-center rounded hover:bg-white/10 transition ${ann.locked ? "text-amber-400 opacity-100" : "text-[#8688a0] opacity-0 group-hover:opacity-100"}`}
                    title={ann.locked ? "Unlock" : "Lock"}>
                    <svg viewBox="0 0 24 24" width="11" height="11"><path d={ann.locked ? "M18 8h-1V6c0-2.76-2.24-5-5-5S7 3.24 7 6v2H6c-1.1 0-2 .9-2 2v10c0 1.1.9 2 2 2h12c1.1 0 2-.9 2-2V10c0-1.1-.9-2-2-2zM9 6c0-1.66 1.34-3 3-3s3 1.34 3 3v2H9V6z" : "M18 8h-1V6c0-2.76-2.24-5-5-5S7 3.24 7 6h2c0-1.66 1.34-3 3-3s3 1.34 3 3v2H6c-1.1 0-2 .9-2 2v10c0 1.1.9 2 2 2h12c1.1 0 2-.9 2-2V10c0-1.1-.9-2-2-2z"} fill="currentColor"/></svg>
                  </button>
                  <button onClick={(e) => { e.stopPropagation(); deleteAnnotation(ann.id); }}
                    className="w-4 h-4 flex items-center justify-center rounded opacity-0 group-hover:opacity-100 text-red-400 hover:text-red-300 transition text-sm leading-none"
                    title="Delete">&times;</button>
                </div>
              ))}
              {annotations.length === 0 && <div className="px-3 py-4 text-center text-[#8688a0] text-xs">No annotations yet.<br />Select a tool and draw on the image.</div>}
              {annotations.length > 0 && labelFilter && annotations.filter((a) => a.label === labelFilter).length === 0 && (
                <div className="px-3 py-4 text-center text-[#8688a0] text-xs">No annotations with label <span className="font-mono">{labelFilter}</span>.</div>
              )}
            </div>
          </div>

          {/* Review */}
          <div className="flex-1">
            <div className="px-3 py-2.5"><h3 className="text-xs uppercase tracking-wider text-[#8688a0] font-semibold">Review</h3></div>
            {currentImage ? <>
              <div className={`mx-3 mb-2 px-3 py-2 rounded text-xs font-semibold text-center capitalize ${statusBgColor(imageStatus)}`}>{imageStatus}</div>
              <textarea value={reviewComment} onChange={(e) => setReviewComment(e.target.value)} placeholder="Add review comments..." rows={2}
                className="block w-[calc(100%-24px)] mx-3 mb-2 p-2 bg-[#1e2030] border border-[#2a2d3e] text-[#e0e0e6] rounded text-xs resize-y focus:border-indigo-500 focus:outline-none transition" />
              <div className="flex gap-1.5 px-3 pb-2">
                <button onClick={() => setStatus("accepted")} className="flex-1 py-1.5 bg-emerald-500 text-white rounded text-xs font-semibold hover:bg-emerald-400 transition flex items-center justify-center gap-1">
                  <svg viewBox="0 0 24 24" width="12" height="12"><path d="M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41z" fill="currentColor"/></svg>
                  Accept <kbd className="text-[9px] opacity-60 ml-1">Q</kbd>
                </button>
                <button onClick={() => setStatus("rejected")} className="flex-1 py-1.5 bg-red-500 text-white rounded text-xs font-semibold hover:bg-red-400 transition flex items-center justify-center gap-1">
                  <svg viewBox="0 0 24 24" width="12" height="12"><path d="M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12z" fill="currentColor"/></svg>
                  Reject <kbd className="text-[9px] opacity-60 ml-1">W</kbd>
                </button>
              </div>
              <div className="flex gap-1.5 px-3 pb-2">
                <button onClick={() => setStatus("annotated")} className="flex-1 py-1.5 border border-[#2a2d3e] text-[#e0e0e6] rounded text-xs font-semibold hover:bg-[#252840] transition">Reset Status</button>
              </div>
              {/* History */}
              {reviewHistory.length > 0 && (
                <div className="px-3 pb-3">
                  <h4 className="text-[10px] uppercase tracking-wider text-[#8688a0] mb-1 font-semibold">History</h4>
                  <div className="max-h-32 overflow-y-auto space-y-1">
                    {[...reviewHistory].reverse().map((h, i) => (
                      <div key={i} className="p-1.5 rounded bg-[#1e2030] text-[11px]">
                        <span className={`font-semibold capitalize ${h.action === "accepted" ? "text-emerald-400" : h.action === "rejected" ? "text-red-400" : "text-indigo-400"}`}>{h.action}</span>
                        {h.comment && <span className="text-[#8688a0]"> &mdash; {h.comment}</span>}
                        <div className="text-[9px] text-[#8688a0] font-mono mt-0.5">{new Date(h.timestamp).toLocaleString()}</div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </> : <div className="px-3 pb-3 text-xs text-[#8688a0]">Select an image to review</div>}
          </div>
        </aside>
      </div>

      {/* Export Modal */}
      {showExportModal && (
        <div className="fixed inset-0 bg-black/60 flex items-center justify-center z-50 backdrop-blur-sm" onClick={() => { setShowExportModal(false); setExportResult(null); }}>
          <div className="bg-[#161822] border border-[#2a2d3e] rounded-lg w-[480px]" onClick={(e) => e.stopPropagation()}>
            <div className="flex items-center justify-between px-5 py-4 border-b border-[#2a2d3e]">
              <h2 className="text-base font-semibold">Export Annotations</h2>
              <button onClick={() => { setShowExportModal(false); setExportResult(null); }} className="text-[#8688a0] hover:text-[#e0e0e6] text-xl leading-none">&times;</button>
            </div>
            <div className="p-5 space-y-3">
              <p className="text-xs text-[#8688a0] mb-2">Rejected images are excluded from all exports.</p>
              <button onClick={doExportCOCO} className="w-full p-4 border border-[#2a2d3e] rounded-lg text-left hover:border-indigo-500 hover:bg-[#252840] transition group">
                <h3 className="text-sm font-semibold group-hover:text-indigo-400 transition">COCO JSON</h3>
                <p className="text-xs text-[#8688a0] mt-1">Standard format. Boxes, polygons, keypoints. Works with Detectron2, MMDetection, YOLOv5+.</p>
              </button>
              <button onClick={doExportYOLO} className="w-full p-4 border border-[#2a2d3e] rounded-lg text-left hover:border-indigo-500 hover:bg-[#252840] transition group">
                <h3 className="text-sm font-semibold group-hover:text-indigo-400 transition">YOLO TXT</h3>
                <p className="text-xs text-[#8688a0] mt-1">Normalized bounding boxes per image. Compatible with Ultralytics YOLOv5/v8.</p>
              </button>
              {exportResult && <div className="p-3 bg-emerald-500/10 rounded text-xs text-emerald-400 font-mono border border-emerald-500/20">{exportResult}</div>}
            </div>
          </div>
        </div>
      )}

      {/* Settings Modal */}
      {showSettingsModal && (
        <div className="fixed inset-0 bg-black/60 flex items-center justify-center z-50 backdrop-blur-sm" onClick={() => setShowSettingsModal(false)}>
          <div className="bg-[#161822] border border-[#2a2d3e] rounded-lg w-[400px]" onClick={(e) => e.stopPropagation()}>
            <div className="flex items-center justify-between px-5 py-4 border-b border-[#2a2d3e]">
              <h2 className="text-base font-semibold">Settings</h2>
              <button onClick={() => setShowSettingsModal(false)} className="text-[#8688a0] hover:text-[#e0e0e6] text-xl leading-none">&times;</button>
            </div>
            <div className="p-5 space-y-4">
              <label className="flex items-center justify-between text-xs">
                <span>Auto-advance after accept/reject</span>
                <input type="checkbox" checked={autoAdvance} onChange={(e) => setAutoAdvance(e.target.checked)} className="accent-indigo-500" />
              </label>
              <label className="flex items-center justify-between text-xs">
                <span>Default annotation fill opacity</span>
                <input type="range" min="0" max="100" value={opacity} onChange={(e) => { setOpacity(+e.target.value); opacityRef.current = +e.target.value; }}
                  className="w-32 h-1 accent-indigo-500" />
              </label>
            </div>
          </div>
        </div>
      )}

      {/* Shortcuts Modal */}
      {showShortcutsModal && (
        <div className="fixed inset-0 bg-black/60 flex items-center justify-center z-50 backdrop-blur-sm" onClick={() => setShowShortcutsModal(false)}>
          <div className="bg-[#161822] border border-[#2a2d3e] rounded-lg w-[520px]" onClick={(e) => e.stopPropagation()}>
            <div className="flex items-center justify-between px-5 py-4 border-b border-[#2a2d3e]">
              <h2 className="text-base font-semibold">Keyboard Shortcuts</h2>
              <button onClick={() => setShowShortcutsModal(false)} className="text-[#8688a0] hover:text-[#e0e0e6] text-xl leading-none">&times;</button>
            </div>
            <div className="p-5">
              <div className="grid grid-cols-2 gap-x-8 gap-y-1">
                <h4 className="col-span-2 text-[10px] uppercase tracking-wider text-[#8688a0] font-semibold mb-1">Tools</h4>
                {[["V","Select"],["B","Bounding Box"],["P","Polygon"],["L","Polyline"],["E","Ellipse"],["K","Keypoint"],["G","Pan"],["Space","Hold to pan"]].map(([k,d])=>(
                  <div key={k} className="flex items-center gap-2 text-xs py-0.5">
                    <kbd className="inline-block px-2 py-0.5 bg-[#1e2030] border border-[#2a2d3e] rounded font-mono text-[10px] min-w-[32px] text-center">{k}</kbd><span>{d}</span>
                  </div>
                ))}
                <h4 className="col-span-2 text-[10px] uppercase tracking-wider text-[#8688a0] font-semibold mt-3 mb-1">Actions</h4>
                {[["1-9","Quick label"],["Del","Delete"],["Ctrl+Z","Undo"],["Ctrl+Shift+Z","Redo"],["Ctrl+S","Save"],["Ctrl+C","Copy annotation"],["Ctrl+V","Paste annotation"],["Ctrl+D","Duplicate"],["Esc","Cancel / Select"]].map(([k,d])=>(
                  <div key={k} className="flex items-center gap-2 text-xs py-0.5">
                    <kbd className="inline-block px-2 py-0.5 bg-[#1e2030] border border-[#2a2d3e] rounded font-mono text-[10px] min-w-[32px] text-center">{k}</kbd><span>{d}</span>
                  </div>
                ))}
                <h4 className="col-span-2 text-[10px] uppercase tracking-wider text-[#8688a0] font-semibold mt-3 mb-1">Navigation</h4>
                {[["Arrow Keys","Prev / Next image"],["+/-","Zoom in/out"],["F","Fit to view"],["H","Toggle annotations"],["Scroll","Zoom at cursor"]].map(([k,d])=>(
                  <div key={k} className="flex items-center gap-2 text-xs py-0.5">
                    <kbd className="inline-block px-2 py-0.5 bg-[#1e2030] border border-[#2a2d3e] rounded font-mono text-[10px] min-w-[32px] text-center">{k}</kbd><span>{d}</span>
                  </div>
                ))}
                <h4 className="col-span-2 text-[10px] uppercase tracking-wider text-[#8688a0] font-semibold mt-3 mb-1">Review</h4>
                {[["Q","Accept"],["W","Reject"]].map(([k,d])=>(
                  <div key={k} className="flex items-center gap-2 text-xs py-0.5">
                    <kbd className="inline-block px-2 py-0.5 bg-[#1e2030] border border-[#2a2d3e] rounded font-mono text-[10px] min-w-[32px] text-center">{k}</kbd><span>{d}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
