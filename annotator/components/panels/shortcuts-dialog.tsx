"use client";

import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";

function ShortcutRow({ k, d }: { k: string; d: string }) {
  return (
    <div className="flex items-center gap-2 text-xs py-0.5">
      <kbd className="inline-flex items-center justify-center min-w-8 h-6 px-2 bg-muted border rounded font-mono text-[10px]">{k}</kbd>
      <span>{d}</span>
    </div>
  );
}

const TOOLS: [string, string][] = [
  ["V", "Select"], ["B", "Bounding box"], ["P", "Polygon"], ["L", "Polyline"],
  ["E", "Ellipse"], ["K", "Keypoint"], ["G", "Pan"], ["Space", "Hold to pan"],
];

const ACTIONS: [string, string][] = [
  ["1-9", "Quick label"], ["Del", "Delete selected / last vertex"],
  ["Enter", "Finish polygon / polyline"], ["Esc", "Cancel / select"],
  ["⌘Z", "Undo"], ["⌘⇧Z", "Redo"], ["⌘S", "Save"],
  ["⌘C", "Copy"], ["⌘V", "Paste"], ["⌘D", "Duplicate"],
];

const NAV: [string, string][] = [
  ["← / →", "Prev / next image"], ["+ / -", "Zoom in / out"],
  ["F", "Fit to view"], ["H", "Toggle annotations"], ["Scroll", "Zoom at cursor"],
];

const REVIEW: [string, string][] = [["Q", "Accept"], ["W", "Reject"]];

export function ShortcutsDialog({
  open,
  onOpenChange,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}) {
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-[560px] max-h-[85vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle>Keyboard shortcuts</DialogTitle>
        </DialogHeader>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-x-8 gap-y-1">
          <h4 className="col-span-full panel-title mt-1">Tools</h4>
          {TOOLS.map(([k, d]) => <ShortcutRow key={k} k={k} d={d} />)}

          <h4 className="col-span-full panel-title mt-3">Actions</h4>
          {ACTIONS.map(([k, d]) => <ShortcutRow key={k} k={k} d={d} />)}

          <h4 className="col-span-full panel-title mt-3">Navigation</h4>
          {NAV.map(([k, d]) => <ShortcutRow key={k} k={k} d={d} />)}

          <h4 className="col-span-full panel-title mt-3">Review</h4>
          {REVIEW.map(([k, d]) => <ShortcutRow key={k} k={k} d={d} />)}
        </div>
      </DialogContent>
    </Dialog>
  );
}
