"use client";

import { cn } from "@/lib/utils";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { ColorPicker } from "@/components/color-picker";
import type { Annotation, LabelDef } from "@/lib/types";

export function LabelsPanel({
  labels,
  annotations,
  activeLabel,
  labelFilter,
  newLabelName,
  newLabelColor,
  swatches,
  onSelectLabel,
  onToggleFilter,
  onRequestRemove,
  onNameChange,
  onColorChange,
  onAdd,
}: {
  labels: LabelDef[];
  annotations: Annotation[];
  activeLabel: string;
  labelFilter: string | null;
  newLabelName: string;
  newLabelColor: string;
  swatches: string[];
  onSelectLabel: (name: string) => void;
  onToggleFilter: (name: string) => void;
  onRequestRemove: (name: string) => void;
  onNameChange: (value: string) => void;
  onColorChange: (hex: string) => void;
  onAdd: () => void;
}) {
  return (
    <section className="border-b p-3">
      <div className="flex items-center justify-between mb-2">
        <h3 className="text-[10px] uppercase tracking-wider text-muted-foreground font-semibold">Labels</h3>
        <span className="text-[10px] text-muted-foreground">Press 1–9</span>
      </div>
      <div className="flex flex-wrap gap-1.5 mb-3">
        {labels.map((l, idx) => {
          const count = annotations.filter((a) => a.label === l.name).length;
          const isActive = activeLabel === l.name;
          const isFilter = labelFilter === l.name;
          return (
            <button
              key={l.name}
              onClick={() => onSelectLabel(l.name)}
              onDoubleClick={() => onToggleFilter(l.name)}
              title={`Click to select, double-click to filter by "${l.name}"`}
              className={cn(
                "group flex items-center gap-1 px-2 h-6 rounded-full text-[11px] font-semibold text-white transition-all",
                isActive && "ring-2 ring-white shadow-md scale-105",
                isFilter && !isActive && "ring-2 ring-white/60",
              )}
              style={{ backgroundColor: l.color }}
            >
              <span className="text-[9px] px-1 rounded bg-black/30 font-mono">{idx + 1}</span>
              {l.name}
              {count > 0 && <span className="text-[9px] px-1 rounded bg-black/40 font-mono tabular-nums">{count}</span>}
              <span
                onClick={(e) => { e.stopPropagation(); onRequestRemove(l.name); }}
                className="ml-0.5 size-3.5 rounded-full flex items-center justify-center text-[9px] bg-black/30 hover:bg-red-500 cursor-pointer opacity-0 group-hover:opacity-100 transition-opacity"
              >×</span>
            </button>
          );
        })}
      </div>
      <div className="flex gap-1.5">
        <Input
          value={newLabelName}
          onChange={(e) => onNameChange(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && onAdd()}
          placeholder="New label…"
          className="h-8 flex-1 text-xs"
        />
        <ColorPicker
          value={newLabelColor}
          onChange={onColorChange}
          swatches={swatches}
          ariaLabel="Pick label color"
        />
        <Button size="sm" onClick={onAdd}>Add</Button>
      </div>
    </section>
  );
}
