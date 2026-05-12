"use client";

import { memo } from "react";
import { cn } from "@/lib/utils";
import { Badge } from "@/components/ui/badge";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { Eye, EyeOff, Lock, Unlock, X, Layers } from "lucide-react";
import type { Annotation } from "@/lib/types";

function AnnotationsListImpl({
  annotations,
  labelFilter,
  selectedAnnotation,
  onSelect,
  onToggleFlag,
  onDelete,
  onClearFilter,
}: {
  annotations: Annotation[];
  labelFilter: string | null;
  selectedAnnotation: string | null;
  onSelect: (id: string) => void;
  onToggleFlag: (id: string, key: "hidden" | "locked") => void;
  onDelete: (id: string) => void;
  onClearFilter: () => void;
}) {
  const visible = annotations.filter((ann) => !labelFilter || ann.label === labelFilter);

  return (
    <section className="border-b">
      <div className="flex items-center justify-between px-3 py-2.5">
        <h3 className="panel-title">Annotations</h3>
        <div className="flex items-center gap-1.5">
          {labelFilter && (
            <Badge variant="secondary" className="text-[10px] cursor-pointer" onClick={onClearFilter}>
              {labelFilter} <X className="size-2.5 ml-1" />
            </Badge>
          )}
          <Badge variant="default" className="tabular-nums">{annotations.length}</Badge>
        </div>
      </div>
      <div className="max-h-64 overflow-y-auto">
        {visible.map((ann) => (
          <div
            key={ann.id}
            onClick={() => onSelect(ann.id)}
            className={cn(
              "flex items-center gap-1.5 px-3 py-1.5 cursor-pointer text-xs transition-colors group border-l-2 border-transparent",
              selectedAnnotation === ann.id
                ? "bg-primary/15 text-foreground border-l-primary"
                : "hover:bg-accent/60 hover:text-accent-foreground",
              ann.hidden && "opacity-40",
            )}
          >
            <span
              className="w-4 h-4 rounded text-[9px] font-bold flex items-center justify-center text-white shrink-0 shadow-sm"
              style={{ backgroundColor: ann.color }}
            >
              {ann.type[0].toUpperCase()}
            </span>
            <span className="flex-1 truncate">{ann.label}</span>
            {ann.attributes && Object.keys(ann.attributes).length > 0 && (
              <Badge variant="outline" className="text-[9px] h-4 px-1">
                {Object.keys(ann.attributes).length}
              </Badge>
            )}
            <Tooltip>
              <TooltipTrigger asChild>
                <button
                  onClick={(e) => { e.stopPropagation(); onToggleFlag(ann.id, "hidden"); }}
                  className={cn(
                    "size-5 sm:size-4 rounded flex items-center justify-center hover:bg-white/10 transition",
                    ann.hidden
                      ? "text-red-400 opacity-100"
                      : "text-muted-foreground opacity-0 group-hover:opacity-100 focus-visible:opacity-100",
                  )}
                  aria-label={ann.hidden ? "Show annotation" : "Hide annotation"}
                >
                  {ann.hidden ? <EyeOff className="size-3" /> : <Eye className="size-3" />}
                </button>
              </TooltipTrigger>
              <TooltipContent>{ann.hidden ? "Show" : "Hide"}</TooltipContent>
            </Tooltip>
            <Tooltip>
              <TooltipTrigger asChild>
                <button
                  onClick={(e) => { e.stopPropagation(); onToggleFlag(ann.id, "locked"); }}
                  className={cn(
                    "size-5 sm:size-4 rounded flex items-center justify-center hover:bg-white/10 transition",
                    ann.locked
                      ? "text-amber-400 opacity-100"
                      : "text-muted-foreground opacity-0 group-hover:opacity-100 focus-visible:opacity-100",
                  )}
                  aria-label={ann.locked ? "Unlock annotation" : "Lock annotation"}
                >
                  {ann.locked ? <Lock className="size-3" /> : <Unlock className="size-3" />}
                </button>
              </TooltipTrigger>
              <TooltipContent>{ann.locked ? "Unlock" : "Lock"}</TooltipContent>
            </Tooltip>
            <button
              onClick={(e) => { e.stopPropagation(); onDelete(ann.id); }}
              className="size-5 sm:size-4 rounded flex items-center justify-center opacity-0 group-hover:opacity-100 focus-visible:opacity-100 text-red-400 hover:text-red-300 transition"
              aria-label="Delete annotation"
            >
              <X className="size-3" />
            </button>
          </div>
        ))}
        {annotations.length === 0 && (
          <div className="px-3 py-6 text-center text-muted-foreground text-xs flex flex-col items-center gap-2">
            <Layers className="size-6 opacity-40" strokeWidth={1.5} />
            <span>No annotations yet.<br />Select a tool and draw on the image.</span>
          </div>
        )}
        {annotations.length > 0 && labelFilter && visible.length === 0 && (
          <div className="px-3 py-4 text-center text-muted-foreground text-xs">
            No annotations with label <span className="font-mono">{labelFilter}</span>.
          </div>
        )}
      </div>
    </section>
  );
}

export const AnnotationsList = memo(AnnotationsListImpl);
