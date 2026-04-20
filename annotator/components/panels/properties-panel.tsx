"use client";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Separator } from "@/components/ui/separator";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Copy, CopyPlus, X } from "lucide-react";
import type { Annotation, LabelDef } from "@/lib/types";

function Row({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex items-center justify-between">
      <span className="text-muted-foreground">{label}</span>
      {children}
    </div>
  );
}

export function PropertiesPanel({
  selectedAnnotation,
  annotations,
  labels,
  attrDraftKey,
  attrDraftVal,
  onAttrDraftKey,
  onAttrDraftVal,
  onChangeLabel,
  onSetAttribute,
  onCopy,
  onDuplicate,
}: {
  selectedAnnotation: string | null;
  annotations: Annotation[];
  labels: LabelDef[];
  attrDraftKey: string;
  attrDraftVal: string;
  onAttrDraftKey: (v: string) => void;
  onAttrDraftVal: (v: string) => void;
  onChangeLabel: (id: string, newLabel: string) => void;
  onSetAttribute: (id: string, key: string, value: string | null) => void;
  onCopy: () => void;
  onDuplicate: () => void;
}) {
  return (
    <section className="border-b p-3">
      <h3 className="text-[10px] uppercase tracking-wider text-muted-foreground font-semibold mb-2">Properties</h3>
      {selectedAnnotation
        ? (() => {
            const ann = annotations.find((a) => a.id === selectedAnnotation);
            if (!ann) return <p className="text-xs text-muted-foreground">Not found</p>;
            const attrEntries = Object.entries(ann.attributes || {});
            return (
              <div className="space-y-2 text-xs">
                <Row label="Type">
                  <Badge
                    variant="outline"
                    className="capitalize"
                    style={{ borderColor: ann.color + "66", color: ann.color }}
                  >
                    {ann.type}
                  </Badge>
                </Row>
                <Row label="Label">
                  <Select value={ann.label} onValueChange={(v) => onChangeLabel(ann.id, v)}>
                    <SelectTrigger size="sm" className="h-7 w-32 text-xs">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {labels.map((l) => (
                        <SelectItem key={l.name} value={l.name}>{l.name}</SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </Row>
                {(ann.type === "bbox" || ann.type === "ellipse") && (
                  <>
                    <Row label="Position">
                      <span className="font-mono tabular-nums text-[11px]">
                        {Math.round(ann.x || 0)}, {Math.round(ann.y || 0)}
                      </span>
                    </Row>
                    <Row label="Size">
                      <span className="font-mono tabular-nums text-[11px]">
                        {Math.round(ann.width || 0)} × {Math.round(ann.height || 0)}
                      </span>
                    </Row>
                  </>
                )}
                {(ann.type === "polygon" || ann.type === "polyline") && (
                  <Row label="Vertices">
                    <span className="tabular-nums">{ann.points?.length || 0}</span>
                  </Row>
                )}
                {ann.type === "keypoint" && (
                  <Row label="Position">
                    <span className="font-mono tabular-nums text-[11px]">
                      {Math.round(ann.x || 0)}, {Math.round(ann.y || 0)}
                    </span>
                  </Row>
                )}

                <Separator />

                <div>
                  <div className="flex items-center justify-between mb-1.5">
                    <span className="text-[10px] uppercase tracking-wider text-muted-foreground font-semibold">Attributes</span>
                    <span className="text-[10px] text-muted-foreground tabular-nums">{attrEntries.length}</span>
                  </div>
                  <div className="space-y-1">
                    {attrEntries.map(([k, v]) => (
                      <div key={k} className="flex items-center gap-1">
                        <span className="text-[10px] text-muted-foreground font-mono w-16 truncate" title={k}>{k}</span>
                        <Input
                          value={v}
                          onChange={(ev) => onSetAttribute(ann.id, k, ev.target.value)}
                          className="h-7 flex-1 text-[11px]"
                        />
                        <Button
                          variant="ghost"
                          size="icon-xs"
                          onClick={() => onSetAttribute(ann.id, k, null)}
                          className="text-destructive"
                        >
                          <X />
                        </Button>
                      </div>
                    ))}
                  </div>
                  <div className="flex gap-1 mt-1.5">
                    <Input
                      value={attrDraftKey}
                      onChange={(ev) => onAttrDraftKey(ev.target.value)}
                      placeholder="key"
                      className="h-7 flex-1 w-0 text-[11px] font-mono"
                    />
                    <Input
                      value={attrDraftVal}
                      onChange={(ev) => onAttrDraftVal(ev.target.value)}
                      placeholder="value"
                      onKeyDown={(ev) => {
                        if (ev.key === "Enter" && attrDraftKey.trim()) {
                          onSetAttribute(ann.id, attrDraftKey.trim(), attrDraftVal);
                          onAttrDraftKey(""); onAttrDraftVal("");
                        }
                      }}
                      className="h-7 flex-1 w-0 text-[11px]"
                    />
                    <Button
                      size="xs"
                      onClick={() => {
                        if (!attrDraftKey.trim()) return;
                        onSetAttribute(ann.id, attrDraftKey.trim(), attrDraftVal);
                        onAttrDraftKey(""); onAttrDraftVal("");
                      }}
                    >
                      Add
                    </Button>
                  </div>
                </div>

                <Separator />

                <div className="flex gap-1.5">
                  <Button variant="outline" size="xs" className="flex-1" onClick={onCopy}>
                    <Copy /> Copy
                  </Button>
                  <Button variant="outline" size="xs" className="flex-1" onClick={onDuplicate}>
                    <CopyPlus /> Duplicate
                  </Button>
                </div>
              </div>
            );
          })()
        : <p className="text-xs text-muted-foreground">Select an annotation to edit</p>}
    </section>
  );
}
