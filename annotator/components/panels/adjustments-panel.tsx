"use client";

import { memo } from "react";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";

function AdjustmentRow({
  label, value, onChange, min, max,
}: {
  label: string; value: number; onChange: (n: number) => void; min: number; max: number;
}) {
  return (
    <div className="flex items-center gap-3">
      <span className="w-16 text-xs text-muted-foreground">{label}</span>
      <Slider
        min={min} max={max} step={1}
        value={[value]}
        onValueChange={(v) => onChange(v[0])}
        className="flex-1"
      />
      <span className="w-10 text-right text-[10px] text-muted-foreground font-mono tabular-nums">{value}%</span>
    </div>
  );
}

function AdjustmentsPanelImpl({
  brightness, setBrightness,
  contrast, setContrast,
  opacity, setOpacity,
  onReset,
}: {
  brightness: number; setBrightness: (n: number) => void;
  contrast: number; setContrast: (n: number) => void;
  opacity: number; setOpacity: (n: number) => void;
  onReset: () => void;
}) {
  const isDirty = brightness !== 100 || contrast !== 100 || opacity !== 0;
  return (
    <section className="panel-section space-y-3">
      <div className="flex items-center justify-between">
        <h3 className="panel-title">Adjustments</h3>
        {isDirty && (
          <span className="text-[10px] text-primary font-medium">modified</span>
        )}
      </div>
      <div className="space-y-2">
        <AdjustmentRow label="Brightness" value={brightness} onChange={setBrightness} min={20} max={200} />
        <AdjustmentRow label="Contrast" value={contrast} onChange={setContrast} min={20} max={200} />
        <AdjustmentRow label="Fill" value={opacity} onChange={setOpacity} min={0} max={100} />
      </div>
      <Button variant="ghost" size="xs" onClick={onReset} disabled={!isDirty}>
        Reset adjustments
      </Button>
    </section>
  );
}

export const AdjustmentsPanel = memo(AdjustmentsPanelImpl);
