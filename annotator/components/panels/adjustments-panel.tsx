"use client";

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

export function AdjustmentsPanel({
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
  return (
    <section className="border-b p-3 space-y-3">
      <h3 className="text-[10px] uppercase tracking-wider text-muted-foreground font-semibold">Adjustments</h3>
      <div className="space-y-2">
        <AdjustmentRow label="Brightness" value={brightness} onChange={setBrightness} min={20} max={200} />
        <AdjustmentRow label="Contrast" value={contrast} onChange={setContrast} min={20} max={200} />
        <AdjustmentRow label="Fill" value={opacity} onChange={setOpacity} min={0} max={100} />
      </div>
      <Button variant="ghost" size="xs" onClick={onReset}>
        Reset adjustments
      </Button>
    </section>
  );
}
