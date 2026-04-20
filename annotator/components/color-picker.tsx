"use client";

import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { Separator } from "@/components/ui/separator";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { cn } from "@/lib/utils";

const COLOR_HEX = /^#?[0-9a-fA-F]{6}$/;

function normalize(hex: string): string {
  return hex.startsWith("#") ? hex : `#${hex}`;
}

export function ColorPicker({
  value,
  onChange,
  swatches,
  ariaLabel = "Pick color",
}: {
  value: string;
  onChange: (hex: string) => void;
  swatches: string[];
  ariaLabel?: string;
}) {
  return (
    <Popover>
      <PopoverTrigger asChild>
        <button
          type="button"
          aria-label={ariaLabel}
          className="size-8 rounded-md border border-input shadow-sm transition-all hover:scale-105 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
          style={{ backgroundColor: value }}
        />
      </PopoverTrigger>
      <PopoverContent className="w-64 p-3 space-y-3" align="end">
        <div>
          <Label className="text-[10px] uppercase tracking-wider text-muted-foreground font-semibold">Swatches</Label>
          <div className="mt-2 grid grid-cols-5 gap-1.5">
            {swatches.map((c) => {
              const active = c.toLowerCase() === value.toLowerCase();
              return (
                <button
                  key={c}
                  type="button"
                  onClick={() => onChange(c)}
                  className={cn(
                    "size-8 rounded-md border-2 transition-all hover:scale-110",
                    active ? "border-foreground ring-2 ring-ring/50" : "border-transparent",
                  )}
                  style={{ backgroundColor: c }}
                  aria-label={c}
                />
              );
            })}
          </div>
        </div>
        <Separator />
        <div className="space-y-1.5">
          <Label htmlFor="cp-hex" className="text-[10px] uppercase tracking-wider text-muted-foreground font-semibold">Custom</Label>
          <div className="flex items-center gap-2">
            <label
              htmlFor="cp-native"
              className="relative size-9 rounded-md border cursor-pointer overflow-hidden"
              style={{ backgroundColor: value }}
            >
              <input
                id="cp-native"
                type="color"
                value={COLOR_HEX.test(value) ? normalize(value) : "#ffffff"}
                onChange={(e) => onChange(e.target.value)}
                className="absolute inset-0 opacity-0 cursor-pointer"
              />
            </label>
            <Input
              id="cp-hex"
              value={value}
              onChange={(e) => {
                const v = e.target.value;
                if (COLOR_HEX.test(v)) onChange(normalize(v));
                else onChange(v); // allow partial typing; validation on commit
              }}
              placeholder="#ff6b6b"
              className="h-9 font-mono text-xs uppercase"
              spellCheck={false}
            />
          </div>
        </div>
      </PopoverContent>
    </Popover>
  );
}
