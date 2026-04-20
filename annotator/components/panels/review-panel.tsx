"use client";

import { cn } from "@/lib/utils";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Check, X } from "lucide-react";
import type { ImageStatus, HistoryEntry } from "@/lib/types";

const statusBadgeVariant: Record<ImageStatus, "default" | "secondary" | "destructive" | "outline"> = {
  unannotated: "secondary",
  annotated: "default",
  accepted: "default",
  rejected: "destructive",
};

export function ReviewPanel({
  currentImage,
  imageStatus,
  reviewComment,
  reviewHistory,
  onCommentChange,
  onStatus,
}: {
  currentImage: string | null;
  imageStatus: ImageStatus;
  reviewComment: string;
  reviewHistory: HistoryEntry[];
  onCommentChange: (value: string) => void;
  onStatus: (status: ImageStatus) => void;
}) {
  return (
    <section className="p-3 space-y-2">
      <h3 className="text-[10px] uppercase tracking-wider text-muted-foreground font-semibold">Review</h3>
      {currentImage ? (
        <>
          <Badge
            variant={statusBadgeVariant[imageStatus]}
            className={cn(
              "w-full justify-center h-7 text-xs capitalize",
              imageStatus === "accepted" && "bg-emerald-500/20 text-emerald-400 border-emerald-500/30",
            )}
          >
            {imageStatus}
          </Badge>
          <Textarea
            value={reviewComment}
            onChange={(e) => onCommentChange(e.target.value)}
            placeholder="Add review comments…"
            rows={2}
            className="text-xs resize-y"
          />
          <div className="flex gap-1.5">
            <Button
              className="flex-1 bg-emerald-500 text-white hover:bg-emerald-400"
              size="sm"
              onClick={() => onStatus("accepted")}
            >
              <Check /> Accept <kbd className="text-[9px] opacity-60 ml-1">Q</kbd>
            </Button>
            <Button
              variant="destructive"
              className="flex-1"
              size="sm"
              onClick={() => onStatus("rejected")}
            >
              <X /> Reject <kbd className="text-[9px] opacity-60 ml-1">W</kbd>
            </Button>
          </div>
          <Button variant="outline" size="sm" className="w-full" onClick={() => onStatus("annotated")}>
            Reset status
          </Button>
          {reviewHistory.length > 0 && (
            <div className="pt-2">
              <h4 className="text-[10px] uppercase tracking-wider text-muted-foreground mb-1 font-semibold">History</h4>
              <div className="max-h-40 overflow-y-auto space-y-1">
                {[...reviewHistory].reverse().map((h, i) => (
                  <div key={i} className="p-1.5 rounded bg-muted text-[11px]">
                    <span className={cn(
                      "font-semibold capitalize",
                      h.action === "accepted" && "text-emerald-400",
                      h.action === "rejected" && "text-red-400",
                      h.action === "annotated" && "text-primary",
                    )}>{h.action}</span>
                    {h.comment && <span className="text-muted-foreground"> — {h.comment}</span>}
                    <div className="text-[9px] text-muted-foreground font-mono mt-0.5">
                      {new Date(h.timestamp).toLocaleString()}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </>
      ) : (
        <p className="text-xs text-muted-foreground">Select an image to review</p>
      )}
    </section>
  );
}
