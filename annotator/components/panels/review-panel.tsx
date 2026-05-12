"use client";

import { memo } from "react";
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

function ReviewPanelImpl({
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
    <section className="panel-section space-y-2 pb-safe">
      <h3 className="panel-title">Review</h3>
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
              className="flex-1 bg-emerald-500 text-white hover:bg-emerald-400 active:scale-95"
              size="sm"
              onClick={() => onStatus("accepted")}
            >
              <Check /> Accept <kbd className="text-[9px] opacity-60 ml-1">Q</kbd>
            </Button>
            <Button
              variant="destructive"
              className="flex-1 active:scale-95"
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
              <h4 className="panel-title mb-1">History</h4>
              <div className="max-h-40 overflow-y-auto space-y-1">
                {[...reviewHistory].reverse().map((h, i) => (
                  <div key={i} className="p-1.5 rounded bg-muted/60 text-[11px] border border-border/40">
                    <span className={cn(
                      "font-semibold capitalize",
                      h.action === "accepted" && "text-emerald-400",
                      h.action === "rejected" && "text-red-400",
                      h.action === "annotated" && "text-primary",
                    )}>{h.action}</span>
                    {h.comment && <span className="text-muted-foreground"> — {h.comment}</span>}
                    <div className="text-[9px] text-muted-foreground font-mono mt-0.5 tabular-nums">
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

export const ReviewPanel = memo(ReviewPanelImpl);
