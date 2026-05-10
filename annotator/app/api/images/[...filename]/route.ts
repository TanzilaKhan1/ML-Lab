import { NextRequest, NextResponse } from "next/server";
import {
  InvalidFilenameError,
  NotFoundError,
  StorageStepError,
  UPLOAD_FOLDERS,
  softDeleteImage,
} from "@/lib/storage";

export const dynamic = "force-dynamic";

export async function DELETE(
  _req: NextRequest,
  { params }: { params: Promise<{ filename: string[] }> },
) {
  const { filename } = await params;
  const key = filename.join("/");
  if (!key) {
    return NextResponse.json({ error: "filename is required" }, { status: 400 });
  }
  // Hard guard: only allow deletes inside known upload folders. Blocks
  // attempts to traverse, double-trash, or hit reserved manifest keys.
  const firstTwo = filename.slice(0, 2).join("/");
  if (!(UPLOAD_FOLDERS as readonly string[]).includes(firstTwo)) {
    return NextResponse.json(
      { error: `Path must start with one of: ${UPLOAD_FOLDERS.join(", ")}` },
      { status: 400 },
    );
  }

  try {
    const trashedKey = await softDeleteImage(key);
    return NextResponse.json({ success: true, trashedKey });
  } catch (err) {
    console.error("[api/images/delete] failed", { key, err });
    if (err instanceof InvalidFilenameError) {
      return NextResponse.json({ error: err.message }, { status: 400 });
    }
    if (err instanceof NotFoundError) {
      return NextResponse.json({ error: err.message }, { status: 404 });
    }
    if (err instanceof StorageStepError) {
      return NextResponse.json(
        { error: "Delete partially failed; check server logs", step: err.step },
        { status: 500 },
      );
    }
    return NextResponse.json({ error: "Internal error" }, { status: 500 });
  }
}
