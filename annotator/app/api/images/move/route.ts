import { NextRequest, NextResponse } from "next/server";
import {
  AlreadyExistsError,
  InvalidFilenameError,
  NotFoundError,
  StorageStepError,
  UPLOAD_FOLDERS,
  moveImage,
  type UploadFolder,
} from "@/lib/storage";

export const dynamic = "force-dynamic";

export async function POST(req: NextRequest) {
  let body: { filename?: unknown; destFolder?: unknown };
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ error: "Invalid JSON body" }, { status: 400 });
  }

  const filename = typeof body.filename === "string" ? body.filename : "";
  const destFolder = typeof body.destFolder === "string" ? body.destFolder : "";

  if (!filename) {
    return NextResponse.json({ error: "filename is required" }, { status: 400 });
  }
  if (!(UPLOAD_FOLDERS as readonly string[]).includes(destFolder)) {
    return NextResponse.json(
      { error: `destFolder must be one of: ${UPLOAD_FOLDERS.join(", ")}` },
      { status: 400 },
    );
  }

  try {
    const newFilename = await moveImage(filename, destFolder as UploadFolder);
    return NextResponse.json({ success: true, filename: newFilename });
  } catch (err) {
    console.error("[api/images/move] failed", { filename, destFolder, err });
    if (err instanceof InvalidFilenameError) {
      return NextResponse.json({ error: err.message }, { status: 400 });
    }
    if (err instanceof NotFoundError) {
      return NextResponse.json({ error: err.message }, { status: 404 });
    }
    if (err instanceof AlreadyExistsError) {
      return NextResponse.json({ error: err.message }, { status: 409 });
    }
    if (err instanceof StorageStepError) {
      return NextResponse.json(
        { error: "Move partially failed; check server logs", step: err.step },
        { status: 500 },
      );
    }
    return NextResponse.json({ error: "Internal error" }, { status: 500 });
  }
}
