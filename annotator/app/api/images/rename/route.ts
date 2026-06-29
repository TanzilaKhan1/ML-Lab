import { NextRequest, NextResponse } from "next/server";
import {
  AlreadyExistsError,
  InvalidFilenameError,
  NotFoundError,
  StorageStepError,
  renameImage,
} from "@/lib/storage";

export const dynamic = "force-dynamic";

export async function POST(req: NextRequest) {
  let body: { filename?: unknown; newBasename?: unknown };
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ error: "Invalid JSON body" }, { status: 400 });
  }

  const filename = typeof body.filename === "string" ? body.filename : "";
  const newBasename = typeof body.newBasename === "string" ? body.newBasename.trim() : "";

  if (!filename) {
    return NextResponse.json({ error: "filename is required" }, { status: 400 });
  }
  if (!newBasename) {
    return NextResponse.json({ error: "newBasename is required" }, { status: 400 });
  }

  try {
    const newFilename = await renameImage(filename, newBasename);
    return NextResponse.json({ success: true, filename: newFilename });
  } catch (err) {
    console.error("[api/images/rename] failed", { filename, newBasename, err });
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
        { error: "Rename partially failed; check server logs", step: err.step },
        { status: 500 },
      );
    }
    return NextResponse.json({ error: "Internal error" }, { status: 500 });
  }
}
