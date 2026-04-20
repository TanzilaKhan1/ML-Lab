import { NextRequest, NextResponse } from "next/server";
import { saveUploadedImage } from "@/lib/storage";

export const dynamic = "force-dynamic";

// Per-file limit for the upload endpoint. 50 MB is generous for labelling images
// (raw DSLR 50 MP JPEGs sit around 15-30 MB) while preventing a bad client from
// starving memory with a multi-GB body. Files larger than this are rejected with
// a 413 so the UI can show a clear error instead of the server OOMing.
const MAX_BYTES_PER_FILE = 50 * 1024 * 1024;

export async function POST(req: NextRequest) {
  const formData = await req.formData();
  const files = formData.getAll("files");
  const uploaded: string[] = [];
  const rejected: { name: string; reason: string }[] = [];

  for (const file of files) {
    if (!(file instanceof File)) continue;
    if (!file.type.startsWith("image/")) {
      rejected.push({ name: file.name, reason: "not an image" });
      continue;
    }
    if (file.size > MAX_BYTES_PER_FILE) {
      rejected.push({
        name: file.name,
        reason: `exceeds ${MAX_BYTES_PER_FILE / 1024 / 1024} MB limit`,
      });
      continue;
    }
    const buffer = Buffer.from(await file.arrayBuffer());
    const name = file.name.replace(/[^a-zA-Z0-9._-]/g, "_");
    await saveUploadedImage(name, buffer);
    uploaded.push(name);
  }

  return NextResponse.json(
    { success: uploaded.length > 0, uploaded, rejected },
    { status: rejected.length > 0 && uploaded.length === 0 ? 413 : 200 },
  );
}
