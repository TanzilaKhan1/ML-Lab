import { NextRequest, NextResponse } from "next/server";
import { saveUploadedImage } from "@/lib/storage";

export const dynamic = "force-dynamic";

export async function POST(req: NextRequest) {
  const formData = await req.formData();
  const files = formData.getAll("files");
  const uploaded: string[] = [];

  for (const file of files) {
    if (!(file instanceof File)) continue;
    const buffer = Buffer.from(await file.arrayBuffer());
    const name = file.name.replace(/[^a-zA-Z0-9._-]/g, "_");
    await saveUploadedImage(name, buffer);
    uploaded.push(name);
  }

  return NextResponse.json({ success: true, uploaded });
}
