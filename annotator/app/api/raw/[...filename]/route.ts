import { NextRequest, NextResponse } from "next/server";
import { getRawImage } from "@/lib/storage";

export const dynamic = "force-dynamic";

export async function GET(
  _req: NextRequest,
  { params }: { params: Promise<{ filename: string[] }> },
) {
  const { filename } = await params;
  const key = filename.join("/");
  const img = await getRawImage(key);
  if (!img) {
    return NextResponse.json({ error: "Not found" }, { status: 404 });
  }
  const ab = new ArrayBuffer(img.body.byteLength);
  new Uint8Array(ab).set(img.body);
  return new NextResponse(ab, {
    status: 200,
    headers: {
      "Content-Type": img.contentType,
      "Content-Length": img.body.byteLength.toString(),
      "Cache-Control": "private, max-age=300",
    },
  });
}
