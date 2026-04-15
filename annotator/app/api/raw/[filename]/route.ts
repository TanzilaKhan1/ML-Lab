import { NextRequest, NextResponse } from "next/server";
import { getRawImage } from "@/lib/storage";

export const dynamic = "force-dynamic";

/**
 * Proxies raw images from R2 through the server so that R2 credentials
 * never need to be shipped to the browser.
 */
export async function GET(
  _req: NextRequest,
  { params }: { params: Promise<{ filename: string }> },
) {
  const { filename } = await params;
  const decoded = decodeURIComponent(filename);
  const img = await getRawImage(decoded);
  if (!img) {
    return NextResponse.json({ error: "Not found" }, { status: 404 });
  }
  // Copy into a fresh ArrayBuffer to satisfy Response's BodyInit typing
  // (Node Buffer's underlying buffer is ArrayBufferLike, not ArrayBuffer).
  const ab = new ArrayBuffer(img.body.byteLength);
  new Uint8Array(ab).set(img.body);
  return new NextResponse(ab, {
    status: 200,
    headers: {
      "Content-Type": img.contentType,
      "Content-Length": img.body.byteLength.toString(),
      // Images in R2 are immutable per-filename; cache briefly to speed up navigation.
      "Cache-Control": "private, max-age=300",
    },
  });
}
