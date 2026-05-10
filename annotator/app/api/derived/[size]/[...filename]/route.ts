import { NextRequest, NextResponse, after } from "next/server";
import sharp from "sharp";
import {
  getObjectBuffer,
  putObject,
  joinKey,
  PREFIX_RAW,
} from "@/lib/r2";

export const dynamic = "force-dynamic";
export const maxDuration = 30;

// Derivative tiers — pixel longest side. WebP quality is tuned per use:
// thumbs are ~1KB targets so quality is irrelevant; preview is what the
// canvas paints first, so we keep it visually clean.
const SIZES: Record<string, { side: number; quality: number }> = {
  thumb:   { side: 256,  quality: 70 },
  preview: { side: 1280, quality: 82 },
};

const PREFIX_DERIVED = () => process.env.R2_PREFIX_DERIVED || "derived";

function derivedKey(size: string, filename: string): string {
  // Strip original extension; derivatives are always .webp.
  const base = filename.replace(/\.[^.]+$/, "");
  return joinKey(PREFIX_DERIVED(), size, base + ".webp");
}

function rawKey(filename: string): string {
  return joinKey(PREFIX_RAW(), filename);
}

/** Reject path traversal segments without false-positives on legitimate
 *  filenames containing two adjacent dots (e.g. "image..backup.png"). */
function isUnsafePath(rel: string): boolean {
  if (!rel) return true;
  if (rel.startsWith("/")) return true;
  for (const seg of rel.split("/")) {
    if (seg === "" || seg === "." || seg === "..") return true;
  }
  return false;
}

export async function GET(
  _req: NextRequest,
  { params }: { params: Promise<{ size: string; filename: string[] }> },
) {
  const { size, filename } = await params;
  const cfg = SIZES[size];
  if (!cfg) {
    return NextResponse.json({ error: `Unknown size: ${size}` }, { status: 400 });
  }
  const rel = filename.join("/");
  if (isUnsafePath(rel)) {
    return NextResponse.json({ error: "Invalid filename" }, { status: 400 });
  }

  const dKey = derivedKey(size, rel);

  // Cache hit — original dims read from the R2 user-defined metadata that
  // was set when the derivative was generated. Browser uses these dims to
  // scale the preview up to true raw resolution before the raw arrives.
  // A transient R2 read error here MUST NOT fail the request — fall through
  // to the resize path so a bucket hiccup doesn't take down the route.
  let cached: Awaited<ReturnType<typeof getObjectBuffer>> = null;
  try {
    cached = await getObjectBuffer(dKey);
  } catch (err) {
    console.error("[derived] cache read failed; falling through to resize", { dKey, err });
  }
  if (cached) {
    // S3/R2 lowercases user-defined metadata keys but preserves hyphens, so
    // "original-width" round-trips intact. (No legacy un-hyphenated keys
    // ever existed — the previous fallback was dead code.)
    const origW = Number(cached.metadata?.["original-width"] || 0);
    const origH = Number(cached.metadata?.["original-height"] || 0);
    const ab = new ArrayBuffer(cached.body.byteLength);
    new Uint8Array(ab).set(cached.body);
    const headers: Record<string, string> = {
      "Content-Type": cached.contentType || "image/webp",
      "Content-Length": cached.body.byteLength.toString(),
      "Cache-Control": "public, max-age=31536000, immutable",
      "X-Derived-Cache": "HIT",
      "Access-Control-Expose-Headers": "X-Original-Width, X-Original-Height, X-Derived-Cache",
    };
    if (origW) headers["X-Original-Width"] = String(origW);
    if (origH) headers["X-Original-Height"] = String(origH);
    return new NextResponse(ab, { status: 200, headers });
  }

  // Cache miss — fetch raw, capture display dims, resize, store, serve.
  const raw = await getObjectBuffer(rawKey(rel));
  if (!raw) {
    return NextResponse.json({ error: "Source image not found" }, { status: 404 });
  }

  // Read display dims (post-EXIF-rotate) before resize so the browser can
  // scale the preview to match the resized webp's source dimensions. Sharp's
  // pipeline state isn't designed to be reused across both phases, so we
  // build two pipelines from the same input Buffer (Buffer is re-readable).
  let origW = 0, origH = 0;
  try {
    const meta = await sharp(raw.body, { failOn: "none" }).rotate().metadata();
    origW = meta.width || 0;
    origH = meta.height || 0;
  } catch (err) {
    const errorId = crypto.randomUUID();
    console.error("[derived] decode/metadata failed", { errorId, dKey, err });
    return NextResponse.json(
      { error: "Image could not be decoded", errorId },
      { status: 500 },
    );
  }

  let webp: Buffer;
  try {
    webp = await sharp(raw.body, { failOn: "none" })
      .rotate()
      .resize({
        width: cfg.side,
        height: cfg.side,
        fit: "inside",
        withoutEnlargement: true,
      })
      .webp({ quality: cfg.quality, effort: 4 })
      .toBuffer();
  } catch (err) {
    const errorId = crypto.randomUUID();
    console.error("[derived] resize/encode failed", { errorId, dKey, err });
    return NextResponse.json(
      { error: "Image could not be resized", errorId },
      { status: 500 },
    );
  }

  // Cache write must outlive the response on serverless platforms — `after()`
  // keeps the Next.js runtime alive past the response so the put isn't frozen.
  // Errors are logged but never reach the client (they ran post-response).
  after(
    putObject(dKey, webp, "image/webp", {
      "original-width": String(origW),
      "original-height": String(origH),
    }).catch((err) => console.error("[derived] cache write failed", { dKey, err })),
  );

  const ab = new ArrayBuffer(webp.byteLength);
  new Uint8Array(ab).set(webp);
  const headers: Record<string, string> = {
    "Content-Type": "image/webp",
    "Content-Length": webp.byteLength.toString(),
    "Cache-Control": "public, max-age=31536000, immutable",
    "X-Derived-Cache": "MISS",
    "Access-Control-Expose-Headers": "X-Original-Width, X-Original-Height, X-Derived-Cache",
  };
  if (origW) headers["X-Original-Width"] = String(origW);
  if (origH) headers["X-Original-Height"] = String(origH);
  return new NextResponse(ab, { status: 200, headers });
}
