import { NextRequest, NextResponse } from "next/server";
import path from "node:path";
import sharp from "sharp";
import { saveUploadedImage, UPLOAD_FOLDERS, type UploadFolder } from "@/lib/storage";

export const dynamic = "force-dynamic";

const MAX_BYTES_PER_FILE = 50 * 1024 * 1024;

const CONVERT_TO_PNG = new Set([".jpg", ".jpeg", ".heic", ".heif", ".bmp", ".tiff", ".tif", ".webp"]);

export async function POST(req: NextRequest) {
  const formData = await req.formData();
  const files = formData.getAll("files");

  // Validate folder — must be one of the four canonical folders
  const folderRaw = formData.get("folder") as string | null;
  const folder: UploadFolder = (UPLOAD_FOLDERS as readonly string[]).includes(folderRaw ?? "")
    ? (folderRaw as UploadFolder)
    : UPLOAD_FOLDERS[0];

  const uploaded: string[] = [];
  const rejected: { name: string; reason: string }[] = [];

  for (const file of files) {
    if (!(file instanceof File)) continue;

    const isImage = file.type.startsWith("image/") || /\.(heic|heif)$/i.test(file.name);
    if (!isImage) {
      rejected.push({ name: file.name, reason: "not an image" });
      continue;
    }
    if (file.size > MAX_BYTES_PER_FILE) {
      rejected.push({ name: file.name, reason: `exceeds ${MAX_BYTES_PER_FILE / 1024 / 1024} MB limit` });
      continue;
    }

    let buffer = Buffer.from(await file.arrayBuffer());
    let basename = file.name.replace(/[^a-zA-Z0-9._-]/g, "_");
    const ext = path.extname(basename).toLowerCase();

    if (CONVERT_TO_PNG.has(ext)) {
      try {
        buffer = Buffer.from(await sharp(buffer).png({ compressionLevel: 8 }).toBuffer());
        basename = basename.slice(0, basename.length - ext.length) + ".png";
      } catch (err) {
        rejected.push({ name: file.name, reason: `conversion failed: ${String(err)}` });
        continue;
      }
    }

    // Full relative path: e.g. "bus/positive/photo.png"
    const relativePath = `${folder}/${basename}`;

    try {
      await saveUploadedImage(relativePath, buffer);
      uploaded.push(relativePath);
    } catch (err) {
      rejected.push({ name: file.name, reason: `upload failed: ${String(err)}` });
    }
  }

  return NextResponse.json(
    { success: uploaded.length > 0, uploaded, rejected },
    { status: rejected.length > 0 && uploaded.length === 0 ? 413 : 200 },
  );
}
