import { NextRequest, NextResponse } from "next/server";
import { clearAllData, initFolders, rebuildManifest } from "@/lib/storage";

export const dynamic = "force-dynamic";
export const maxDuration = 60;

export async function DELETE(_req: NextRequest) {
  const result = await clearAllData();
  return NextResponse.json({ success: true, ...result });
}

export async function POST(_req: NextRequest) {
  const folders = await initFolders();
  return NextResponse.json({ success: true, folders });
}

export async function PUT(_req: NextRequest) {
  const m = await rebuildManifest();
  return NextResponse.json({
    success: true,
    images: Object.keys(m.images).length,
    updatedAt: m.updatedAt,
  });
}
