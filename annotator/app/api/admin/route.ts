import { NextRequest, NextResponse } from "next/server";
import { clearAllData, initFolders } from "@/lib/storage";

export const dynamic = "force-dynamic";

export async function DELETE(_req: NextRequest) {
  const result = await clearAllData();
  return NextResponse.json({ success: true, ...result });
}

export async function POST(_req: NextRequest) {
  const folders = await initFolders();
  return NextResponse.json({ success: true, folders });
}
