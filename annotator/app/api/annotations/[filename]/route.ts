import { NextRequest, NextResponse } from "next/server";
import { getAnnotation, saveAnnotation, updateStatus } from "@/lib/storage";
import type { ImageStatus } from "@/lib/types";

export const dynamic = "force-dynamic";

export async function GET(
  _req: NextRequest,
  { params }: { params: Promise<{ filename: string }> }
) {
  const { filename } = await params;
  return NextResponse.json(getAnnotation(decodeURIComponent(filename)));
}

export async function POST(
  req: NextRequest,
  { params }: { params: Promise<{ filename: string }> }
) {
  const { filename } = await params;
  const body = await req.json();
  body.filename = decodeURIComponent(filename);
  saveAnnotation(body);
  return NextResponse.json({ success: true });
}

export async function PATCH(
  req: NextRequest,
  { params }: { params: Promise<{ filename: string }> }
) {
  const { filename } = await params;
  const body = await req.json();
  updateStatus(decodeURIComponent(filename), body.status as ImageStatus, body.comment);
  return NextResponse.json({ success: true });
}
