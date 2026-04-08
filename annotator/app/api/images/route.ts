import { NextRequest, NextResponse } from "next/server";
import { getImages } from "@/lib/storage";
import type { ImageStatus } from "@/lib/types";

export const dynamic = "force-dynamic";

export async function GET(req: NextRequest) {
  const filter = req.nextUrl.searchParams.get("filter") as ImageStatus | "all" | null;
  return NextResponse.json(getImages(filter || "all"));
}
