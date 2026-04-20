import type { Metadata } from "next";
import { Inter, JetBrains_Mono, Geist } from "next/font/google";
import "./globals.css";
import { cn } from "@/lib/utils";
import { Providers } from "./providers";

const geist = Geist({ subsets: ["latin"], variable: "--font-sans" });
const inter = Inter({ variable: "--font-inter", subsets: ["latin"] });
const jetbrains = JetBrains_Mono({ variable: "--font-mono", subsets: ["latin"] });

export const metadata: Metadata = {
  title: "Image Annotator",
  description: "Professional image annotation tool for data labeling",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="en"
      className={cn("h-full", inter.variable, jetbrains.variable, geist.variable, "font-sans")}
      suppressHydrationWarning
    >
      <body className="h-full bg-background text-foreground overflow-hidden antialiased">
        <Providers>{children}</Providers>
      </body>
    </html>
  );
}
