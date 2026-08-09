import type { Metadata } from "next";
import { Be_Vietnam_Pro, JetBrains_Mono } from "next/font/google";
import "./globals.css";

// Be Vietnam Pro: dựng riêng cho tiếng Việt (dấu chuẩn) — ưu tiên đúng chữ hơn font mặc định.
const sans = Be_Vietnam_Pro({
  subsets: ["latin", "vietnamese"],
  weight: ["400", "500", "600", "700"],
  variable: "--font-sans",
  display: "swap",
});

const mono = JetBrains_Mono({
  subsets: ["latin", "vietnamese"],
  weight: ["400", "500"],
  variable: "--font-mono",
  display: "swap",
});

export const metadata: Metadata = {
  title: "HIT · MIRA — Trợ lý đa phương thức CLB Tin học",
  description:
    "Hỏi đáp về tư liệu ảnh/video và nội quy CLB Tin học HIT, có trích dẫn nguồn.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="vi" className={`${sans.variable} ${mono.variable}`}>
      <body className="font-sans">{children}</body>
    </html>
  );
}
