"use client";

import { useState } from "react";
import { ImageIcon } from "./icons";

// Ảnh có placeholder khi tải lỗi (US-503.1 AC-2: không vỡ layout).
export function MediaImage({ src, alt }: { src: string; alt: string }) {
  const [failed, setFailed] = useState(false);

  if (failed) {
    return (
      <div className="flex aspect-[4/3] w-full flex-col items-center justify-center gap-2 bg-zinc-100 text-zinc-400">
        <ImageIcon width={22} height={22} />
        <span className="text-xs">Không tải được ảnh</span>
      </div>
    );
  }

  return (
    // eslint-disable-next-line @next/next/no-img-element
    <img
      src={src}
      alt={alt}
      loading="lazy"
      onError={() => setFailed(true)}
      className="aspect-[4/3] w-full object-cover"
    />
  );
}
