"use client";

import { useEffect, useRef, useState } from "react";
import { validateImage } from "@/lib/image";
import type { Override } from "@/lib/types";
import { ArrowUp, ImageIcon } from "./icons";

type Mode = "auto" | Override;

const MODES: { key: Mode; label: string }[] = [
  { key: "auto", label: "Tự động" },
  { key: "media", label: "Ảnh & video" },
  { key: "regulation", label: "Nội quy" },
  { key: "both", label: "Cả hai" },
];

export function Composer({
  value,
  onChange,
  onSubmit,
  mode,
  onMode,
  busy,
  image,
  imagePreview,
  onPickImage,
}: {
  value: string;
  onChange: (v: string) => void;
  onSubmit: () => void;
  mode: Mode;
  onMode: (m: Mode) => void;
  busy: boolean;
  image: File | null;
  imagePreview: string | null;
  onPickImage: (f: File | null) => void;
}) {
  const ref = useRef<HTMLTextAreaElement>(null);
  const fileRef = useRef<HTMLInputElement>(null);
  const [imgError, setImgError] = useState<string | null>(null);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 180)}px`;
  }, [value]);

  const canSend = (value.trim().length > 0 || !!image) && !busy;

  function handleKey(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      if (canSend) onSubmit();
    }
  }

  function pick(e: React.ChangeEvent<HTMLInputElement>) {
    const f = e.target.files?.[0];
    e.target.value = ""; // cho phép chọn lại cùng file
    if (!f) return;
    const err = validateImage(f);
    if (err) {
      setImgError(err);
      return;
    }
    setImgError(null);
    onPickImage(f);
  }

  return (
    <div className="rounded-2xl border border-zinc-200/90 bg-white p-2.5 shadow-[0_16px_50px_-18px_rgba(24,24,27,0.2)] ring-1 ring-black/[0.02]">
      <div className="mb-2.5 flex flex-wrap gap-1 px-1 pt-0.5">
        {MODES.map((m) => (
          <button
            key={m.key}
            onClick={() => onMode(m.key)}
            className={`rounded-lg px-2.5 py-1.5 text-xs font-medium transition-colors ${
              mode === m.key
                ? "bg-accent-soft text-accent-ink"
                : "text-zinc-400 hover:text-zinc-600"
            }`}
          >
            {m.label}
          </button>
        ))}
      </div>

      {imagePreview && (
        <div className="mb-2 flex items-center gap-2 px-1">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={imagePreview}
            alt="ảnh đính kèm"
            className="h-14 w-14 rounded-lg border border-zinc-200 object-cover"
          />
          <button
            onClick={() => onPickImage(null)}
            className="text-xs text-zinc-400 transition-colors hover:text-red-600"
          >
            Bỏ ảnh
          </button>
        </div>
      )}

      <div className="flex items-end gap-2">
        <input
          ref={fileRef}
          type="file"
          accept="image/jpeg,image/png,image/webp"
          onChange={pick}
          className="hidden"
        />
        <button
          onClick={() => fileRef.current?.click()}
          aria-label="Đính kèm ảnh"
          className="grid h-10 w-10 shrink-0 place-items-center rounded-xl text-zinc-400 transition-colors hover:bg-zinc-100 hover:text-zinc-700"
        >
          <ImageIcon width={18} height={18} />
        </button>
        <textarea
          ref={ref}
          rows={1}
          value={value}
          onChange={(e) => onChange(e.target.value)}
          onKeyDown={handleKey}
          placeholder={image ? "Thêm mô tả (tùy chọn)…" : "Hỏi về ảnh, video hoặc nội quy CLB…"}
          className="scroll-slim max-h-44 min-h-[40px] flex-1 resize-none bg-transparent px-1 py-2 text-sm leading-relaxed text-zinc-800 outline-none placeholder:text-zinc-400"
        />
        <button
          onClick={() => canSend && onSubmit()}
          disabled={!canSend}
          aria-label="Gửi"
          className="grid h-10 w-10 shrink-0 place-items-center rounded-xl bg-accent text-white transition-all enabled:hover:bg-accent-ink enabled:active:scale-95 disabled:cursor-not-allowed disabled:bg-zinc-200 disabled:text-zinc-400"
        >
          <ArrowUp width={18} height={18} />
        </button>
      </div>

      {imgError && <p className="px-2 pt-1.5 text-xs text-red-600">{imgError}</p>}
    </div>
  );
}
