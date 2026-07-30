"use client";

import { useState } from "react";
import { SOURCE_LABEL } from "@/lib/format";
import type { ChatTurn } from "@/lib/types";
import { ResultCard } from "./ResultCard";
import { CheckIcon, CopyIcon, SparkIcon } from "./icons";

function Avatar() {
  return <img src="/logo.png" alt="MIRA" className="h-9 w-9 shrink-0 rounded-xl border border-white shadow-sm" />;
}

function SourceBadge({ source, count }: { source: string; count: number }) {
  return (
    <span className="inline-flex items-center gap-1.5 rounded-full border border-zinc-200 bg-zinc-50 px-2.5 py-1 text-[11px] font-medium text-zinc-500">
      <SparkIcon width={12} height={12} className="text-accent" />
      {SOURCE_LABEL[source] ?? source}
      {count > 0 && ` · ${count} nguồn`}
    </span>
  );
}

function LoadingDots() {
  return (
    <div className="flex items-center gap-2 pt-1 text-sm text-zinc-500">
      <span className="font-medium text-zinc-700">MIRA đang tìm kiếm</span>
      <span>
        <span className="typing-dot">.</span>
        <span className="typing-dot">.</span>
        <span className="typing-dot">.</span>
      </span>
    </div>
  );
}

/** Blinking cursor khi đang stream */
function StreamCursor() {
  return (
    <span className="ml-0.5 inline-block h-[1em] w-[2px] animate-pulse rounded-full bg-zinc-400 align-text-bottom" />
  );
}

export function MessageBubble({ turn, onRetry }: { turn: ChatTurn; onRetry?: () => void }) {
  const [copied, setCopied] = useState(false);

  // ── User bubble ──────────────────────────────────────────────────────────
  if (turn.role === "user") {
    return (
      <div className="flex animate-msg-in flex-col items-end gap-2">
        {turn.imagePreview && (
          // eslint-disable-next-line @next/next/no-img-element
          <img
            src={turn.imagePreview}
            alt="ảnh đã gửi"
            className="max-h-64 max-w-[85%] rounded-2xl rounded-br-md border border-zinc-200 object-cover shadow-sm"
          />
        )}
        {turn.text && (
          <div className="max-w-[85%] rounded-2xl rounded-br-md bg-zinc-900 px-4 py-3 text-sm leading-6 text-zinc-50 shadow-sm">
            {turn.text}
          </div>
        )}
      </div>
    );
  }

  // ── Assistant bubble ─────────────────────────────────────────────────────
  return (
    <div className="flex animate-msg-in gap-3 sm:gap-4">
      <Avatar />
      <div className="min-w-0 flex-1">

        {/* Loading: chưa có token nào */}
        {turn.status === "loading" && (
          <>
            <LoadingDots />
            <p className="mt-1 text-xs text-zinc-400">Đang đối chiếu các nguồn liên quan</p>
          </>
        )}

        {/* Streaming: đang nhận token */}
        {turn.status === "streaming" && (
          <div className="surface rounded-2xl rounded-tl-md px-5 py-4">
            <p className="whitespace-pre-wrap text-[15px] leading-7 text-zinc-800">
              {turn.streamText || ""}
              <StreamCursor />
            </p>
          </div>
        )}

        {/* Error */}
        {turn.status === "error" && (
          <div className="rounded-2xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
            <p className="font-medium">Không thể hoàn thành yêu cầu</p>
            <p className="mt-1 text-red-600">{turn.text || "Kiểm tra kết nối tới máy chủ."}</p>
            {onRetry && (
              <button
                onClick={onRetry}
                className="mt-3 rounded-lg bg-red-600 px-3 py-1.5 text-xs font-medium text-white active:scale-[.97]"
              >
                Thử lại
              </button>
            )}
          </div>
        )}

        {/* Done — LangGraph (finalText) */}
        {turn.status === "done" && turn.finalText && (
          <div className="space-y-4">
            <div className="surface rounded-2xl rounded-tl-md px-5 py-4">
              <p className="whitespace-pre-wrap text-[15px] leading-7 text-zinc-800">
                {turn.finalText}
              </p>
              <div className="mt-4 flex items-center justify-end border-t border-zinc-100 pt-3">
                <button
                  onClick={async () => {
                    await navigator.clipboard.writeText(turn.finalText ?? "");
                    setCopied(true);
                    setTimeout(() => setCopied(false), 1600);
                  }}
                  className="inline-flex items-center gap-1.5 rounded-lg px-2 py-1 text-xs text-zinc-400 transition hover:bg-zinc-100 hover:text-zinc-700"
                >
                  {copied ? <CheckIcon width={14} /> : <CopyIcon width={14} />}
                  {copied ? "Đã sao chép" : "Sao chép"}
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Done — FastAPI legacy (reply.answer + items) */}
        {turn.status === "done" && turn.reply && !turn.finalText && (
          <div className="space-y-4">
            <div className="surface rounded-2xl rounded-tl-md px-5 py-4">
              <p className="whitespace-pre-wrap text-[15px] leading-7 text-zinc-800">
                {turn.reply.answer}
              </p>
              <div className="mt-4 flex items-center justify-between border-t border-zinc-100 pt-3">
                <SourceBadge source={turn.reply.source} count={turn.reply.items.length} />
                <button
                  onClick={async () => {
                    await navigator.clipboard.writeText(turn.reply?.answer ?? "");
                    setCopied(true);
                    setTimeout(() => setCopied(false), 1600);
                  }}
                  className="inline-flex items-center gap-1.5 rounded-lg px-2 py-1 text-xs text-zinc-400 transition hover:bg-zinc-100 hover:text-zinc-700"
                >
                  {copied ? <CheckIcon width={14} /> : <CopyIcon width={14} />}
                  {copied ? "Đã sao chép" : "Sao chép"}
                </button>
              </div>
            </div>
            {turn.reply.items.length > 0 && (
              <div>
                <div className="mb-2.5 flex items-center justify-between">
                  <p className="text-xs font-semibold uppercase tracking-[.12em] text-zinc-400">
                    Nguồn tham khảo
                  </p>
                  <span className="text-xs text-zinc-400">{turn.reply.items.length} kết quả</span>
                </div>
                <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
                  {turn.reply.items.map((item, i) => (
                    <ResultCard key={item.id ?? `${turn.id}-${i}`} item={item} />
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

      </div>
    </div>
  );
}
