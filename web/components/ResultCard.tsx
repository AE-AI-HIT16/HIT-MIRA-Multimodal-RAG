"use client";

import { useState } from "react";
import { clipUrl, frameUrl, mediaUrl, streamUrl } from "@/lib/api";
import { mmss } from "@/lib/format";
import type { ChatItem } from "@/lib/types";
import { BookIcon, ClockIcon, LinkIcon, PlayIcon } from "./icons";
import { MediaImage } from "./MediaImage";

const CLAMP_CHARS = 170; // caption dài hơn mức này → cắt 3 dòng + "Xem thêm"
const MAX_CHIPS = 5; // số mốc thời gian hiện mặc định, còn lại gập sau "+N"

function SourceLink({ url }: { url?: string | null }) {
  if (!url) {
    return <span className="text-xs text-zinc-400">Nguồn nội bộ</span>;
  }
  return (
    <a
      href={url}
      target="_blank"
      rel="noreferrer"
      className="inline-flex items-center gap-1 text-xs text-accent transition-colors hover:text-accent-ink"
    >
      <LinkIcon width={13} height={13} />
      Bài gốc
    </a>
  );
}

// Caption bài đăng có thể là cả bức tường chữ → clamp 3 dòng, bấm mở/thu gọn.
function ClampText({ text, fallback }: { text?: string; fallback: string }) {
  const [open, setOpen] = useState(false);
  const t = text?.trim();
  if (!t) {
    return <p className="text-sm leading-snug text-zinc-400">{fallback}</p>;
  }
  const long = t.length > CLAMP_CHARS;
  return (
    <div>
      <p
        className={`whitespace-pre-line text-sm leading-snug text-zinc-700 ${
          long && !open ? "line-clamp-3" : ""
        }`}
      >
        {t}
      </p>
      {long && (
        <button
          onClick={() => setOpen(!open)}
          className="mt-1 text-xs font-medium text-accent transition-colors hover:text-accent-ink"
        >
          {open ? "Thu gọn" : "Xem thêm"}
        </button>
      )}
    </div>
  );
}

// Hàng chip mốc thời gian: tối đa MAX_CHIPS, còn lại gập sau "+N".
function TimeChips({
  times,
  activeTs,
  onPick,
}: {
  times: number[];
  activeTs?: number;
  onPick: (ts: number) => void;
}) {
  const [all, setAll] = useState(false);
  if (times.length === 0) return null;
  const shown = all ? times : times.slice(0, MAX_CHIPS);
  const hidden = times.length - shown.length;
  return (
    <div className="flex flex-wrap items-center gap-1.5">
      <span className="text-xs text-zinc-400">Mốc khớp:</span>
      {shown.map((ts) => (
        <button
          key={ts}
          onClick={() => onPick(ts)}
          className={`rounded-full px-2.5 py-1 font-mono text-xs transition-colors ${
            ts === activeTs
              ? "bg-accent text-white"
              : "bg-accent-soft text-accent-ink hover:bg-accent-ring/40"
          }`}
        >
          {mmss(ts)}
        </button>
      ))}
      {hidden > 0 && (
        <button
          onClick={() => setAll(true)}
          className="rounded-full border border-zinc-200 px-2.5 py-1 font-mono text-xs text-zinc-500 transition-colors hover:border-accent-ring hover:text-accent-ink"
        >
          +{hidden}
        </button>
      )}
      {all && times.length > MAX_CHIPS && (
        <button
          onClick={() => setAll(false)}
          className="text-xs text-zinc-400 transition-colors hover:text-zinc-600"
        >
          thu gọn
        </button>
      )}
    </div>
  );
}

// Card frame video: thumbnail keyframe (ảnh tĩnh, không tốn ffmpeg) → bấm mới nạp clip.
// Chip mốc thời gian seek NGAY TRONG CARD (đổi clip), không mở tab mới.
function VideoFrameCard({ item }: { item: ChatItem }) {
  const vid = (item.video_id ?? item.id) as number;
  const [activeTs, setActiveTs] = useState(item.timestamp as number);
  const [playing, setPlaying] = useState(false);
  const [thumbFailed, setThumbFailed] = useState(false);

  const times = (item.frames ?? []).map((f) => f.timestamp);
  if (times.length === 0) times.push(item.timestamp as number);

  function pick(ts: number) {
    setActiveTs(ts);
    setThumbFailed(false);
  }

  return (
    <article className="overflow-hidden rounded-2xl border border-zinc-200 bg-white">
      {playing ? (
        <video
          key={activeTs}
          controls
          autoPlay
          preload="auto"
          className="aspect-video w-full bg-black"
        >
          <source src={clipUrl(vid, activeTs)} type="video/mp4" />
        </video>
      ) : (
        <button
          onClick={() => setPlaying(true)}
          className="group relative block aspect-video w-full bg-zinc-900"
          aria-label={`Phát clip tại ${mmss(activeTs)}`}
        >
          {!thumbFailed && (
            // eslint-disable-next-line @next/next/no-img-element
            <img
              src={frameUrl(vid, activeTs)}
              alt={`Khung hình tại ${mmss(activeTs)}`}
              loading="lazy"
              onError={() => setThumbFailed(true)}
              className="absolute inset-0 h-full w-full object-cover"
            />
          )}
          <span className="absolute inset-0 flex items-center justify-center bg-black/20 transition-colors group-hover:bg-black/30">
            <span className="flex h-12 w-12 items-center justify-center rounded-full bg-white/90 text-accent shadow-md transition-transform group-hover:scale-105 group-active:scale-95">
              <PlayIcon width={20} height={20} />
            </span>
          </span>
          <span className="absolute bottom-2 right-2 rounded-md bg-black/60 px-1.5 py-0.5 font-mono text-xs text-white">
            {mmss(activeTs)}
          </span>
        </button>
      )}

      <div className="space-y-2 p-4">
        <div className="flex items-start justify-between gap-2">
          <ClampText
            text={item.caption}
            fallback={`Video #${vid} · khớp tại ${mmss(activeTs)}`}
          />
          <a
            href={`${streamUrl(vid)}#t=${activeTs}`}
            target="_blank"
            rel="noreferrer"
            className="shrink-0 whitespace-nowrap text-xs text-accent transition-colors hover:text-accent-ink"
          >
            Toàn video
          </a>
        </div>
        {times.length > 1 && (
          <TimeChips
            times={times}
            activeTs={activeTs}
            onPick={(ts) => pick(ts)}
          />
        )}
      </div>
    </article>
  );
}

// Một item kết quả → card theo loại (ảnh / frame video / transcript / nội quy).
export function ResultCard({ item }: { item: ChatItem }) {
  // 1) Transcript video (gộp theo video_id, có moments)
  if (item.moments && item.moments.length > 0) {
    const vid = item.video_id ?? item.id;
    const starts = item.moments
      .map((m) => m.start_sec)
      .filter((s): s is number => s != null);
    return (
      <article className="overflow-hidden rounded-2xl border border-zinc-200 bg-white">
        {vid != null && (
          <video controls preload="metadata" className="aspect-video w-full bg-black">
            <source src={streamUrl(vid)} />
          </video>
        )}
        <div className="space-y-2 p-4">
          <div className="flex items-center gap-1.5 text-xs font-medium text-zinc-500">
            <ClockIcon width={13} height={13} /> Lời nói trong video
          </div>
          <div className="flex flex-wrap gap-1.5">
            {starts.slice(0, MAX_CHIPS + 1).map((s, i) => (
              <a
                key={i}
                href={vid != null ? `${streamUrl(vid)}#t=${s}` : undefined}
                target="_blank"
                rel="noreferrer"
                className="rounded-full bg-accent-soft px-2.5 py-1 font-mono text-xs text-accent-ink transition-colors hover:bg-accent-ring/40"
              >
                {mmss(s)}
              </a>
            ))}
          </div>
          {item.moments[0]?.text && (
            <ClampText text={`“${item.moments[0].text}”`} fallback="" />
          )}
        </div>
      </article>
    );
  }

  // 2) Nội quy (điều/khoản)
  if (item.article != null || item.clause != null) {
    const label =
      item.article != null && item.clause != null
        ? `Điều ${item.article} · Khoản ${item.clause}`
        : item.article != null
          ? `Điều ${item.article}`
          : "Nội quy";
    return (
      <article className="rounded-2xl border border-zinc-200 bg-white p-4">
        <div className="mb-2 inline-flex items-center gap-1.5 rounded-md bg-zinc-100 px-2 py-1 text-xs font-semibold text-zinc-700">
          <BookIcon width={13} height={13} /> {label}
        </div>
        <blockquote className="border-l-2 border-accent-ring pl-3 text-sm leading-relaxed text-zinc-700">
          {item.text}
        </blockquote>
      </article>
    );
  }

  // 3) Frame video khớp (có timestamp) — thumbnail + seek trong card
  if (item.timestamp != null && (item.video_id != null || item.id != null)) {
    return <VideoFrameCard item={item} />;
  }

  // 4) Ảnh
  const id = item.media_id ?? item.id;
  return (
    <article className="overflow-hidden rounded-2xl border border-zinc-200 bg-white">
      {id != null && (
        <a href={mediaUrl(id)} target="_blank" rel="noreferrer">
          <MediaImage src={mediaUrl(id)} alt={item.caption || "ảnh kết quả"} />
        </a>
      )}
      <div className="flex items-start justify-between gap-3 p-4">
        <div className="min-w-0 flex-1">
          {(item.event || item.year) && (
            <div className="mb-1.5 inline-flex items-center rounded-md bg-zinc-100 px-2 py-0.5 text-xs font-medium text-zinc-600">
              {[item.event, item.year].filter(Boolean).join(" · ")}
            </div>
          )}
          <ClampText text={item.caption} fallback="Chưa có mô tả" />
        </div>
        <div className="shrink-0 pt-0.5">
          <SourceLink url={item.source_url} />
        </div>
      </div>
    </article>
  );
}
