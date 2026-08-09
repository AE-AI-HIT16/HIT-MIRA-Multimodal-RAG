"use client";

import { useState } from "react";
import { mediaFileUrl, videoUrl } from "@/lib/api";
import { mmss } from "@/lib/format";
import type { ChatItem } from "@/lib/types";
import { BookIcon, ClockIcon, LinkIcon, PlayIcon } from "./icons";
import { MediaImage } from "./MediaImage";
import { RichText } from "./RichText";

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

/**
 * Số trích dẫn `[n]` — cùng con số agent thấy trong `context` khi viết câu trả lời.
 *
 * Chỉ đúng khi thẻ được dựng từ chính payload agent đã đọc, nên `soTrichDan` để
 * trống ở nhánh tra lại bằng REST và huy hiệu này tự biến mất theo.
 */
function SoTrichDan({ so, tren }: { so?: number[]; tren?: "anh" }) {
  if (!so || so.length === 0) return null;
  // Thẻ gộp mang nhiều số: rút gọn dải liền mạch để "[5][6][7][8]" không chiếm
  // mất góc ảnh, nhưng dải đứt quãng thì in đủ chứ không giấu số nào.
  const lienMach = so.length > 1 && so[so.length - 1] - so[0] === so.length - 1;
  const nhan = lienMach ? `${so[0]}–${so[so.length - 1]}` : so.join(", ");
  return (
    <span
      className={
        tren === "anh"
          ? "absolute left-2 top-2 rounded-md bg-black/60 px-1.5 py-0.5 font-mono text-xs text-white"
          : "shrink-0 rounded-md bg-zinc-100 px-1.5 py-0.5 font-mono text-xs text-zinc-500"
      }
    >
      [{nhan}]
    </span>
  );
}

// Card frame video: thumbnail keyframe (ảnh tĩnh, không tốn ffmpeg) → bấm mới nạp clip.
// Chip mốc thời gian seek NGAY TRONG CARD (đổi clip), không mở tab mới.
function VideoFrameCard({ item }: { item: ChatItem }) {
  const vid = item.video_uid ?? null;
  const [activeTs, setActiveTs] = useState(item.timestamp as number);
  const [playing, setPlaying] = useState(false);
  const [thumbFailed, setThumbFailed] = useState(false);

  const times = (item.frames ?? []).map((f) => f.timestamp);
  if (times.length === 0) times.push(item.timestamp as number);

  // Thumbnail phải là keyframe của ĐÚNG mốc đang chọn. Bám vào `item.object_key`
  // thì đổi chip xong ảnh vẫn là khung cũ, tức là thẻ nói "02:15" mà cho xem
  // hình ở 00:40 — vẫn còn nhìn thấy được nhưng đã là chú thích sai.
  const khungHienTai = (item.frames ?? []).find((f) => f.timestamp === activeTs);
  const anhKey = khungHienTai?.frame_path ?? item.object_key;

  function pick(ts: number) {
    setActiveTs(ts);
    setThumbFailed(false);
  }

  return (
    <article className="overflow-hidden rounded-2xl border border-zinc-200 bg-white shadow-sm transition-shadow hover:shadow-md">
      {playing && vid ? (
        // Không cắt clip: phát video gốc và tua bằng fragment #t=. MinIO trả
        // 206 Partial Content nên trình duyệt nhảy thẳng tới đúng giây.
        <video
          key={activeTs}
          controls
          autoPlay
          preload="auto"
          className="aspect-video w-full bg-black"
          src={videoUrl(vid, activeTs)}
        />
      ) : (
        <button
          onClick={() => setPlaying(true)}
          className="group relative block aspect-video w-full bg-zinc-900"
          aria-label={`Phát clip tại ${mmss(activeTs)}`}
        >
          {!thumbFailed && anhKey && (
            // Thumbnail là chính keyframe đã trích sẵn trong MinIO, không phải
            // ảnh sinh ra lúc chạy — nên không tốn ffmpeg và luôn khớp mốc giây.
            // eslint-disable-next-line @next/next/no-img-element
            <img
              key={anhKey}
              src={mediaFileUrl(anhKey)}
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
          {/* Sau lớp phủ để số không bị nhuộm tối theo. */}
          <SoTrichDan so={item.soTrichDan} tren="anh" />
          <span className="absolute bottom-2 right-2 rounded-md bg-black/60 px-1.5 py-0.5 font-mono text-xs text-white">
            {mmss(activeTs)}
          </span>
        </button>
      )}

      {/* Không hiện caption: mô tả do VLM sinh ra, còn ở đây khung hình đã là câu
          trả lời. Caption vẫn nằm nguyên trong payload/Qdrant vì truy xuất và
          agent đọc nó — đây chỉ là tầng hiển thị. Cả khối tự biến mất khi không
          có gì để hiện, chứ không để lại một dải trắng cao 4rem. */}
      {(vid || times.length > 1 || item.source_url) && (
        <div className="flex flex-wrap items-center gap-2 px-4 py-3">
          {times.length > 1 && (
            <div className="min-w-0 flex-1">
              <TimeChips
                times={times}
                activeTs={activeTs}
                onPick={(ts) => pick(ts)}
              />
            </div>
          )}
          {/* Hai đích khác nhau, đứng cạnh nhau nên phải đọc ra được là khác:
              "Toàn video" mở file trong MinIO, "Bài gốc" mở bài đăng ngoài. */}
          <div className="ml-auto flex shrink-0 items-center gap-3">
            {vid && (
              <a
                href={videoUrl(vid, activeTs)}
                target="_blank"
                rel="noreferrer"
                className="whitespace-nowrap text-xs text-accent transition-colors hover:text-accent-ink"
              >
                Toàn video
              </a>
            )}
            {/* Khác thẻ ảnh: ở đây KHÔNG hiện dự phòng "Nguồn nội bộ". Điểm
                index trước US-405.1 không có `source_url` (retriever.py:248),
                mà thẻ video đã có "Toàn video" nói rõ nguồn là file nội bộ. */}
            {item.source_url && <SourceLink url={item.source_url} />}
          </div>
        </div>
      )}
    </article>
  );
}

/**
 * Nội quy: dải trích dẫn gập được, KHÔNG phải thẻ lớn như ảnh.
 *
 * Ảnh thì bản thân nó là câu trả lời nên thẻ to là đúng. Nội quy thì ngược lại —
 * câu trả lời đã nằm ở phần chữ phía trên, đoạn văn bản ở đây chỉ là căn cứ để
 * đối chiếu. Đổ cả mục ra màn hình (đo được 969 ký tự cho mục dài nhất) làm câu
 * trả lời thật bị đẩy khuất, và người đọc phải tự dò xem câu nào là câu liên quan.
 *
 * Gập sẵn, bấm mới mở. Thân mục đi qua `RichText` nên gạch đầu dòng ra đúng danh
 * sách thay vì dính thành một đoạn văn liền — chuỗi có sẵn ký tự xuống dòng, chỉ
 * là trước đây bị nuốt mất.
 */
function RegulationCard({ label, text }: { label: string; text: string }) {
  const [mo, setMo] = useState(false);
  const coThan = text.trim().length > 0;

  return (
    <article className="overflow-hidden rounded-xl border border-zinc-200 bg-white transition-shadow hover:shadow-sm">
      <button
        type="button"
        onClick={() => coThan && setMo((truoc) => !truoc)}
        disabled={!coThan}
        aria-expanded={mo}
        className="flex w-full items-center gap-2.5 px-3.5 py-2.5 text-left transition-colors hover:bg-zinc-50 disabled:cursor-default disabled:hover:bg-transparent"
      >
        <BookIcon width={14} height={14} className="shrink-0 text-zinc-400" />
        <span className="min-w-0 flex-1 text-sm font-medium leading-snug text-zinc-800">
          {label}
        </span>
        {coThan && (
          <span className="shrink-0 text-xs text-zinc-400">
            {mo ? "Thu gọn" : "Xem"}
          </span>
        )}
      </button>
      {mo && coThan && (
        <div className="border-t border-zinc-100 px-3.5 pb-3.5 pt-3">
          <div className="border-l-2 border-accent-ring pl-3 text-sm">
            <RichText text={text} />
          </div>
        </div>
      )}
    </article>
  );
}

// Một item kết quả → card theo loại (ảnh / frame video / transcript / nội quy).
export function ResultCard({ item }: { item: ChatItem }) {
  // 1) Transcript video (gộp theo video_id, có moments)
  if (item.moments && item.moments.length > 0) {
    const vid = item.video_uid ?? null;
    const starts = item.moments
      .map((m) => m.start_sec)
      .filter((s): s is number => s != null);
    return (
      <article className="overflow-hidden rounded-2xl border border-zinc-200 bg-white shadow-sm transition-shadow hover:shadow-md">
        {vid && (
          <video
            controls
            preload="metadata"
            className="aspect-video w-full bg-black"
            src={videoUrl(vid)}
          />
        )}
        <div className="space-y-2 px-4 py-3">
          <div className="flex items-center gap-1.5 text-xs font-medium text-zinc-500">
            <ClockIcon width={13} height={13} /> Lời nói trong video
            <SoTrichDan so={item.soTrichDan} />
            {/* Thẻ này đã nhúng sẵn trình phát nên không có "Toàn video";
                "Bài gốc" là hành động duy nhất, đặt cuối hàng nhãn. */}
            {item.source_url && (
              <span className="ml-auto">
                <SourceLink url={item.source_url} />
              </span>
            )}
          </div>
          {/* Đoạn khớp có thể thiếu `start_sec` (đã lọc ở trên): không còn mốc
              nào thì bỏ luôn hàng chip, chứ một `div` rỗng vẫn ăn khoảng cách
              của `space-y-2`. */}
          {starts.length > 0 && (
            <div className="flex flex-wrap gap-1.5">
              {starts.slice(0, MAX_CHIPS + 1).map((s, i) => (
                <a
                  key={i}
                  href={vid ? videoUrl(vid, s) : undefined}
                  target="_blank"
                  rel="noreferrer"
                  className="rounded-full bg-accent-soft px-2.5 py-1 font-mono text-xs text-accent-ink transition-colors hover:bg-accent-ring/40"
                >
                  {mmss(s)}
                </a>
              ))}
            </div>
          )}
          {/* Không in lời thoại: cùng lý do với caption ở hai thẻ dưới — thẻ này
              để xem và đối chiếu, không phải để đọc. Văn bản vẫn đi tới agent
              qua `context`, và mốc thời gian vẫn dẫn thẳng tới đúng giây đã nói
              câu đó, nên số `[n]` vẫn kiểm chứng được. */}
        </div>
      </article>
    );
  }

  // 2) Nội quy (điều/khoản)
  if (item.label || item.article != null || item.clause != null) {
    // Nội quy thật không đánh số điều/khoản mà chia theo mục có tiêu đề, nên
    // ưu tiên nhãn chuỗi; nhánh số giữ lại cho dữ liệu kiểu cũ.
    const label =
      item.label ??
      (item.article != null && item.clause != null
        ? `Điều ${item.article} · Khoản ${item.clause}`
        : item.article != null
          ? `Điều ${item.article}`
          : "Nội quy");
    return <RegulationCard label={label} text={item.text ?? ""} />;
  }

  // 3) Frame video khớp (có timestamp) — thumbnail + seek trong card
  if (item.timestamp != null && item.video_uid) {
    return <VideoFrameCard item={item} />;
  }

  // 4) Ảnh
  const src = item.object_key ? mediaFileUrl(item.object_key) : null;
  return (
    <article className="overflow-hidden rounded-2xl border border-zinc-200 bg-white shadow-sm transition-shadow hover:shadow-md">
      {src && (
        <a href={src} target="_blank" rel="noreferrer" className="relative block">
          <MediaImage src={src} alt={item.caption || "ảnh kết quả"} />
          <SoTrichDan so={item.soTrichDan} tren="anh" />
        </a>
      )}
      {/* Không hiện caption — xem chú thích ở `VideoFrameCard`. Chân thẻ co lại
          thành một hàng: nhãn sự kiện/năm bên trái, link bài gốc bên phải.
          `items-center` chứ không `items-start` vì không còn khối chữ nhiều dòng
          để căn theo. */}
      <div className="flex items-center gap-3 px-4 py-3">
        {(item.event || item.year) && (
          <div className="min-w-0 truncate rounded-md bg-zinc-100 px-2 py-0.5 text-xs font-medium text-zinc-600">
            {[item.event, item.year].filter(Boolean).join(" · ")}
          </div>
        )}
        <div className="ml-auto shrink-0">
          <SourceLink url={item.source_url} />
        </div>
      </div>
    </article>
  );
}
