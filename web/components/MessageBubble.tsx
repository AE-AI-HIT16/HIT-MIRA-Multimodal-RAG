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

/**
 * Thẻ kết quả thật từ `/api/media/search` + `/api/retrieval/search`.
 *
 * Tách thành component riêng để **đặt được vào trong bong bóng trả lời**: ảnh là
 * một phần của câu trả lời chứ không phải phụ lục ở cuối. Vẫn giữ trạng thái
 * riêng (`hitsStatus`) vì hai đường chạy song song — chữ hỏng thì ảnh vẫn phải
 * xem được, và ngược lại.
 */
/** Số thẻ hiện ngay; phần còn lại nằm sau nút "Xem thêm".
 *
 * Đặt trong bong bóng trả lời nên độ dài là chuyện của câu trả lời, không phải
 * của phụ lục: đo thật một câu hỏi media thì lưới ra 16 thẻ, phần lớn là video
 * player cao ngang màn hình. 4 thẻ là 2 hàng, vừa đủ thấy mà không đẩy câu trả
 * lời tiếp theo ra khỏi tầm nhìn.
 */
const SO_THE_HIEN_TRUOC = 4;

function KhoiKetQua({ turn, trongBongBong }: { turn: ChatTurn; trongBongBong?: boolean }) {
  const [moRong, setMoRong] = useState(false);
  const [moKhoiYeu, setMoKhoiYeu] = useState(false);

  if (turn.role !== "assistant" || !turn.hitsStatus) return null;

  const tatCa = turn.hits ?? [];
  const chinh = tatCa.filter((item) => !item.lienQuanYeu);
  const yeu = tatCa.filter((item) => item.lienQuanYeu);
  const soKetQua = chinh.length;
  const hienThi = moRong ? chinh : chinh.slice(0, SO_THE_HIEN_TRUOC);
  const conLai = chinh.length - hienThi.length;

  return (
    <div className={trongBongBong ? "mt-4 border-t border-zinc-100 pt-4" : "mt-4"}>
      <div className="mb-2.5 flex items-center justify-between">
        <p className="text-xs font-semibold uppercase tracking-[.12em] text-zinc-400">
          {trongBongBong ? "Kết quả liên quan" : "Nguồn tham khảo"}
        </p>
        {turn.hitsStatus === "done" && (
          <span className="text-xs text-zinc-400">{soKetQua} kết quả</span>
        )}
      </div>

      {turn.hitsStatus === "loading" && (
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
          {[0, 1].map((i) => (
            <div
              key={i}
              className="h-44 animate-pulse rounded-2xl border border-zinc-200 bg-zinc-100"
            />
          ))}
        </div>
      )}

      {turn.hitsStatus === "error" && (
        <p className="rounded-xl border border-amber-200 bg-amber-50 px-3 py-2 text-xs text-amber-700">
          Không tải được nguồn tham khảo: {turn.hitsError ?? "lỗi không rõ"}
        </p>
      )}

      {turn.hitsStatus === "done" && soKetQua === 0 && yeu.length === 0 && (
        <p className="text-xs text-zinc-400">Không tìm thấy nguồn nào khớp.</p>
      )}

      {turn.hitsStatus === "done" && soKetQua > 0 && (
        <>
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
            {hienThi.map((item, i) => (
              <ResultCard key={`${turn.id}-hit-${i}`} item={item} />
            ))}
          </div>
          {(conLai > 0 || moRong) && (
            <button
              onClick={() => setMoRong((truoc) => !truoc)}
              className="mt-3 w-full rounded-xl border border-zinc-200 py-2 text-xs font-medium text-zinc-500 transition hover:bg-zinc-50 hover:text-zinc-700"
            >
              {moRong ? "Thu gọn" : `Xem thêm ${conLai} kết quả`}
            </button>
          )}
        </>
      )}

      {/* Nhánh nội quy trả về nguyên kho cho mọi câu hỏi (4 chunk, top_k 4), nên
          câu hỏi về ảnh vẫn kéo theo nội quy. Gấp lại chứ không vứt: điểm thấp
          là "nhiều khả năng lạc đề", không phải "chắc chắn sai". */}
      {turn.hitsStatus === "done" && yeu.length > 0 && (
        <div className={soKetQua > 0 ? "mt-3" : ""}>
          <button
            onClick={() => setMoKhoiYeu((truoc) => !truoc)}
            className="w-full rounded-xl border border-dashed border-zinc-200 py-2 text-xs text-zinc-400 transition hover:bg-zinc-50 hover:text-zinc-600"
          >
            {moKhoiYeu
              ? "Ẩn các mục khớp yếu"
              : `${yeu.length} mục nội quy khớp yếu — có thể không liên quan`}
          </button>
          {moKhoiYeu && (
            <div className="mt-3 grid grid-cols-1 gap-3 sm:grid-cols-2">
              {yeu.map((item, i) => (
                <ResultCard key={`${turn.id}-yeu-${i}`} item={item} />
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export function MessageBubble({ turn, onRetry }: { turn: ChatTurn; onRetry?: () => void }) {
  const [copied, setCopied] = useState(false);

  // Có bong bóng trả lời thì ảnh nằm trong đó; lúc chưa có (đang chờ, hoặc câu
  // trả lời lỗi) thì vẫn phải hiện độc lập, nếu không mất luôn phần bằng chứng.
  const coBongBongTraLoi =
    turn.status === "streaming" || (turn.status === "done" && !!turn.finalText);

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
            {/* Nhánh truy xuất thường xong trước stream, nên ảnh hiện ngay
                trong lúc chữ còn đang chạy. */}
            <KhoiKetQua turn={turn} trongBongBong />
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
              <KhoiKetQua turn={turn} trongBongBong />
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

        {/* Chưa có bong bóng để đặt vào (đang chờ token đầu, hoặc câu trả lời
            lỗi) thì hiện độc lập — bằng chứng không được biến mất theo chữ. */}
        {!coBongBongTraLoi && <KhoiKetQua turn={turn} />}

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
