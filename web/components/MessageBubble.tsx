import { SOURCE_LABEL } from "@/lib/format";
import type { ChatTurn } from "@/lib/types";
import { ResultCard } from "./ResultCard";
import { SparkIcon } from "./icons";

function Avatar() {
  return (
    // eslint-disable-next-line @next/next/no-img-element
    <img src="/logo.png" alt="MIRA" className="h-8 w-8 shrink-0 rounded-lg" />
  );
}

function SourceBadge({ source }: { source: string }) {
  return (
    <span className="inline-flex items-center gap-1 rounded-full border border-zinc-200 bg-white px-2 py-0.5 text-xs font-medium text-zinc-500">
      <SparkIcon width={12} height={12} className="text-accent" />
      {SOURCE_LABEL[source] ?? source}
    </span>
  );
}

function LoadingCards() {
  return (
    <div className="mt-3 grid grid-cols-1 gap-3 sm:grid-cols-2">
      {[0, 1].map((i) => (
        <div key={i} className="overflow-hidden rounded-2xl border border-zinc-200 bg-white">
          <div className="shimmer aspect-[4/3] w-full" />
          <div className="space-y-2 p-4">
            <div className="shimmer h-3 w-3/4 rounded" />
            <div className="shimmer h-3 w-1/2 rounded" />
          </div>
        </div>
      ))}
    </div>
  );
}

export function MessageBubble({
  turn,
  onRetry,
}: {
  turn: ChatTurn;
  onRetry?: () => void;
}) {
  if (turn.role === "user") {
    return (
      <div className="flex animate-msg-in flex-col items-end gap-1.5">
        {turn.imagePreview && (
          // eslint-disable-next-line @next/next/no-img-element
          <img
            src={turn.imagePreview}
            alt="ảnh đã gửi"
            className="max-h-56 max-w-[85%] rounded-2xl rounded-br-md border border-zinc-200 object-cover"
          />
        )}
        {turn.text && (
          <div className="max-w-[85%] rounded-2xl rounded-br-md bg-zinc-900 px-4 py-2.5 text-sm leading-relaxed text-zinc-50">
            {turn.text}
          </div>
        )}
      </div>
    );
  }

  return (
    <div className="flex animate-msg-in gap-3">
      <Avatar />
      <div className="min-w-0 flex-1">
        {turn.status === "loading" && (
          <>
            <div className="flex items-center gap-1.5 text-sm text-zinc-400">
              đang tìm
              <span className="typing-dot">.</span>
              <span className="typing-dot">.</span>
              <span className="typing-dot">.</span>
            </div>
            <LoadingCards />
          </>
        )}

        {turn.status === "error" && (
          <div className="rounded-2xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
            <p>{turn.text || "Không gửi được. Kiểm tra kết nối tới máy chủ."}</p>
            {onRetry && (
              <button
                onClick={onRetry}
                className="mt-2 rounded-lg bg-red-600 px-3 py-1.5 text-xs font-medium text-white transition-transform active:scale-[0.97]"
              >
                Gửi lại
              </button>
            )}
          </div>
        )}

        {turn.status === "done" && turn.reply && (
          <div className="space-y-3">
            <div className="rounded-2xl rounded-tl-md border border-zinc-200 bg-white px-4 py-3">
              <p className="whitespace-pre-wrap text-sm leading-relaxed text-zinc-800">
                {turn.reply.answer}
              </p>
              <div className="mt-2.5">
                <SourceBadge source={turn.reply.source} />
              </div>
            </div>

            {turn.reply.items.length > 0 && (
              <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
                {turn.reply.items.map((item, i) => (
                  <ResultCard key={item.id ?? `${turn.id}-${i}`} item={item} />
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
