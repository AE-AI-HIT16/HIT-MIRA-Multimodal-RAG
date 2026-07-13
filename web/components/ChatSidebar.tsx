"use client";

import { CloseIcon, MessageIcon, PlusIcon, TrashIcon } from "./icons";

export interface ChatSessionSummary {
  id: string;
  title: string;
  updatedAt: number;
}

function relativeTime(time: number) {
  const diff = Date.now() - time;
  if (diff < 60_000) return "Vừa xong";
  if (diff < 3_600_000) return `${Math.floor(diff / 60_000)} phút trước`;
  if (diff < 86_400_000) return `${Math.floor(diff / 3_600_000)} giờ trước`;
  return new Intl.DateTimeFormat("vi-VN", { day: "2-digit", month: "2-digit" }).format(time);
}

export function ChatSidebar({
  sessions,
  activeId,
  open,
  onClose,
  onNew,
  onSelect,
  onDelete,
}: {
  sessions: ChatSessionSummary[];
  activeId: string;
  open: boolean;
  onClose: () => void;
  onNew: () => void;
  onSelect: (id: string) => void;
  onDelete: (id: string) => void;
}) {
  return (
    <>
      {open && <button aria-label="Đóng lịch sử" onClick={onClose} className="fixed inset-0 z-30 bg-zinc-950/30 backdrop-blur-sm lg:hidden" />}
      <aside className={`fixed bottom-0 left-0 top-16 z-40 flex w-[286px] min-h-0 flex-col border-r border-zinc-200/80 bg-white transition-transform lg:sticky lg:top-16 lg:z-10 lg:h-[calc(100dvh-4rem)] lg:translate-x-0 ${open ? "translate-x-0" : "-translate-x-full"}`}>
        <div className="flex items-center gap-2 p-4">
          <button onClick={onNew} className="flex flex-1 items-center justify-center gap-2 rounded-xl bg-zinc-900 px-4 py-2.5 text-sm font-medium text-white shadow-sm transition hover:bg-zinc-800 active:scale-[.98]">
            <PlusIcon width={16} height={16} /> Cuộc trò chuyện mới
          </button>
          <button onClick={onClose} aria-label="Đóng" className="grid h-10 w-10 place-items-center rounded-xl text-zinc-500 hover:bg-zinc-100 lg:hidden"><CloseIcon /></button>
        </div>
        <div className="px-4 pb-2 pt-3 text-[11px] font-semibold uppercase tracking-[.14em] text-zinc-400">Gần đây</div>
        <div className="scroll-slim min-h-0 flex-1 space-y-1 overflow-y-auto px-2 pb-4">
          {sessions.length === 0 ? (
            <div className="mx-2 rounded-xl border border-dashed border-zinc-200 px-4 py-8 text-center">
              <MessageIcon className="mx-auto text-zinc-300" />
              <p className="mt-2 text-xs leading-relaxed text-zinc-400">Lịch sử hội thoại sẽ xuất hiện tại đây.</p>
            </div>
          ) : sessions.map((session) => (
            <div key={session.id} className={`group flex items-center rounded-xl transition ${activeId === session.id ? "bg-accent-soft" : "hover:bg-zinc-100"}`}>
              <button onClick={() => { onSelect(session.id); onClose(); }} className="min-w-0 flex-1 px-3 py-2.5 text-left">
                <p className={`truncate text-sm ${activeId === session.id ? "font-medium text-accent-ink" : "text-zinc-700"}`}>{session.title}</p>
                <p className="mt-0.5 text-[11px] text-zinc-400">{relativeTime(session.updatedAt)}</p>
              </button>
              <button onClick={() => onDelete(session.id)} aria-label="Xóa hội thoại" className="mr-2 grid h-8 w-8 shrink-0 place-items-center rounded-lg text-zinc-400 opacity-0 transition hover:bg-white hover:text-red-600 group-hover:opacity-100 focus:opacity-100"><TrashIcon width={15} height={15} /></button>
            </div>
          ))}
        </div>
        <div className="shrink-0 border-t border-zinc-100 bg-white px-4 pb-[max(1rem,env(safe-area-inset-bottom))] pt-3 text-[11px] leading-5 text-zinc-400">
          Lịch sử lưu trên trình duyệt này.
        </div>
      </aside>
    </>
  );
}
