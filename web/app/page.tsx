"use client";

import { useEffect, useRef, useState } from "react";
import { ChatSidebar, type ChatSessionSummary } from "@/components/ChatSidebar";
import { Composer } from "@/components/Composer";
import { EmptyState } from "@/components/EmptyState";
import { MessageBubble } from "@/components/MessageBubble";
import { TopBar } from "@/components/TopBar";
import { MenuIcon } from "@/components/icons";
import {
  buildHumanMessage,
  extractTextFromLGMessage,
  lgCreateThread,
  lgStream,
} from "@/lib/api";
import type { ChatTurn, LGMessage, Override } from "@/lib/types";

type Mode = "auto" | Override;
type StoredSession = ChatSessionSummary & {
  turns: ChatTurn[];
  threadId: string | null;
};

const STORAGE_KEY = "hit_mira_chat_sessions_v2";
const freshId = () => `chat-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`;

export default function ChatPage() {
  const [turns, setTurns] = useState<ChatTurn[]>([]);
  const [input, setInput] = useState("");
  const [mode, setMode] = useState<Mode>("auto");
  const [busy, setBusy] = useState(false);
  const [threadId, setThreadId] = useState<string | null>(null);
  const [image, setImage] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [sessions, setSessions] = useState<StoredSession[]>([]);
  const [activeId, setActiveId] = useState(freshId);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [hydrated, setHydrated] = useState(false);
  const idRef = useRef(0);
  const bottomRef = useRef<HTMLDivElement>(null);
  const abortRef = useRef<AbortController | null>(null);

  const nextId = () => `t${Date.now()}-${++idRef.current}`;

  // Khôi phục sessions từ localStorage
  useEffect(() => {
    try {
      const saved = JSON.parse(
        localStorage.getItem(STORAGE_KEY) || "[]",
      ) as StoredSession[];
      setSessions(saved);
      if (saved[0]) {
        setActiveId(saved[0].id);
        setTurns(saved[0].turns);
        setThreadId(saved[0].threadId);
      }
    } catch {
      localStorage.removeItem(STORAGE_KEY);
    }
    setHydrated(true);
  }, []);

  // Lưu sessions vào localStorage khi turns thay đổi
  useEffect(() => {
    if (!hydrated || turns.length === 0) return;
    setSessions((prev) => {
      const firstQuestion =
        turns.find((t) => t.role === "user")?.text || "Tra cứu bằng hình ảnh";
      // Xoá imagePreview trước khi lưu (objectURL không serialize được)
      const cleanTurns = turns.map((t) => ({
        ...t,
        imagePreview: undefined,
        streamText: undefined, // không lưu text streaming dở
      }));
      const current: StoredSession = {
        id: activeId,
        title: firstQuestion.slice(0, 54),
        updatedAt: Date.now(),
        turns: cleanTurns,
        threadId,
      };
      const next = [current, ...prev.filter((s) => s.id !== activeId)].slice(0, 30);
      localStorage.setItem(STORAGE_KEY, JSON.stringify(next));
      return next;
    });
  }, [turns, threadId, activeId, hydrated]);

  // Auto-scroll xuống cuối
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [turns]);

  function pickImage(f: File | null) {
    setImagePreview((prev) => {
      if (prev) URL.revokeObjectURL(prev);
      return f ? URL.createObjectURL(f) : null;
    });
    setImage(f);
  }

  function newChat() {
    if (busy) {
      abortRef.current?.abort();
    }
    setActiveId(freshId());
    setTurns([]);
    setThreadId(null);
    setInput("");
    pickImage(null);
    setSidebarOpen(false);
  }

  function selectSession(id: string) {
    if (busy) return;
    const session = sessions.find((s) => s.id === id);
    if (!session) return;
    setActiveId(id);
    setTurns(session.turns);
    setThreadId(session.threadId);
    setInput("");
    pickImage(null);
  }

  function deleteSession(id: string) {
    const next = sessions.filter((s) => s.id !== id);
    setSessions(next);
    localStorage.setItem(STORAGE_KEY, JSON.stringify(next));
    if (id === activeId) newChat();
  }

  /**
   * Hàm gửi message chính — kết nối LangGraph qua SSE stream.
   */
  async function send() {
    const q = input.trim();
    const img = image;
    if ((!q && !img) || busy) return;

    const preview = imagePreview ?? undefined;
    setInput("");
    if (img) pickImage(null);

    const userTurnId = nextId();
    const asstTurnId = nextId();
    const ov: Override | undefined = mode === "auto" ? undefined : mode;

    // Thêm lượt user + placeholder assistant
    setTurns((prev) => [
      ...prev,
      {
        id: userTurnId,
        role: "user",
        text: q,
        imagePreview: img ? preview : undefined,
        status: "done",
      },
      {
        id: asstTurnId,
        role: "assistant",
        text: "",
        streamText: "",
        status: "loading",
        query: q,
      },
    ]);

    setBusy(true);
    const controller = new AbortController();
    abortRef.current = controller;

    try {
      // 1. Tạo thread nếu chưa có (cuộc trò chuyện mới)
      let tid = threadId;
      if (!tid) {
        tid = await lgCreateThread();
        setThreadId(tid);
      }

      // 2. Build message (text + ảnh base64 nếu có)
      const humanMsg = await buildHumanMessage(q, img);

      // 3. Stream
      let accumulated = "";
      let gotFirstToken = false;

      for await (const chunk of lgStream(tid, [humanMsg], ov, controller.signal)) {
        // stream_mode="messages" → event = "messages/partial" hoặc "messages/complete"
        if (
          chunk.event === "messages/partial" ||
          chunk.event === "messages/complete"
        ) {
          const dataArr = Array.isArray(chunk.data) ? chunk.data : [chunk.data];
          const msg = dataArr.find((m: any) => m && m.type === "ai") as LGMessage | undefined;

          // Chỉ lấy message của AI (type = "ai"), bỏ qua tool messages
          if (!msg) continue;

          const text = extractTextFromLGMessage(msg);
          if (!text) continue;

          accumulated = text; // LangGraph gửi toàn bộ text tích luỹ, không chỉ delta

          if (!gotFirstToken) {
            gotFirstToken = true;
          }

          setTurns((prev) =>
            prev.map((t) =>
              t.id === asstTurnId
                ? { ...t, status: "streaming", streamText: accumulated }
                : t,
            ),
          );
        }
      }

      // 4. Stream xong → set done
      setTurns((prev) =>
        prev.map((t) =>
          t.id === asstTurnId
            ? {
                ...t,
                status: "done",
                streamText: undefined,
                finalText: accumulated || "(Không có phản hồi)",
                text: accumulated || "(Không có phản hồi)",
              }
            : t,
        ),
      );
    } catch (e) {
      if ((e as Error).name === "AbortError") {
        // Người dùng cancel — giữ nguyên những gì đã stream
        setTurns((prev) =>
          prev.map((t) =>
            t.id === asstTurnId && t.status === "streaming"
              ? {
                  ...t,
                  status: "done",
                  finalText: (t.streamText ?? "") + " _(đã dừng)_",
                  text: t.streamText ?? "",
                  streamText: undefined,
                }
              : t,
          ),
        );
      } else {
        const msg = e instanceof Error ? e.message : "Có lỗi xảy ra";
        setTurns((prev) =>
          prev.map((t) =>
            t.id === asstTurnId
              ? { ...t, status: "error", text: msg, streamText: undefined }
              : t,
          ),
        );
      }
    } finally {
      setBusy(false);
      abortRef.current = null;
    }
  }

  /** Gửi lại câu hỏi khi lỗi */
  async function retry(asstTurnId: string, query: string) {
    if (busy) return;
    const ov: Override | undefined = mode === "auto" ? undefined : mode;

    setTurns((prev) =>
      prev.map((t) =>
        t.id === asstTurnId
          ? { ...t, status: "loading", text: "", streamText: "" }
          : t,
      ),
    );
    setBusy(true);
    const controller = new AbortController();
    abortRef.current = controller;

    try {
      let tid = threadId;
      if (!tid) {
        tid = await lgCreateThread();
        setThreadId(tid);
      }

      const humanMsg = await buildHumanMessage(query);
      let accumulated = "";

      for await (const chunk of lgStream(tid, [humanMsg], ov, controller.signal)) {
        if (
          chunk.event === "messages/partial" ||
          chunk.event === "messages/complete"
        ) {
          const dataArr = Array.isArray(chunk.data) ? chunk.data : [chunk.data];
          const msg = dataArr.find((m: any) => m && m.type === "ai") as LGMessage | undefined;
          if (!msg) continue;
          const text = extractTextFromLGMessage(msg);
          if (!text) continue;
          accumulated = text;
          setTurns((prev) =>
            prev.map((t) =>
              t.id === asstTurnId
                ? { ...t, status: "streaming", streamText: accumulated }
                : t,
            ),
          );
        }
      }

      setTurns((prev) =>
        prev.map((t) =>
          t.id === asstTurnId
            ? {
                ...t,
                status: "done",
                streamText: undefined,
                finalText: accumulated || "(Không có phản hồi)",
                text: accumulated || "(Không có phản hồi)",
              }
            : t,
        ),
      );
    } catch (e) {
      const msg = e instanceof Error ? e.message : "Có lỗi xảy ra";
      setTurns((prev) =>
        prev.map((t) =>
          t.id === asstTurnId
            ? { ...t, status: "error", text: msg, streamText: undefined }
            : t,
        ),
      );
    } finally {
      setBusy(false);
      abortRef.current = null;
    }
  }

  return (
    <div className="min-h-[100dvh] bg-[var(--bg)]">
      <TopBar />
      <div className="mx-auto flex max-w-7xl">
        <ChatSidebar
          sessions={sessions}
          activeId={activeId}
          open={sidebarOpen}
          onClose={() => setSidebarOpen(false)}
          onNew={newChat}
          onSelect={selectSession}
          onDelete={deleteSession}
        />
        <section className="relative flex min-h-[calc(100dvh-4rem)] min-w-0 flex-1 flex-col">
          <div className="app-grid pointer-events-none absolute inset-x-0 top-0 h-72 opacity-40" />
          <div className="relative mx-auto w-full max-w-4xl flex-1 px-4 pb-8 sm:px-8">
            <div className="flex h-14 items-center justify-between border-b border-zinc-200/70 lg:hidden">
              <button
                onClick={() => setSidebarOpen(true)}
                className="flex items-center gap-2 rounded-lg px-2 py-1.5 text-sm text-zinc-600 hover:bg-white"
              >
                <MenuIcon width={18} /> Lịch sử
              </button>
              <button
                onClick={newChat}
                className="text-sm font-medium text-accent-ink"
              >
                Cuộc trò chuyện mới
              </button>
            </div>

            {turns.length === 0 ? (
              <EmptyState onPick={(q) => setInput(q)} />
            ) : (
              <div className="space-y-8 py-8">
                {turns.map((turn) => (
                  <MessageBubble
                    key={turn.id}
                    turn={turn}
                    onRetry={
                      turn.status === "error" && turn.query
                        ? () => retry(turn.id, turn.query as string)
                        : undefined
                    }
                  />
                ))}
              </div>
            )}
            <div ref={bottomRef} />
          </div>

          <div className="sticky bottom-0 z-20 bg-gradient-to-t from-[var(--bg)] via-[var(--bg)] to-transparent pt-8">
            <div className="mx-auto max-w-4xl px-4 pb-4 sm:px-8">
              <Composer
                value={input}
                onChange={setInput}
                onSubmit={send}
                mode={mode}
                onMode={setMode}
                busy={busy}
                image={image}
                imagePreview={imagePreview}
                onPickImage={pickImage}
              />
              <p className="mt-2 text-center text-[11px] text-zinc-400">
                MIRA có thể chưa chính xác. Hãy kiểm tra các nguồn được trích dẫn.
              </p>
            </div>
          </div>
        </section>
      </div>
    </div>
  );
}
