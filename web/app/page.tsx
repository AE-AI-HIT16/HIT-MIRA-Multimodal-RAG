"use client";

import { useEffect, useRef, useState } from "react";
import { ChatSidebar, type ChatSessionSummary } from "@/components/ChatSidebar";
import { Composer } from "@/components/Composer";
import { EmptyState } from "@/components/EmptyState";
import { MessageBubble } from "@/components/MessageBubble";
import { TopBar } from "@/components/TopBar";
import { MenuIcon } from "@/components/icons";
import { sendMessage, sendMessageImage } from "@/lib/api";
import type { ChatReply, ChatTurn, Override } from "@/lib/types";

type Mode = "auto" | Override;
type StoredSession = ChatSessionSummary & { turns: ChatTurn[]; conversationId: number | null };
const STORAGE_KEY = "hit_mira_chat_sessions_v1";
const freshId = () => `chat-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`;

export default function ChatPage() {
  const [turns, setTurns] = useState<ChatTurn[]>([]);
  const [input, setInput] = useState("");
  const [mode, setMode] = useState<Mode>("auto");
  const [busy, setBusy] = useState(false);
  const [conversationId, setConversationId] = useState<number | null>(null);
  const [image, setImage] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [sessions, setSessions] = useState<StoredSession[]>([]);
  const [activeId, setActiveId] = useState(freshId);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [hydrated, setHydrated] = useState(false);
  const idRef = useRef(0);
  const bottomRef = useRef<HTMLDivElement>(null);
  const nextId = () => `t${Date.now()}-${++idRef.current}`;

  useEffect(() => {
    try {
      const saved = JSON.parse(localStorage.getItem(STORAGE_KEY) || "[]") as StoredSession[];
      setSessions(saved);
      if (saved[0]) {
        setActiveId(saved[0].id);
        setTurns(saved[0].turns);
        setConversationId(saved[0].conversationId);
      }
    } catch { localStorage.removeItem(STORAGE_KEY); }
    setHydrated(true);
  }, []);

  useEffect(() => {
    if (!hydrated || turns.length === 0) return;
    setSessions((prev) => {
      const firstQuestion = turns.find((t) => t.role === "user")?.text || "Tra cứu bằng hình ảnh";
      const cleanTurns = turns.map((t) => ({ ...t, imagePreview: undefined }));
      const current: StoredSession = { id: activeId, title: firstQuestion.slice(0, 54), updatedAt: Date.now(), turns: cleanTurns, conversationId };
      const next = [current, ...prev.filter((s) => s.id !== activeId)].slice(0, 30);
      localStorage.setItem(STORAGE_KEY, JSON.stringify(next));
      return next;
    });
  }, [turns, conversationId, activeId, hydrated]);

  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "end" }); }, [turns]);

  function pickImage(f: File | null) {
    setImagePreview((prev) => { if (prev) URL.revokeObjectURL(prev); return f ? URL.createObjectURL(f) : null; });
    setImage(f);
  }

  function newChat() {
    if (busy) return;
    setActiveId(freshId()); setTurns([]); setConversationId(null); setInput(""); pickImage(null); setSidebarOpen(false);
  }

  function selectSession(id: string) {
    if (busy) return;
    const session = sessions.find((s) => s.id === id);
    if (!session) return;
    setActiveId(id); setTurns(session.turns); setConversationId(session.conversationId); setInput(""); pickImage(null);
  }

  function deleteSession(id: string) {
    const next = sessions.filter((s) => s.id !== id);
    setSessions(next); localStorage.setItem(STORAGE_KEY, JSON.stringify(next));
    if (id === activeId) newChat();
  }

  async function runQuery(asstId: string, sender: () => Promise<ChatReply>) {
    setBusy(true);
    setTurns((prev) => prev.map((t) => t.id === asstId ? { ...t, status: "loading", text: "" } : t));
    try {
      const reply = await sender();
      setConversationId((c) => reply.conversation_id ?? c);
      setTurns((prev) => prev.map((t) => t.id === asstId ? { ...t, status: "done", reply, text: "" } : t));
    } catch (e) {
      const msg = e instanceof Error ? e.message : "Có lỗi xảy ra";
      setTurns((prev) => prev.map((t) => t.id === asstId ? { ...t, status: "error", text: msg } : t));
    } finally { setBusy(false); }
  }

  function send() {
    const q = input.trim(); const img = image;
    if ((!q && !img) || busy) return;
    const preview = imagePreview ?? undefined;
    setInput(""); if (img) pickImage(null);
    const asstId = nextId(); const ov = mode === "auto" ? undefined : mode;
    setTurns((prev) => [...prev, { id: nextId(), role: "user", text: q, imagePreview: img ? preview : undefined }, { id: asstId, role: "assistant", text: "", status: "loading", query: img ? undefined : q }]);
    const sender = img ? () => { const form = new FormData(); form.append("image", img); if (q) form.append("text", q); if (ov) form.append("override", ov); if (conversationId != null) form.append("conversation_id", String(conversationId)); return sendMessageImage(form); } : () => sendMessage({ text: q, override: ov, conversationId });
    void runQuery(asstId, sender);
  }

  return (
    <div className="min-h-[100dvh] bg-[var(--bg)]">
      <TopBar />
      <div className="mx-auto flex max-w-7xl">
        <ChatSidebar sessions={sessions} activeId={activeId} open={sidebarOpen} onClose={() => setSidebarOpen(false)} onNew={newChat} onSelect={selectSession} onDelete={deleteSession} />
        <section className="relative flex min-h-[calc(100dvh-4rem)] min-w-0 flex-1 flex-col">
          <div className="app-grid pointer-events-none absolute inset-x-0 top-0 h-72 opacity-40" />
          <div className="relative mx-auto w-full max-w-4xl flex-1 px-4 pb-8 sm:px-8">
            <div className="flex h-14 items-center justify-between border-b border-zinc-200/70 lg:hidden">
              <button onClick={() => setSidebarOpen(true)} className="flex items-center gap-2 rounded-lg px-2 py-1.5 text-sm text-zinc-600 hover:bg-white"><MenuIcon width={18} /> Lịch sử</button>
              <button onClick={newChat} className="text-sm font-medium text-accent-ink">Cuộc trò chuyện mới</button>
            </div>
            {turns.length === 0 ? <EmptyState onPick={(q) => setInput(q)} /> : <div className="space-y-8 py-8">{turns.map((turn) => <MessageBubble key={turn.id} turn={turn} onRetry={turn.status === "error" && turn.query ? () => runQuery(turn.id, () => sendMessage({ text: turn.query as string, override: mode === "auto" ? undefined : mode, conversationId })) : undefined} />)}</div>}
            <div ref={bottomRef} />
          </div>
          <div className="sticky bottom-0 z-20 bg-gradient-to-t from-[var(--bg)] via-[var(--bg)] to-transparent pt-8">
            <div className="mx-auto max-w-4xl px-4 pb-4 sm:px-8">
              <Composer value={input} onChange={setInput} onSubmit={send} mode={mode} onMode={setMode} busy={busy} image={image} imagePreview={imagePreview} onPickImage={pickImage} />
              <p className="mt-2 text-center text-[11px] text-zinc-400">MIRA có thể chưa chính xác. Hãy kiểm tra các nguồn được trích dẫn.</p>
            </div>
          </div>
        </section>
      </div>
    </div>
  );
}
