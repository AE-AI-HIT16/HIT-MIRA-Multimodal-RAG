"use client";

import { useEffect, useRef, useState } from "react";
import { Composer } from "@/components/Composer";
import { EmptyState } from "@/components/EmptyState";
import { MessageBubble } from "@/components/MessageBubble";
import { TopBar } from "@/components/TopBar";
import { sendMessage, sendMessageImage } from "@/lib/api";
import type { ChatReply, ChatTurn, Override } from "@/lib/types";

type Mode = "auto" | Override;

export default function ChatPage() {
  const [turns, setTurns] = useState<ChatTurn[]>([]);
  const [input, setInput] = useState("");
  const [mode, setMode] = useState<Mode>("auto");
  const [busy, setBusy] = useState(false);
  const [conversationId, setConversationId] = useState<number | null>(null);
  const [image, setImage] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);

  const idRef = useRef(0);
  const bottomRef = useRef<HTMLDivElement>(null);
  const nextId = () => `t${++idRef.current}`;

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

  async function runQuery(asstId: string, sender: () => Promise<ChatReply>) {
    setBusy(true);
    setTurns((prev) =>
      prev.map((t) => (t.id === asstId ? { ...t, status: "loading", text: "" } : t)),
    );
    try {
      const reply = await sender();
      setConversationId((c) => reply.conversation_id ?? c);
      setTurns((prev) =>
        prev.map((t) =>
          t.id === asstId ? { ...t, status: "done", reply, text: "" } : t,
        ),
      );
    } catch (e) {
      const msg = e instanceof Error ? e.message : "Có lỗi xảy ra";
      setTurns((prev) =>
        prev.map((t) => (t.id === asstId ? { ...t, status: "error", text: msg } : t)),
      );
    } finally {
      setBusy(false);
    }
  }

  function send() {
    const q = input.trim();
    const img = image;
    if ((!q && !img) || busy) return;
    const preview = imagePreview ?? undefined;

    setInput("");
    if (img) pickImage(null);

    const asstId = nextId();
    const ov = mode === "auto" ? undefined : mode;
    setTurns((prev) => [
      ...prev,
      { id: nextId(), role: "user", text: q, imagePreview: img ? preview : undefined },
      // ảnh không giữ File để retry → chỉ text mới có nút Gửi lại
      { id: asstId, role: "assistant", text: "", status: "loading", query: img ? undefined : q },
    ]);

    const sender: () => Promise<ChatReply> = img
      ? () => {
          const form = new FormData();
          form.append("image", img);
          if (q) form.append("text", q);
          if (ov) form.append("override", ov);
          if (conversationId != null) form.append("conversation_id", String(conversationId));
          return sendMessageImage(form);
        }
      : () => sendMessage({ text: q, override: ov, conversationId });

    void runQuery(asstId, sender);
  }

  return (
    <div className="min-h-[100dvh]">
      <TopBar />

      <main className="mx-auto max-w-3xl px-4 pb-52 pt-2">
        {turns.length === 0 ? (
          <EmptyState onPick={(q) => setInput(q)} />
        ) : (
          <div className="space-y-6 py-6">
            {turns.map((turn) => (
              <MessageBubble
                key={turn.id}
                turn={turn}
                onRetry={
                  turn.status === "error" && turn.query
                    ? () => runQuery(turn.id, () =>
                        sendMessage({
                          text: turn.query as string,
                          override: mode === "auto" ? undefined : mode,
                          conversationId,
                        }),
                      )
                    : undefined
                }
              />
            ))}
          </div>
        )}
        <div ref={bottomRef} />
      </main>

      <div className="fixed inset-x-0 bottom-0 z-20 border-t border-zinc-200 bg-[var(--bg)]/90 backdrop-blur-md">
        <div className="mx-auto max-w-3xl px-4 py-3">
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
            MIRA có thể trả lời chưa chính xác — hãy đối chiếu với nguồn được trích dẫn.
          </p>
        </div>
      </div>
    </div>
  );
}
