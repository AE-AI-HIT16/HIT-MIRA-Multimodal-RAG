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
  searchMedia,
  searchMediaByImage,
  searchRegulations,
} from "@/lib/api";
import { gopKetQua, type KetQuaGop } from "@/lib/results";
import { laXaGiao } from "@/lib/smalltalk";
import { docToolCall, type KeHoachTraCuu, type NguonTraCuu } from "@/lib/toolcall";
import { docKetQuaTool, type KetQuaTool } from "@/lib/toolresult";
import type {
  ChatTurn,
  LGMessage,
  MediaSearchResponse,
  Override,
  RegulationSearchResponse,
} from "@/lib/types";

type Mode = "auto" | Override;
type StoredSession = ChatSessionSummary & {
  turns: ChatTurn[];
  threadId: string | null;
};

const STORAGE_KEY = "hit_mira_chat_sessions_v2";

/**
 * Tra lại bằng REST — **đường dự phòng**, không còn là đường chính.
 *
 * Đường chính là đọc thẳng kết quả tool trong stream (`lib/toolresult.ts`), vì
 * đó đúng là thứ agent đã đọc để viết câu trả lời. Hàm này chỉ chạy trong hai
 * trường hợp tool không phục vụ được:
 *
 * 1. **Truy vấn có ảnh.** MCP `search_media` chỉ nhận chữ, nên kết quả tool của
 *    lượt đó là "LLM tả ảnh thành chữ rồi đi tìm". `/api/media/search-image`
 *    nhúng chính tấm ảnh và so với vector ảnh trong kho (US-303.1) — kết quả
 *    khác hẳn và tốt hơn, nên ở đây REST thắng.
 * 2. **Agent sập hoặc tool lỗi.** Không có kết quả nào để đọc, mà "chữ hỏng thì
 *    ảnh vẫn phải xem được" là tính chất có chủ ý của màn hình này.
 *
 * `loc` là bộ lọc agent suy ra từ câu hỏi. **API truy xuất không tự đọc câu chữ**
 * — nó là vector search thuần, viết "năm 2025" trong query trả về cả 2022 lẫn
 * 2024 (đã đo). Bước suy luận duy nhất nằm ở agent khi nó điền tham số tool, nên
 * ở đây chỉ chép lại chứ tuyệt đối không tự đoán năm/sự kiện lần nữa: hai chỗ
 * đoán độc lập là hai chỗ có thể lệch nhau ngay trong cùng một câu trả lời.
 */
async function napTheKetQua(
  q: string,
  mode: Mode,
  img?: File | null,
  keHoach: KeHoachTraCuu | null = null,
): Promise<KetQuaGop> {
  const modeChoMedia = mode === "auto" || mode === "media" || mode === "both";
  // Nội quy là văn bản thuần — không có chữ thì không có gì để tra. Đính kèm
  // ảnh mà vẫn gõ chữ thì vẫn tra bình thường.
  const modeChoNoiQuy =
    !!q && (mode === "auto" || mode === "regulation" || mode === "both");

  // `keHoach === null` là "agent chưa kịp nói gì" (nó sập trước khi gọi tool):
  // giữ hành vi cũ, tra mọi nguồn mà chế độ cho phép, để câu trả lời hỏng không
  // kéo theo mất luôn nguồn tham khảo. Có kế hoạch rồi thì THEO ĐÚNG nó —
  // agent chỉ gọi search_regulations mà mình vẫn tra media là tự chế thêm 12
  // thẻ video không ai hỏi.
  const canMedia = modeChoMedia && (keHoach === null || !!keHoach.media);
  const canNoiQuy = modeChoNoiQuy && (keHoach === null || !!keHoach.regulation);
  if (!canMedia && !canNoiQuy) return { items: [], notes: [] };

  // Agent thường viết lại câu hỏi gọn hơn khi gọi tool ("ảnh Open Day 2024");
  // dùng bản đó thì thẻ khớp với thứ agent thật sự đã tra.
  const locMedia = keHoach?.media ?? {};
  const cauTraMedia = locMedia.query?.trim() || q;
  const cauTraNoiQuy = keHoach?.regulation?.query?.trim() || q;
  const { years, events } = locMedia;

  const [media, noiQuy] = await Promise.all([
    canMedia
      ? // Có ảnh → truy vấn bằng vector ảnh (US-303.1); ảnh không đi qua LLM.
        (img
          ? searchMediaByImage(img, { text: cauTraMedia, years, events })
          : searchMedia(cauTraMedia, { years, events })
        ).catch(() => null)
      : Promise.resolve(null),
    // Nội quy không có trường năm/sự kiện nên không nhận bộ lọc nào.
    canNoiQuy ? searchRegulations(cauTraNoiQuy).catch(() => null) : Promise.resolve(null),
  ]);
  if (!media && !noiQuy) throw new Error("Không gọi được API truy xuất");
  return gopKetQua(media, noiQuy);
}

/**
 * Có nên chạy nhánh truy xuất cho lượt này không.
 *
 * Trả `false` thì khối thẻ kết quả **không hiện gì cả** (`hitsStatus` để trống),
 * chứ không phải hiện "0 kết quả" — chào một câu mà bị báo "Không tìm thấy nguồn
 * nào khớp" thì vẫn là nhiễu, chỉ đỡ hơn một chút.
 *
 * Chỉ chặn ở chế độ "Tự động". Người dùng bấm tay sang "Ảnh & video" / "Nội quy"
 * / "Cả hai" là đã nói rõ họ muốn tra cứu — không đoán lại thay họ.
 */
function coTraCuu(q: string, mode: Mode, img?: File | null): boolean {
  if (img) return true; // ảnh chính là câu truy vấn (US-303.1), không bao giờ là xã giao
  if (!q) return false;
  if (mode !== "auto") return true;
  return !laXaGiao(q);
}
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
  const composerRef = useRef<HTMLTextAreaElement>(null);

  const nextId = () => `t${Date.now()}-${++idRef.current}`;

  /**
   * Bấm thẻ gợi ý thì đổ câu hỏi vào ô nhập **và** đưa con trỏ xuống đó.
   *
   * Không gửi thẳng: thẻ gợi ý là điểm khởi đầu để sửa lại, gửi luôn thì mất cơ
   * hội đổi "2024" thành năm khác. Nhưng chỉ `setInput` thôi thì chữ hiện ở một
   * chỗ khác với chỗ mắt đang nhìn, người dùng tưởng bấm hụt.
   */
  function chonGoiY(q: string) {
    setInput(q);
    requestAnimationFrame(() => {
      const el = composerRef.current;
      if (!el) return;
      el.focus();
      el.setSelectionRange(el.value.length, el.value.length);
    });
  }

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
      // Xoá imagePreview + imageFile trước khi lưu: objectURL và `File` đều
      // không serialize được, và tải lại trang thì cũng không còn ảnh để gửi.
      const cleanTurns = turns.map((t) => ({
        ...t,
        imagePreview: undefined,
        imageFile: undefined,
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

  /**
   * Dọn ô nhập sau khi gửi — **không** thu hồi objectURL.
   *
   * Bong bóng lượt hỏi vừa nhận chính chuỗi `blob:` đó làm `src`. `pickImage(null)`
   * thu hồi nó ngay trong cùng một handler, tức là trước cả lần vẽ đầu tiên, nên
   * ảnh người dùng vừa đính kèm hiện ra vỡ. Quyền sở hữu chuyển sang lượt hỏi:
   * URL sống tới khi rời trang. Rò một objectURL cho mỗi ảnh đã gửi là cái giá
   * rẻ hơn nhiều so với việc không nhìn thấy thứ mình vừa gửi.
   */
  function nhaAnhKhoiONhap() {
    setImage(null);
    setImagePreview(null);
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
   * Bộ điều khiển nhánh thẻ kết quả cho MỘT lượt trả lời.
   *
   * **Thẻ dựng từ chính kết quả tool trong stream** — cùng một lượt truy xuất
   * với câu trả lời, nên không còn cảnh chữ nói "10 kết quả" mà dưới hiện 6 thẻ
   * khác hẳn (đo được: agent gọi `top_k: 10, source: "clip"`, giao diện tra lại
   * `top_k: 6, source: "both"`). Kèm theo đó, số `[n]` trong `context` mà agent
   * trích dẫn trỏ đúng vào thẻ thứ n.
   *
   * REST chỉ còn là đường dự phòng — xem chú thích của `napTheKetQua`.
   *
   * `send` và `retry` dùng chung: hai bản sao của luật này sẽ trôi khỏi nhau, mà
   * triệu chứng là gửi lại một câu thì thẻ khác hẳn lần đầu.
   */
  function taoNhanhTheKetQua(
    asstTurnId: string,
    q: string,
    mode: Mode,
    img?: File | null,
  ) {
    let daGoiTool = false;
    let daHienThat = false; // đã hiện thẻ dựng từ kết quả tool
    let daChayREST = false;
    let luot = 0;
    // Gom dần qua từng tool call: agent có thể tra cả hai nguồn trong một lượt
    // (chế độ "Cả hai", hoặc câu hỏi mập mờ). Ghi đè theo nguồn nên lần gọi sau
    // của cùng một nguồn thay bộ lọc cũ, còn nguồn kia vẫn được giữ.
    const keHoach: KeHoachTraCuu = {};
    // Kết quả thật, gom theo nguồn giống hệt kế hoạch ở trên.
    const ketQua: {
      media?: MediaSearchResponse | null;
      noiQuy?: RegulationSearchResponse | null;
    } = {};

    const capNhat = (luotNay: number, thayDoi: Partial<ChatTurn>) =>
      setTurns((prev) =>
        // Agent có thể gọi tool lần hai để tự sửa bộ lọc; kết quả lần một về
        // muộn thì không được đè lên lần hai.
        luotNay !== luot
          ? prev
          : prev.map((t) => (t.id === asstTurnId ? { ...t, ...thayDoi } : t)),
      );

    return {
      daChay: () => daHienThat || daChayREST,

      /**
       * Ghi nhận tham số của một tool call.
       *
       * Đường ảnh dùng nó ngay để tra REST theo đúng bộ lọc agent suy ra. Đường
       * chữ chỉ **cất đi**: kết quả thật sắp tới nơi rồi, nhưng nếu tool lỗi thì
       * đây là bộ lọc để tra lại.
       */
      ghiNhanToolCall(nguon: NguonTraCuu, loc: KeHoachTraCuu[NguonTraCuu]) {
        daGoiTool = true;
        keHoach[nguon] = loc;
        if (img) this.chay({ ...keHoach });
      },

      /** Kết quả thật của một tool — nguồn chính để dựng thẻ. */
      ghiNhanKetQua(kq: KetQuaTool) {
        // Lượt có ảnh đã tra bằng vector ảnh rồi; đè kết quả text-only của tool
        // lên đó là đánh đổi ngược, nên bỏ qua.
        if (img) return;
        if (kq.payload === null) {
          // Tool hỏng thì không có gì để hiện — tra lại bằng REST để câu trả lời
          // hỏng không kéo theo mất luôn nguồn tham khảo.
          if (!daHienThat) this.chay({ ...keHoach });
          return;
        }
        if (kq.nguon === "media") ketQua.media = kq.payload;
        else ketQua.noiQuy = kq.payload;

        daHienThat = true;
        // Tăng lượt để một lần tra REST đang bay không đè lên kết quả thật.
        const luotNay = ++luot;
        const gop = gopKetQua(ketQua.media ?? null, ketQua.noiQuy ?? null);
        capNhat(luotNay, {
          hits: gop.items,
          hitsNotes: gop.notes,
          hitsStatus: "done",
        });
      },

      chay(keHoachHienTai: KeHoachTraCuu | null) {
        daChayREST = true;
        const luotNay = ++luot;
        napTheKetQua(q, mode, img, keHoachHienTai)
          .then((gop) =>
            capNhat(luotNay, {
              hits: gop.items,
              hitsNotes: gop.notes,
              hitsStatus: "done",
            }),
          )
          .catch((e: unknown) =>
            capNhat(luotNay, {
              hitsStatus: "error",
              hitsError: e instanceof Error ? e.message : "Lỗi truy xuất",
            }),
          );
      },

      /**
       * Agent đã gọi tool nhưng stream kết thúc mà không kết quả nào tới nơi.
       *
       * Không nên xảy ra, nhưng để mặc thì khối thẻ đứng nguyên ở trạng thái
       * skeleton mãi mãi — im lặng và trông như đang tải.
       */
      buNeuThieu() {
        if (daGoiTool && !daHienThat && !daChayREST) this.chay({ ...keHoach });
      },

      /** Không hiện khối thẻ nào cả — khác hẳn với hiện "0 kết quả". */
      an() {
        setTurns((prev) =>
          prev.map((t) =>
            t.id === asstTurnId ? { ...t, hitsStatus: undefined } : t,
          ),
        );
      },
    };
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
    if (img) nhaAnhKhoiONhap();

    const userTurnId = nextId();
    const asstTurnId = nextId();
    const traCuu = coTraCuu(q, mode, img);

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
        // Giữ lại để "Gửi lại" hỏi đúng câu đã hỏi. Thiếu nó thì một câu hỏi
        // kèm ảnh gửi lại thành câu hỏi chỉ có chữ — một câu hỏi khác hẳn, mà
        // không có dấu hiệu nào cho người dùng biết.
        imageFile: img,
        hitsStatus: traCuu ? "loading" : undefined,
      },
    ]);

    await chayLuot(asstTurnId, q, img, traCuu);
  }

  /**
   * Một lượt hỏi–đáp: tạo thread, gửi message, đọc stream, dựng thẻ.
   *
   * `send` và `retry` **dùng chung đúng hàm này**. Trước đây mỗi bên giữ một bản
   * sao của vòng lặp stream, và chúng đã trôi khỏi nhau ngay lần sửa đầu tiên —
   * đó là cùng một lý do khiến bộ điều khiển thẻ kết quả được viết dùng chung.
   */
  async function chayLuot(
    asstTurnId: string,
    q: string,
    img: File | null | undefined,
    traCuu: boolean,
  ) {
    const ov: Override | undefined = mode === "auto" ? undefined : mode;
    setBusy(true);
    const controller = new AbortController();
    abortRef.current = controller;

    const the = taoNhanhTheKetQua(asstTurnId, q, mode, img);

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

      for await (const chunk of lgStream(tid, [humanMsg], ov, controller.signal)) {
        // stream_mode="messages" → event = "messages/partial" hoặc "messages/complete"
        if (
          chunk.event === "messages/partial" ||
          chunk.event === "messages/complete"
        ) {
          const dataArr = Array.isArray(chunk.data) ? chunk.data : [chunk.data];

          // Duyệt MỌI message chứ không chỉ message của AI: kết quả truy xuất
          // đi về dưới dạng ToolMessage (`type: "tool"`), và đó mới là thứ dựng
          // nên các thẻ. Bản trước `find(type === "ai")` nên nó bị bỏ qua sạch.
          for (const msg of dataArr as LGMessage[]) {
            if (!msg || typeof msg !== "object") continue;

            if (msg.type === "tool") {
              if (!traCuu) continue;
              const kq = docKetQuaTool(msg);
              if (kq) the.ghiNhanKetQua(kq);
              continue;
            }
            if (msg.type !== "ai") continue;

            // Đây là chỗ duy nhất biết "năm 2025" phải thành `years: [2025]`.
            // Phải đọc TRƯỚC khi bỏ qua message không có chữ: message mang tool
            // call có `content` rỗng, nên nó rơi đúng vào nhánh `continue` dưới.
            if (traCuu) {
              for (const tc of msg.tool_calls ?? []) {
                const doc = docToolCall(tc);
                if (doc) the.ghiNhanToolCall(doc.nguon, doc.loc);
              }
            }

            const text = extractTextFromLGMessage(msg);
            if (!text) continue;

            accumulated = text; // LangGraph gửi toàn bộ text tích luỹ, không chỉ delta

            setTurns((prev) =>
              prev.map((t) =>
                t.id === asstTurnId
                  ? { ...t, status: "streaming", streamText: accumulated }
                  : t,
              ),
            );
          }
        }
      }

      // Agent trả lời xong mà không tra cứu gì (nó tự biết câu này không cần
      // kho) → không có nguồn nào để hiện. Đây là cùng một quyết định với
      // `laXaGiao`, nhưng do agent đưa ra nên chính xác hơn nhiều.
      the.buNeuThieu();
      if (!the.daChay()) the.an();

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
        if (!the.daChay()) the.an();
      } else {
        const msg = e instanceof Error ? e.message : "Có lỗi xảy ra";
        setTurns((prev) =>
          prev.map((t) =>
            t.id === asstTurnId
              ? { ...t, status: "error", text: msg, streamText: undefined }
              : t,
          ),
        );
        // Agent sập TRƯỚC khi kịp gọi tool. Không có bộ lọc nào để chép, nhưng
        // "chữ hỏng thì ảnh vẫn phải xem được" là tính chất có chủ ý của màn
        // hình này — nên vẫn tra, chỉ là không lọc. Thà thẻ rộng còn hơn không
        // có nguồn nào khi câu trả lời đã mất.
        if (traCuu && !the.daChay()) the.chay(null);
        else if (!the.daChay()) the.an();
      }
    } finally {
      setBusy(false);
      abortRef.current = null;
    }
  }

  /**
   * Gửi lại câu hỏi khi lỗi — **gửi lại đúng câu đã hỏi, kể cả ảnh đính kèm.**
   *
   * Bản trước chỉ mang theo phần chữ, nên "ảnh này chụp ở đâu" + một tấm ảnh gửi
   * lại thành một câu hỏi chỉ có chữ: câu khác, kho tra khác, mà không có dấu
   * hiệu nào cho người dùng biết là ảnh đã bị bỏ rơi.
   */
  async function retry(asstTurnId: string, query: string, img?: File | null) {
    if (busy) return;
    const traCuu = coTraCuu(query, mode, img);

    setTurns((prev) =>
      prev.map((t) =>
        t.id === asstTurnId
          ? {
              ...t,
              status: "loading",
              text: "",
              streamText: "",
              hits: undefined,
              hitsNotes: undefined,
              hitsError: undefined,
              hitsStatus: traCuu ? "loading" : undefined,
            }
          : t,
      ),
    );

    // Gửi lại thì nguồn tham khảo cũng phải mới, theo đúng lượt tra của lần này
    // — chứ không giữ lại kết quả của lượt hỏng.
    await chayLuot(asstTurnId, query, img, traCuu);
  }

  return (
    <div className="min-h-[100dvh] bg-[var(--bg)]">
      <TopBar />
      {/* Không giới hạn bề ngang ở đây: bọc cả sidebar trong `max-w-7xl` làm
          sidebar lơ lửng giữa trang trên màn rộng, đọc ra như trang bị lệch chứ
          không phải trang thoáng. Chỉ cột hội thoại mới cần giới hạn. */}
      <div className="flex">
        <ChatSidebar
          sessions={sessions}
          activeId={activeId}
          open={sidebarOpen}
          onClose={() => setSidebarOpen(false)}
          onNew={newChat}
          onSelect={selectSession}
          onDelete={deleteSession}
        />
        {/* Trừ cả đường viền dưới của TopBar (4rem là phần ruột), không thì trang
            cao hơn màn đúng 1px và màn chào rỗng vẫn mọc thanh cuộn. */}
        <section className="relative flex min-h-[calc(100dvh-4rem-1px)] min-w-0 flex-1 flex-col">
          <div className="app-grid pointer-events-none absolute inset-x-0 top-0 h-72 opacity-40" />
          <div className="relative mx-auto flex w-full max-w-5xl flex-1 flex-col px-4 pb-8 sm:px-8">
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
              // Căn giữa theo chiều dọc phần trống còn lại: neo trên bằng padding
              // cứng thì màn cao thừa khoảng chết giữa khối chào và ô nhập, còn
              // màn laptop thấp lại để hai khối chạm nhau.
              <div className="flex flex-1 items-center">
                <EmptyState onPick={chonGoiY} />
              </div>
            ) : (
              <div className="space-y-8 py-8">
                {turns.map((turn) => (
                  <MessageBubble
                    key={turn.id}
                    turn={turn}
                    onRetry={
                      // Lượt chỉ có ảnh thì `query` rỗng — vẫn phải gửi lại được,
                      // nếu không đúng lượt hỏi bằng ảnh lại là lượt không có nút.
                      turn.status === "error" && (turn.query || turn.imageFile)
                        ? () => retry(turn.id, turn.query ?? "", turn.imageFile)
                        : undefined
                    }
                  />
                ))}
              </div>
            )}
            <div ref={bottomRef} />
          </div>

          <div className="sticky bottom-0 z-20 bg-gradient-to-t from-[var(--bg)] via-[var(--bg)] to-transparent pt-8">
            <div className="mx-auto max-w-5xl px-4 pb-4 sm:px-8">
              <Composer
                inputRef={composerRef}
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
              <p className="mt-2 text-center text-[11px] text-zinc-500">
                MIRA có thể chưa chính xác. Hãy kiểm tra các nguồn được trích dẫn.
              </p>
            </div>
          </div>
        </section>
      </div>
    </div>
  );
}
