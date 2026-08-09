import type {
  AdminStats,
  MediaSearchResponse,
  RegulationSearchResponse,
  ChatReply,
  EvalReport,
  IndexJobStatus,
  LGChunk,
  LGMessage,
  Override,
  RegulationResult,
  TokenOut,
  UploadResult,
  UserOut,
} from "./types";

// ── FastAPI (legacy — Auth / Admin / Upload vẫn dùng) ─────────────────────

export const API_URL =
  process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

const TOKEN_KEY = "hit_mira_token";

export function getToken(): string | null {
  if (typeof window === "undefined") return null;
  return window.localStorage.getItem(TOKEN_KEY);
}

export function setToken(token: string | null) {
  if (typeof window === "undefined") return;
  if (token) window.localStorage.setItem(TOKEN_KEY, token);
  else window.localStorage.removeItem(TOKEN_KEY);
}

function authHeaders(): Record<string, string> {
  const t = getToken();
  return t ? { Authorization: `Bearer ${t}` } : {};
}

async function parseError(res: Response): Promise<string> {
  try {
    const body = await res.json();
    if (typeof body?.detail === "string") return body.detail;
    if (Array.isArray(body?.detail)) return body.detail[0]?.msg ?? "Dữ liệu không hợp lệ";
  } catch {
    /* ignore */
  }
  return `Lỗi ${res.status}`;
}

export interface SendPayload {
  text: string;
  override?: Override;
  conversationId?: number | null;
}

export async function sendMessage(payload: SendPayload): Promise<ChatReply> {
  const res = await fetch(`${API_URL}/chat/message`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...authHeaders() },
    body: JSON.stringify({
      text: payload.text,
      override: payload.override,
      conversation_id: payload.conversationId ?? undefined,
    }),
  });
  if (!res.ok) throw new Error(await parseError(res));
  return res.json();
}

// Truy vấn bằng ảnh (multipart: image + text? + override? + conversation_id?).
export async function sendMessageImage(form: FormData): Promise<ChatReply> {
  const res = await fetch(`${API_URL}/chat/message-image`, {
    method: "POST",
    headers: authHeaders(),
    body: form,
  });
  if (!res.ok) throw new Error(await parseError(res));
  return res.json();
}

export async function login(email: string, password: string): Promise<TokenOut> {
  const res = await fetch(`${API_URL}/api/auth/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email, password }),
  });
  if (!res.ok) throw new Error(await parseError(res));
  return res.json();
}

export async function register(
  email: string,
  password: string,
  name?: string,
): Promise<UserOut> {
  const res = await fetch(`${API_URL}/api/auth/register`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email, password, name: name || undefined }),
  });
  if (!res.ok) throw new Error(await parseError(res));
  return res.json();
}

export async function me(): Promise<UserOut | null> {
  const t = getToken();
  if (!t) return null;
  const res = await fetch(`${API_URL}/api/auth/me`, { headers: authHeaders() });
  if (!res.ok) return null;
  return res.json();
}

// ---- Admin ----
export async function adminStats(): Promise<AdminStats> {
  const res = await fetch(`${API_URL}/api/admin/stats`, { headers: authHeaders() });
  if (!res.ok) throw new Error(await parseError(res));
  return res.json();
}

// multipart: KHÔNG set Content-Type để trình duyệt tự thêm boundary.
export async function uploadMedia(form: FormData): Promise<UploadResult> {
  const res = await fetch(`${API_URL}/api/ingest/upload`, {
    method: "POST",
    headers: authHeaders(),
    body: form,
  });
  if (!res.ok) throw new Error(await parseError(res));
  return res.json();
}

export async function uploadRegulations(form: FormData): Promise<RegulationResult> {
  const res = await fetch(`${API_URL}/ingest/regulations`, {
    method: "POST",
    headers: authHeaders(),
    body: form,
  });
  if (!res.ok) throw new Error(await parseError(res));
  return res.json();
}

// Chạy pipeline index cho asset chưa index — subprocess nền phía API (spec P4-3 ③).
// Nội quy KHÔNG có ở đây: `/api/documents/upload` đã nhúng và ghi Qdrant ngay
// lúc nạp, nên không có bước index riêng nào để chạy.
export async function startIndex(
  target: "media" | "videos",
): Promise<IndexJobStatus> {
  const res = await fetch(`${API_URL}/api/admin/index/${target}`, {
    method: "POST",
    headers: authHeaders(),
  });
  if (!res.ok) throw new Error(await parseError(res));
  return res.json();
}

/** Trạng thái mọi job nền — gồm cả job `eval`, không chỉ hai job index. */
export async function indexStatus(): Promise<IndexJobStatus[]> {
  const res = await fetch(`${API_URL}/api/admin/index/status`, { headers: authHeaders() });
  if (!res.ok) throw new Error(await parseError(res));
  return res.json();
}

/**
 * Khởi chạy đánh giá. Trả về **trạng thái job**, không phải báo cáo.
 *
 * Một lượt đánh giá nhúng lại toàn bộ tập truy vấn và mất hàng chục giây tới
 * vài phút — treo vào một request HTTP là cầm chắc timeout ở proxy, và mất
 * luôn kết quả của một lượt chạy đã tốn quota. Theo dõi bằng `indexStatus()`
 * (target `eval`) rồi đọc `evalReport()` khi job xong.
 */
export async function runEval(): Promise<IndexJobStatus> {
  const res = await fetch(`${API_URL}/api/admin/eval/run`, {
    method: "POST",
    headers: authHeaders(),
  });
  if (!res.ok) throw new Error(await parseError(res));
  return res.json();
}

export async function evalReport(): Promise<EvalReport | null> {
  const res = await fetch(`${API_URL}/api/admin/eval/report`, { headers: authHeaders() });
  if (res.status === 404) return null;
  if (!res.ok) throw new Error(await parseError(res));
  return res.json();
}

// ── Media files ────────────────────────────────────────────────────────────
// Backend chuyển hướng 307 sang presigned URL của MinIO. Id là UUID chứ không
// phải số tự tăng — bản đầu khai `mediaUrl(id: number)` nên không khớp dòng nào.

/** Ảnh / keyframe theo object key — dạng dùng chính, vì kết quả truy xuất trả sẵn key. */
export const mediaFileUrl = (objectKey: string) =>
  `${API_URL}/api/media-files/by-key?object_key=${encodeURIComponent(objectKey)}`;

/** Media theo `media_id` (UUID). */
export const mediaByIdUrl = (mediaId: string) =>
  `${API_URL}/api/media-files/${encodeURIComponent(mediaId)}`;

/**
 * Video gốc theo `video_id`, tua bằng fragment `#t=`.
 *
 * Không cắt clip bằng ffmpeg: presigned URL của MinIO hỗ trợ HTTP Range (đo
 * được 206 Partial Content), nên trình duyệt nhảy thẳng tới đúng giây. Đổi lại
 * là tải video đầy đủ chứ không phải một đoạn ngắn.
 */
export const videoUrl = (videoId: string, ts?: number) => {
  const base = `${API_URL}/api/media-files/video/${encodeURIComponent(videoId)}`;
  return ts != null ? `${base}#t=${Math.max(0, Math.floor(ts))}` : base;
};

// ── Tìm kiếm (đường đã chạy thật) ──────────────────────────────────────────

/**
 * `years`/`events` KHÔNG được suy ra từ câu chữ ở đây — API là vector search
 * thuần, viết "năm 2025" trong `query` không lọc gì cả (đo được: trả về cả 2022
 * lẫn 2024). Bộ lọc phải do người gọi truyền vào; nguồn duy nhất sinh ra nó là
 * tool call của agent, xem `locTuToolCall` trong `app/page.tsx`.
 */
export async function searchMedia(
  query: string,
  opts: {
    topK?: number;
    source?: "clip" | "transcript" | "both";
    years?: number[];
    events?: string[];
  } = {},
): Promise<MediaSearchResponse> {
  const res = await fetch(`${API_URL}/api/media/search`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      query,
      top_k: opts.topK ?? 6,
      source: opts.source ?? "both",
      // Chỉ gửi khi có: mảng rỗng và thiếu khoá là như nhau ở API, nhưng gửi
      // thừa làm log khó đọc khi truy vết "vì sao lọc ra rỗng".
      ...(opts.years?.length ? { years: opts.years } : {}),
      ...(opts.events?.length ? { events: opts.events } : {}),
    }),
  });
  if (!res.ok) throw new Error(await parseError(res));
  return res.json();
}

/**
 * Truy vấn bằng ẢNH (US-302.1, US-303.1) — multipart, KHÔNG đi qua LLM.
 *
 * Đây là điểm khác cốt lõi so với việc đính ảnh vào khung chat: ảnh gửi cho
 * LLM chỉ được nó *tả thành chữ* rồi mới đi tìm, còn ở đây ảnh được nhúng
 * thành vector và so trực tiếp với vector ảnh trong kho.
 *
 * Không set `Content-Type` để trình duyệt tự thêm boundary.
 */
export async function searchMediaByImage(
  image: File,
  opts: {
    text?: string;
    topK?: number;
    source?: "clip" | "transcript" | "both";
    years?: number[];
    events?: string[];
  } = {},
): Promise<MediaSearchResponse> {
  const form = new FormData();
  form.append("image", image);
  // Có chữ thì gửi kèm: chữ lo nhánh lời thoại, thứ mà vector ảnh không dò được.
  if (opts.text?.trim()) form.append("query", opts.text.trim());
  form.append("top_k", String(opts.topK ?? 6));
  form.append("source", opts.source ?? "both");
  // Form lặp khoá cho mảng — FastAPI đọc `years=2024&years=2025` thành list.
  for (const nam of opts.years ?? []) form.append("years", String(nam));
  for (const su_kien of opts.events ?? []) form.append("events", su_kien);

  const res = await fetch(`${API_URL}/api/media/search-image`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) throw new Error(await parseError(res));
  return res.json();
}

export async function searchRegulations(
  query: string,
  opts: { topK?: number; rewrite?: boolean } = {},
): Promise<RegulationSearchResponse> {
  const res = await fetch(`${API_URL}/api/retrieval/search`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      query,
      top_k: opts.topK ?? 4,
      // Viết lại truy vấn gọi thêm một lượt LLM. Bật mặc định thì mỗi lần gõ
      // phải chờ thêm vài giây, mà thẻ kết quả chỉ để xem kèm câu trả lời.
      rewrite: opts.rewrite ?? false,
    }),
  });
  if (!res.ok) throw new Error(await parseError(res));
  return res.json();
}

// ── LangGraph API ──────────────────────────────────────────────────────────

export const LANGGRAPH_URL =
  process.env.NEXT_PUBLIC_LANGGRAPH_URL ?? "http://localhost:2024";

/** Tạo thread mới trên LangGraph. Trả về thread_id (UUID string). */
export async function lgCreateThread(): Promise<string> {
  const res = await fetch(`${LANGGRAPH_URL}/threads`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({}),
  });
  if (!res.ok) throw new Error(`LangGraph: không tạo được thread (${res.status})`);
  const data = await res.json() as { thread_id: string };
  return data.thread_id;
}

/**
 * Đọc File thành base64 data URL (vd: "data:image/jpeg;base64,...").
 * Dùng để nhét ảnh vào HumanMessage multipart.
 */
export function fileToBase64(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result as string);
    reader.onerror = () => reject(new Error("Không đọc được file"));
    reader.readAsDataURL(file);
  });
}

/** Xây dựng LangChain HumanMessage multipart (text + ảnh base64). */
export async function buildHumanMessage(
  text: string,
  image?: File | null,
): Promise<Record<string, unknown>> {
  type ContentPart =
    | { type: "text"; text: string }
    | { type: "image_url"; image_url: { url: string } };

  const content: ContentPart[] = [];

  if (text.trim()) {
    content.push({ type: "text", text: text.trim() });
  }

  if (image) {
    const b64 = await fileToBase64(image);
    content.push({ type: "image_url", image_url: { url: b64 } });
  }

  // Nếu chỉ có text thuần (không ảnh) → dùng string content thay array để gọn hơn
  if (!image && text.trim()) {
    return { type: "human", content: text.trim() };
  }

  return { type: "human", content };
}

/**
 * Gửi message vào thread và stream response từ LangGraph.
 * Yield từng LGChunk SSE.
 *
 * @param threadId  - thread_id từ lgCreateThread()
 * @param messages  - mảng LangChain messages (kết quả buildHumanMessage)
 * @param override  - "media" | "regulation" | "both" | undefined
 * @param signal    - AbortSignal để cancel
 */
export async function* lgStream(
  threadId: string,
  messages: Record<string, unknown>[],
  override?: Override,
  signal?: AbortSignal,
): AsyncGenerator<LGChunk> {
  const body: Record<string, unknown> = {
    assistant_id: "agent",
    input: { messages },
    stream_mode: ["messages"],
  };

  if (override) {
    body.config = { configurable: { override } };
  }

  const res = await fetch(`${LANGGRAPH_URL}/threads/${threadId}/runs/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });

  if (!res.ok) {
    const txt = await res.text().catch(() => "");
    throw new Error(`LangGraph stream lỗi ${res.status}: ${txt}`);
  }

  // Đọc SSE line-by-line
  const reader = res.body?.getReader();
  if (!reader) throw new Error("Không đọc được response stream");

  const decoder = new TextDecoder();
  let buffer = "";
  let currentEvent = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";

    for (const line of lines) {
      if (line.startsWith("event:")) {
        currentEvent = line.slice(6).trim();
      } else if (line.startsWith("data:")) {
        const raw = line.slice(5).trim();
        if (!raw || raw === "[DONE]") continue;
        try {
          const parsed = JSON.parse(raw) as unknown;
          yield { event: currentEvent, data: parsed } as LGChunk;
        } catch {
          // bỏ qua dòng JSON lỗi
        }
      }
    }
  }
}

/**
 * Trích xuất text thuần từ LGMessage.content
 * (content có thể là string hoặc array ContentPart)
 */
export function extractTextFromLGMessage(msg: LGMessage): string {
  if (typeof msg.content === "string") return msg.content;
  if (Array.isArray(msg.content)) {
    return msg.content
      .filter((p) => p.type === "text")
      .map((p) => p.text ?? "")
      .join("");
  }
  return "";
}
