import type {
  AdminStats,
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
  const res = await fetch(`${API_URL}/auth/login`, {
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
  const res = await fetch(`${API_URL}/auth/register`, {
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
  const res = await fetch(`${API_URL}/auth/me`, { headers: authHeaders() });
  if (!res.ok) return null;
  return res.json();
}

// ---- Admin ----
export async function adminStats(): Promise<AdminStats> {
  const res = await fetch(`${API_URL}/admin/stats`, { headers: authHeaders() });
  if (!res.ok) throw new Error(await parseError(res));
  return res.json();
}

// multipart: KHÔNG set Content-Type để trình duyệt tự thêm boundary.
export async function uploadMedia(form: FormData): Promise<UploadResult> {
  const res = await fetch(`${API_URL}/ingest/upload`, {
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
export async function startIndex(
  target: "media" | "videos" | "regulations",
): Promise<IndexJobStatus> {
  const res = await fetch(`${API_URL}/admin/index/${target}`, {
    method: "POST",
    headers: authHeaders(),
  });
  if (!res.ok) throw new Error(await parseError(res));
  return res.json();
}

export async function indexStatus(): Promise<IndexJobStatus[]> {
  const res = await fetch(`${API_URL}/admin/index/status`, { headers: authHeaders() });
  if (!res.ok) throw new Error(await parseError(res));
  return res.json();
}

export async function runEval(): Promise<EvalReport> {
  const res = await fetch(`${API_URL}/eval/run`, {
    method: "POST",
    headers: authHeaders(),
  });
  if (!res.ok) throw new Error(await parseError(res));
  return res.json();
}

export async function evalReport(): Promise<EvalReport | null> {
  const res = await fetch(`${API_URL}/eval/report`, { headers: authHeaders() });
  if (res.status === 404) return null;
  if (!res.ok) throw new Error(await parseError(res));
  return res.json();
}

export const mediaUrl = (id: number) => `${API_URL}/media/${id}`;
export const clipUrl = (id: number, ts: number) =>
  `${API_URL}/media/${id}/clip?ts=${ts}`;
export const streamUrl = (id: number) => `${API_URL}/media/${id}/stream`;
export const frameUrl = (id: number, ts: number) =>
  `${API_URL}/media/${id}/frame?ts=${ts}`;

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
