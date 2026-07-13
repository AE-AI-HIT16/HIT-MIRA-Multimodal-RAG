import type {
  AdminStats,
  ChatReply,
  EvalReport,
  IndexJobStatus,
  Override,
  RegulationResult,
  TokenOut,
  UploadResult,
  UserOut,
} from "./types";

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
