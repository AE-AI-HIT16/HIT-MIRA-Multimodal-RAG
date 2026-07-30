export type Override = "media" | "regulation" | "both";

export interface Moment {
  start_sec?: number | null;
  end_sec?: number | null;
  text?: string | null;
}

// Frame khớp trong một video (retrieve_media gộp theo video_id).
export interface FrameMoment {
  timestamp: number;
  score?: number;
  frame_path?: string | null;
}

// items của ChatReply — hình dạng khác nhau theo nguồn (ảnh / frame video / transcript / nội quy).
export interface ChatItem {
  id?: number;
  score?: number;
  media_id?: number;
  media_type?: string;
  video_id?: number;
  timestamp?: number;
  frame_path?: string;
  caption?: string;
  event?: string | null;
  year?: number | null;
  source_url?: string | null;
  article?: number | null;
  clause?: number | null;
  text?: string;
  heading?: string | null;
  regulation_id?: number;
  moments?: Moment[];
  frames?: FrameMoment[];
}

export interface ChatReply {
  answer: string;
  source: string; // media | regulation | both
  items: ChatItem[];
  conversation_id: number | null;
}

export interface TokenOut {
  access_token: string;
  token_type: string;
}

export interface UserOut {
  id: number;
  email: string;
  name: string | null;
  role: string;
}

export interface AdminStats {
  posts: number;
  images: number;
  videos: number;
  regulations_active: number;
  rule_chunks: number;
  indexed_images: number;
  indexed_videos: number;
  indexed_rule_chunks: number;
}

export interface UploadResult {
  created_ids: number[];
  skipped: { file: string; reason: string }[];
}

export interface RegulationResult {
  regulation_id: number;
  n_chunks: number;
  needs_review: boolean;
}

export interface EvalReport {
  n_queries: number;
  recall: Record<string, number>;
  recall_by_category: Record<string, number> | null;
  mrr: number;
  latency_ms: { avg?: number; p95?: number };
  routing_accuracy: number | null;
  groundedness: number | null;
  targets: Record<string, number>;
  passed: Record<string, boolean>;
  notes: string | null;
}

// Job index pipeline chạy nền (POST /admin/index/{target}).
export interface IndexJobStatus {
  target: "media" | "videos" | "regulations" | string;
  state: "idle" | "running" | "done" | "failed" | string;
  returncode: number | null;
  started_at: number | null;
  log_tail: string;
}

// ── LangGraph ──────────────────────────────────────────────────────────────

/** Một message chunk trả về từ LangGraph SSE stream (stream_mode="messages"). */
export interface LGChunk {
  event: string; // "messages/partial" | "messages/complete" | "error" | ...
  data: LGMessage | LGError | unknown;
}

export interface LGMessage {
  type: string;          // "ai" | "human" | "tool" | ...
  content: string | LGContentPart[];
  id?: string;
  name?: string | null;
  tool_calls?: unknown[];
  additional_kwargs?: Record<string, unknown>;
}

export interface LGContentPart {
  type: "text" | "image_url";
  text?: string;
  image_url?: { url: string };
}

export interface LGError {
  message: string;
  error?: string;
}

// ── ChatTurn (mở rộng hỗ trợ LangGraph streaming) ─────────────────────────

/** Trạng thái 1 lượt hội thoại ở client. */
export interface ChatTurn {
  id: string;
  role: "user" | "assistant";

  // Nội dung văn bản:
  // - khi streaming: streamText tích luỹ dần, finalText chưa có
  // - khi done: finalText là nội dung hoàn chỉnh
  text: string;
  streamText?: string;
  finalText?: string;

  reply?: ChatReply;          // dùng khi gọi FastAPI (legacy)
  status?: "loading" | "streaming" | "error" | "done";
  query?: string;             // câu hỏi gốc — dùng để "Gửi lại" khi lỗi
  imagePreview?: string;      // objectURL ảnh đính kèm (hiển thị ở bong bóng user)

  // LangGraph thread
  threadId?: string;
}
