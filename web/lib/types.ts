export type Override = "media" | "regulation" | "both";

// ── Hình dạng THẬT do API trả về ───────────────────────────────────────────
// Khớp payload Qdrant mô tả trong CLAUDE.md. Ảnh tĩnh cố ý KHÔNG có
// `timestamp_sec` và `video_id` — để một trích dẫn không thể bịa ra khoảnh
// khắc trong một tấm ảnh chụp.

export interface MediaClipHit {
  score: number;
  media_kind: "image" | "video_frame";
  unit_id: string | null;
  video_id: string | null;
  image_media_id: string | null;
  frame_media_id?: string | null;
  timestamp_sec: number | null;
  frame_index: number | null;
  caption: string;
  ocr_text: string;
  bucket_name: string | null;
  frame_object_key: string | null;
  post_id: string | null;
  source_url: string | null;
  detected_objects?: unknown[];
}

export interface TranscriptMomentHit {
  score: number;
  start_sec: number | null;
  end_sec: number | null;
  text: string;
  unit_id: string | null;
}

export interface TranscriptVideoHit {
  video_id: string;
  score: number;
  post_id: string | null;
  source_url: string | null;
  language: string | null;
  moments: TranscriptMomentHit[];
}

export interface MediaSearchResponse {
  query: string;
  source: string;
  clips: MediaClipHit[];
  videos: TranscriptVideoHit[];
  context: string;
  total: number;
  found: boolean;
  errors?: string[];
}

export interface RegulationHit {
  score: number;
  text: string;
  chunk_id: string | null;
  document_id: string | null;
  chunk_index: number | null;
  filename: string | null;
  section: string | null;
  page: number | null;
}

export interface RegulationSearchResponse {
  query: string;
  rewritten_query?: string | null;
  results: RegulationHit[];
}

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
// `object_key` / `video_uid` là các trường THẬT dùng để dựng URL; `media_id` số
// là tàn dư của bản thiết kế cũ, giữ lại để không phá phần code còn dùng.
export interface ChatItem {
  id?: number;
  score?: number;
  media_id?: number;
  media_type?: string;
  video_id?: number;
  timestamp?: number;
  frame_path?: string;
  object_key?: string | null;  // khoá object trong MinIO -> mediaFileUrl()
  video_uid?: string | null;   // video_id dạng UUID -> videoUrl()
  label?: string | null;       // nhãn hiện trên thẻ nội quy (thay article/clause)
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

  // Thẻ kết quả lấy từ /api/media/search và /api/retrieval/search, chạy song
  // song với stream của LangGraph. Câu trả lời là chữ do agent viết; phần này
  // là bằng chứng nhìn được kèm theo — ảnh, mốc thời gian, link bài gốc.
  hits?: ChatItem[];
  hitsStatus?: "loading" | "done" | "error";
  hitsError?: string;
  status?: "loading" | "streaming" | "error" | "done";
  query?: string;             // câu hỏi gốc — dùng để "Gửi lại" khi lỗi
  imagePreview?: string;      // objectURL ảnh đính kèm (hiển thị ở bong bóng user)

  // LangGraph thread
  threadId?: string;
}
