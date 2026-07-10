# Spec P0 — Nền móng & Provider

> Phase 0 · Epic E0 + E3 · Sprint 1 · Owner chính: DE (infra) + AIE-1/AIE-2 (provider)
> Mục tiêu: **có chỗ chạy thật** (Postgres + Qdrant + CI) và **4 adapter model** sau lớp abstraction.
> Exit gate: `docker compose up` → services healthy + `GET /health` OK · CI xanh · 4 provider khởi tạo được với key/model thật.

## Phạm vi

| ID | Chức năng | Task | Owner | Ghi chú |
|---|---|---|---|---|
| P0-1 | Hạ tầng container | T-01 | DE | |
| P0-2 | Migration Alembic | T-02 | DE | |
| P0-3 | Data model 19 bảng | T-05 | DE | 🔴 đường găng |
| P0-4 | CI lint + test | T-03 | AIE-2 | |
| P0-5 | Provider Embeddings | T-20 | AIE-1 | |
| P0-6 | Provider ASR | T-21 | AIE-2 | |
| P0-7 | Provider Captioner | T-22 | AIE-2 | |
| P0-8 | Provider LLM | T-23 | AIE-2 | |
| P0-9 | Config & DI wiring | — | AIE-2 | quy tắc kiến trúc bắt buộc |

---

## P0-1 · Hạ tầng container `[T-01]`

**File:** `docker-compose.yml` (hiện rỗng)

**Yêu cầu dịch vụ:**

| Service | Image | Port | Volume | Healthcheck |
|---|---|---|---|---|
| `postgres` | `postgres:16-alpine` | 5432 | `pgdata:/var/lib/postgresql/data` | `pg_isready -U hit` |
| `qdrant` | `qdrant/qdrant:latest` | 6333 (REST), 6334 (gRPC) | `qdrant_storage:/qdrant/storage` | tcp 6333 |
| `minio` | `minio/minio:latest` | 9000 (S3), 9001 (console) | `minio_data:/data` | `mc ready local` |
| `api` | build `api/Dockerfile` | 8000 | `./data:/app/data` | `curl -f localhost:8000/health` |

**Biến môi trường (`.env`, xem `.env.example`):**
```
DATABASE_URL=postgresql+psycopg://hit:hit@postgres:5432/hit_mira
QDRANT_URL=http://qdrant:6333
STORAGE_BACKEND=minio            # dev/test không docker để mặc định filesystem
MINIO_ENDPOINT=minio:9000        # trong docker; ngoài host là localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
MINIO_BUCKET=hit-mira-media
LLM_API_KEY=<gemini key>
```

**Object storage (P0-1):** media gốc lưu qua `shared/providers/storage.py::StorageProvider`
(hạ tầng CHO SẴN, cùng nhóm với `QdrantStore`). Hai backend một interface:
`FilesystemStorage` (mặc định dev/test, zero-config — pytest KHÔNG cần dựng MinIO, giống
sqlite in-memory thay Postgres) và `MinioStorage` (bật khi `STORAGE_BACKEND=minio`).
Factory `app/deps.py::get_storage` chọn backend theo config; bucket tự tạo qua
`ensure_bucket()` lần đầu chạm. Ingest (P1-3) `store.put`, media (P1-2) `store.open/exists`.

**Hành vi:**
- `api` phụ thuộc `postgres` + `qdrant` + `minio` với `condition: service_healthy`.
- Không có `.env` → api vẫn boot với default (`sqlite:///./data/dev.db` + qdrant localhost + storage filesystem) — dev không docker vẫn chạy được.
- Model nặng (torch/faster-whisper) KHÔNG bắt buộc trong image v1 — pipeline offline chạy ngoài container được (CLI trên host có GPU).

**Edge:**
- Qdrant/MinIO chưa sẵn sàng khi api boot → api không được crash: provider lazy-init, chỉ connect khi request đầu chạm.
- Volume mất → Qdrant tự tạo collection lại nhờ `ensure_collection()` (P1-8), MinIO tự tạo bucket lại nhờ `ensure_bucket()`, nhưng dữ liệu phải re-index / re-upload — ghi rõ trong README.

**DoD / Test:**
- `docker compose up -d` → `docker compose ps` cả 4 healthy.
- `curl localhost:8000/health` → `{"status":"ok","env":"dev"}`.
- `curl localhost:6333/collections` → 200.
- MinIO console `localhost:9001` mở được; bucket `hit-mira-media` tồn tại sau request ingest đầu.

---

## P0-2 · Migration Alembic `[T-02]`

**File:** `api/shared/db/migrations/` · Phụ thuộc: P0-3

**Yêu cầu:**
- `alembic init` cấu hình đọc `DATABASE_URL` từ `app.config.settings` (qua env, KHÔNG import `app` trong `env.py` nếu tránh được — dùng `os.environ` fallback).
- Revision đầu: autogenerate từ `Base.metadata` sau khi P0-3 xong.
- Chạy được trên cả sqlite (dev) và postgres (docker).

**Edge:** sqlite không hỗ trợ `ALTER` đầy đủ → dùng `render_as_batch=True` trong `env.py`.

**DoD:** `alembic upgrade head` dựng đủ bảng khớp models; `alembic downgrade base` sạch; chạy 2 lần liên tiếp không lỗi (idempotent).

---

## P0-3 · Data model — 19 bảng ORM `[T-05]` 🔴 đường găng

**File:** `api/shared/db/models.py` (hiện chỉ có `Consent`)

**Nhóm bảng theo PRD §5** (tên cột chuẩn hoá, mọi bảng có `id` PK + `created_at`):

**Nhóm nguồn dữ liệu:**
| Bảng | Cột chính | FK / Ghi chú |
|---|---|---|
| `consents` | scope, effective_date, signed_by, file_path, active | đã có |
| `posts` | fb_post_id, posted_at, caption_original, source_url, consent_id | → consents; unique fb_post_id |
| `media_assets` | post_id, media_type(image\|video), file_path, checksum, needs_review | → posts; unique checksum |
| `events` | name, started_at, ended_at | posts.event_id → events (nullable) |

**Nhóm dẫn xuất pipeline:**
| Bảng | Cột chính | FK |
|---|---|---|
| `video_frames` | media_id, timestamp_sec, frame_path | → media_assets |
| `transcripts` | media_id, start_sec, end_sec, text, confidence | → media_assets |
| `captions` | target_type(image\|frame), target_id, text, lang, generated_by(auto\|manual), low_confidence | polymorphic target |
| `ocr_texts` (v2) | media_id, text, lang, confidence | → media_assets |
| `embeddings` | target_type, target_id, model, dim, vector_id, collection | trỏ point Qdrant |

**Nhóm nội quy:**
| Bảng | Cột chính | FK |
|---|---|---|
| `regulations` | title, version, effective_date, is_active, file_path | chỉ 1 bản is_active/title |
| `rule_chunks` | regulation_id, article(điều), clause(khoản), title, text | → regulations |

**Nhóm app:**
| Bảng | Cột chính | FK |
|---|---|---|
| `users` | email, password_hash, role(admin\|member\|guest) | unique email |
| `conversations` | user_id, started_at | → users |
| `messages` | conversation_id, role(user\|assistant), text, image_path, source_label, payload_json | → conversations |

**Nhóm eval:**
| Bảng | Cột chính | Ghi chú |
|---|---|---|
| `eval_queries` | query_text, query_image_path, expected_media_ids(json), expected_rule_chunk_ids(json), category | ~30–50 dòng |
| `eval_runs` | ran_at, recall_at_k, mrr, latency_avg, latency_p95, notes | kết quả từng lần đo |

(+ v2: `faces`, `face_matches`, `privacy_optouts`, `dedup_log` — khai báo skeleton, chưa dùng.)

**Quy tắc:**
- `posted_at` ISO datetime; `media_type` ràng Enum; thiếu trường bắt buộc → `needs_review=True` và **loại khỏi tập index** (US-104.1).
- Không xoá cứng bản ghi nguồn (crawl v2 đánh dấu deleted).

**DoD:** `Base.metadata.create_all` chạy sạch trên sqlite in-memory (conftest); FK/relationship truy cập được 2 chiều; test tạo 1 chuỗi post→media→frame→embedding không lỗi.

---

## P0-4 · CI `[T-03]`

**File:** `.github/workflows/ci.yml`

```yaml
# khung
on: [pull_request, push]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - checkout · setup-python 3.11 · pip install -r api/requirements.txt (bản nhẹ, KHÔNG torch)
      - cd api && ruff check .
      - cd api && pytest   # fake-based, không cần GPU/Qdrant; ffmpeg: apt-get install -y ffmpeg
```

**Lưu ý:** requirements đầy đủ kéo torch nhiều GB → tách `requirements-ci.txt` (fastapi, pydantic, sqlalchemy, pytest, ruff) vì toàn bộ test dùng fake. Cài `ffmpeg` để không skip nhóm test frames.

**DoD:** PR chạy lint + test tự động, fail chặn merge.

---

## P0-5 · Provider Embeddings `[T-20]`

**File:** `api/shared/providers/embeddings.py`

**Contract (ABC):**
```python
class TextEmbedder(ABC):
    dim: int
    def embed(self, texts: list[str]) -> list[list[float]]

class ImageEmbedder(ABC):
    dim: int
    def embed(self, image_paths: list[str]) -> list[list[float]]
    def embed_query(self, text: str) -> list[float]   # text → không gian ảnh
```

**Impl:**
- `VietnameseTextEmbedder` — SentenceTransformer `AITeamVN/Vietnamese_Embedding` (nền bge-m3), `normalize_embeddings=True`. Dùng cho transcript + nội quy.
- `JinaClipEmbedder` — SentenceTransformer `jinaai/jina-clip-v2`, `trust_remote_code=True`. `embed()` mở PIL image; `embed_query()` nhúng text **cùng không gian** với ảnh (cho phép text→ảnh trực tiếp).

**Bất biến:**
- Lazy import torch/sentence-transformers TRONG `__init__` — `import shared.providers.embeddings` không kéo torch.
- Vector đã normalize → cosine ≈ dot; Qdrant collection dùng distance Cosine.
- Đổi model → phải re-embed toàn collection (không trộn không gian — US-202.1 edge).

**DoD:** embed batch cùng dim; test fake-injectable (mọi hàm nhận `embedder=`).

---

## P0-6 · Provider ASR `[T-21]`

**File:** `api/shared/providers/asr.py`

**Contract:**
```python
@dataclass
class TranscriptSegment:
    start_sec: float; end_sec: float; text: str; confidence: float | None = None

class ASRModel(ABC):
    def transcribe(self, audio_path: str) -> list[TranscriptSegment]
```

**Impl:** `FasterWhisperASR(model_name="large-v3", *, device="auto", model=None)` — backend faster-whisper (CTranslate2); `language="vi"`, `vad_filter=True`; `confidence` = avg_logprob. Param `model=` để inject fake.

**DoD:** audio mẫu → ≥1 segment, timestamp tăng dần trong thời lượng.

---

## P0-7 · Provider Captioner `[T-22]`

**File:** `api/shared/providers/captioner.py`

**Contract:** `Captioner.caption(image_path: str) -> str`

**Impl:** `GeminiVisionCaptioner(api_key, model="gemini-2.5-flash", client=None)` — mở PIL image, `generate_content([_PROMPT, image])`; `_PROMPT` yêu cầu 1 câu tiếng Việt mô tả người/hoạt động/bối cảnh. Trả rỗng → `RuntimeError` (tầng pipeline nuốt thành `""`).

**DoD:** ảnh mẫu → caption tiếng Việt; thiếu key → lỗi rõ ngay khi khởi tạo.

---

## P0-8 · Provider LLM `[T-23]`

**File:** `api/shared/providers/llm.py`

**Contract:** `LLMClient.generate(prompt: str, *, temperature: float = 0.2) -> str`

**Impl:** `GeminiClient(api_key, model="gemini-2.5-flash", client=None)`; rỗng (safety block/quota) → `RuntimeError("Gemini trả rỗng…")` để tầng answer fallback.

**Ràng buộc vận hành:** free-tier ~15 req/phút → caption/summary tính OFFLINE, online chỉ 1 call LLM/lượt chat; câu đơn giản có thể chuyển `gemini-2.5-flash-lite` qua `LLM_MODEL`.

**DoD:** `generate` trả text ổn định; thiếu key → RuntimeError hướng dẫn đặt `.env`.

---

## P0-9 · Config & DI wiring

**File:** `api/app/config.py` + `api/app/deps.py`

**Quy tắc kiến trúc (bất biến toàn dự án):**
1. `shared/` KHÔNG import `app.config` — mọi wiring đọc config nằm ở `app/deps.py`.
2. Factory `@lru_cache`: `get_llm · get_text_embedder · get_image_embedder · get_asr · get_captioner · get_media_store · get_transcript_store · get_regulation_store` — tạo 1 lần, lazy import bên trong.
3. Mọi hàm lõi nhận provider **tùy chọn** (`embedder=None, store=None, llm=None…`), `None` → build từ deps. Test luôn inject fake — không cần GPU/network/Qdrant.

**Config keys:** `database_url · qdrant_url · qdrant_collection_{media,transcript,regulation} · llm_provider/llm_api_key/llm_model · data_dir · jwt_secret`.
