# Spec P1 — Ingest & Pipeline offline

> Phase 1 · Epic E1 + E2 · Sprint 2–3 · Owner chính: DE
> Mục tiêu: biến media/nội quy thô thành **dữ liệu tìm được** trong Qdrant + Postgres.
> Nguyên tắc NFR: chạy ở worker/CLI, **không nằm trên request path**; lỗi 1 mục → log + bỏ qua, KHÔNG chặn batch.
> Exit gate: 1 video thật qua `python -m pipeline.run` → vector trong 3 collection; nạp nội quy → truy "Điều X" đúng; upload không consent → 403.

## Phạm vi

| ID | Chức năng | Task | Ghi chú |
|---|---|---|---|
| P1-1 | Consent gate | T-04 | cho sẵn — domain mẫu |
| P1-2 | Media storage `get_media` | T-06 | |
| P1-3 | Ingest upload | T-07 | |
| P1-4 | Nạp nội quy tách điều/khoản | T-08 | |
| P1-5 | Trích keyframe | T-10 | |
| P1-6 | Caption | T-12 | |
| P1-7 | Embed ảnh/text | T-13 | |
| P1-8 | Build/upsert Qdrant | T-14 | 🔴 đường găng |
| P1-9 | ASR + chunk transcript | T-11 | |
| P1-10 | Orchestrator `run.py` | T-15 | |

## Sơ đồ luồng offline

```
video ─┬─► frames (ffmpeg keyframe + ts) ─► caption (Gemini, opt-in) ─► embed ảnh (Jina-CLIP) ─► Qdrant "media_clip"
       └─► audio 16k mono (ffmpeg) ─► ASR (PhoWhisper) ─► chunk (~800 char, 15% overlap) ─► embed text (Vietnamese_Embedding) ─► Qdrant "video_transcript"

file nội quy ─► tách điều/khoản (rule_chunks) ─► embed text ─► Qdrant "regulation_text"
```

Hai nhánh video **độc lập**: hỏng frame vẫn có transcript và ngược lại.

---

## P1-1 · Consent gate `[T-04]` (cho sẵn — domain mẫu)

**File:** `app/domains/consent/*` + `app/deps.py::require_consent`

**Hành vi:** mọi endpoint ingestion `Depends(require_consent)`; không có consent active → `403 {"detail": "Chưa có quyền sử dụng dữ liệu"}` (US-101.1 AC-1). Lô dữ liệu nạp được gắn `consent_id` (AC-2).

**DoD:** `test_consent.py` xanh. *Domain này scaffold cho sẵn làm mẫu — đọc để học pattern `router.py · service.py · schemas.py` trước khi code domain của mình.*

---

## P1-2 · Media storage `[T-06]`

**File:** `app/domains/media/service.py` + `router.py` · Phụ thuộc: P0-3

**API:**
```
GET /media/{id}          → file (FileResponse) — dùng cho <img>/<video> ở web
GET /media/{id}/meta     → MediaAssetOut {id, media_type, posted_at, caption_original, source_url, event}
```

**Chữ ký service (đề xuất):**
```python
def get_media(session: Session, media_id: int) -> MediaAsset      # 404 nếu không có
def get_media_file(session: Session, media_id: int) -> Path        # kiểm file tồn tại
```

**Edge:**
- `id` không tồn tại → `HTTPException(404, "Không tìm thấy media")`.
- Bản ghi còn nhưng file mất trên storage → `500` message rõ + set `needs_review=True` (đánh dấu hỏng, US-105.1 edge).

**DoD / Test `TC-105`:** lưu 1 media (fixture) → GET trả đúng file+metadata; id lạ → 404.

---

## P1-3 · Ingest upload `[T-07]`

**File:** `app/domains/ingest/service.py::save_upload` + router · Phụ thuộc: P0-3, P1-1

**API:**
```
POST /ingest/upload   (multipart, Depends(require_consent), Depends(require_admin))
  files[]: ảnh/video
  metadata: {source_url, posted_at, caption_original?, event_name?}   # bắt buộc trừ ?
→ 200 {"created_ids": [...], "skipped": [{"file":..., "reason":...}]}
```

**Chữ ký service (đề xuất):**
```python
ALLOWED = {".jpg", ".jpeg", ".png", ".webp", ".mp4", ".mov"}

def save_upload(session, files: list[UploadFile], meta: UploadMeta,
                consent_id: int, *, data_dir: str | None = None) -> UploadResult
```

**Hành vi:**
1. Validate metadata bắt buộc (`source_url`, `posted_at` ISO) → thiếu → `422` chỉ rõ trường (US-102.1 AC-2).
2. Mỗi file: check đuôi ∈ ALLOWED → sai → vào `skipped` kèm lý do, không chặn file khác.
3. Tính `checksum` (sha256) → trùng bản ghi cũ → `skipped: "duplicate"` không tạo mới (nối BR-106).
4. Lưu file `data/raw/{images|videos}/<checksum><ext>`; tạo `posts` (1 post/batch metadata) + `media_assets` gắn `consent_id`.
5. `media_type` suy từ đuôi; thiếu trường chuẩn hoá → `needs_review=True` (US-104.1).

**DoD / Test `TC-102`:** upload batch → số bản ghi = số file hợp lệ; thiếu `posted_at` → 422 nêu tên trường; upload lại file trùng → không nhân đôi.

---

## P1-4 · Nạp nội quy `[T-08]`

**File:** `app/domains/ingest/service.py::load_regulations` · Phụ thuộc: P0-3

**API:**
```
POST /ingest/regulations   (Depends(require_admin))
  file: .md/.txt (v1; .docx/.pdf → v2)
  meta: {title, version, effective_date}
→ 200 {"regulation_id": N, "n_chunks": M, "needs_review": bool}
```

**Chữ ký service (đề xuất):**
```python
def load_regulations(session, text: str, meta: RegulationMeta) -> Regulation
def split_rules(text: str) -> list[RuleChunk]   # pure function, test riêng
```

**Thuật toán `split_rules` (v1, regex):**
- Bắt heading `Điều \d+` (± tiêu đề cùng dòng) → mở article mới.
- Trong article, bắt `Khoản \d+` / `\d+\.` đầu dòng → clause; không có khoản → cả điều là 1 chunk.
- Ra `{article: int, clause: int|None, title: str, text: str}`.

**Hành vi phiên bản:** nạp version mới cùng `title` → bản cũ `is_active=False` (giữ lịch sử), chỉ bản active được index/phục vụ (US-107.1 AC-2).

**Edge:** file không match cấu trúc điều/khoản → tách theo heading/đoạn trống + `needs_review=True`, vẫn nạp (AC-1 nới).

**Nối P2:** sau khi nạp → embed từng chunk (P1-7 `embed_texts`) → `build_index("regulation_text", …)` payload `{article, clause, text, regulation_id}`.

**DoD / Test `TC-107`:** nạp văn bản mẫu → truy "Điều 3" trả đúng nội dung điều 3; nạp version 2 → version 1 không được phục vụ.

---

## P1-5 · Trích keyframe `[T-10]`

**File:** `pipeline/frames.py`

**Chữ ký:**
```python
@dataclass
class Frame: media_asset_id: int; timestamp_sec: float; frame_path: str

def extract_keyframes(video_path, media_asset_id, *, step_sec=2.0,
                      max_frames=60, out_dir="./data/frames") -> list[Frame]
```

**Hành vi:** `ffprobe` đọc duration → mẫu đều mỗi `step_sec` (chặn `max_frames`) → `ffmpeg -ss t -i video -frames:v 1 out.jpg`.

**Edge (US-201.1):** video hỏng/codec lạ → `[]` + log, KHÔNG raise; timestamp tăng dần, không trùng, < duration, sai số ≤0.5s; video dài → `max_frames` chặn tải.

**DoD:** `test_frames_have_timestamp`, `test_corrupt_video_returns_empty`.

---

## P1-6 · Caption `[T-12]`

**File:** `pipeline/caption.py`

**Chữ ký:** `generate_caption(image_path: str, *, captioner=None) -> str`

**Hành vi:** delegate `captioner.caption(image_path)`; **mọi Exception → log warning + trả `""`** (gắn cờ review, không chặn pipeline — US-203.1 edge). `""` sau này = `low_confidence` khi ghi bảng `captions`.

**DoD:** `test_caption_generated`, `test_caption_error_returns_empty`.

---

## P1-7 · Embed `[T-13]`

**File:** `pipeline/embed.py`

**Chữ ký:**
```python
def embed_images(paths: list[str], *, embedder=None) -> list[list[float]]
def embed_texts(texts: list[str], *, embedder=None) -> list[list[float]]
```

**Bất biến:** lọc `os.path.exists` TRƯỚC khi embed + log số bị bỏ → caller phải lọc frame theo cùng điều kiện để **vector↔payload 1:1** (xem P1-10).

**DoD:** `test_embed_same_dim`, `test_embed_skips_missing_file`.

---

## P1-8 · Build/upsert Qdrant `[T-14]` 🔴 đường găng

**File:** `pipeline/index.py`

**Chữ ký:** `build_index(collection, dim, ids, vectors, payloads, *, store=None) -> None`

**Hành vi:** validate `len(ids)==len(vectors)==len(payloads)` và mỗi `len(v)==dim` → sai → `ValueError("sai chiều")` KHÔNG upsert gì (US-205.1 AC-2); `store.ensure_collection()` → `store.upsert(ids, vectors, payloads)`.

**Payload contract (3 collection — retrieval P2 đọc theo tên khoá):**
| Collection | Payload |
|---|---|
| `media_clip` | `{video_id, timestamp, frame_path, caption}` |
| `video_transcript` | `{video_id, start_sec, end_sec, text}` |
| `regulation_text` | `{article/dieu, clause/khoan, text, regulation_id?}` |

**DoD:** `test_index_upsert`, `test_index_rejects_wrong_dim`.

---

## P1-9 · ASR + chunk `[T-11]`

**File:** `pipeline/asr.py`

**Chữ ký:**
```python
def extract_transcript(video_path, media_asset_id, *, asr=None,
                       out_dir="./data/audio") -> list[TranscriptSegment]
def chunk_transcript(segments, media_asset_id, *,
                     max_chars=800, overlap_chars=120) -> list[dict]
```

**Hành vi:**
- `_extract_audio`: ffmpeg → WAV 16kHz mono; video không audio stream → `None` → trả `[]` KHÔNG gọi ASR, không lỗi (US-208.1 AC-2).
- `chunk_transcript`: cửa sổ ~`max_chars` (≈200–300 token TV) + overlap ~15%; giữ `start_sec` chunk = start segment đầu, `end_sec` = end segment cuối; chống trùng đoạn cuối bằng con trỏ `emitted_upto`.
- Output dict khớp payload `video_transcript`.

**DoD:** `test_transcript_segments`, `test_transcript_empty_when_no_audio`, `test_chunk_transcript_*`.

---

## P1-10 · Orchestrator `[T-15]`

**File:** `pipeline/run.py`

**Chữ ký:**
```python
def process_video(video_path, media_asset_id, *,
    collection="media_clip", out_dir="./data/frames",
    embedder=None, store=None,            # nhánh media
    captioner=None,                        # bật caption (opt-in)
    asr=None, text_embedder=None,          # bật transcript (opt-in qua asr)
    transcript_store=None, transcript_collection="video_transcript",
) -> dict  # {media_asset_id, collection, n_frames, n_indexed, n_captioned, n_transcript}
```

**Luồng:**
1. **Nhánh media (luôn chạy):** keyframe → lọc file tồn tại (giữ 1:1) → embed → (caption nếu có captioner, lỗi→`""`) → `build_index` payload kèm `caption`.
2. **Nhánh transcript (khi `asr` truyền vào):** extract → rỗng → bỏ nhánh êm; chunk → embed_texts → `build_index`.
3. Lệch số lượng frame↔vector → log error + bỏ nhánh, không raise.

**CLI:** `cd api && python -m pipeline.run <video_path> [media_asset_id]` — không truyền provider → build thật từ `app.deps` (cần model + Qdrant).

**Quy ước dọn dẹp:** mọi artifact ghi theo `out_dir` truyền vào; test dùng `tmp_path` — KHÔNG để rơi `./data/` khi test.

**DoD:** `test_offline_chain_indexes_media`, `_captions_media`, `_indexes_transcript`.

---

## Hạng mục tích hợp để đóng P1

1. **P1-2/3/4** (storage + ingest + nội quy) — cần P0-3 models trước.
2. **Ghi DB song song Qdrant:** spec v1 của `process_video` chỉ ghi Qdrant; khi có models (P0-3) → ghi thêm `video_frames`/`transcripts`/`captions`/`embeddings` (1 transaction/video).
3. **Batch runner:** vòng lặp `process_video` trên toàn kho `media_assets` chưa index (query `needs_review=False`), log tổng kết.
