# Phase Plan & Functional Spec — HIT-MIRA Multimodal RAG

> Chia dự án thành **5 Phase** theo dòng chảy giá trị (nền → dữ liệu → truy xuất → trả lời → app/đo).
> Mỗi Phase liệt kê **từng chức năng** kèm spec: mục tiêu · input/output (chữ ký hàm) · hành vi & edge · DoD (test).
> Nguồn: [tasks.md](tasks.md) (T-xx), [prd.md](prd.md) (US/AC), [tech-pipeline.md](tech-pipeline.md) (model). Bám code base `api/`.
> **Spec chi tiết từng Phase** (API contract, schema, luồng, test plan): [specs/](specs/README.md).
>
> **Đường găng:** `T-05 → T-14 → T-30 → T-41 → T-50 → T-60 → T-74`.
> **Theo dõi tiến độ:** trên GitHub Project / [tasks.md](tasks.md) — tài liệu này mô tả *cái cần làm*, không đánh dấu done/todo.

## Bản đồ Phase

| Phase | Tên | Epic | Sprint | Mục tiêu chốt Phase |
|---|---|---|---|---|
| **P0** | Nền móng & Provider | E0, E3 | 1 | Hạ tầng chạy (docker, DB, CI) + 4 provider (embed/ASR/caption/LLM) hoạt động |
| **P1** | Ingest & Pipeline offline | E1, E2 | 2–3 | Nạp media/nội quy có consent → keyframe·caption·ASR·embed·index Qdrant |
| **P2** | Truy xuất & Router | E4, T-40 | 3 | 3 nhánh retrieve + rank ngưỡng + tool registry + router phân loại |
| **P3** | Sinh câu trả lời & Chat | E5, E6 | 4 | RAG synth + trích dẫn + orchestrator `/chat/message` + auth + media serving |
| **P4** | Frontend · Eval · Demo | E7, E8 | 5 | Web chat đa phương thức + Recall@k/MRR/latency + demo 4 luồng |

---

# P0 · Nền móng & Provider

**Mục tiêu Phase:** dựng chỗ chạy thật (Postgres + Qdrant + CI) và 4 adapter model sau lớp abstraction. Đây là điều kiện để mọi Phase sau chạy end-to-end với dữ liệu thật.

**Cổng ra Phase (exit gate):** `docker compose up` → 2 service healthy, `GET /health` OK · `pytest` xanh trong CI · 4 provider khởi tạo được (có key/model thật).

### P0-1 · Hạ tầng container `[T-01]`
- **File:** `docker-compose.yml` (đang rỗng)
- **Mục tiêu:** postgres + qdrant + api + (web) + volume bền, mạng nội bộ.
- **Output:** service `postgres:16` (volume `pgdata`), `qdrant` (volume `qdrant_storage`, port 6333), `api` (build `api/Dockerfile`, env từ `.env`), healthcheck cho từng service.
- **Edge:** thiếu `.env` → dùng default trong `config.py` (sqlite + qdrant localhost); qdrant chưa sẵn sàng → api retry connect.
- **DoD:** `docker compose up` → postgres+qdrant healthy, `curl /health` trả `{"status":"ok"}`.

### P0-2 · Migration schema `[T-02]`
- **File:** `shared/db/migrations/` (Alembic)
- **Mục tiêu:** version hoá schema thay cho `create_all`.
- **Phụ thuộc:** P0-3 (models).
- **DoD:** `alembic upgrade head` dựng đủ bảng khớp `models.py`; `downgrade` sạch.

### P0-3 · Data model — 19 bảng ORM `[T-05]` 🔴 đường găng
- **File:** `shared/db/models.py` (hiện chỉ có `consent`)
- **Mục tiêu:** khai báo toàn bộ bảng quan hệ PRD §5.
- **Bảng chính:** `posts`, `media_assets`, `events`, `video_frames`, `transcripts`, `captions`, `regulations`, `rule_chunks`, `embeddings`, `conversations`, `messages`, `users`, `consent`, `eval_queries`, `ocr_texts`(v2), `faces`(v2)…
- **Ràng buộc:** FK `media_assets.post_id→posts`, `video_frames.media_id→media_assets`, `rule_chunks.regulation_id→regulations`; cột chuẩn hoá `posted_at`(ISO), `media_type`, `source_url`, `needs_review`.
- **Edge:** bản ghi thiếu trường bắt buộc → `needs_review=true`, loại khỏi tập index (US-104.1).
- **DoD:** `Base.metadata.create_all` chạy sạch; FK/relationship load được; `conftest` sqlite dựng đủ bảng.

### P0-4 · CI lint + test `[T-03]`
- **File:** `.github/workflows/ci.yml`
- **DoD:** PR tự chạy `ruff check .` + `pytest` (cd api); fail → chặn merge.

### P0-5 · Provider Embeddings `[T-20]`
- **File:** `shared/providers/embeddings.py`
- **Chữ ký (ABC + impl):**
  - `TextEmbedder.embed(texts: list[str]) -> list[list[float]]`, `.dim`
  - `ImageEmbedder.embed(image_paths: list[str]) -> list[list[float]]`, `.embed_query(text: str) -> list[float]`, `.dim`
  - `VietnameseTextEmbedder` = SentenceTransformer `AITeamVN/Vietnamese_Embedding`, `normalize_embeddings=True`
  - `JinaClipEmbedder` = SentenceTransformer `jinaai/jina-clip-v2` — **ảnh và text query chung 1 không gian** (bật text→ảnh)
- **Hành vi:** lazy import torch/ST trong `__init__`; `trust_remote_code=True`.
- **DoD:** embed cùng `dim`; query text tiếng Việt → ảnh liên quan ngữ nghĩa trong top-k (US-202.1).

### P0-6 · Provider ASR `[T-21]`
- **File:** `shared/providers/asr.py`
- **Chữ ký:** `ASRModel.transcribe(audio_path: str) -> list[TranscriptSegment]`; `TranscriptSegment(start_sec, end_sec, text, confidence)`.
- **Impl:** `FasterWhisperASR(model_name="large-v3", language="vi", vad_filter=True)` — backend faster-whisper (CTranslate2) cho PhoWhisper.
- **DoD:** audio mẫu → ≥1 segment có text + timestamp tăng dần (`test_transcript_segments`).

### P0-7 · Provider Captioner `[T-22]`
- **File:** `shared/providers/captioner.py`
- **Chữ ký:** `Captioner.caption(image_path: str) -> str`; `GeminiVisionCaptioner(api_key, model="gemini-2.5-flash")`.
- **Hành vi:** mở PIL image → `generate_content([_PROMPT_vi, image])`; rỗng → raise (để P1-6 nuốt thành `""`).
- **DoD:** ảnh mẫu → caption tiếng Việt 1 câu.

### P0-8 · Provider LLM `[T-23]`
- **File:** `shared/providers/llm.py`
- **Chữ ký:** `LLMClient.generate(prompt: str, *, temperature=0.2) -> str`; `GeminiClient(api_key, model="gemini-2.5-flash")`.
- **Hành vi:** thiếu key → RuntimeError rõ; Gemini rỗng (safety/quota) → raise (để tầng answer fallback).
- **DoD:** `generate(prompt)` trả text ổn định.

### P0-9 · Config & Wiring (DI)
- **File:** `app/config.py` (pydantic-settings), `app/deps.py` (`@lru_cache` factories)
- **Quy tắc kiến trúc:** `shared/` KHÔNG import `app.config`; mọi wiring đọc config nằm ở `deps.py`; provider lazy-import.
- **Factories:** `get_llm/get_text_embedder/get_image_embedder/get_asr/get_captioner/get_media_store/get_transcript_store/get_regulation_store`.

---

# P1 · Ingest & Pipeline offline

**Mục tiêu Phase:** biến media/nội quy thô thành dữ liệu tìm được. Chạy ở worker/CLI, **không nằm trên request path**. Lỗi 1 mục → log + bỏ qua, không chặn batch.

**Cổng ra Phase:** 1 video thật qua `python -m pipeline.run` → có frame + (caption) + transcript + vector trong 3 collection Qdrant; nạp 1 file nội quy → truy "Điều X" đúng.

### P1-1 · Consent gate `[T-04]`
- **File:** `app/domains/consent/*`, `app/deps.py::require_consent`
- **Spec:** chưa có consent hiệu lực → mọi endpoint ingestion trả `403 "Chưa có quyền sử dụng dữ liệu"` (US-101.1).
- **DoD:** `test_consent.py` xanh.

### P1-2 · Media storage `[T-06]`
- **File:** `app/domains/media/service.py::get_media`
- **Chữ ký (đề xuất):** `get_media(session, media_id: int) -> MediaAsset` + route `GET /media/{id}`.
- **Output:** file media + metadata theo id.
- **Edge:** id không tồn tại → 404; bản ghi còn nhưng file mất → lỗi rõ + đánh dấu hỏng (US-105.1).
- **DoD:** `TC-105`: GET đúng file+metadata; id lạ → 404.

### P1-3 · Ingest upload `[T-07]`
- **File:** `app/domains/ingest/service.py::save_upload`
- **Chữ ký (đề xuất):** `save_upload(session, files, metadata) -> list[int]` (ids đã tạo), sau `require_consent`.
- **Output:** bản ghi `posts` + `media_assets`, file vào storage.
- **Edge:** định dạng không hỗ trợ → từ chối kèm lý do; thiếu trường metadata bắt buộc → chặn, chỉ rõ trường thiếu; checksum trùng → cảnh báo, không nhân đôi (US-102.1/104.1).
- **DoD:** `TC-102`: số bản ghi = số file hợp lệ; thiếu trường → chặn.

### P1-4 · Nạp nội quy `[T-08]`
- **File:** `app/domains/ingest/service.py::load_regulations`
- **Chữ ký (đề xuất):** `load_regulations(session, file, meta) -> Regulation` + list `rule_chunks`.
- **Output:** `regulations` + `rule_chunks{article/dieu, clause/khoan, title, text}` truy theo số điều; gắn phiên bản + ngày hiệu lực.
- **Edge:** file không rõ cấu trúc → tách theo heading/đoạn + `needs_review`; phiên bản mới → giữ lịch sử, chỉ bản hiệu lực được phục vụ (US-107.1).
- **DoD:** `TC-107`: truy "Điều X" trả đúng nội dung.

### P1-5 · Trích keyframe `[T-10]`
- **File:** `pipeline/frames.py::extract_keyframes`
- **Chữ ký:** `extract_keyframes(video_path, media_asset_id, *, step_sec=2.0, max_frames=60, out_dir="./data/frames") -> list[Frame]`; `Frame(media_asset_id, timestamp_sec, frame_path)`.
- **Hành vi:** `ffprobe` lấy duration → lấy mẫu theo `step_sec` → `ffmpeg -ss t -frames:v 1`.
- **Edge:** video hỏng/codec lạ → `[]` (không raise, không chặn batch); timestamp tăng dần, sai số ≤0.5s (US-201.1).
- **DoD:** `test_frames_have_timestamp`; `test_corrupt_video_returns_empty`.

### P1-6 · Caption `[T-12]`
- **File:** `pipeline/caption.py::generate_caption`
- **Chữ ký:** `generate_caption(image_path, *, captioner=None) -> str`.
- **Hành vi:** delegate `captioner.caption`; **mọi lỗi → `""`** (gắn cờ review, không chặn pipeline).
- **DoD:** `test_caption_generated`; `test_caption_error_returns_empty`.

### P1-7 · Embed `[T-13]`
- **File:** `pipeline/embed.py`
- **Chữ ký:** `embed_images(paths, *, embedder=None) -> list[vec]`; `embed_texts(texts, *, embedder=None) -> list[vec]`.
- **Hành vi:** lọc file không tồn tại trước khi embed → **giữ vector↔payload 1:1**; log số bị bỏ.
- **DoD:** `test_embed_same_dim`; `test_embed_skips_missing_file`.

### P1-8 · Build/upsert index `[T-14]` 🔴 đường găng
- **File:** `pipeline/index.py::build_index`
- **Chữ ký:** `build_index(collection, dim, ids, vectors, payloads, *, store=None)`.
- **Hành vi:** validate độ dài bằng nhau + mỗi `len(v)==dim` (sai → `ValueError "sai chiều"`); `ensure_collection()` → `upsert()`.
- **DoD:** `test_index_upsert`; `test_index_rejects_wrong_dim`.

### P1-9 · ASR transcript + chunk `[T-11]`
- **File:** `pipeline/asr.py`
- **Chữ ký:** `extract_transcript(video_path, media_asset_id, *, asr=None, out_dir="./data/audio") -> list[TranscriptSegment]`; `chunk_transcript(segments, media_asset_id, *, max_chars=800, overlap_chars=120) -> list[dict]`.
- **Hành vi:** `ffmpeg` tách audio 16kHz mono WAV; câm/không audio → `None` → `[]` (không gọi ASR, không lỗi). Chunk ~200–300 token (~800 char TV), ~15% overlap, giữ timestamp; payload `{video_id, start_sec, end_sec, text}`.
- **Edge:** video câm → `[]` (US-208.1 AC-2).
- **DoD:** `test_transcript_segments`; `test_transcript_empty_when_no_audio`; `test_chunk_transcript_*`.

### P1-10 · Orchestrator pipeline `[T-15]`
- **File:** `pipeline/run.py::process_video`
- **Chữ ký:** `process_video(video_path, media_asset_id, *, collection="media_clip", out_dir="./data/frames", embedder, store, captioner, asr, text_embedder, transcript_store, transcript_collection="video_transcript") -> summary`.
- **Hành vi:** 2 nhánh **độc lập** — media luôn chạy (keyframe→caption opt-in→embed→index, payload `{video_id, timestamp, frame_path, caption}`); transcript chỉ chạy khi truyền `asr`. Summary `{n_frames, n_indexed, n_captioned, n_transcript}`. CLI: `python -m pipeline.run <video> [media_id]`.
- **DoD:** `test_offline_chain_indexes_media/_captions_media/_indexes_transcript`.

---

# P2 · Truy xuất & Router

**Mục tiêu Phase:** từ câu hỏi → chọn nguồn → truy xuất top-k từ Qdrant → lọc ngưỡng. Toàn bộ inject provider/store để test bằng fake.

**Cổng ra Phase:** router phân đúng media/regulation; 3 hàm retrieve trả top-k đúng payload; rank cắt ngưỡng "không tìm thấy"; tool trả `ToolResult` đúng nhãn.

### P2-1 · retrieve_media `[T-30]` 🔴 đường găng
- **File:** `app/domains/retrieval/service.py::retrieve_media`
- **Chữ ký:** `retrieve_media(query, top_k=5, *, embedder=None, store=None) -> list[dict]`.
- **Hành vi:** `embedder.embed_query(query)` (text→ảnh chung không gian) → `store.search` → `_hits_to_items` map `{id, score, **payload}`.
- **Edge:** ngoài miền → để P2-4 rank cắt (US-301.1).
- **DoD:** `TC-301` recall@k; map đủ trường + tôn trọng top_k.

### P2-2 · retrieve_regulations `[T-31]`
- **File:** cùng service, `retrieve_regulations(query, top_k=5, *, embedder=None, store=None)`.
- **Hành vi:** `embedder.embed([query])[0]` (Vietnamese_Embedding) → search collection nội quy → trả `{article/dieu, clause/khoan, text, score}`.
- **DoD:** `TC-307`: điều/khoản đúng theo điểm giảm dần.

### P2-3 · retrieve_by_transcript `[T-32]`
- **File:** cùng service, `retrieve_by_transcript(query, top_k=5, *, embedder=None, store=None)`.
- **Hành vi:** search collection transcript → **gộp hit theo `video_id`** (giữ max score, gom `moments` = các đoạn start/end) → 1 video không nhân đôi.
- **DoD:** `TC-308`: đúng video+đoạn, không nhân đôi.

### P2-4 · rank + ngưỡng `[T-33]`
- **File:** cùng service, `rank(candidates, threshold=0.0)`.
- **Hành vi:** lọc `score >= threshold`, sort giảm dần; tất cả dưới ngưỡng → `[]` (tín hiệu "không tìm thấy" cho P3).
- **DoD:** `TC-306` `test_ranking_and_threshold`.

### P2-5 · Tools + registry `[T-34]`
- **File:** `app/tools/{base,registry,media_tool,regulation_tool}.py`
- **Chữ ký:** `BaseTool.run(query, *, top_k=5, threshold=0.0, **kwargs) -> ToolResult(source, items)`.
- **Hành vi:** `rank(retrieve_*(query, top_k), threshold)` → `ToolResult(source="media"|"regulation", items=hits)`. Registry là đường nâng lên Agentic v2.
- **DoD:** `test_tools.py`: `ToolResult` đúng nhãn + tôn trọng threshold.

### P2-6 · Router — luật `[BR-507]`
- **File:** `app/routing/intent.py::route`
- **Chữ ký:** `route(text: str|None, has_image: bool, override: Source|None) -> Source` (MEDIA/REGULATION/BOTH).
- **Hành vi:** override → dùng luôn; có ảnh → MEDIA; từ khóa nội quy (`REGULATION_KEYWORDS`) → REGULATION; else MEDIA.
- **DoD:** câu hỏi nội quy → REGULATION; có ảnh → MEDIA.

### P2-7 · Router — classifier fallback `[T-40]`
- **File:** `app/routing/intent.py`
- **Mục tiêu:** khi luật không rõ (không ảnh, không từ khóa) → dùng classifier nhẹ (LLM zero-shot hoặc embedding-similarity) đoán MEDIA/REGULATION/BOTH thay vì mặc định MEDIA.
- **Chữ ký (đề xuất):** `classify(text, *, llm=None) -> Source`, gọi trong `route` trước nhánh default; injectable để test.
- **Edge:** mơ hồ giữa 2 nguồn → BOTH (chạy cả hai rồi gộp); cho phép override thủ công.
- **DoD:** `test_ambiguous_uses_classifier`; đo routing accuracy ở P4 (BR-606).

---

# P3 · Sinh câu trả lời & Chat

**Mục tiêu Phase:** ghép lõi RAG thành 1 luồng `/chat/message`, sinh câu trả lời bám nguồn + trích dẫn, không bịa; kèm auth và media serving để trả clip/preview.

**Cổng ra Phase:** `POST /chat/message` (text ± ảnh) → định tuyến đúng → câu trả lời có trích dẫn (media) / điều-khoản+disclaimer (nội quy) / "không tìm thấy" khi rỗng; non-admin bị chặn endpoint admin.

### P3-1 · synthesize_answer `[T-41]` 🔴 đường găng
- **File:** `app/domains/answer/service.py::synthesize_answer`
- **Chữ ký (đề xuất):** `synthesize_answer(query, hits, *, llm=None) -> Answer{text, sources}`.
- **Hành vi:** prompt grounded, trích dẫn đánh số `[n]` map tới nguồn; `_snippet` ưu tiên `caption`/`transcript`/`text`.
- **Edge:** hits rỗng → `handle_not_found` (KHÔNG gọi LLM); LLM lỗi → `_raw_fallback` (liệt kê nguồn thô + báo lỗi nhẹ) (US-401.1).
- **DoD:** bám nguồn, không thêm sự kiện ngoài ngữ cảnh; LLM lỗi → fallback không sập.

### P3-2 · answer_regulation `[T-42]`
- **File:** cùng service, `answer_regulation(query, rule_hits, *, llm=None)`.
- **Hành vi:** prompt neo `[Điều X Khoản Y]`; **luôn** append `_DISCLAIMER` ("thông tin tham khảo, BCN quyết định cuối").
- **Edge:** rỗng → "nội quy không quy định nội dung này"; LLM lỗi → `_raw_regulation_fallback` (vẫn kèm disclaimer); không suy diễn vượt văn bản (US-407.1).
- **DoD:** `TC-407`: citation điều/khoản + disclaimer.

### P3-3 · handle_not_found `[T-43]`
- **File:** cùng service, `handle_not_found(query) -> Answer`.
- **Hành vi:** thông báo cố định "chưa tìm thấy dữ liệu phù hợp" + gợi ý chỉnh truy vấn; không gọi LLM.
- **DoD:** `TC-406`: ngoài miền → không bịa.

### P3-4 · handle_message (orchestrator) `[T-50]` 🔴 đường găng — mắt xích nối lõi RAG
- **File:** `app/domains/chat/service.py::handle_message` (đang `NotImplementedError`)
- **Chữ ký:** `handle_message(text: str|None, image_path: str|None=None, *, override=None, ...) -> ChatReply`.
- **Hành vi (ráp 3 bước):**
  1. `route(text, has_image=image_path is not None, override)` → nguồn
  2. `registry.get(source).run(query)` → `ToolResult` (MEDIA/REGULATION/BOTH; BOTH → gọi cả 2 rồi gộp)
  3. MEDIA → `synthesize_answer(items)`; REGULATION → `answer_regulation(items)`; rỗng → `handle_not_found`
- **Output:** câu trả lời + nhãn nguồn đã dùng + items đa phương thức; route `POST /chat/message`.
- **Edge:** ảnh + text mâu thuẫn → ưu tiên ảnh (mặc định); ép override thủ công.
- **DoD:** `TC-507` `test_router.py`: định tuyến đúng + override; end-to-end với fake.

### P3-5 · Auth JWT `[T-51]`
- **File:** `app/domains/auth/*`, `app/deps.py::require_admin`
- **Chữ ký:** `login(cred) -> {token}`; `require_admin` giải mã JWT, kiểm role.
- **Edge:** non-admin gọi endpoint admin → 403.
- **DoD:** `TC-505`.

### P3-6 · Media serving — clip + stream `[T-52]`
- **File:** `app/domains/media/service.py`
- **Chức năng:** cắt clip ~6s (3s trước + 3s sau timestamp, ffmpeg); stream/seek video tới timestamp (`#t=`/range request).
- **Edge:** frame sát đầu/cuối → clip co theo biên; codec không seek → tải đoạn quanh timestamp (US-403/404).
- **DoD:** `TC-403` clip ≤6s chứa timestamp; `TC-404` seek ±1s.

### P3-7 · Lưu hội thoại `[T-53]`
- **File:** `app/domains/chat/*`
- **Chức năng:** lưu `conversations` + `messages` mỗi lượt gửi/nhận.
- **DoD:** `TC-501`: gửi/nhận được lưu.

---

# P4 · Frontend · Eval · Demo

**Mục tiêu Phase:** giao diện chat đa phương thức + đo chất lượng khách quan + checklist demo. **Lưu ý thứ tự:** bộ eval có nhãn (P4-4) nên làm sớm từ sprint 1–2 vì cần cho benchmark model.

**Cổng ra Phase:** web chat gửi/nhận + render ảnh/clip/link inline; báo cáo Recall@k/MRR/latency; 4 luồng lõi demo pass.

### P4-1 · Web chat `[T-60]` 🔴 đường găng
- **File:** `web/` (Next.js)
- **Chức năng:** khung chat text + upload ảnh; gọi `/chat/message`; responsive (Chrome/Firefox/Safari, desktop+mobile).
- **Edge:** timeout → lỗi + gửi lại, không mất nội dung; file quá lớn/sai định dạng → chặn (US-501/502/505).
- **DoD:** `TC-501/502/505`.

### P4-2 · Render đa phương thức `[T-61]`
- **File:** `web/`
- **Chức năng:** card kết quả inline — ảnh + mô tả, player clip, link nguồn (`source_url`, không có → "nguồn nội bộ").
- **Edge:** media lỗi → placeholder + thử lại, không vỡ layout (US-402/403/405/503).
- **DoD:** `TC-503`: ảnh + clip render/phát inline.

### P4-3 · Màn admin `[T-62]`
- **File:** `web/`
- **Chức năng:** nạp dữ liệu · trigger pipeline · xem eval.
- **DoD:** thao tác nạp/chạy pipeline/xem eval được.

### P4-4 · Bộ eval có nhãn `[T-70]` (làm sớm sprint 1–2)
- **File:** `data/eval/` + `eval_queries`
- **Chức năng:** ~30–50 truy vấn (text & ảnh) kèm media/điều-khoản kỳ vọng.
- **DoD:** `TC-601`: ≥N truy vấn có đáp án.

### P4-5 · Recall@k + MRR `[T-71]`
- **File:** `app/domains/eval/service.py`
- **Chữ ký:** `recall_at_k(results, gold, k)`, `mrr(results, gold)`.
- **DoD:** `TC-602` `test_recall_and_mrr_match_by_hand` (khớp tính tay).

### P4-6 · Latency `[T-72]`
- **File:** cùng service; đo avg + p95 luồng `/chat/message`.
- **DoD:** `TC-603`: avg ≤5s có p95.

### P4-7 · Eval nội quy & routing `[T-73]`
- **File:** cùng service; 3 chỉ số: đúng điều/khoản · groundedness · routing accuracy.
- **DoD:** `TC-606`.

### P4-8 · Demo checklist `[T-74]` 🔴 đích
- **File:** `docs/`
- **Chức năng:** 4 luồng lõi end-to-end (text→media, ảnh→sự kiện, nội quy, video-transcript).
- **DoD:** `TC-605`: 4 luồng pass.

---

## Phụ thuộc giữa Phase

```
P0 (T-05 models, T-01 docker) ──┬─→ P1 (ingest + pipeline → Qdrant)
P0 (T-20..23 providers) ────────┘        │
                                         ▼
                          P2 (retrieve + rank + router)
                                         │
                                         ▼
                          P3 (answer + T-50 chat + auth + media)
                                         │
                                         ▼
                          P4 (web + eval + demo)
```

## Gợi ý thứ tự thực thi (để chạy thật sớm nhất)

1. **P0-1** docker-compose (Qdrant+Postgres sống)
2. **P3-4** `handle_message` (nối route→tool→answer thành `/chat/message`)
3. Nạp 1–2 video thật qua `pipeline.run` → hỏi qua API → xác nhận luồng sống
4. Bồi **P0-3** models + **P1-2..4** ingest → dữ liệu vào có kỷ luật
5. **P4** web + eval + demo
