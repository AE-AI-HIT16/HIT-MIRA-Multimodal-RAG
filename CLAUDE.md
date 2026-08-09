# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

HIT-MIRA Multimodal RAG — a 3-person student capstone: a **Router RAG** (not yet agentic; agentic = v2) system that answers Vietnamese questions about club media (video/image) and club regulations, with citations.

**Docs and comments are written in Vietnamese — keep that convention.** All planning docs live in `docs/`: `brd.md` (BR-xxx business requirements) → `prd.md` (US-xxx stories, TC-xxx test cases, data model) → `tasks.md` (T-xx task board with owners/DoD) → `tech-pipeline.md` (model/technology decisions and rationale) → `structure.md` (directory layout rationale). Traceability chain BR → US → domain → TC/test is load-bearing; when adding code, reference the relevant IDs.

> `docs/structure.md` and parts of `docs/tech-pipeline.md` still describe an earlier planned layout. **This file describes what is actually in the tree.** When the two disagree, trust the code.

## Commands

All backend work happens from `API/` (imports assume `API/` is the working directory):

```bash
cd API
uvicorn src.server:app --reload      # dev server (health check: GET /health)
pytest                               # all tests (fakes + sqlite, no live services needed)
pytest tests/test_image_pipeline.py  # one file
pytest tests/test_image_pipeline.py -k image_route   # one test
ruff check .                         # lint
```

The MCP server is a separate process (`mcp/`, default port 8091):

```bash
cd mcp && PYTHONPATH=src python src/server.py
```

`docker-compose.yml` brings up postgres + qdrant + minio. Config comes from `.env` substituted into `API/Resources/{dev,model,prompt}.yaml` via `Template.safe_substitute(os.environ)` (see `API/src/config/config.py`); `AppConfig` in `API/src/configuration.py` is the single accessor. **Many settings also accept a direct env override that wins over the YAML** — e.g. `MEDIA_VISION_*` and `MEDIA_IMAGE_EMBEDDING_*`. Check `configuration.py` before assuming a value comes from YAML.

## Architecture

`API/src/` is the backend. The online/offline separation is an NFR — do not put heavy ML work on the request path.

- **`server.py` — app factory.** `create_app()` registers seven routers under `/api` (`documents`, `retrieval`, `media_retrieval`, `media_files`, `admin`, `auth`, `ingest`) plus `/health`. There is no `app/` package and no `domains/` package; router modules live in `src/routers/`.
  - **`include_api_router` copies routes by re-registering `route.endpoint`**, so anything attached to the `APIRouter` object rather than the endpoint function is silently dropped — including router-level `dependencies=[...]`. **Auth guards must therefore be per-endpoint `Depends`**, and a router-level guard would look correct while protecting nothing.
  - **A guard must be the *first* parameter with a `Depends`.** FastAPI resolves dependencies in declaration order, so an admin guard placed after a provider dependency lets an unauthenticated request run that provider first — `POST /api/ingest/upload` was opening MinIO connections before rejecting the caller. `tests/test_ingest_upload.py::test_can_quyen_admin` pins this.
- **`src/jobs/runner.py` — background jobs as subprocesses.** Backs the admin screen's index/eval buttons. It shells out to the existing `scripts/*.py --apply` rather than reimplementing indexing in-process: two copies of that logic would drift, and a multi-minute index on the request path violates the online/offline NFR. **Job state lives in one process's RAM** — multiple uvicorn workers or `--reload` each see a different job table.
- **`rag_video_anh/` — the media pipeline and media retrieval.** Split into `pipeline/` (offline: validate → route → keyframes → OCR/caption → detection → ASR → transcript mapping → normalize), `retrieval/` (units builder, indexing service, retriever, retrieval service), `embedding/`, `vector_store/`, `repository/` (SQLAlchemy models + Unit of Work), `schemas/`.
- **`rag_noiquy/` — the regulation/document path.** Its own `embedding/`, `pipeline/`, `retrieval/`, `vector_store/`. Backs `routers/documents.py` and `routers/retrieval.py`.
- **`src/rag/` is a broken orphan** — imports fail. Do not extend it; do not import from it.
- **`scripts/` — operational CLIs** (upload crawl to MinIO, register media rows, caption, index, RunPod jobs). Every destructive one is **dry-run by default and requires `--apply`**; keep that convention for new scripts.
- **`runpod_worker/` — GPU worker** (`handler.py`). Artifact-only: it receives presigned URLs, runs `scripts/export_video_artifacts.py`, and PUTs one ZIP back. It never sees PostgreSQL or MinIO credentials.
- **`embedding_server/` — self-hosted `jina-clip-v2`**, có hai lớp chạy chung một `encoder.py`: FastAPI/Jina wire protocol trong `server.py`, và RunPod Serverless queue adapter trong `runpod_handler.py`. Serverless dùng cho benchmark/index ngoại tuyến (active workers = 0); nếu sau benchmark vẫn cần Jina trên đường query online thì phải chọn worker luôn bật hoặc topology khác, không đẩy cold start vào request người dùng. Tách khỏi `runpod_worker/` vì worker kia xử lý video artifact và có dependency khác. Pins `transformers<5`: 5.x removed `clip_loss`, which Jina's remote code imports at module level.
- **`mcp/` — MCP server.** A tool is registered by adding an entry to `Resources/tools.yaml` **and** a same-named method on `FeatureManager`; `ToolRegistry` binds them by `getattr`. `log_exceptions` deliberately converts exceptions into `{"success": False, "error_type": ..., "message": ...}` — that is the MCP contract, not a swallowed error.

Providers are injected as optional constructor parameters (`image_embedder=`, `text_embedder=`, `vector_store=`, `storage=`, `builder=`, `vision_service=`…) so tests pass fakes — follow that pattern rather than instantiating models inside functions.

### Where each model is actually used

| Concern | Model | Wired in |
| --- | --- | --- |
| Images, video keyframes, transcript text, **and user queries** | **Jina-CLIP v2** (`jina-clip-v2`, 1024-d, cosine) | `rag_video_anh/embedding/embedding_service.py`, transport chosen by `embedding/provider.py` |
| Regulation/document text | `EMBEDDING_MODEL` = `jina-clip-v2` (same model — see below) | `rag_noiquy/embedding/embedding_service.py` |
| ASR | `hynt/Zipformer-30M-RNNT-6000h` via `sherpa_onnx` | `pipeline/asr_service.py` |
| Caption + OCR (one call returns both) | `MEDIA_VISION_MODEL_NAME` — any OpenAI-compatible endpoint that accepts `image_url`; no provider default | `pipeline/qwen_vision_service.py` |
| Query rewriting (regulation path) | `LLM_PROVIDER:LLM_MODEL` | `rag_noiquy/retrieval/query_rewriter.py` |

Answer synthesis lives in `ChatBot/` (LangGraph), outside `API/src/`.

**One model, two ways to reach it.** `build_media_embedder()` in `embedding/provider.py` is the only place that decides: a direct Jina-protocol endpoint (`MEDIA_IMAGE_EMBEDDING_BASE_URL`, always-on, no cold start) or a RunPod Serverless queue (`JINA_RUNPOD_ENDPOINT_ID`, `active workers = 0`, ~190s cold start). `runpod_transport.py` makes the queue look like the direct protocol at the `http_post=` seam, so the client keeps its batching, downscaling and error classification. Auto-detects, direct wins when both are set, `MEDIA_EMBEDDING_PROVIDER` forces. Detection reads env **before** YAML because `AppConfig` freezes these as dataclass defaults at import time. Online callers pass `for_online_queries=True`: the queue is not blocked there, but it warns and drops the job deadline from 900s to 240s.

Three things that surprise people:

1. **Transcripts are embedded with Jina-CLIP v2, not a dedicated Vietnamese text model.** `indexing_service.py` does `self.text_embedder = text_embedder or self.image_embedder`. That is what makes one query vector rank images *and* speech together — the joint space is the feature, not an oversight. `tech-pipeline.md` still names `AITeamVN/Vietnamese_Embedding`; changing to it would improve transcript retrieval but break the single-vector property.
2. **`jina-clip-v2` only accepts `task="retrieval.query"`.** Sending `retrieval.passage` returns HTTP 422. Asymmetric query/passage embedding belongs to `jina-embeddings-v3`, a different model.
3. **On this corpus the CLIP model also *wins* at text-to-text.** Measured 06/08/2026 on 505 post messages against 76 qrels, Jina-CLIP v2 beat Azure `text-embedding-3-small` on every metric — Recall@5 0.688 vs 0.565, MRR@10 0.728 vs 0.626, nDCG@10 0.693 vs 0.594 — with a rejection gap ~3× wider (+0.042 vs +0.015). See `docs/text-embedding-jina-vs-azure.md`. Both numbers are floors (qrels are not exhaustive); the *ordering* is what the measurement settles.
### One embedding model — a deliberate decision

**Every collection in this system is embedded by `jina-clip-v2` at 1024 dimensions**: `media_clip` (images + video keyframes), `video_transcript` (speech), and `rag_documents` (regulations). One key, one rate limit, one dimension. Keep it that way unless the trigger below fires.

The constraint that forces the choice: **only a multimodal model can embed images at all.** So "one model everywhere" necessarily means a CLIP-family model. Picking the strongest Vietnamese text model (`baai/bge-m3`, `AITeamVN/Vietnamese_Embedding`) instead means it cannot embed images, forcing a second model.

This section used to say "one model" and "best text retrieval" were mutually exclusive, on the assumption that the CLIP family is weakest at text-to-text. **Measurement on 06/08/2026 contradicted that** — see surprise #3 above. On post messages the multimodal model was also the better text retriever, so one vector space is not the trade-off it was written up as. The claim below about regulations is a *different corpus* and still unmeasured.

What that buys, and what it costs:

- **Buys:** `VideoRetrievalService.retrieve()` embeds the query **once** and searches `media_clip` and `video_transcript` in the same space. A separate text model would force two query embeddings per request. Fewer providers also means fewer outages — an expired vision-provider balance once took out captioning and the whole regulation path in one go, which is why the vision config is now provider-agnostic.
- **Costs:** measured on the real regulation chunks, Jina-CLIP v2 answered 6/7 queries top-1, but absolute scores sat at 0.3–0.6 and the rank-1-to-rank-2 gap was as thin as 0.01. That gets fragile as the corpus grows, and it makes a "not found" threshold hard to place — which matters because this system must never fabricate.

**Trigger to revisit:** when `rag_documents` exceeds roughly 50 chunks, re-measure top-1 on real questions. If it falls below ~80%, give **regulations only** a dedicated text model. That split costs nothing architecturally — regulations have their own collection, their own MCP tool (`search_regulations`), and their own router, and never share a query vector with images. Do **not** split transcripts out; that is what would break the single-query property.

Whichever model is chosen, **changing it means re-embedding the whole collection.** Two models are two vector spaces; mixing them in one collection makes ranking meaningless. Also note `EMBEDDING_CHECK_CTX_LENGTH=false`: LangChain otherwise fetches a HuggingFace tokenizer named after the model and dies with `OSError` for providers that have no HF repo, `jina-clip-v2` included.

### Online Router RAG flow

`POST /api/media/search` → `VideoRetrievalService.retrieve()` embeds the query **once** and searches both collections in that shared space, merges, and builds a citation context string. The tool registry in `mcp/` is deliberately kept as the upgrade path to agentic RAG in v2.

**`POST /api/media/search-image` is the image-query path** (multipart, US-302.1/US-303.1). The uploaded image is embedded by the same `jina-clip-v2` (`embed_image_blobs`, bytes only — never written to disk, BR-703) and compared against `media_clip` directly, so this is image→image, not "describe the image then search by text". Routing rules that are easy to get wrong:

- **An image never queries `video_transcript`.** Image-only requests skip that branch *by routing* and say so in the response's `notes` — treating an empty `videos` list as "nothing was said about this" is the misreading that field exists to prevent. Send text alongside the image and the text vector takes the transcript branch while the image vector takes the clip branch (two embed calls, only in this case).
- **Image wins over text on the clip branch** (US-502.1: "ảnh + text mâu thuẫn → ưu tiên ảnh"). Text-only requests still embed exactly once — do not regress that.
- MCP's `search_media` is still text-only. A tool call cannot carry an image, so the supervisor's *prose* answer is still based on the LLM reading the picture and inventing keywords; only the web's result cards use true image retrieval. Closing that needs an image-carrying MCP tool plus a LangGraph node that calls it before the supervisor.

**Filtering by year** — `years: [2024, 2025]` on both search routes (and on MCP `search_media`) filters by `post_created_at`, the **post's** timestamp, not the media row's (`media.created_at` is merely when the registration script ran, identical for every image). Three details are load-bearing:

- **Year boundaries are cut in UTC+7**, because the payload stores UTC (`2024-07-06T12:59:02Z`) while the person asking means the Vietnamese calendar. Cutting in UTC pushes a post made after 19:00 on 31/12 Hanoi time into the next year — rare, but that is exactly when year-in-review posts happen.
- **`video_ids` AND `years`, but year OR year.** `QdrantVideoVectorStore.search_filter` nests a `should` group per criterion inside one `must`; flattening them into a single `should` still returns plausible-looking results, which is why it is pinned by test.
- **Empty because of the filter is not empty because the corpus is empty** — the service appends a `notes` entry naming the years, same rule as the image/transcript routing note above.

Points indexed before these keys existed carry no `post_created_at` and are therefore invisible to any year filter. All 1,628 `media_clip` points were backfilled via `index_image_units.py --apply --payload-only`; new collections get the datetime payload index automatically from `ensure_collection`.

**Filtering by event** — `events: ["HIT Open Day"]` on the same three entry points, matching `event_key`, the **series** slug. It composes with `years` through the same `must`/`should` nesting, so "Open Day 2024" is an intersection.

- **The filter key is not what the user types.** The payload holds `hit-open-day`; the question says "HIT Open Day". Both sides go through `common_utils/text_keys.slugify` — that module exists precisely so the writer (`repository/events.py`) and the reader (`vector_store.py`) cannot drift, because drift here returns *zero results for a real event* with no error anywhere.
- **`đ` again.** `slugify` maps `đ→d` before stripping accents, for the same reason the ASR boilerplate gate had to; NFKD deletes `đ` outright, so "Đại hội" would key as `ai-hoi`.
- **Unknown event names are ignored, not rejected** — unlike years, which have a valid range and raise. There is nothing to validate a name against, so a wrong one simply matches nothing; the service says so in `notes`.
- **`event_key` is the series, never the occurrence.** `event_occurrences` are fine-grained (`offline-hang-thang` has 38 occurrences across 38 posts); keying payloads on them would make almost every filter return one or two points. The occurrence table earns its keep as data, not as a filter key.

**How events got there** (`scripts/extract_post_events.py`, run over 496 posts, ≈$1 of LLM per run):

- **`data/events/su_kien_chuan.yaml` is the source of truth for series names**, not the model. It is the club's own annual calendar, and it is committed (a `.gitignore` exception, same as `data/eval/eval_queries.yaml`) because losing it means series names revert to whatever the LLM invents. Its names go straight into the pass-1 prompt, so a post on the calendar gets the exact canonical name and **skips clustering entirely**.
- **Letting the model define the taxonomy got two things wrong**, both invisible without the club's list: it merged "Tuyển CTV" into "Tuyển thành viên HIT" (the club tracks these as separate drives), and it dropped all 14 of the 8/3 · 20/10 · 20/11 greeting posts because the prompt said holiday greetings are not events — the club puts them on the calendar. Entries are tagged `[CLB]` (from the club's list) or `[KHO]` (recurring in the corpus but not on the club's list — my inference, and the tag says so).
- **Two LLM passes, and pass 2 still matters for everything off-calendar.** Pass 1 extracts a name per post; pass 2 clusters only the names that did *not* match the calendar (external contests, seminars, one-off classes). Before anchoring, the corpus yielded 142 names where "Team Building", "Teambuilding" and "Team Building with HIT" were three separate `event_key`s — a filter that silently returns a third of the right answer.
- **Pass 2 is batched because one call timed out.** 142 names in a single prompt exceeded the `cx/gpt-5.5` proxy's patience. Batches of 40, **sorted by normalized name** so variants land together, then one final merge call over the canonical names. The merge round is deliberately *not* recursive — a model that groups nothing would otherwise loop forever on the same list.
- **A fabricated event is worse than a missing one**, so every extraction must carry a quote copied verbatim from the post; if the quote isn't in the post (whitespace/emoji/accents normalized away), the result is dropped. Posts with no event carry no `event_key` at all rather than a guess.
- Assignment is stored with `assigned_by="llm"`, the confidence, and the quote in `evidence` — so a wrong one is traceable to what the model read.
- **The calendar file outranks stored aliases** (`add_alias(..., ghi_de=True)`, used only when seeding from the YAML). Everywhere else an alias that already points elsewhere is left alone, because silently re-pointing one changes what a name means with no signal. Without that one exception, the bad `"Tuyển CTV" → tuyen-thanh-vien-hit` alias from the first run would have survived every later correction.
- **`--redo` deletes series that end up with no posts.** A re-run under a better taxonomy strands the old one's series; left in place they still look like valid filter choices that return nothing. Incremental runs skip this — there, "no posts yet" just means the posts have not been processed.
- **Writing the DB does not change payloads.** Run `backfill_media_text_post_metadata.py --collection <name> --apply` afterwards, once per collection. Posts that lost their event get the key *deleted*, not blanked — otherwise a stale `event_key` keeps matching a filter forever.

## Constraints worth knowing

- **Jina rate limit is 100,000 tokens/minute (sliding window); one 512px image costs exactly 4,000 tokens.** `ImageEmbeddingService._reserve_tokens` paces requests *proactively*; do not replace it with react-to-429 backoff, which measured ~10 images/min versus ~20–25. Images are downscaled to 512px before embedding (`MEDIA_IMAGE_EMBEDDING_MAX_SIDE`) — a 2048px image costs 6× the tokens for a cosine-0.994 identical vector.
- **The captioning VLM is `Qwen3-VL-8B-Instruct`, self-hosted on RunPod Serverless** (endpoint `o6k6y1xaoghddz`, vLLM via `runpod/worker-v1-vllm`). Two things about it bite:
  - **A vLLM template that sets only `MODEL_NAME` will not boot on a 48 GB GPU.** vLLM defaults `max_model_len` to `max_position_embeddings` = 262,144, which needs 36 GiB of KV cache (144 KiB/token × 36 layers × 8 KV heads) on top of ~16 GiB of weights — over budget, so the worker dies at startup and the endpoint reports `unhealthy` with no job ever running. `MAX_MODEL_LEN=32768` fixes it.
  - **`MEDIA_VISION_REPETITION_PENALTY=1.05` is required with a vLLM backend.** Without it, text-dense images (posters, banners) send the model into degenerate repetition until it hits the 2,048-token ceiling: measured 61.7 s and truncated JSON, versus 6.6 s and 194 tokens with the penalty. It is a vLLM extension, not standard OpenAI, so it is **off by default** and sent via `extra_body` only when configured — enabling it globally would 400 on any non-vLLM endpoint and take out the whole caption stage.
  - Measured 07/08/2026 on prompt `qwen-vision-v2`: 4–6 s/image warm, versus 84–361 s/image on the previous `cx/gpt-5.5` proxy. Cold start is ~3 min (no network volume, so weights re-download).
- **`media_clip` holds both static images and video keyframes**, told apart by the `media_kind` payload key (`"image"` / `"video_frame"`). Points indexed before that key existed are inferred from the presence of `video_id`.
- **Qdrant payload keys are a contract** — retrieval and answer synthesis read them by name:
  - image: `{media_kind, unit_id, image_media_id, post_id, bucket_name, frame_object_key, caption, ocr_text, vision_metadata, detected_objects, object_counts}`
  - video keyframe: same, plus `{video_id, frame_media_id, frame_index, timestamp_sec, transcript_context, transcript_context_start_sec, transcript_context_end_sec}`
  - transcript: `{video_id, unit_id, post_id, start_sec, end_sec, text, language, source_segment_ids}`
  - **Images deliberately carry no `timestamp_sec` and no `video_id`** so a citation can never invent a moment in a still photo.
- **Captions arrive after indexing.** Analysis and indexing are separate steps, so re-embedding to attach a caption is wasted quota — use `index_image_units.py --apply --payload-only`, which rewrites payloads and leaves vectors alone (1,628 points in ~16s versus ~78 minutes).
- **Point IDs are deterministic** (`QdrantVideoVectorStore.point_id`, uuid5 of the business key), so re-indexing overwrites instead of duplicating, and "is this already indexed?" is answerable without a marker table.
- **One shared SQLAlchemy engine per process** (`repository/database.py:get_session_manager`). Constructing a `DatabaseSessionManager` per Unit of Work leaks a connection pool and exhausts PostgreSQL (`sorry, too many clients already`) within a few hundred calls.
- **A stage that is skipped by routing must say so.** `_pipeline_status` treats `SKIPPED`/`NOT_FOUND` as incomplete *unless* the reason contains `"disabled by route"`. Get this wrong and every run reports `partial_success`, hiding real failures.
- **Never fabricate:** empty retrieval → not-found response (no LLM call); LLM error → fallback that still lists sources; caption error → `""` (recorded as `FAILED`, does not block); corrupt/silent video → `[]`. Regulation answers always append the mandatory disclaimer.
- **The video pipeline has two silent-corruption traps, both found by running one real video on 07/08/2026.** Neither raises; both write plausible-looking wrong data, and the run reports success.
  - **A silent video makes Zipformer emit the single character `"<"`,** which the old hallucination gate passed (not empty, matches no boilerplate phrase) and stored as a `DONE` transcript — a citable line of speech in a video where nobody speaks. Measured on all 6 silent videos in the corpus: every one produced exactly `"<"`. `_co_noi_dung_doc_duoc` now requires at least one alphanumeric character. Corpus: 52/59 videos have real audio, 6 are digitally silent (-91 dB), 1 has no audio track at all (that one returns `NOT_FOUND`, not `ERROR` — no audio is a normal state here, and an `ERROR` would make every run look failed).
  - **`đ` is its own letter — NFKD does not decompose it,** so `encode("ascii", "ignore")` deleted it and `"đăng ký kênh"` normalized to `"ang ky kenh"`. Four of the thirteen boilerplate phrases were written with `d` and had therefore never matched anything. Fixed by mapping `đ→d` *before* stripping accents; fixing the phrase list instead would just re-set the trap for the next person.
  - **Keyframe dedup is blind to bright video.** It compares raw 32×32 RGB with cosine at threshold 0.90, and cosine on un-centered pixels measures overall brightness, not content: three visually distinct app screens scored 0.9819–0.9931, collapsing a 28.8 s video to **one** keyframe. Mean-centering, 128×128 grayscale and dHash were all measured and none separate them — at that resolution the on-screen text is simply gone. `KEYFRAME_MAX_GAP_SEC` (10 s) does not fix the metric; it bounds the damage by never comparing frames that are far apart in time, so temporal coverage survives a blind metric. A semantic embedding is the real fix, and it is v2.
- Vector/payload alignment in indexing must stay 1:1 (filter frames to existing objects *before* embedding). Media and transcript indexing are **independent branches** — a frame-corrupt video must still yield a transcript.
- Long batch jobs must be **resumable and fail loudly**: skip work already marked `DONE` (not merely "a row exists"), and abort after N consecutive failures instead of marking a thousand items broken in four minutes.
- OCR, hybrid keyword search, object filtering, reranking, and the `privacy` domain are explicitly deferred to v2 — don't build them into v1.

## Tests

`API/tests/` — 72 tests, all offline (fakes for embedders, vector store, MinIO, vision; sqlite/in-memory where a DB is needed). Each test maps to a TC-xxx in `docs/prd.md`; a task is Done only when its test passes. When implementing a feature, implement the function, then assert against the Acceptance Criteria in the PRD.

`ruff check .` from `API/` reports pre-existing findings in files this project inherited; leave them alone unless you are editing that file for another reason, and keep files you touch clean.
