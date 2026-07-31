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

`docker-compose.yml` brings up postgres + qdrant + minio. Config comes from `.env` substituted into `API/Resources/{dev,model,prompt}.yaml` via `Template.safe_substitute(os.environ)` (see `API/src/config/config.py`); `AppConfig` in `API/src/configuration.py` is the single accessor. **Many settings also accept a direct env override that wins over the YAML** — e.g. `MEDIA_VISION_*` beats `OPENROUTER_*`. Check `configuration.py` before assuming a value comes from YAML.

## Architecture

`API/src/` is the backend. The online/offline separation is an NFR — do not put heavy ML work on the request path.

- **`server.py` — app factory.** `create_app()` registers exactly three routers under `/api` (`documents`, `retrieval`, `media_retrieval`) plus `/health`. There is no `app/` package and no `domains/` package; router modules live in `src/routers/`.
- **`rag_video_anh/` — the media pipeline and media retrieval.** Split into `pipeline/` (offline: validate → route → keyframes → OCR/caption → detection → ASR → transcript mapping → normalize), `retrieval/` (units builder, indexing service, retriever, retrieval service), `embedding/`, `vector_store/`, `repository/` (SQLAlchemy models + Unit of Work), `schemas/`.
- **`rag_noiquy/` — the regulation/document path.** Its own `embedding/`, `pipeline/`, `retrieval/`, `vector_store/`. Backs `routers/documents.py` and `routers/retrieval.py`.
- **`src/rag/` is a broken orphan** — imports fail. Do not extend it; do not import from it.
- **`scripts/` — operational CLIs** (upload crawl to MinIO, register media rows, caption, index, RunPod jobs). Every destructive one is **dry-run by default and requires `--apply`**; keep that convention for new scripts.
- **`runpod_worker/` — GPU worker** (`handler.py`). Artifact-only: it receives presigned URLs, runs `scripts/export_video_artifacts.py`, and PUTs one ZIP back. It never sees PostgreSQL or MinIO credentials.
- **`mcp/` — MCP server.** A tool is registered by adding an entry to `Resources/tools.yaml` **and** a same-named method on `FeatureManager`; `ToolRegistry` binds them by `getattr`. `log_exceptions` deliberately converts exceptions into `{"success": False, "error_type": ..., "message": ...}` — that is the MCP contract, not a swallowed error.

Providers are injected as optional constructor parameters (`image_embedder=`, `text_embedder=`, `vector_store=`, `storage=`, `builder=`, `vision_service=`…) so tests pass fakes — follow that pattern rather than instantiating models inside functions.

### Where each model is actually used

| Concern | Model | Wired in |
| --- | --- | --- |
| Images, video keyframes, transcript text, **and user queries** | **Jina-CLIP v2** (`jina-clip-v2`, 1024-d, cosine) | `rag_video_anh/embedding/embedding_service.py` |
| Regulation/document text | `baai/bge-m3` via an OpenAI-compatible endpoint | `rag_noiquy/embedding/embedding_service.py` |
| ASR | `hynt/Zipformer-30M-RNNT-6000h` via `sherpa_onnx` | `pipeline/asr_service.py` |
| Caption + OCR (one call returns both) | `MEDIA_VISION_MODEL_NAME`, else `OPENROUTER_MODEL_NAME` | `pipeline/qwen_vision_service.py` |
| Query rewriting (regulation path) | `LLM_PROVIDER:LLM_MODEL` | `rag_noiquy/retrieval/query_rewriter.py` |

Answer synthesis lives in `ChatBot/` (LangGraph), outside `API/src/`.

Two things that surprise people:

1. **Transcripts are embedded with Jina-CLIP v2, not a dedicated Vietnamese text model.** `indexing_service.py` does `self.text_embedder = text_embedder or self.image_embedder`. That is what makes one query vector rank images *and* speech together — the joint space is the feature, not an oversight. `tech-pipeline.md` still names `AITeamVN/Vietnamese_Embedding`; changing to it would improve transcript retrieval but break the single-vector property.
2. **`jina-clip-v2` only accepts `task="retrieval.query"`.** Sending `retrieval.passage` returns HTTP 422. Asymmetric query/passage embedding belongs to `jina-embeddings-v3`, a different model.

### Online Router RAG flow

`POST /api/media/search` → `VideoRetrievalService.retrieve()` embeds the query **once** and searches both collections in that shared space, merges, and builds a citation context string. The tool registry in `mcp/` is deliberately kept as the upgrade path to agentic RAG in v2.

## Constraints worth knowing

- **Jina rate limit is 100,000 tokens/minute (sliding window); one 512px image costs exactly 4,000 tokens.** `ImageEmbeddingService._reserve_tokens` paces requests *proactively*; do not replace it with react-to-429 backoff, which measured ~10 images/min versus ~20–25. Images are downscaled to 512px before embedding (`MEDIA_IMAGE_EMBEDDING_MAX_SIDE`) — a 2048px image costs 6× the tokens for a cosine-0.994 identical vector.
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
- Vector/payload alignment in indexing must stay 1:1 (filter frames to existing objects *before* embedding). Media and transcript indexing are **independent branches** — a frame-corrupt video must still yield a transcript.
- Long batch jobs must be **resumable and fail loudly**: skip work already marked `DONE` (not merely "a row exists"), and abort after N consecutive failures instead of marking a thousand items broken in four minutes.
- OCR, hybrid keyword search, object filtering, reranking, and the `privacy` domain are explicitly deferred to v2 — don't build them into v1.

## Tests

`API/tests/` — 72 tests, all offline (fakes for embedders, vector store, MinIO, vision; sqlite/in-memory where a DB is needed). Each test maps to a TC-xxx in `docs/prd.md`; a task is Done only when its test passes. When implementing a feature, implement the function, then assert against the Acceptance Criteria in the PRD.

`ruff check .` from `API/` reports pre-existing findings in files this project inherited; leave them alone unless you are editing that file for another reason, and keep files you touch clean.
