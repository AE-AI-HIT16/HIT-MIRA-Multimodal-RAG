# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

HIT-MIRA Multimodal RAG — a 3-person student capstone: a **Router RAG** (not yet agentic; agentic = v2) system that answers Vietnamese questions about club media (video/image) and club regulations, with citations. Backend is a teaching scaffold: many functions are stubs that students implement task-by-task against `docs/tasks.md`.

**Docs and comments are written in Vietnamese — keep that convention.** All planning docs live in `docs/`: `brd.md` (BR-xxx business requirements) → `prd.md` (US-xxx stories, TC-xxx test cases, data model) → `tasks.md` (T-xx task board with owners/DoD) → `tech-pipeline.md` (model/technology decisions and rationale) → `structure.md` (directory layout rationale). Traceability chain BR → US → domain → TC/test is load-bearing; when adding code, reference the relevant IDs.

## Commands

All backend work happens from `api/` (imports assume `api/` is the working directory):

```bash
cd api
uvicorn app.main:app --reload        # dev server (health check: GET /health)
pytest                               # all tests (sqlite in-memory, no services needed)
pytest tests/test_retrieval.py       # one file
pytest tests/test_retrieval.py -k retrieve_media   # one test
ruff check .                         # lint
python -m pipeline.run <video_path> [media_id]     # offline pipeline for one video
```

No frontend exists yet (`web/` holds only a README). `docker-compose.yml` (P0-1) brings up postgres + qdrant + minio + api. Config comes from pydantic-settings reading `.env` (see `.env.example`); dev defaults to sqlite (`./data/dev.db`) + filesystem storage, docker/prod overrides to Postgres + MinIO.

## Architecture

`api/` splits into three layers — the online/offline separation is an NFR, do not put heavy ML work on the request path:

- **`app/` — ONLINE request path (lightweight).** `main.py` is the app factory; domain routers are registered there **only after a domain is implemented** (currently just `consent`). Each `app/domains/<x>/` maps to a BR group and contains exactly `router.py` (FastAPI routes) / `service.py` (logic) / `schemas.py` (pydantic). One person owns one domain.
- **`pipeline/` — OFFLINE worker/CLI (heavy jobs).** `run.py` orchestrates two **independent** branches per video (so a frame-corrupt video can still yield transcript): media (keyframes → optional caption → image embed → Qdrant `media_clip`) and transcript (ASR → `chunk_transcript` deterministic char-window chunking → text embed → Qdrant `video_transcript`). A failing item is logged and skipped, never blocks the batch.
- **`shared/` — used by both.** `db/` (SQLAlchemy models + session), `vectorstore/qdrant.py` (3 collections: media / transcript / regulation), `providers/` (swappable adapters for embeddings, ASR, captioner, LLM, and object **storage** — model choices are pinned in `docs/tech-pipeline.md`: Jina-CLIP v2 for images *and* text queries into one joint space, AITeamVN/Vietnamese_Embedding for transcript/regulation text, PhoWhisper via faster-whisper for ASR, Gemini 2.5 Flash free-tier for caption/answer). `providers/storage.py` (`StorageProvider`) is infra-provided like `QdrantStore`: `FilesystemStorage` (dev/test, zero-config — pytest needs no MinIO) vs `MinioStorage` (docker/prod), selected by `STORAGE_BACKEND` in `get_storage()`. **`shared/` must not import `app.config`** — all config-reading provider wiring lives in `app/deps.py` as `@lru_cache` factories, and every concrete provider **lazy-imports its heavy deps inside `__init__`** so a plain `import` never pulls torch/genai/qdrant/minio/PIL.

**Online Router RAG flow** (`POST /chat/message`): `app/routing/intent.py` picks the source (media / regulation / both; rules first, classifier fallback, user override) → `app/tools/registry.py` resolves the matching `BaseTool` (`media_tool`, `regulation_tool` wrap retrieval) → `domains/retrieval` queries Qdrant → `domains/answer` does LLM synthesis with citations and disclaimer. The tool registry is deliberately kept as the upgrade path to agentic RAG in v2.

Providers are injected as optional parameters (`embedder=`, `store=`, `captioner=`...) so tests pass fakes — follow that pattern rather than instantiating models inside functions.

## Tests

`tests/conftest.py` provides `engine` / `db` / `client` fixtures (sqlite in-memory `StaticPool` + `TestClient` with `get_session` overridden). `tests/test_learning_todo.py` is the backlog of skipped skeleton tests: when implementing a feature, implement the function, move/unskip the test, and assert against the Acceptance Criteria in `docs/prd.md`. Each test maps to a TC-xxx; a task is Done only when its test passes.

## Constraints worth knowing

- Gemini free-tier quota (~15 req/min) is a real limit — prefer offline precomputation (captions, summaries) and caching over per-request LLM calls; Flash-Lite for simple queries. Free-tier also trains on prompts/responses (a privacy concern once real member data is processed — see `tech-pipeline.md`).
- **Qdrant payload keys are a contract** — retrieval and answer synthesis read them by name: media `{video_id, timestamp, frame_path, caption}`, transcript `{video_id, start_sec, end_sec, text}`, regulation `{article/dieu, clause/khoan, text}`. `retrieve_by_transcript` merges hits by `video_id` (max score + collected `moments`) so a video isn't double-counted.
- **Never fabricate:** empty retrieval → `handle_not_found` (no LLM call); LLM error → raw fallback that still lists sources; caption error → `""` (flagged, doesn't block); corrupt/silent video → `[]`. Regulation answers always append a mandatory disclaimer.
- Vector/payload alignment in indexing must stay 1:1 (filter frames to existing files *before* embedding; see `pipeline/run.py` guards). Don't let pipeline artifacts leak into the repo — thread `out_dir` (tests pass `tmp_path`); the `./data/...` default is gitignored but leaks if a test omits it.
- OCR, hybrid keyword search, object filtering, reranking, and the `privacy` domain are explicitly deferred to v2 — don't build them into v1.
