# Cấu trúc thư mục — HIT-MIRA Multimodal RAG

> Kiến trúc: **Router RAG** (chưa agentic). Monorepo · backend domain-theo-nhóm-BR · tách ONLINE/OFFLINE.
> Nâng lên Agentic RAG đầy đủ (phân rã truy vấn/multi-hop) → v2.

## Cây thư mục

```
HIT-MIRA-Multimodal-RAG/
├── api/                          # BACKEND (Python, FastAPI)
│   ├── app/                      # ── ONLINE: request path, nhẹ ──
│   │   ├── main.py               #    app factory + đăng ký router
│   │   ├── config.py             #    pydantic-settings đọc .env
│   │   ├── deps.py               #    dependency chung (db session, current_user)
│   │   ├── core/                 #    security(JWT/hash/role) · errors · logging
│   │   ├── tools/                #    Tool registry cho Router RAG
│   │   │   ├── base.py           #      BaseTool + ToolResult
│   │   │   ├── registry.py       #      đăng ký/tra tool theo tên
│   │   │   ├── media_tool.py     #      bọc retrieval media (text/ảnh/transcript)
│   │   │   └── regulation_tool.py#      bọc retrieval nội quy
│   │   ├── routing/              #    Bộ định tuyến ý định  [BR-507]
│   │   │   └── intent.py         #      luật trước → classifier fallback → override
│   │   └── domains/              #    MODULE theo nhóm BR (router.py·service.py·schemas.py)
│   │       ├── auth/             #      BR-505
│   │       ├── consent/          #      BR-101,701
│   │       ├── ingest/           #      BR-102/104/105/107  upload + nạp nội quy
│   │       ├── media/            #      BR-105/403/404  serve · clip · stream
│   │       ├── retrieval/        #      BR-301/302/303/306/307/308
│   │       ├── answer/           #      BR-401/405/406/407  RAG synth · trích dẫn
│   │       ├── chat/             #      BR-501/502/503  orchestrate: routing→tools→answer
│   │       ├── eval/             #      BR-601..606
│   │       └── privacy/          #      BR-702/703/705  (v2)
│   │
│   ├── pipeline/                 # ── OFFLINE: job nặng, chạy worker/CLI  [NFR] ──
│   │   ├── frames.py             #    BR-201  keyframe + timestamp
│   │   ├── asr.py                #    BR-208  audio → ZipFormer RNNT → transcript
│   │   ├── caption.py            #    BR-203  caption ảnh/frame
│   │   ├── ocr.py                #    BR-204  (v2)
│   │   ├── embed.py              #    BR-202/207/208  sinh embedding
│   │   ├── index.py              #    BR-205/206  build/upsert Qdrant
│   │   └── run.py                #    entrypoint: xử lý 1 video / 1 batch
│   │
│   ├── shared/                   # ── DÙNG CHUNG app + pipeline ──
│   │   ├── db/  (session · models · migrations/)   # 19 bảng Postgres — PRD §5
│   │   ├── vectorstore/qdrant.py                    # 3 collection: media/transcript/nội quy
│   │   └── providers/  (embeddings · asr · captioner · llm · storage)  # adapter đổi được
│   │                                               # storage: filesystem (dev/test) | MinIO (docker/prod)
│   │
│   ├── tests/  ·  requirements.txt  ·  Dockerfile
│
├── web/                          # FRONTEND Next.js  [BR-500]
├── data/                         # ARTIFACTS (gitignored, giữ .gitkeep)
│   ├── raw/{images,videos,posts} · frames · transcripts · clips · regulations · processed · eval
├── docs/                         # brd.md · prd.md · structure.md · (hld.md)
├── docker-compose.yml            # postgres + qdrant + minio + api (+ web)
└── .env.example
```

## Quy ước module (domain)

Mỗi `domains/<x>/` gồm: `router.py` (FastAPI routes) · `service.py` (logic) · `schemas.py` (pydantic).
Một người ôm trọn 1 domain, không giẫm chân nhau. Traceability: BR → domain → test hook.

## Luồng Router RAG (online)

```
POST /chat/message
   → routing/intent.route(text, has_image, override)   # chọn nguồn: media | regulation | both
   → tools.registry.get(source).run(query)             # gọi tool tương ứng
   → domains/retrieval (Qdrant)                         # truy xuất top-k
   → domains/answer  (LLM synth + trích dẫn + disclaimer)
```

## Ánh xạ từ template "AI Agent Project Structure" (bản gốc tham chiếu)

| Template gốc (agent) | Ở dự án này (Router RAG) | Ghi chú |
|---|---|---|
| `agents/` | — (bỏ) | Agentic → v2 |
| `orchestration/` (task_planner, state_manager) | `app/routing/` (nhẹ) | Chỉ định tuyến, chưa plan/state |
| `tools/` (base_tool, registry) | `app/tools/` | **Giữ** — hợp Router RAG & là đường lên agentic |
| `core/` (config, logger, exceptions) | `app/config.py` + `app/core/` | |
| `memory/` (vector_store, embeddings) | `shared/vectorstore/` + `shared/providers/embeddings` | |
| `knowledge/` (loader, splitter, retriever, rag_pipeline) | `pipeline/` + `domains/retrieval` + `domains/answer` | Nên xoá thư mục `knowledge/` rỗng cũ |
| `models/` (llm_factory, model_router) | `shared/providers/llm` | |
| `api/` (routes, main) | `app/` (main + domains) | |
| `utils/` | `app/core/` | Gộp cho gọn |
```
