# Cấu trúc thư mục — HIT-MIRA Multimodal RAG

> Kiến trúc: **Router RAG** (chưa agentic). Monorepo · tách ONLINE/OFFLINE.
> Nâng lên Agentic RAG đầy đủ (phân rã truy vấn/multi-hop) → v2.
>
> **Tài liệu này mô tả cây code đang thật sự tồn tại.** Bản trước (viết lúc chưa
> code) mô tả một layout `api/app/domains/...` chưa bao giờ được dựng; giữ nó lại
> chỉ khiến người mới đi lạc. Khi tài liệu và code lệch nhau, tin code.

## Cây thư mục

```
HIT-MIRA-Multimodal-RAG/
├── API/                          # BACKEND FastAPI — mọi lệnh chạy từ trong đây
│   ├── src/
│   │   ├── server.py             #   app factory: đăng ký 3 router dưới /api + /health
│   │   ├── configuration.py      #   AppConfig — đầu mối DUY NHẤT để đọc cấu hình
│   │   ├── config/config.py      #   nạp Resources/*.yaml, thay ${BIEN} bằng biến môi trường
│   │   ├── routers/              # ── ONLINE: đường đi của request, phải nhẹ ──
│   │   │   ├── documents.py      #     nạp & quản lý tài liệu nội quy
│   │   │   ├── retrieval.py      #     truy xuất nội quy
│   │   │   └── media_retrieval.py#     POST /api/media/search — truy xuất ảnh + video
│   │   │
│   │   ├── rag_video_anh/        # ── NHÁNH MEDIA (ảnh, video) ──
│   │   │   ├── pipeline/         #   OFFLINE: validate → route → keyframe → OCR/caption
│   │   │   │                     #            → detection → ASR → map transcript → normalize
│   │   │   ├── retrieval/        #   units builder · indexing · retriever · retrieval service
│   │   │   ├── embedding/        #   Jina-CLIP v2 (kèm bộ giữ nhịp token)
│   │   │   ├── vector_store/     #   Qdrant: media_clip · video_transcript
│   │   │   ├── repository/       #   model SQLAlchemy + Unit of Work + session manager dùng chung
│   │   │   └── schemas/          #   pydantic cho media/video/transcript
│   │   │
│   │   ├── rag_noiquy/           # ── NHÁNH NỘI QUY (văn bản) ──
│   │   │   ├── pipeline/         #   parser → cleaner → chunker → ingest
│   │   │   ├── retrieval/        #   query rewriter · retriever · retrieval service
│   │   │   ├── embedding/        #   cùng model với nhánh media (xem CLAUDE.md)
│   │   │   └── vector_store/     #   Qdrant: rag_documents
│   │   │
│   │   ├── rag/                  # ⚠️ MỒ CÔI — import gãy. Không dùng, không mở rộng.
│   │   ├── common_utils/ · log/  #   tiện ích dùng chung
│   │   │
│   │   └── (Resources/ ở cấp API: dev.yaml · model.yaml · prompt.yaml · prompts/)
│   └── tests/                    #   pytest, chạy offline hoàn toàn (fake + sqlite)
│
├── ChatBot/                      # TẦNG SINH CÂU TRẢ LỜI (LangGraph) — tiến trình riêng
│   ├── src/graph/
│   │   ├── graph.py              #   dựng & biên dịch đồ thị (đồng bộ — xem tests/)
│   │   ├── nodes/ · agents/      #   một nút SUPERVISOR duy nhất ở v1
│   │   ├── client_tools/         #   nối MCP server qua langchain-mcp-adapters
│   │   ├── middlewares/          #   chèn thời gian vào prompt · chuẩn hoá tham số tool
│   │   └── state.py · configuration.py
│   ├── Resources/                #   agents.yaml · models.yaml · prompts.yaml · prompts/
│   └── tests/                    #   pytest offline: không cần MCP server, không gọi LLM
│
├── mcp/                          # MCP SERVER (mặc định cổng 8091) — cầu nối ChatBot ↔ API
│   ├── src/{server,feature_manager}.py · tools/{registry,manager}.py · clients/
│   └── Resources/tools.yaml      #   thêm tool = thêm mục ở đây + method cùng tên ở FeatureManager
│
├── scripts/                      # CLI vận hành (upload MinIO · caption · index · job RunPod)
│                                 # Mọi script phá huỷ đều dry-run mặc định, phải có --apply
├── runpod_worker/                # WORKER GPU — chỉ nhận URL presigned, trả về 1 file ZIP
│                                 # Không hề biết thông tin đăng nhập PostgreSQL hay MinIO
├── web/                          # FRONTEND Next.js (app/ · components/ · lib/)
├── migrations/ · schema.sql      # DDL PostgreSQL
├── docs/                         # brd · prd · tasks · tech-pipeline · structure · specs/
├── public/                       # tài liệu tham chiếu (2 paper AI Challenge)
├── docker-compose.yml            # postgres + qdrant + minio
└── .env.example
```

## Vì sao chia theo nhánh dữ liệu, không chia theo BR

Bản kế hoạch cũ định chia `domains/` theo nhóm BR (auth, ingest, media, retrieval,
answer, chat, eval, privacy). Thực tế code hội tụ về **hai nhánh dữ liệu**:
`rag_video_anh` (ảnh/video) và `rag_noiquy` (văn bản). Lý do là ranh giới thật của
hệ thống nằm ở *loại dữ liệu*, không ở *nhóm yêu cầu*: hai nhánh có pipeline khác
nhau, collection Qdrant khác nhau, và hỏng độc lập với nhau. Traceability BR → US →
TC vẫn giữ nguyên, chỉ là neo vào test và docstring thay vì vào tên thư mục.

Đường ranh **ONLINE/OFFLINE** thì vẫn đúng như kế hoạch và là một NFR: `routers/`
chỉ nhúng câu hỏi rồi tra Qdrant; mọi việc nặng (keyframe, ASR, caption, nhúng
hàng loạt) nằm ở `pipeline/` và `scripts/`, chạy ngoài request.

## Luồng Router RAG (online)

```
người dùng → web/ → ChatBot (LangGraph, nút SUPERVISOR)
   → MCP server: search_regulations | search_media
   → API: /api/retrieval/search | /api/media/search
        └─ VideoRetrievalService.retrieve(): nhúng câu hỏi MỘT lần
           → tìm song song media_clip + video_transcript trong cùng không gian vector
           → gộp, dựng chuỗi trích dẫn đánh số sẵn [1] [2] [3]
   → LLM tổng hợp câu trả lời + trích dẫn (+ disclaimer nếu là nội quy)
```

Việc chọn nguồn (media / nội quy / cả hai) do LLM quyết bằng cách chọn tool, không
phải bằng một bộ luật `routing/intent.py` như bản kế hoạch — nên "Router" ở đây
nằm trong prompt của supervisor. `mcp/` được giữ đúng dạng registry để v2 nâng lên
agentic mà không phải viết lại tầng tool.

## Ba tiến trình, chạy độc lập

| Tiến trình | Lệnh | Ghi chú |
|---|---|---|
| API | `cd API && uvicorn src.server:app --reload` | import giả định thư mục làm việc là `API/` |
| MCP server | `cd mcp && PYTHONPATH=src python src/server.py` | mặc định cổng 8091 |
| ChatBot | `cd ChatBot && langgraph dev` | cần MCP server sống thì tool mới có |

Hạ tầng (`postgres`, `qdrant`, `minio`) lên bằng `docker-compose up -d`.
