# Chạy demo — bốn tầng, đúng thứ tự

> Kiểm bằng cách chạy thật ngày **02/08/2026**: web → LangGraph → MCP → API →
> Qdrant + MinIO + pod nhúng. Câu hỏi tiếng Việt thật trả về câu trả lời có link
> Facebook thật kèm thẻ ảnh xem được.

## Thứ tự khởi động (từng tầng phụ thuộc tầng dưới)

```bash
cd /home/ubuntu/HIT-MIRA-Multimodal-RAG
PY=/home/ubuntu/miniconda3/envs/nhhoang/bin/python

# 0. Hạ tầng + nguồn nhúng
docker-compose up -d                                  # postgres, qdrant, minio
$PY scripts/deploy_embedding_pod.py --status          # phải có pod đang RUNNING
#    chưa có thì: $PY scripts/deploy_embedding_pod.py --community --apply
#    rồi cập nhật MEDIA_IMAGE_EMBEDDING_BASE_URL / EMBEDDING_BASE_URL / JINA_API_KEY

# 1. API  (cổng 8000 — MCP_BASE_URL trong .env trỏ đúng cổng này)
cd API && $PY -m uvicorn src.server:app --port 8000

# 2. MCP  (cổng 8091 — ChatBot cần nó sống thì supervisor mới có tool)
cd mcp && PYTHONPATH=src $PY src/server.py

# 3. LangGraph  (cổng 2024 — nơi web gửi tin nhắn)
cd ChatBot && langgraph dev --no-browser

# 4. Web  (cổng 3000)
cd web && npm run dev
```

Kiểm nhanh cả bốn tầng:

```bash
curl -s localhost:8000/health                                     # {"status":"ok"}
curl -s -o /dev/null -w "%{http_code}\n" localhost:8091/mcp       # không phải 000
curl -s -X POST localhost:2024/threads -H 'Content-Type: application/json' -d '{}'
curl -s -o /dev/null -w "%{http_code}\n" localhost:3000           # 200
```

## Web nối vào đâu

| Việc | Đường đi |
|---|---|
| Câu trả lời | `web` → `LANGGRAPH_URL/threads/{id}/runs/stream` (SSE) → supervisor → MCP → API |
| Thẻ nguồn tham khảo | `web` → `API/api/media/search` + `API/api/retrieval/search`, **chạy song song với stream** |
| Ảnh và keyframe | `API/api/media-files/by-key?object_key=…` → 307 → presigned MinIO |
| Video, tua đúng giây | `API/api/media-files/video/{video_id}#t=…` → 307 → presigned MinIO (Range 206) |

Hai đường tách nhau **có chủ ý**: agent viết câu trả lời, còn thẻ kết quả là
bằng chứng nhìn được. Chờ agent xong mới gọi thì người dùng đợi hai lượt; và
nhánh thẻ hỏng thì câu trả lời vẫn hiện bình thường (có trạng thái riêng
`hitsStatus`, hỏng thì báo bằng một dải cảnh báo chứ không nuốt).

## Ba điều dễ vấp

- **`.env` phải để `MCP_BASE_URL=http://localhost:8000`.** Chạy API ở cổng khác
  thì MCP gọi vào chỗ trống, supervisor báo không có dữ liệu, mà không có lỗi
  nào hiện ra ở web.
- **Không có pod nhúng thì mọi thứ chết cùng lúc**: cả chat lẫn thẻ kết quả,
  vì câu hỏi nào cũng phải nhúng. Xem `embedding_server/README.md`.
- **`web/` cần `langgraph dev` chạy song song** — chat không đi qua FastAPI. Đây
  là hệ quả của việc cố ý không làm `/chat/message` (xem `docs/tasks.md`, T-50).

## Còn thiếu gì trên web

- **Màn admin** (`/admin`) — cập nhật 05/08/2026: bốn endpoint đã có, đều đứng
  sau quyền admin.

  | Chức năng | Endpoint | Ghi chú |
  |---|---|---|
  | Nạp ảnh/video | `POST /api/ingest/upload` | file vào MinIO theo key `events/<post>/media/<tên>`, metadata vào `posts`+`media` |
  | Chạy index | `POST /api/admin/index/{media\|videos}` | job nền, tiến trình con |
  | Trạng thái job | `GET /api/admin/index/status` | trả **cả** job `eval` |
  | Chạy đánh giá | `POST /api/admin/eval/run` | job nền, trả trạng thái job chứ không phải báo cáo |

  Còn hai chỗ chưa dùng được:

  - **Nạp nội quy** (`/ingest/regulations`) vẫn 404. Không phải chuyện đổi
    đường dẫn: form gửi `.md`/`.txt` + `title` + `version`, còn
    `/api/documents/upload` chỉ nhận `.pdf`/`.docx`, không có chỗ cho tiêu đề
    lẫn phiên bản, và trả về hình dạng khác. Muốn dùng thì phải mở rộng
    `DocumentParser` chứ không phải trỏ lại link.
  - **Video vừa nạp chưa index được ngay.** `POST /api/ingest/upload` chỉ ghi
    `media`, KHÔNG ghi `videos` — dòng đó là kết quả của pipeline parse
    (duration, fps, keyframe). Phải chạy pipeline trước, rồi nút "Index video"
    mới thấy nó.
- **Đăng nhập** (`/login`) gọi `/api/auth/*` — đã có (`register`/`login`/`me`).
- **Truy vấn bằng ảnh** (05/08/2026): **thẻ kết quả** đã tìm bằng vector ảnh
  thật — web gọi `POST /api/media/search-image`, ảnh được nhúng rồi so trực
  tiếp với vector ảnh trong `media_clip`. Còn **câu trả lời chữ** thì vẫn do
  supervisor nhìn ảnh rồi tự nghĩ ra từ khoá gọi `search_media`: tool call
  không mang được ảnh, muốn phần chữ cũng bám vector ảnh thì phải thêm một
  node LangGraph gọi MCP bằng ảnh trước khi supervisor chạy.
  Chưa chạy lại `demo_checklist.py` (L6) vì Qdrant/MinIO đang tắt.
