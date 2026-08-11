# HIT-MIRA-Multimodal-RAG

Hệ thống hỏi–đáp tiếng Việt về **kho media của CLB** (ảnh, video) và **nội quy
CLB**, trả lời kèm nguồn xem được: thẻ ảnh, keyframe tua đúng giây, đoạn lời
thoại, mục nội quy.

Kiến trúc là **Router RAG** — một câu hỏi được định tuyến tới đúng nhánh truy
hồi rồi mới tổng hợp câu trả lời; agentic RAG là v2. Luật xuyên suốt: **không
truy hồi được thì trả lời "không tìm thấy", không bịa.**

Tài liệu và chú thích viết bằng tiếng Việt.

## Một câu hỏi đi qua những đâu

```
trình duyệt ──► web (3000) ──► chatbot (2024, LangGraph) ──► mcp (8091) ──► api (8000)
                   │                     │ ToolMessage                           │
                   └── REST dự phòng / truy vấn ảnh ───────────────────────────┘
                                                                    Qdrant · Postgres · MinIO
```

Supervisor viết câu trả lời và trả kết quả truy xuất về cùng stream dưới dạng
`ToolMessage`. Web dùng chính payload đó để dựng thẻ nguồn, nên câu trả lời và
thẻ luôn thuộc cùng một lượt tìm kiếm; số trích dẫn `[n]` cũng trỏ đúng thẻ thứ
`n`. REST chỉ là đường dự phòng khi tool/agent lỗi, hoặc là đường chính của truy
vấn bằng ảnh vì MCP `search_media` hiện chỉ nhận văn bản. Nhánh thẻ hỏng không
nuốt phần chữ: `hitsStatus` báo lỗi riêng trên giao diện.

| Việc | Đường đi |
| --- | --- |
| Câu trả lời | `web` → `/threads/{id}/runs/stream` (SSE) → supervisor → MCP → API |
| Thẻ nguồn tham khảo | Đường chính: `ToolMessage` trong cùng stream; dự phòng: `web` → API retrieval |
| Tìm bằng ảnh | `web` → `POST /api/media/search-image` (nhúng trực tiếp ảnh người dùng gửi) |
| Ảnh và keyframe | `GET /api/media-files/by-key?object_key=…` → 307 → presigned MinIO |
| Video, tua đúng giây | `GET /api/media-files/video/{video_id}#t=…` → 307 → presigned MinIO (Range 206) |

Sơ đồ chi tiết: [docs/luong-query-den-cau-tra-loi.drawio](docs/luong-query-den-cau-tra-loi.drawio)
(trực tuyến) và [docs/luong-offline.drawio](docs/luong-offline.drawio) (ngoại tuyến).

## Chạy hệ thống trên `main`

### 1. Yêu cầu

- Docker Engine có Compose v2, dùng cho PostgreSQL, Qdrant và MinIO.
- Python 3.12 (mã nguồn khai báo tối thiểu 3.11) và Node.js 20.
- Một endpoint LLM tương thích OpenAI; endpoint embedding cho media và nội quy
  tùy nhánh muốn sử dụng.

`main` hiện chỉ đóng gói `api` cùng ba dịch vụ hạ tầng trong Compose. ChatBot,
MCP và Web chưa có Dockerfile trên nhánh này, vì vậy cách chạy được hỗ trợ bên
dưới là **hạ tầng trong Docker, bốn tầng ứng dụng chạy trên máy thật**.

### 2. Cấu hình

```bash
cp .env.example .env
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

Điền kết quả lệnh thứ hai vào `AUTH_SECRET_KEY`, sau đó cấu hình các nhóm cần
dùng trong `.env`:

| Khả năng | Biến cần cấu hình |
| --- | --- |
| Chat và viết lại truy vấn | `LLM_MODEL`, `LLM_API_KEY` hoặc `OPENAI_API_KEY`; `OPENAI_BASE_URL` nếu dùng proxy |
| Đăng nhập | `AUTH_SECRET_KEY` tối thiểu 16 ký tự |
| Tìm media | `MEDIA_IMAGE_EMBEDDING_BASE_URL` + `JINA_API_KEY`, hoặc `JINA_RUNPOD_ENDPOINT_ID` + `RUNPOD_API_KEY` |
| Tìm nội quy | `EMBEDDING_BASE_URL`, `EMBEDDING_API_KEY`, `EMBEDDING_MODEL` |
| Caption/OCR | `MEDIA_VISION_MODEL_NAME`, `MEDIA_VISION_API_BASE_URL`, `MEDIA_VISION_API_KEY` |
| Media qua MinIO | `STORAGE_BACKEND=minio`, `MINIO_*` |

Không chép `RUNPOD_API_KEY` sang `JINA_API_KEY`: hai biến xác thực hai giao thức
khác nhau. Xem chú thích trong [.env.example](.env.example) trước khi điền.

Code vision trên `main` hỗ trợ cả một lượt và hai lượt, nhưng mặc định là một
lượt. Theo cấu hình vận hành của dự án, OCR phải chạy trước để caption nhận chữ
OCR làm bối cảnh; thêm dòng sau vào `.env`:

```bash
MEDIA_VISION_TWO_PASS=true
```

Cờ này được API đọc khi chạy local. Service `api` trong Compose hiện chỉ nhận
các biến liệt kê trực tiếp ở `docker-compose.yml`, nên không dùng container API
nếu cần bảo đảm chế độ hai lượt mà chưa cập nhật Compose.

### 3. Khởi tạo hạ tầng và database

```bash
docker compose up -d postgres qdrant minio
docker compose ps

docker compose exec -T postgres sh -c \
  'psql -v ON_ERROR_STOP=1 -U "$POSTGRES_USER" -d "$POSTGRES_DB"' < schema.sql

for migration in migrations/*.sql; do
  docker compose exec -T postgres sh -c \
    'psql -v ON_ERROR_STOP=1 -U "$POSTGRES_USER" -d "$POSTGRES_DB"' < "$migration"
done
```

Chỉ chạy `schema.sql` trên database mới. Các migration có đánh ngày dùng để
nâng database đã tồn tại; repo chưa dùng Alembic.

Giá trị mặc định của Compose là user/password `hit`/`hit`. Nếu `.env` dùng mật
khẩu khác, `POSTGRES_PASSWORD` và mật khẩu trong `DATABASE_URL` phải trùng nhau.

### 4. Cài dependency

Chạy từ thư mục gốc repo:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r API/requirements.txt -r mcp/requirements.txt -e ChatBot
python -m pip install "langgraph-cli[inmem]" pytest ruff
(cd web && npm ci)
```

### 5. Khởi động bốn tầng ứng dụng

Mỗi lệnh chạy trong một terminal riêng, đều bắt đầu từ thư mục gốc và dùng cùng
virtualenv:

```bash
(cd API && python -m uvicorn src.server:app --reload --port 8000)
(cd mcp && PYTHONPATH=src python src/server.py)
(cd ChatBot && langgraph dev --no-browser)
(cd web && npm run dev)
```

Luồng phụ thuộc là **web → chatbot → mcp → api**. Khi lần lỗi, kiểm tra theo
chiều ngược lại, bắt đầu từ API.

Kiểm nhanh:

```bash
curl -fsS localhost:8000/health
curl -s -o /dev/null -w "%{http_code}\n" localhost:8091/mcp
curl -fsS -X POST localhost:2024/threads -H 'Content-Type: application/json' -d '{}'
curl -s -o /dev/null -w "%{http_code}\n" localhost:3000
```

`GET /mcp` có thể trả 406 khi thiếu header `Accept`; chỉ cần mã khác `000` để
xác nhận tiến trình đang lắng nghe.

### 6. Nạp dữ liệu ban đầu

Stack mới chỉ có bảng rỗng. Nạp thử nội quy sau khi đã cấu hình text embedding:

```bash
curl -f -F 'file=@public/Nội Quy CLB 2022.docx.pdf' \
  http://localhost:8000/api/documents/upload
```

**Giới hạn đã biết trên `main`:** ba endpoint ghi dưới `/api/documents` chưa có
guard admin, dù PRD quy định ingestion chỉ dành cho admin. Không public cổng
8000 và không dùng endpoint upload/delete trên mạng không tin cậy. Form nạp nội
quy của Web cũng đang gọi đường cũ `/ingest/regulations`; lệnh `curl` phía trên
là đường hoạt động đúng.

Với media đã có trong MinIO dưới prefix `raw/google-drive/data`, chạy dry-run
trước rồi mới thêm `--apply`:

```bash
python scripts/register_minio_images.py --prefix raw/google-drive/data
python scripts/register_minio_videos.py --prefix raw/google-drive/data

python scripts/register_minio_images.py --prefix raw/google-drive/data --apply
python scripts/register_minio_videos.py --prefix raw/google-drive/data --apply
python scripts/caption_images.py --limit 5 --apply
python scripts/index_image_units.py --limit 5 --apply
```

Video phải qua worker tạo artifact rồi import trước khi
`scripts/index_all_videos.py --apply` có dữ liệu để index. Xem
[luồng RunPod video](docs/runpod-serverless-video-pipeline.md) và
[kiểm kê dữ liệu](docs/data-audit.md).

Tài khoản đầu tiên đăng ký qua `/login` nhận quyền `admin`; các tài khoản sau là
`user`.

### 7. Truy cập từ máy khác

Khi chạy local, Web mặc định gọi thẳng API ở cổng 8000 và LangGraph ở cổng
2024. Muốn trình duyệt chỉ truy cập origin của Web, tạo `web/.env.local` trước
khi chạy hoặc build Web:

```bash
NEXT_PUBLIC_API_URL=/backend
NEXT_PUBLIC_LANGGRAPH_URL=/langgraph
BACKEND_API_ORIGIN=http://127.0.0.1:8000
BACKEND_LANGGRAPH_ORIGIN=http://127.0.0.1:2024
```

Ảnh và video vẫn nhận phản hồi 307 rồi được trình duyệt tải trực tiếp từ
presigned URL của MinIO. Vì vậy đặt endpoint S3 mà máy người dùng truy cập được
trong `.env` ở gốc repo:

```bash
# Dev/LAN
MINIO_PUBLIC_ENDPOINT=<ip-may-chu>:9000
MINIO_PUBLIC_SECURE=false

# Production qua reverse proxy HTTPS
MINIO_PUBLIC_ENDPOINT=media.example.com
MINIO_PUBLIC_SECURE=true
```

Compose publish 5432, 6333/6334, 8000, 9000 và 9001 trên `0.0.0.0`. Trên máy
public phải dùng firewall/security group, đổi mật khẩu mặc định và bind các
cổng nội bộ về `127.0.0.1`. Chỉ public cổng Web và endpoint S3 đã được bảo vệ.

## Cây thư mục

| Thư mục | Nội dung |
| --- | --- |
| [API/](API/) | FastAPI. `src/rag_video_anh/` (media: pipeline ngoại tuyến + truy hồi), `src/rag_noiquy/` (nội quy), `src/routers/` (7 router dưới `/api`), `src/jobs/` (job nền cho màn admin), `tests/` (chạy ngoại tuyến) |
| [ChatBot/](ChatBot/) | Supervisor LangGraph — **nơi duy nhất sinh câu trả lời** |
| [mcp/](mcp/) | Máy chủ MCP. Thêm tool = thêm mục trong `Resources/tools.yaml` **và** một method cùng tên trên `FeatureManager` |
| [web/](web/) | Next.js: khung chat, hỏi bằng ảnh, thẻ kết quả, màn admin — xem [web/README.md](web/README.md) |
| [embedding_server/](embedding_server/) | `jina-clip-v2` tự host: FastAPI hoặc RunPod Serverless, chung một `encoder.py` |
| [runpod_worker/](runpod_worker/) | Worker GPU xử lý video, chỉ nhận URL presigned và trả về một ZIP artifact — không thấy credential của Postgres/MinIO |
| [scripts/](scripts/) | CLI vận hành: nạp MinIO, đăng ký media, caption, index, gán sự kiện, đánh giá. Phần lớn batch script có dry-run, nhưng không phải tất cả — xem cảnh báo dưới đây |
| [docs/](docs/) | BRD → PRD → tasks → tech-pipeline; báo cáo đo; sơ đồ |
| [data/](data/) · [migrations/](migrations/) | Dữ liệu tham chiếu (lịch sự kiện, bộ câu hỏi đánh giá) · schema DB |

Kiến trúc, ràng buộc và các bẫy đã đo được nằm ở [CLAUDE.md](CLAUDE.md) — đọc
trước khi sửa code. Chỗ nào `docs/structure.md` và code không khớp thì **tin
code**.

## Mô hình dùng ở đâu

| Việc | Mô hình | Không gian vector |
| --- | --- | --- |
| Ảnh, keyframe video, lời thoại, **và câu truy vấn** | `jina-clip-v2` (1024-d, cosine) | `media_clip` + `video_transcript` — **dùng chung**, đây là tính chất load-bearing |
| Văn bản nội quy | Azure `text-embedding-3-small` (1536-d) | `rag_documents_azure_1536` — tách hẳn |
| ASR | `hynt/Zipformer-30M-RNNT-6000h` (sherpa-onnx) | |
| OCR trước → caption sau (caption nhận chữ OCR làm bối cảnh) | `Qwen3-VL-8B-Instruct` tự host trên RunPod (vLLM) | |
| Tổng hợp câu trả lời | `LLM_PROVIDER:LLM_MODEL` trong `ChatBot/` | |

Media và lời thoại **chung một không gian** là lý do `VideoRetrievalService`
nhúng câu hỏi **đúng một lần** rồi tìm cả hai collection. Đổi mô hình nhúng
đồng nghĩa với nhúng lại toàn bộ collection.

Truy vấn lọc được theo `years` (cắt mốc năm theo UTC+7, không phải UTC) và
`events` (khớp theo slug chuỗi sự kiện, không phải chữ người dùng gõ).

## An toàn khi chạy script

Các batch script như `register_minio_*`, `caption_images.py`,
`index_image_units.py` và `index_all_videos.py` mặc định chỉ lập kế hoạch; thêm
`--apply` mới ghi. Ngược lại, các lệnh chuyên biệt như
`import_media_outputs.py`, `index_video_retrieval_units.py`,
`runpod_video_job.py` và `add_issues_to_project.sh` có thể ghi hoặc gọi dịch vụ
thật ngay khi chạy. Luôn đọc `python <script> --help` và kiểm tra `.env` trước
khi dùng script không có cờ `--apply`.

## Kiểm thử

Bộ test dùng fake embedder/vector store/MinIO và SQLite nên không cần dịch vụ
sống hay API key. Chạy cùng phạm vi với CI:

```bash
(cd API && ruff check src tests && pytest -q)
(cd ChatBot && pytest -q)
(cd web && npx tsc --noEmit && npm run build)
```

Không ghi cứng số lượng test vào README vì pytest có parametrization và con số
thay đổi theo từng task. Các test quan trọng được neo vào TC-xxx trong
[docs/prd.md](docs/prd.md); một task chỉ được coi là xong khi CI của nó chạy qua.

## Kết quả đo

Đo trên kho thật ngày 02/08/2026 — Recall@5 **0,642** (mục tiêu ≥ 0,80, **chưa
đạt**), MRR **0,776** (mục tiêu ≥ 0,60, đạt), độ trễ trung bình **0,85s** (p95
1,71s; mục tiêu ≤ 5s, đạt). Nhãn được gán bằng luật từ khoá độc lập với không
gian vector, **không** lấy từ đầu ra của hệ thống. Yếu nhất là nhánh lời thoại
video. Chi tiết và cách chạy lại: [docs/eval-report.md](docs/eval-report.md).

So sánh mô hình nhúng văn bản (Jina-CLIP v2 thắng Azure trên chính kho này):
[docs/text-embedding-jina-vs-azure.md](docs/text-embedding-jina-vs-azure.md).

## Phạm vi v1 và những gì để lại cho v2

Đã chốt **không** làm trong v1: agentic RAG, tìm kiếm lai theo từ khoá, rerank
bằng mô hình/cross-encoder, lọc theo đối tượng phát hiện được, miền `privacy`.
Nhánh nội quy vẫn có bước xếp lại nhẹ bằng keyword boost sau truy hồi vector.
Ngoài ra còn hai chỗ đã biết là chưa đủ tốt:

- **Tool `search_media` của MCP mới chỉ nhận văn bản.** Hỏi bằng ảnh thì thẻ kết
  quả là truy hồi ảnh→ảnh thật, nhưng phần *lời văn* của câu trả lời vẫn dựa
  trên việc LLM nhìn ảnh rồi tự nghĩ ra từ khoá.
- **Khử trùng lặp keyframe mù với video sáng** — cosine trên pixel thô đo độ
  sáng chứ không đo nội dung. `KEYFRAME_MAX_GAP_SEC` chỉ chặn thiệt hại; cách
  sửa thật là dùng embedding ngữ nghĩa, và đó là v2.

## Tài liệu

| Tài liệu | Dùng khi |
| --- | --- |
| [CLAUDE.md](CLAUDE.md) | Sửa code — ràng buộc kiến trúc và các bẫy đã đo |
| [docs/brd.md](docs/brd.md) · [docs/prd.md](docs/prd.md) · [docs/tasks.md](docs/tasks.md) | Chuỗi truy vết BR-xxx → US-xxx → TC-xxx → T-xx |
| [docs/tech-pipeline.md](docs/tech-pipeline.md) | Lý do chọn từng mô hình/công nghệ |
| [docs/handoff.md](docs/handoff.md) | Nhận bàn giao môi trường |
| [docs/chay-demo.md](docs/chay-demo.md) | Trước khi demo |
| [docs/data-audit.md](docs/data-audit.md) | Kiểm kê dữ liệu thô trên MinIO |
| [docs/runpod-serverless-video-pipeline.md](docs/runpod-serverless-video-pipeline.md) · [docs/runpod-serverless-embedding.md](docs/runpod-serverless-embedding.md) | Vận hành GPU qua RunPod |
