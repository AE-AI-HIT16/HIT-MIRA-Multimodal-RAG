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

## Yêu cầu môi trường

- Docker Engine có Compose v2; GPU NVIDIA + NVIDIA Container Toolkit chỉ cần
  khi bật profile `gpu`.
- Khi chạy ứng dụng ngoài Docker: Python 3.12 (tối thiểu 3.11) và Node.js 20.
- Một endpoint LLM tương thích OpenAI; endpoint embedding cho media và nội quy
  tuỳ nhánh muốn sử dụng.

## Chạy toàn bộ hệ thống bằng Docker

### 1. Cấu hình

```bash
cp .env.example .env
```

Các nhóm biến quan trọng trong `.env`:

| Khả năng | Biến cần cấu hình |
| --- | --- |
| Chat và viết lại truy vấn | `LLM_MODEL` + `LLM_API_KEY` hoặc `OPENAI_API_KEY`; `OPENAI_BASE_URL` nếu dùng proxy |
| Đăng nhập | `AUTH_SECRET_KEY` tối thiểu 16 ký tự |
| Tìm media | Một trong hai đường: `MEDIA_IMAGE_EMBEDDING_BASE_URL` + `JINA_API_KEY`, hoặc `JINA_RUNPOD_ENDPOINT_ID` + `RUNPOD_API_KEY` |
| Tìm nội quy | `EMBEDDING_BASE_URL`, `EMBEDDING_API_KEY`, `EMBEDDING_MODEL` |
| Caption/OCR và xử lý video | Nhóm `MEDIA_VISION_*` và `RUNPOD_*`; chỉ cần khi chạy pipeline ngoại tuyến/admin |
| Xem media từ máy khác | `MINIO_PUBLIC_ENDPOINT` và `MINIO_PUBLIC_SECURE` |

Không chép `RUNPOD_API_KEY` sang `JINA_API_KEY`: khoá thứ hai là Bearer token
của endpoint Jina trực tiếp. Xem chú thích ngay trong [.env.example](.env.example)
để chọn đúng một provider.

### 2. Khởi tạo database lần đầu

Compose không tự chạy `schema.sql`. Với một volume Postgres mới, khởi động hạ
tầng rồi tạo schema và áp dụng migration theo thứ tự tên file:

```bash
docker compose up -d postgres qdrant minio

docker compose exec -T postgres sh -c \
  'psql -v ON_ERROR_STOP=1 -U "$POSTGRES_USER" -d "$POSTGRES_DB"' < schema.sql

for migration in migrations/*.sql; do
  docker compose exec -T postgres sh -c \
    'psql -v ON_ERROR_STOP=1 -U "$POSTGRES_USER" -d "$POSTGRES_DB"' < "$migration"
done
```

Chỉ chạy `schema.sql` trên database mới; file này không phải migration lặp lại.
Các migration có đánh ngày dùng để nâng một database đã tồn tại. Repo chưa dùng
Alembic, vì vậy khi thêm migration mới phải ghi rõ thứ tự và cách rollback.

### 3. Khởi động ứng dụng

```bash
docker compose up -d --build
docker compose ps         # 7 service: postgres · qdrant · minio · api · mcp · chatbot · web
```

Mở <http://localhost:3000>. Trình duyệt chỉ nói chuyện với cổng 3000; Next chuyển
tiếp nội bộ sang `api` (8000) và `chatbot` (2024), nên hai cổng đó không cần mở
ra ngoài và không có chuyện CORS.

| Service | Cổng | Vai trò |
| --- | --- | --- |
| `web` | 3000 | Giao diện Next.js |
| `chatbot` | 2024 | LangGraph dev server, tổng hợp câu trả lời |
| `mcp` | 8091 | Máy chủ MCP, nơi chatbot gọi tool |
| `api` | 8000 | FastAPI: truy hồi media + nội quy |
| `postgres` · `qdrant` · `minio` | 5432 · 6333 · 9000/9001 | Metadata · vector · file gốc |

Chuỗi phụ thuộc là **web → chatbot → mcp → api**. Lần lỗi thì đi ngược chiều đó:
web trắng kết quả thường là `api` trả 503 vì thiếu biến môi trường, chứ hiếm khi
là lỗi giao diện. `docker compose logs -f api` là chỗ nhìn đầu tiên.

### 4. Nạp dữ liệu ban đầu

Một stack mới **không có dữ liệu nghiệp vụ**: Postgres chỉ có bảng, MinIO chưa
có media và Qdrant chưa có vector. Container healthy không có nghĩa là chatbot
đã có gì để trả lời.

Nạp thử nội quy có sẵn trong repo sau khi đã cấu hình nhánh text embedding:

```bash
curl -f -F 'file=@public/Nội Quy CLB 2022.docx.pdf' \
  http://localhost:8000/api/documents/upload
```

Với bộ media đã upload vào MinIO dưới prefix `raw/google-drive/data`, luôn chạy
khô để kiểm số lượng/prefix trước, rồi mới ghi và index:

```bash
python scripts/register_minio_images.py --prefix raw/google-drive/data
python scripts/register_minio_videos.py --prefix raw/google-drive/data

python scripts/register_minio_images.py --prefix raw/google-drive/data --apply
python scripts/register_minio_videos.py --prefix raw/google-drive/data --apply
python scripts/caption_images.py --limit 5 --apply
python scripts/index_image_units.py --limit 5 --apply
```

Video phải qua worker tạo artifact rồi import vào PostgreSQL/MinIO trước khi
`scripts/index_all_videos.py --apply` có dữ liệu để index. Xem
[docs/runpod-serverless-video-pipeline.md](docs/runpod-serverless-video-pipeline.md)
cho luồng đầy đủ và [docs/data-audit.md](docs/data-audit.md) cho layout của bộ
dữ liệu gốc. Các lệnh trên ghi vào dịch vụ thật được cấu hình trong `.env`; đừng
thêm `--apply` trước khi kết quả dry-run đúng.

Tài khoản đầu tiên đăng ký qua `/login` sẽ nhận quyền `admin`; hãy tạo tài khoản
này trước khi đưa server ra internet.

Hai điểm dễ vấp:

- **Bốn biến của `web` đọc lúc build, không phải lúc chạy** — đổi cái nào cũng
  phải `docker compose up -d --build web`. `BACKEND_*_ORIGIN` vì Next đọc
  `rewrites()` một lần rồi ghi cứng vào `routes-manifest.json`; `NEXT_PUBLIC_*`
  vì chúng được nướng thẳng vào bundle trình duyệt.
- **Đang chạy `next dev` / `uvicorn` ở máy thật thì tắt trước**, nếu không các
  cổng 3000/8000 bị chiếm và container không bind được.

### Chạy trên máy chủ có IP công khai

Không cần sửa gì trong ảnh: `NEXT_PUBLIC_API_URL=/backend` và
`NEXT_PUBLIC_LANGGRAPH_URL=/langgraph` là đường dẫn **tương đối**, nên trình duyệt
chỉ gọi về đúng cổng 3000 nó đang mở, dù đó là IP nào. Đừng điền IP tuyệt đối vào
hai biến này — làm thế là khoá cứng ảnh vào một máy chủ.

Địa chỉ public của object storage là phần phụ thuộc môi trường triển khai:

```bash
# Máy dev/LAN, truy cập MinIO trực tiếp
MINIO_PUBLIC_ENDPOINT=<ip-may-chu>:9000
MINIO_PUBLIC_SECURE=false

# Production, ưu tiên domain HTTPS qua reverse proxy
MINIO_PUBLIC_ENDPOINT=media.example.com
MINIO_PUBLIC_SECURE=true
```

Ảnh và video không được proxy qua API — API trả **307 sang presigned URL của
MinIO** và trình duyệt tự đi tiếp. URL đó do máy chủ ký nhưng người dùng mới là
người mở, nên nếu để trống thì nó mang host `minio:9000` (tên service nội bộ):
API vẫn 200, log vẫn sạch, chỉ **ảnh trong thẻ kết quả là vỡ hết**. Endpoint S3
phải truy cập được từ trình duyệt; production nên đưa nó qua HTTPS thay vì mở
thẳng cổng 9000 ra internet.

> **Cấu hình compose hiện tại chỉ phù hợp máy dev tin cậy.** Nó publish 5432,
> 6333/6334, 8000, 8091, 2024 và 9001 trên `0.0.0.0`; LangGraph còn chạy
> `auth=noop`. Trên máy chủ công khai phải bind các cổng nội bộ về `127.0.0.1`,
> đổi toàn bộ mật khẩu mặc định, chặn 9001, đặt firewall/security group và dùng
> HTTPS qua reverse proxy. Chỉ public 3000 và endpoint S3 cần cho presigned URL;
> không mở MinIO 9000 với `minioadmin/minioadmin`.

`embedding_server` (jina-clip-v2 tự host) nằm ngoài mặc định vì cần GPU và ~3,5GB
trọng số. Muốn tự host riêng phần media embedding:

```bash
docker compose --profile gpu up -d embedding-server
# rồi trỏ MEDIA_IMAGE_EMBEDDING_BASE_URL=http://embedding-server:8100
```

Việc này không tự biến cả hệ thống thành ngoại tuyến: LLM, text embedding và
VLM vẫn phải được trỏ tới endpoint tự host nếu muốn không gọi dịch vụ ngoài.
RunPod Serverless `active=0` hợp với batch index nhưng có cold start, không nên
dùng cho truy vấn online cần phản hồi ngay.

`runpod_worker/` giữ nguyên Dockerfile riêng — nó là job hàng đợi chạy trên GPU
của RunPod, không phải service sống lâu trong compose.

## Chạy từng tầng ở máy thật (khi lập trình)

Tạo môi trường và cài dependency một lần từ thư mục gốc repo:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r API/requirements.txt -r mcp/requirements.txt -e ChatBot
python -m pip install "langgraph-cli[inmem]" pytest ruff
(cd web && npm ci)
```

Hạ tầng vẫn nên để trong Docker. Sau khi đã khởi tạo schema như phần Docker,
chạy bốn tầng ứng dụng trong **bốn terminal riêng**, đúng thứ tự dưới lên; mọi
lệnh dưới đây bắt đầu từ thư mục gốc repo và dùng cùng virtualenv:

```bash
docker compose up -d postgres qdrant minio

(cd API && python -m uvicorn src.server:app --reload --port 8000)  # terminal 1
(cd mcp && PYTHONPATH=src python src/server.py)                    # terminal 2
(cd ChatBot && langgraph dev --no-browser)                         # terminal 3
(cd web && npm run dev)                                            # terminal 4
```

Kiểm nhanh cả bốn tầng:

```bash
curl -s localhost:8000/health                                 # {"status":"ok"}
curl -s -o /dev/null -w "%{http_code}\n" localhost:8091/mcp   # khác 000 là được
curl -s -X POST localhost:2024/threads -H 'Content-Type: application/json' -d '{}'
curl -s -o /dev/null -w "%{http_code}\n" localhost:3000       # 200
```

> `.env` phải để `MCP_BASE_URL=http://localhost:8000` khi chạy tay. Chạy API ở
> cổng khác thì MCP gọi vào chỗ trống, supervisor báo "không có dữ liệu", mà
> **không có lỗi nào hiện ra ở web**.

Chi tiết và danh sách kiểm trước khi demo: [docs/chay-demo.md](docs/chay-demo.md).

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
| Caption + OCR (một lượt gọi trả cả hai) | `Qwen3-VL-8B-Instruct` tự host trên RunPod (vLLM) | |
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

Đã chốt **không** làm trong v1: agentic RAG, tìm kiếm lai theo từ khoá, rerank,
lọc theo đối tượng phát hiện được, miền `privacy`. Ngoài ra còn hai chỗ đã biết
là chưa đủ tốt:

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
