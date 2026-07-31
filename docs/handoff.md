# Bàn giao — HIT-MIRA Multimodal RAG

> Chốt ngày **31/07/2026**, nhánh `integration/v1` (chưa đẩy lên remote).
> Viết cho người/agent tiếp nhận. Đọc hết phần §1 trước khi gõ bất cứ lệnh nào —
> có một thứ đang chặn toàn bộ hệ thống.

**Đọc kèm:** `CLAUDE.md` (ràng buộc kiến trúc, phần quan trọng nhất) →
`docs/structure.md` (cây code thật) → `docs/prd.md` (US/TC để neo test).

---

## 1. ĐANG BỊ CHẶN: tài khoản Jina hết số dư

```
HTTP 403  AUTHZ_INSUFFICIENT_BALANCE
"Insufficient account balance. Top up your account at https://jina.ai/api-dashboard/key-manager"
```

Cả hệ thống dùng **một model nhúng duy nhất** (`jina-clip-v2`, xem `CLAUDE.md`
§ "One embedding model"), nên một tài khoản chết là chết cả ba đường:

| Đường | Trạng thái |
|---|---|
| `POST /api/media/search` | HTTP 500 |
| `POST /api/retrieval/search` | HTTP 500 |
| Mọi script index | nhúng 0 điểm |

**Không có cách nào lách.** Phải nạp tiền hoặc thay `JINA_API_KEY` trong `.env`.
Kiểm tra nhanh xem đã sống lại chưa:

```bash
curl -s https://api.jina.ai/v1/embeddings \
  -H "Authorization: Bearer $JINA_API_KEY" -H "Content-Type: application/json" \
  -d '{"model":"jina-clip-v2","task":"retrieval.query","input":[{"text":"thử"}]}' | head -c 200
```

> `task` **chỉ nhận** `retrieval.query`. Gửi `retrieval.passage` là HTTP 422 —
> nhúng bất đối xứng thuộc về `jina-embeddings-v3`, một model khác.

### Sự cố đi kèm, đã vá

Khi hết số dư, `IndexingService` **nuốt lỗi**: nó bắt hết `Exception` để một
keyframe hỏng không giết cả video — đúng ý đồ, nhưng một nhà cung cấp chết trông
y hệt một ảnh hỏng. Kết quả: **23 video liên tiếp index được 0 điểm trong khi
script vẫn in `OK`** (script chỉ nhìn mã thoát). Chuyện chỉ lộ ra vì có người
tình cờ hỏi chatbot.

Đã sửa: `ImageEmbeddingProviderFatalError` cho HTTP **401/402/403** — không thử
lại, không bị nuốt, mẻ chạy dừng ngay và nói rõ lý do.
Xem `API/src/rag_video_anh/embedding/embedding_service.py` và
`API/tests/test_embedding_fatal_errors.py` (5 test).

**Bài học cho người tiếp nhận:** đừng tin dòng `OK` của script batch. Kiểm bằng
số điểm thật trong Qdrant.

---

## 2. Trạng thái dữ liệu hôm nay

| Thứ | Số thật | Ghi chú |
|---|---|---|
| `media_clip` | **2.359** điểm | 1.628 ảnh tĩnh + 731 keyframe video |
| `video_transcript` | **50** điểm | |
| `rag_documents` | **4** điểm | corpus nội quy còn rất nhỏ |
| Video đã index | **21 / 59** | 38 video còn lại, ~1.700 keyframe |
| Caption keyframe | **2.428 / 2.429** | |
| Test `API/` | **82 passed** | offline hoàn toàn |
| Test `ChatBot/` | **11 passed, 1 xfailed** | xfail là chủ ý, xem §4 |

Một keyframe **không bao giờ** caption được:
`frames/161c0e31-0769-4e2e-8784-8b2c1a03853a/video_01_frame_012375.jpg` — có dòng
trong PostgreSQL nhưng object không tồn tại trong MinIO. Thử lại vô ích; lúc index
nó bị lọc ra. Nên xoá dòng DB cho sạch.

Trong Qdrant còn **8 collection rác** tên `*_test*` (tổng 34 điểm), vô hại nhưng
nên dọn: `media_clip_api_test`, `media_clip_jina_test`, `…_2`, `…_3`, `…_4`,
`video_transcript_api_test`, `video_transcript_jina_test_3`, `…_4`.

---

## 3. Việc đầu tiên khi Jina sống lại: index nốt 38 video

An toàn để chạy lại từ đầu — point ID sinh bằng uuid5 của khoá nghiệp vụ, nên
index lại là **ghi đè, không nhân bản**. Video đã xong chỉ tốn công nhúng lại.

```bash
cd /home/ubuntu/HIT-MIRA-Multimodal-RAG
PY=/home/ubuntu/miniconda3/envs/nhhoang/bin/python

# Lấy danh sách video đã parse xong từ PostgreSQL (tự viết truy vấn theo
# repository/models.py), rồi chạy từng cái một:
while read -r id; do
  "$PY" scripts/index_video_retrieval_units.py "$id"
done < danh_sach_video.txt
```

Sau đó **kiểm bằng số điểm**, đừng tin log:

```bash
"$PY" - <<'PY'
import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
load_dotenv("/home/ubuntu/HIT-MIRA-Multimodal-RAG/.env")
c = QdrantClient(url=os.environ["QDRANT_URL"], api_key=os.environ.get("QDRANT_API_KEY") or None)
for name in ("media_clip", "video_transcript", "rag_documents"):
    print(name, c.count(name, exact=True).count)
PY
```

Ước lượng: Jina cho **100.000 token/phút**, một ảnh 512px tốn đúng **4.000 token**
→ trần ~25 ảnh/phút → 1.700 keyframe ≈ **70 phút**. Đừng thay bộ giữ nhịp chủ động
(`_reserve_tokens`) bằng backoff phản ứng-với-429: đo được ~10 ảnh/phút thay vì 20–25.

---

## 4. Việc còn lại, theo thứ tự ưu tiên

### Cần quyết định của anh Hoàng (không tự làm)

1. **Nạp Jina** — chặn mọi thứ.
2. **Đẩy nhánh** — `integration/v1` có **79 commit chưa từng lên remote**.
3. **Xoay khoá API** — file `.env.bak.0640` từng suýt lọt vào commit (đã gỡ khỏi
   index trước khi đẩy, chưa rò ra ngoài, nhưng khoá đã nằm trên đĩa một thời gian).

### Lệch PRD — code chưa đủ so với tài liệu

4. **US-401.1 AC-2** — timeout phải trả về *kết quả truy xuất thô kèm báo lỗi nhẹ*.
   Hiện `SupervisorAgent._timeout_response` chỉ trả một lời xin lỗi: người dùng chờ
   hết giờ rồi nhận về con số không.
   → Đã có test đánh dấu sẵn: `ChatBot/tests/test_supervisor_agent.py::test_timeout_van_kem_theo_ket_qua_tho`
   là `xfail(strict=True)`. Làm xong thì test XPASS và **bắt buộc phải gỡ marker** —
   đó là cách mốc này tự báo là đã đóng.
5. **US-405.1 `source_url`** — câu trả lời phải gắn link bài gốc; payload media hiện
   chỉ có `post_id`. Thêm được, nhưng phải sửa payload Qdrant, mà nhánh video **không
   có** `--payload-only` (chỉ `scripts/index_image_units.py` có) → với video là phải
   nhúng lại thật. **Nên gộp vào lượt index ở §3**, đừng tách ra thành lượt riêng.

### Dọn dẹp

6. Xoá 8 collection `*_test*` trong Qdrant (§2).
7. Xoá dòng DB của keyframe mồ côi (§2).
8. Một transcript chạy bằng `large-v3` thay vì Zipformer như phần còn lại → chạy lại
   cho đồng nhất.

---

## 5. Bẫy đã dẫm, đừng dẫm lại

**Môi trường**

- Python: `/home/ubuntu/miniconda3/envs/nhhoang/bin/python` (3.12). Gọi bằng
  **đường dẫn tuyệt đối** — `conda activate` không sống qua từng lệnh.
- Mọi script rời phải `load_dotenv("/home/ubuntu/HIT-MIRA-Multimodal-RAG/.env")`,
  nếu không PostgreSQL báo `password authentication failed for user "hit"`.
- `uv` **chưa được cài** trên máy này, nên `ChatBot/Makefile` (dùng `uv run`) không
  chạy được. Chạy thẳng: `cd ChatBot && python -m pytest -q`.

**Git**

- **Không bao giờ `git add -A`.** Stage từng file một. `.env.bak.0640` chứa khoá
  thật đã suýt vào commit đúng vì thói quen này.
- Docs, comment và commit message viết **tiếng Việt** — đây là quy ước của dự án.

**Code**

- Bắt hết `Exception` trong vòng lặp batch che mất lỗi hệ thống. Nếu thêm nhánh
  `except` mới ở `IndexingService`, phải để `ImageEmbeddingProviderFatalError` đi
  xuyên qua (`except ImageEmbeddingProviderFatalError: raise` đặt **trước** nhánh
  `except Exception`).
- `ChatBot/src/graph/graph.py`: `init_root_graph()` **phải đồng bộ**. Bản cũ khai
  `async` rồi gọi `asyncio.run()` lúc import → chết trong mọi tiến trình đã có event
  loop. Có test canh: `tests/test_graph_khoi_tao.py`.
- Tool của supervisor được liệt kê **tường minh** ở `REQUIRED_TOOLS`. Thêm tool vào
  `mcp/Resources/tools.yaml` là **chưa đủ** — agent sẽ không thấy nó. Đây chính là
  lỗi từng làm chatbot từ chối mọi câu hỏi về ảnh/video.
- System prompt bị chạy qua `format_map`. Dán một mẫu JSON có `{` lẻ vào là **mọi
  lượt chat** ném `ValueError` trước khi kịp gọi model. Có test canh:
  `tests/test_prompt_supervisor.py`.
- `LANGCHAIN_TRACING_V2` để `false`. Bật với khoá rỗng thì mỗi lượt chat spam 403.

**Kiểm chứng**

- Đừng `grep -E "^--- AIMessage"` để đọc câu trả lời của chatbot — nó chỉ khớp
  dòng đầu và làm câu trả lời trông như bị cụt. (Đã tưởng nhầm có bug vì chuyện này.)

---

## 6. Lệnh hay dùng

```bash
# Backend (import giả định thư mục làm việc là API/)
cd API && uvicorn src.server:app --reload      # health: GET /health
cd API && python -m pytest -q                  # 82 test, offline
cd API && ruff check .

# MCP server (cổng 8091) — ChatBot cần nó sống thì mới có tool
cd mcp && PYTHONPATH=src python src/server.py

# ChatBot
cd ChatBot && python -m pytest -q              # 11 passed, 1 xfailed
cd ChatBot && langgraph dev

# Hạ tầng
docker-compose up -d                           # postgres + qdrant + minio
```
