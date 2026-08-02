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

> **Cập nhật 02/08/2026 — đã có đường thoát, xem §1b.** Trọng số `jina-clip-v2`
> tự host được trên GPU RunPod, và **đã đo là cho ra đúng vector như API**. Nạp
> tiền giờ là một lựa chọn, không còn là điều kiện bắt buộc.

Kiểm tra nhanh xem tài khoản đã sống lại chưa:

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

## 1b. Đường thoát: tự host chính model đó — `embedding_server/`

**Giữ nguyên `jina-clip-v2`, chỉ đổi chỗ chạy.** Đổi sang model khác là vứt toàn
bộ 2.409 điểm đang có (hai model là hai không gian vector), mà vẫn phải chạy nốt
38 video — mất cả chì lẫn chài.

**Đã đo parity, và đạt.** `scripts/check_embedding_parity.py` lấy điểm thật
trong Qdrant, tải đúng ảnh đó từ MinIO, tiền xử lý bằng chính
`ImageEmbeddingService._image_as_base64`, nhúng lại rồi so cosine:

| Collection | n | Thấp nhất | Trung bình |
|---|---|---|---|
| `media_clip` | 4 | 0.999841 | 0.999904 |
| `video_transcript` | 4 | 0.998703 | 0.999558 |

→ **cùng không gian vector, index cũ dùng tiếp được.** Hai model khác nhau cho
cosine quanh 0 ở 1024 chiều, chứ không phải 0.99. Script chạy được cả khi Jina
đang chết vì nó chỉ đọc Qdrant + MinIO.

Client **không phải sửa một dòng nào** — server nói đúng giao thức của Jina, chỉ
đổi cấu hình:

```bash
MEDIA_IMAGE_EMBEDDING_BASE_URL=https://<pod>-8100.proxy.runpod.net/v1/embeddings
JINA_API_KEY=local                     # service chặn nếu khoá rỗng
MEDIA_EMBEDDING_TOKENS_PER_MINUTE=0    # tắt giữ nhịp, hạn mức Jina không còn
```

Chi tiết deploy, ba tham số không được sai, và vì sao phải ghim
`transformers<5`: đọc `embedding_server/README.md`.

**Hai điều đã đo được, đừng phát hiện lại:**

- **Máy API hiện tại không chạy nổi model.** 7,6GB RAM / 2 nhân: nạp xong chiếm
  3,68GB thường trú, đỉnh 5,05GB, forward pass đầu tiên bị OOM giết (exit 137).
  `bfloat16` không cứu được (đỉnh vẫn 4,89GB). Phép đo trên chỉ chạy xong nhờ
  tạm thêm 8GB swap. Nên **nhúng câu hỏi cũng phải đi qua pod RunPod**, và pod
  phải **thường trực** chứ không serverless — cold start 30–60 giây thì
  `/api/media/search` còn tệ hơn bây giờ.
- **`transformers` 5.x làm vỡ remote code của Jina** ở hai nấc (`hasattr(torch,
  torch_dtype)` rồi `ImportError: clip_loss`). Nấc đầu lách được bằng cách
  truyền tên kiểu dưới dạng chuỗi; nấc sau thì không. Đã ghim `<5`.

---

## 2. Trạng thái dữ liệu hôm nay

| Thứ | Số thật | Ghi chú |
|---|---|---|
| `media_clip` | **2.359** điểm | 1.628 ảnh tĩnh + 731 keyframe video |
| `video_transcript` | **50** điểm | |
| `rag_documents` | **4** điểm | corpus nội quy còn rất nhỏ |
| Video đã index | **21 / 59** | 38 video còn lại, ~1.700 keyframe |
| Caption keyframe | **2.428 / 2.429** | |
| Test `API/` | **96 passed** | offline hoàn toàn |
| Test `ChatBot/` | **15 passed** | offline hoàn toàn |

Toàn bộ **2.359 điểm hiện có đã được gắn `source_url`** bằng đường payload-only
(không nhúng lại, không tốn token — xem §3). Các video index sau này tự có sẵn.

**8 collection rác** `*_test*` trong Qdrant: **đã xoá** (34 điểm). Ba collection
thật giữ nguyên số điểm.

Một keyframe mồ côi — `frames/161c0e31-…/video_01_frame_012375.jpg`: object không
tồn tại trong MinIO (`NoSuchKey`, trong khi 116 frame anh em cùng thư mục đều còn;
frame cuối thật sự là `…_012371.jpg`). **Chưa xoá dòng DB**, vì khi kiểm lại thấy
nó *có* `caption_results` trạng thái `DONE` kèm caption thật — mâu thuẫn với giả
định "chưa bao giờ caption được" ban đầu. Dòng này vô hại (lúc index bị lọc ra vì
thiếu object); ai muốn dọn thì xoá `media_id = 00062686-a5d6-45b8-b014-74ecc01c89b8`,
FK sẽ CASCADE sang `frames`/`caption_results`/`ocr_results`/`object_results`.

---

## 3. Việc đầu tiên khi có đường nhúng trở lại: index nốt 38 video

> Áp dụng cho cả hai đường — pod RunPod (§1b) hay tài khoản Jina đã nạp. Nếu
> dùng RunPod thì nhớ `MEDIA_EMBEDDING_TOKENS_PER_MINUTE=0`, bộ giữ nhịp sinh
> ra để né hạn mức của Jina, giờ chỉ làm chậm.

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

### Sửa payload mà không nhúng lại

Khi chỉ cần thêm/sửa **một khoá payload** (như `source_url` vừa rồi), đừng index
lại — vector không hề đổi, nhúng lại là đốt quota vô ích. Cả hai nhánh đều có
đường payload-only, **chạy được cả khi Jina hết tiền**:

```bash
"$PY" scripts/index_image_units.py --payload-only --apply          # ảnh tĩnh
"$PY" scripts/index_video_retrieval_units.py <video_media_id> --payload-only
```

Đo thực tế: 1.628 ảnh mất ~1 phút, so với ~78 phút nếu nhúng lại.

---

## 4. Việc còn lại, theo thứ tự ưu tiên

### Cần quyết định của anh Hoàng (không tự làm)

1. **Chọn đường nhúng:** deploy `embedding_server/` lên pod RunPod thường trực
   (§1b — không tốn Jina, parity đã đo là đạt), hay nạp lại Jina, hay cả hai
   (RunPod lo index, Jina lo câu hỏi). Cho tới khi chọn thì `/api/media/search`
   và mọi script index vẫn đứng.
2. **Xoay khoá API** — file `.env.bak.0640` từng suýt lọt vào commit (đã gỡ khỏi
   index trước khi đẩy, chưa rò ra ngoài, nhưng khoá đã nằm trên đĩa một thời gian).

~~3. Đẩy nhánh~~ — `integration/v1` **đã lên remote**, local và
`origin/integration/v1` trùng nhau ở 81 commit. Nhánh local chưa set upstream nên
`git status` không in dòng "ahead of origin"; gắn bằng
`git branch --set-upstream-to=origin/integration/v1`. **Chưa có PR** nào mở về
`develop`.

### Lệch PRD — **đã đóng**, giữ lại để biết đường mà kiểm

4. ~~**US-401.1 AC-2**~~ — timeout nay kèm kết quả truy xuất thô. `ToolResultCollector`
   ghi lại `context` của từng tool ngay khi tool trả về, nên khi `asyncio.wait_for`
   cắt ngang thì vẫn còn thứ để đưa cho người dùng. Test:
   `ChatBot/tests/test_supervisor_agent.py`.
5. ~~**US-405.1 `source_url`**~~ — cột dữ liệu vốn đã có sẵn (`posts.post_url`,
   496/496 bài đều có), chỉ thiếu đường ống. Nay đi hết: unit → payload Qdrant →
   kết quả API → chuỗi `context` → prompt. Không có link thì ghi "nguồn nội bộ"
   (AC-2), **không bao giờ** dựng URL. Test: `API/tests/test_source_url_citation.py`.

### Dọn dẹp

6. ~~Xoá 8 collection `*_test*`~~ — đã xoá.
7. Dòng DB của keyframe mồ côi — **cố tình chưa xoá**, lý do ở §2.
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
