# Bàn giao — HIT-MIRA Multimodal RAG

> Cập nhật **02/08/2026**, nhánh `integration/v1` (đã lên remote).
> Viết cho người/agent tiếp nhận. Đọc §1 và §1b trước khi gõ bất cứ lệnh nào —
> hệ thống **đang chạy hoàn chỉnh trên embedding server tự host**, không còn phụ
> thuộc API trả phí. Việc duy nhất cần trông: pod GPU tính tiền theo giờ, nhớ
> tắt khi không dùng (§4.1).

**Đọc kèm:** `CLAUDE.md` (ràng buộc kiến trúc, phần quan trọng nhất) →
`docs/structure.md` (cây code thật) → `docs/prd.md` (US/TC để neo test).

---

## 1. Tài khoản Jina hết số dư — KHÔNG còn là ngõ cụt (xem §1b)

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

**Giữ nguyên `jina-clip-v2`, chỉ đổi chỗ chạy.** Lúc quyết định, đổi sang model
khác nghĩa là vứt toàn bộ 2.409 điểm đang có (hai model là hai không gian
vector) mà vẫn phải chạy nốt 38 video — mất cả chì lẫn chài. Giữ nguyên model
nên index cũ sống, và giờ con số đó đã là **4.248 điểm**.

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

**Cả ba đường đã chạy thật qua pod ngày 02/08/2026**, không chỉ qua test:

| Đường | Kiểm bằng | Kết quả |
|---|---|---|
| index 38 video | đếm điểm trong Qdrant | 38/38, 0 lỗi, `media_clip` 2.359 → 4.056 |
| `/api/media/search` | câu hỏi tiếng Việt thật | `context` có link Facebook thật, ảnh + lời thoại cùng ra |
| `/api/retrieval/search` | câu hỏi nội quy thật | qua LangChain → pod, có cả bước viết lại truy vấn |

Một endpoint phục vụ được cả ba vì server nhận **hai định dạng `input`**: kiểu
Jina (`[{"text": …}]`) cho nhánh media, và kiểu OpenAI (`["chuỗi"]`) cho nhánh
nội quy đi qua LangChain. Bản đầu chỉ nhận dạng object và nhánh nội quy ăn 422 —
lỗi hiện ra tận trong LangChain nên rất khó lần ngược về server.

**Chi phí thật:** cả đợt index 38 video hết **\$0,07**. Pod RTX A5000 là
**\$0,27/giờ** (không phải \$0,16 như bảng giá cộng đồng) → **~\$195/tháng** nếu
để thường trực. Số dư \$8,19 chỉ trụ được ~30 giờ, nên **pod thường trực không
phải phương án nuôi được bằng số dư hiện tại** — xem §4.

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
| `media_clip` | **4.056** điểm | 1.628 ảnh tĩnh + 2.428 keyframe video |
| `video_transcript` | **248** điểm | 56 video; từng là 192/45 trước khi vá bug mốc giây 0 (§2b) |
| `rag_documents` | **4** điểm | corpus nội quy còn rất nhỏ — xem §4 |
| Video đã index | **59 / 59** | xong ngày 02/08/2026 trên GPU RunPod |
| Caption keyframe | **2.428 / 2.429** | |
| Test `API/` | **101 passed** | offline hoàn toàn |
| Test `ChatBot/` | **15 passed** | offline hoàn toàn |

**`source_url` phủ 100%**: 4.056/4.056 điểm `media_clip` và 192/192 điểm
`video_transcript`. Các điểm cũ được gắn bằng đường payload-only (không nhúng
lại, không tốn token — xem §3); các video index sau đó tự mang sẵn.

Đường trích dẫn đã **chạy thật một lần** ngày 02/08/2026, không chỉ qua test:
một câu hỏi tiếng Việt đi hết `/api/media/search` và trả về `context` có link
Facebook thật, ảnh và lời thoại cùng ra trong một lượt, ảnh không mang mốc giây
còn video thì có.

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

## 2b. Bug mốc giây 0 — đã vá 02/08/2026

`0.0` là falsy trong Python, nên `segment.get("start_sec") or segment.get("start_time")`
ở `scripts/import_media_outputs.py` rơi sang nhánh sau và ghi **None**. Segment
đầu của **mọi** transcript bắt đầu đúng ở giây 0, nên cả 56 transcript đều mất
mốc, bộ dựng unit loại chúng vì `invalid_timestamp`, và **11 video chỉ có một
segment thì biến mất hoàn toàn** khỏi `video_transcript`.

| | Trước | Sau |
|---|---|---|
| Điểm `video_transcript` | 192 | **248** |
| Video có lời thoại tìm được | 45 | **56** |
| Ký tự lời thoại mất | 18.777 / 112.006 = **16,8%** | 0 |

Phần mất là mở đầu video — chỗ nói tên câu lạc bộ và tên sự kiện, tức phần nhận
dạng rõ nhất. Sau khi vá, câu hỏi "sinh nhật lần thứ mười ba của câu lạc bộ tin
học" trả về đúng đoạn `[0..60s]` ở **hạng 1**.

Đã vá cả ba tầng:

- **Code**: hàm `so_dau_tien()` coi `0.0` là số hợp lệ, + 2 test chặn tái diễn.
- **Dữ liệu**: `update transcript_segments set start_time=0.0 where start_time is null`
  (56 dòng). An toàn vì đã kiểm trước: cả 56 đều là segment ĐẦU của transcript,
  và các segment là cửa sổ 60 giây liền mạch nên giá trị đúng chắc chắn là 0.
- **Index**: thêm cờ `--transcript-only` cho `index_video_retrieval_units.py`.
  Hai nhánh vốn độc lập, nên vá lời thoại **không cần** nhúng lại 2.428 keyframe
  cho ra đúng vector cũ. 56 video, 0 lỗi, `media_clip` giữ nguyên 4.056 điểm.

> **Bài học:** `summary.json` của lần build thật **đã ghi sẵn** `"invalid_timestamp": 1`
> kèm cả `segment_id`, ngay từ đầu. Bằng chứng nằm trên đĩa hàng tuần mà không ai
> đọc — đúng cùng một hạng lỗi với "đừng tin dòng OK của script batch" ở §1.
> Đọc `skipped_by_reason` sau mỗi lần build; số khác 0 nghĩa là có dữ liệu bị bỏ.

---

## 3. Index — ĐÃ XONG, giữ lại để biết đường chạy lại

> **59/59 video đã index xong ngày 02/08/2026** trên pod GPU RunPod, 38 video
> trong khoảng 70 phút, 0 lỗi, tốn khoảng \$0,30. Phần dưới giữ lại cho lần sau
> (thêm video mới, hoặc phải nhúng lại vì đổi model).
>
> Nhớ `MEDIA_EMBEDDING_TOKENS_PER_MINUTE=0` khi dùng pod tự host: bộ giữ nhịp
> sinh ra để né hạn mức của Jina, với pod riêng thì chỉ làm chậm.

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

1. ~~**Chọn nguồn nhúng cho CÂU HỎI**~~ — **đã chốt: tự host, không dùng API trả
   phí.** Pod GPU `tbhzy5i5ypqxuv` (RTX A4000, **community cloud, \$0,17/giờ**)
   dựng ngày 02/08/2026, và `.env` đã trỏ sang nó. Đã kiểm thật, không chỉ qua
   test:

   | Kiểm | Kết quả |
   |---|---|
   | parity với điểm đang có trong Qdrant | `media_clip` thấp nhất **0.999894** · `video_transcript` **0.999916** → cùng không gian, index cũ nguyên giá trị |
   | `POST /api/media/search` | HTTP 200 trong **1,7s**, ảnh có caption + `source_url` Facebook thật |
   | `POST /api/retrieval/search` | HTTP 200 trong **5,5s** (gồm bước viết lại truy vấn qua LLM), qua LangChain → pod |

   **Việc còn lại là chuyện tiền, không phải chuyện kỹ thuật.** Pod tính tiền
   theo giờ kể cả lúc ngồi không, mà nhúng một câu hỏi chỉ tốn ~20 token:

   | Phương án | Chi phí | Số dư \$8,13 trụ được |
   |---|---|---|
   | Pod community thường trực (đang chạy) | \$0,17/giờ ≈ \$122/tháng | ~48 giờ |
   | Pod secure thường trực | \$0,27/giờ ≈ \$195/tháng | ~30 giờ |
   | Bật lúc demo rồi tắt | vài xu mỗi buổi | rất lâu |

   Máy API **không** chạy CPU thay thế được (7,6GB RAM, OOM — xem §1b).
   **Nhớ tắt khi không dùng:**
   `python scripts/deploy_embedding_pod.py --terminate tbhzy5i5ypqxuv`

   > Community cloud rẻ hơn nhưng **hay hết máy**: A5000 trả `SUPPLY_CONSTRAINT`
   > ngay lần đầu. Script nay tự chuyển sang GPU dự phòng trong danh sách ưu
   > tiên (A5000 → A4000 → …), nên hết máy không còn là bế tắc.
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

### Ingest — việc còn lại (soát toàn diện ngày 02/08/2026)

6. **2 video chưa xử lý xong.** `video_01.mp4` của post `1028940839054401` và
   `1104021571546327`: đều dài ~220s, **có tiếng (aac, đã ffprobe)**, nhưng không
   có dòng `transcripts` nào và job ASR vẫn `PENDING`. Keyframe cũng chỉ chọn được
   **2 và 3** khung từ 6.617/5.278 khung thô, trong khi video khác cỡ 10–15
   khung/phút. Chạy dở dang, không phải video câm. `data/asr_test_clips/` chứa
   sẵn clip cắt từ đúng hai video này — ai đó đã từng gỡ lỗi chúng.
7. **Kho nội quy gần như trống**: `rag_documents` chỉ **4 chunk, ~2.000 ký tự**,
   toàn bộ là "Nội quy sử dụng phòng" + địa chỉ CLB. `search_regulations` hầu như
   không có gì để trả lời. Đây là thiếu nội dung, không phải lỗi code.

> Ngưỡng trong `CLAUDE.md` — "khi `rag_documents` vượt ~50 chunk thì đo lại top-1"
> — còn rất xa. Chưa cần tách model text riêng cho nhánh nội quy.

### Đánh giá — đã có số (02/08/2026)

Bộ đánh giá dựng xong: 50 truy vấn có nhãn, chỉ số, và checklist demo chạy được.
Số đo và **ba cảnh báo phải đọc trước khi trích số**: `docs/eval-report.md`.

| | Đo được | Mục tiêu | |
|---|---|---|---|
| Recall@5 | 0,642 | ≥ 0,80 | chưa đạt |
| MRR | 0,776 | ≥ 0,60 | đạt |
| Latency trung bình | 0,85s (p95 1,71s) | ≤ 5s | đạt |
| Checklist demo | 5 PASS · 0 FAIL · 2 chưa hỗ trợ | mọi luồng pass | một phần |

Hai kết luận đáng nhớ:

- **T-33 chốt được rồi: đừng dùng ngưỡng cosine cho "không tìm thấy".** Điểm câu
  ngoài miền (cao nhất 0,389) chồng lấn điểm câu trong miền (thấp nhất 0,254),
  nên không ngưỡng nào tách được. Việc từ chối phải ở tầng trả lời — và
  `supervisor_prompt.md` đang làm đúng vậy.
- **Tên sự kiện chỉ nằm trong OCR, không có trong caption.** "Open Day",
  "hackathon", "seminar", "gala" xuất hiện 0 lần trong caption. Vector là của
  hình ảnh, nên loại câu hỏi đó v1 về bản chất không trả lời được — đúng phần đã
  hoãn sang v2. Đo riêng được 0,367/0,500.

### Đã soát và SẠCH

496/496 post có media · 59/59 video có dòng `videos` + frame · **MinIO khớp DB**
(4.116 dòng, đúng 1 object thiếu là keyframe mồ côi đã biết) · `source_url` phủ
100% · caption/OCR/detection **4.057/4.057 DONE** · **ASR phủ trung bình 100,7%
thời lượng video, không video nào dưới 50%** (đã nghi bị cắt ngắn, đo ra là không).

### Dọn dẹp

8. ~~Xoá 8 collection `*_test*`~~ — đã xoá.
9. Dòng DB của keyframe mồ côi — **cố tình chưa xoá**, lý do ở §2.
10. Một transcript chạy bằng `large-v3` thay vì Zipformer như phần còn lại → chạy
    lại cho đồng nhất. Một transcript `FAILED` (`video_07.mp4` của post
    `1320967086518440`). Một caption `DONE` mà rỗng.
11. 27 job `PENDING` treo từ 22/07 (asr 3, caption 8, ocr 8, object_detection 8) —
    rác lịch sử, việc thật đã chạy xong qua script.

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

- **Đừng viết `a or b` cho số.** `0` và `0.0` là falsy nên luôn rơi sang `b`.
  Đây chính là bug ở §2b, giấu mất 16,8% lời thoại suốt nhiều tuần mà không
  script nào báo lỗi. Dùng `so_dau_tien()` trong `scripts/import_media_outputs.py`,
  hoặc kiểm `is None` tường minh.
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
cd API && python -m pytest -q                  # 97 test, offline
cd API && ruff check .

# MCP server (cổng 8091) — ChatBot cần nó sống thì mới có tool
cd mcp && PYTHONPATH=src python src/server.py

# ChatBot
cd ChatBot && python -m pytest -q              # 15 test, offline
cd ChatBot && langgraph dev

# Hạ tầng
docker-compose up -d                           # postgres + qdrant + minio

# Đánh giá (E8) — cần nguồn nhúng đang sống
python scripts/build_eval_labels.py --apply    # giải nhãn từ kho -> eval_queries.json
python scripts/run_eval.py                     # Recall@k / MRR / latency
python scripts/demo_checklist.py               # checklist demo, exit != 0 nếu có FAIL

# Nguồn nhúng (§1b) — pod tính tiền theo giờ, nhớ tắt
python scripts/deploy_embedding_pod.py --apply
python scripts/deploy_embedding_pod.py --status
python scripts/deploy_embedding_pod.py --terminate <POD_ID>
python scripts/check_embedding_parity.py --url <endpoint> --api-key <khoá>
```
