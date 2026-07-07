# PRD — Trợ lý ảo Multimodal RAG cho Fanpage CLB Tin học HIT

> Product Requirements Document · Version tài liệu: 1.0 · Ngày: 2026-06-10 · Mode: capstone
> Nguồn BRD: [docs/brd.md](brd.md) · Phạm vi PRD này: **v1 + v2** (BR-506 thuộc v3 — chỉ placeholder)

## 1. Tổng quan & Liên kết BRD

**Mục tiêu sản phẩm.** Trợ lý ảo hội thoại (chatbot trên web app) giúp hỏi-đáp và truy xuất chuyên sâu trong kho multimedia 5 năm gần nhất của Fanpage CLB Tin học HIT bằng **văn bản và hình ảnh**, trả về **ảnh kèm mô tả + clip/preview video** đúng ngữ cảnh. PRD này bung mỗi BR (v1 + v2) thành User Story đủ điều kiện để code & test.

**Phạm vi PRD này phủ các BR:**
- **v1 (Must/Should):** BR-101, BR-102, BR-104, BR-105, **BR-107**, BR-201, BR-202, BR-203, BR-205, **BR-207**, **BR-208**, BR-301, BR-302, BR-303, BR-306, **BR-307**, **BR-308**, BR-401, BR-402, BR-403, BR-404, BR-405, BR-406, **BR-407**, BR-501, BR-502, BR-503, BR-505, **BR-507**, BR-601, BR-602, BR-603, BR-604, BR-605, **BR-606**, BR-701, BR-704.
- **v2:** BR-103, BR-106, BR-204, BR-206, BR-304, BR-305, BR-504, BR-702, BR-703, BR-705.
- **Ngoài phạm vi:** BR-506 (Messenger fanpage — v3).

> **Bài toán #4 — Hỏi-đáp nội quy CLB (v1).** Ngoài 3 bài toán media (text→media, ảnh→sự kiện, ảnh→ảnh), trợ lý còn **tra cứu nội quy + trả lời câu hỏi tình huống** (text RAG riêng, neo điều/khoản + disclaimer — BR-107/207/307/407). Toàn hệ vận hành theo **Router RAG**: một bộ định tuyến (router) chọn nguồn phù hợp cho mỗi câu hỏi (kho media ↔ văn bản nội quy ↔ cả hai) — ưu tiên luật khi tín hiệu rõ, fallback classifier khi mơ hồ (BR-507), đo bằng bộ đánh giá riêng (BR-606). *(Agentic RAG đầy đủ — phân rã truy vấn/multi-hop/tổng hợp đa nguồn — để dành v2.)*

> **Stack đề xuất (giả định — sửa nếu sai).** Backend Python + **FastAPI**; embedding đa phương thức **multilingual CLIP** (OpenCLIP/Jina-CLIP); captioning **BLIP-2/VLM**; OCR (v2) **VietOCR/EasyOCR**; face (v2) **InsightFace**; **vector DB Qdrant** (FAISS cho POC); **PostgreSQL** cho metadata/quan hệ; **object storage** (filesystem/MinIO) cho media; **ffmpeg** trích frame + cắt clip; **LLM sinh câu trả lời = API free-tier** (Gemini/GPT/Claude) sau lớp abstraction; frontend **Next.js (React)** với khung chat. Lựa chọn cuối chốt ở `/write-hld`.

> **Ghi chú giảng dạy (capstone).** Phần data pipeline & data contract (BR-100/BR-200) được đặc tả kỹ ở Data Model + US tương ứng để vai **Data Engineer** có hạng mục chấm điểm rõ: schema bảng, trường bắt buộc, và test hook nạp/đo đều quy về một tiêu chí nghiệm thu.

## 2. User Journey

| Bước | Hành động | BR liên quan |
|---|---|---|
| 1 | Admin lưu hồ sơ cấp quyền dữ liệu từ Ban CN; bật cờ cho phép thu thập | BR-101, BR-701 |
| 2 | Admin/Data Engineer nạp dữ liệu (tải tay v1 / crawl v2) + chuẩn hóa metadata, lưu kho thô | BR-102, BR-103, BR-104, BR-105, BR-106 |
| 2b | Admin nạp **văn bản nội quy** CLB, tách điều/khoản | BR-107 |
| 3 | Pipeline xử lý: trích frame video, **tách audio → ASR transcript**, sinh caption/OCR, sinh embedding, xây/cập nhật vector index media + **index transcript** + **index nội quy riêng** | BR-201, BR-208, BR-203, BR-204, BR-202, BR-205, BR-206, BR-207 |
| 4 | Người dùng đăng nhập, mở chat, nhập câu hỏi text hoặc upload ảnh; **router tự định tuyến** tới nguồn phù hợp (media / nội quy / cả hai); câu hỏi về **nội dung nói trong video** khớp transcript | BR-501, BR-502, BR-505, BR-507, BR-301, BR-302, BR-303, BR-307, BR-308, BR-304, BR-305 |
| 5 | Hệ thống truy xuất + xếp hạng + sinh câu trả lời RAG có trích dẫn (media: trích nguồn; **nội quy: dẫn điều/khoản + disclaimer**) | BR-306, BR-401, BR-405, BR-407, BR-704 |
| 6 | Hiển thị kết quả đa phương thức: ảnh+mô tả, clip 3s/preview video, **câu trả lời nội quy có dẫn điều khoản**; xử lý "không tìm thấy"; hỏi nối tiếp (v2) | BR-402, BR-403, BR-404, BR-406, BR-503, BR-504 |
| 7 | Admin đánh giá Recall@k/MRR/độ trễ/chất lượng; xem log; xử lý gỡ bỏ & quyền riêng tư (v2) | BR-601–605, BR-702, BR-703, BR-705 |

## 3. Functional Requirements (User Stories)

> Mỗi US truy vết về một BR, đủ 5 mục L3 (Mục tiêu / Input / Output / Edge / Test) + Acceptance Criteria dạng Given/When/Then. Nhãn `[v1]`/`[v2]` theo version BR.

### Từ BR-101 · Xin quyền truy cập dữ liệu `[v1]`

#### US-101.1 — Lưu hồ sơ cấp quyền & cổng chặn thu thập
*Là `Quản trị viên (Admin/Data Engineer)`, tôi muốn lưu bằng chứng được Ban CN cấp quyền và chỉ cho phép thu thập sau đó, để đảm bảo tuân thủ.*

- **Mục tiêu**: Ràng buộc mọi thao tác nạp dữ liệu phải đứng sau một bản ghi consent hợp lệ.
- **Input**: Email/văn bản đồng ý (phạm vi sử dụng, ngày hiệu lực, người ký, file đính kèm).
- **Output**: Bản ghi `consent` (scope, ngày, file) được lưu; cờ `ingest_enabled=true`.
- **Edge**: Chưa có consent (hoặc consent hết hạn) → mọi endpoint ingestion trả `403` kèm thông báo rõ.
- **Test**: Gọi `POST /ingest/upload` khi chưa có consent → 403; tạo consent → gọi lại → 200.

**Acceptance Criteria**
- **AC-1** — Given chưa có bản ghi consent hợp lệ, When admin mở/gọi chức năng nạp dữ liệu, Then hệ thống chặn và hiển thị "Chưa có quyền sử dụng dữ liệu".
- **AC-2** — Given đã lưu consent hợp lệ, When admin nạp một lô dữ liệu, Then lô đó được gắn `consent_id` và cho phép tiếp tục.

### Từ BR-102 · Thu thập thủ công `[v1]`

#### US-102.1 — Tải tay bài/ảnh/video vào kho thô
*Là `Quản trị viên (Admin/Data Engineer)`, tôi muốn upload thủ công media + metadata, để có tập dữ liệu mẫu cho POC.*

- **Mục tiêu**: Nạp media gốc kèm metadata vào kho có cấu trúc.
- **Input**: File ảnh/video + form metadata (link bài gốc, ngày đăng, caption gốc, sự kiện).
- **Output**: Bản ghi `posts` + `media_assets`, file lưu storage, trả về danh sách ID đã tạo.
- **Edge**: Định dạng không hỗ trợ → từ chối kèm lý do; file trùng checksum → cảnh báo, không tạo bản ghi nhân đôi (nối BR-106).
- **Test**: Upload 1 batch sự kiện tiêu biểu → số bản ghi = số file hợp lệ; upload lại file trùng → không nhân đôi.

**Acceptance Criteria**
- **AC-1** — Given consent hợp lệ và file đúng định dạng, When admin upload kèm metadata bắt buộc, Then hệ thống tạo bản ghi và trả ID.
- **AC-2** — Given thiếu trường metadata bắt buộc, When submit, Then bị chặn và chỉ rõ trường thiếu.

### Từ BR-103 · Thu thập tự động (crawl) `[v2]`

#### US-103.1 — Crawl bài/ảnh/video từ fanpage qua quyền Admin
*Là `Data Engineer`, tôi muốn crawl tự động dữ liệu 5 năm kèm metadata, để phủ toàn bộ kho mà không tải tay.*

- **Mục tiêu**: Tự động thu thập và cập nhật bài mới từ fanpage.
- **Input**: Token/quyền Admin page, khoảng thời gian cần crawl (vd 5 năm gần nhất).
- **Output**: Bản ghi `posts`/`media_assets` được nạp + log số lượng; lần chạy sau chỉ lấy bài mới.
- **Edge**: API rate-limit/lỗi mạng → retry có backoff, ghi job thất bại để chạy lại; bài bị xóa trên nguồn → đánh dấu, không xóa cứng.
- **Test**: Chạy crawl tập con → đếm bản ghi khớp; chạy lần 2 → chỉ thêm bài mới (incremental).

**Acceptance Criteria**
- **AC-1** — Given có quyền Admin và consent, When chạy job crawl khoảng thời gian, Then media + metadata được nạp vào kho.
- **AC-2** — Given đã crawl trước đó, When chạy lại, Then chỉ bài mới được thêm, không nhân đôi (checksum/fb_post_id).

### Từ BR-104 · Chuẩn hóa metadata `[v1]`

#### US-104.1 — Chuẩn hóa & xác thực metadata bắt buộc
*Là `Data Engineer`, tôi muốn mỗi mục có đủ metadata chuẩn, để truy vấn và lọc về sau chính xác.*

- **Mục tiêu**: Đảm bảo 100% mục trong kho có đủ trường metadata bắt buộc.
- **Input**: Bản ghi media thô (từ tải tay/crawl).
- **Output**: Bản ghi đã chuẩn hóa: `posted_at`, `caption_original`, `source_url`, `media_type`, `event_id?`.
- **Edge**: Bài cũ thiếu ngày/caption → gắn cờ `needs_review`, không đưa vào index cho tới khi đủ trường tối thiểu.
- **Test**: Quét kho → báo cáo % mục đủ trường; mục thiếu bị loại khỏi tập index.

**Acceptance Criteria**
- **AC-1** — Given một bản ghi media mới, When lưu vào kho, Then các trường metadata bắt buộc được kiểm tra và chuẩn hóa định dạng (ngày ISO, loại media hợp lệ).
- **AC-2** — Given mục thiếu trường bắt buộc, When chạy kiểm tra, Then mục bị gắn `needs_review` và loại khỏi tập sẽ index.

### Từ BR-105 · Lưu trữ kho dữ liệu thô `[v1]`

#### US-105.1 — Lưu & truy xuất media theo ID
*Là `hệ thống`, tôi muốn lưu media gốc + metadata truy lại được theo ID, để các tầng sau tham chiếu ổn định.*

- **Mục tiêu**: Kho media bền vững, truy xuất theo ID trả đúng file + metadata.
- **Input**: File media + metadata đã chuẩn hóa.
- **Output**: `media_assets.id` + đường dẫn storage; `GET /media/{id}` trả file + metadata.
- **Edge**: ID không tồn tại → 404; file mất trên storage nhưng còn bản ghi → trả lỗi rõ + đánh dấu hỏng.
- **Test**: Lưu rồi `GET /media/{id}` → đúng file + metadata; ID lạ → 404.

**Acceptance Criteria**
- **AC-1** — Given media đã lưu, When gọi `GET /media/{id}`, Then trả đúng file/metadata tương ứng.
- **AC-2** — Given ID không tồn tại, When gọi, Then trả 404 với thông báo rõ.

### Từ BR-106 · Khử trùng lặp `[v2]`

#### US-106.1 — Phát hiện & gộp bản trùng
*Là `Data Engineer`, tôi muốn phát hiện ảnh/bài trùng, để index không phình và kết quả không lặp.*

- **Mục tiêu**: Không tạo bản ghi/embedding nhân đôi cho cùng nội dung.
- **Input**: Media mới + kho hiện có.
- **Output**: Trùng → bỏ qua hoặc gộp vào bản ghi gốc; báo cáo số mục đã khử.
- **Edge**: Ảnh gần giống (resize/crop) → dùng perceptual hash ngưỡng cấu hình; nghi ngờ → đưa vào hàng `needs_review`.
- **Test**: Nạp lại file trùng + biến thể resize → không tạo bản ghi mới, log "deduped".

**Acceptance Criteria**
- **AC-1** — Given file trùng checksum, When nạp, Then bị nhận diện trùng và không tạo bản ghi/embedding mới.
- **AC-2** — Given ảnh biến thể gần giống vượt ngưỡng perceptual-hash, When nạp, Then được đánh dấu nghi trùng để review.

### Từ BR-107 · Nạp & cấu trúc văn bản nội quy `[v1]`

#### US-107.1 — Nạp & tách nội quy theo điều/khoản
*Là `Quản trị viên`, tôi muốn nạp văn bản nội quy CLB và tách theo điều/khoản, để làm nguồn tri thức cho hỏi-đáp nội quy.*

- **Mục tiêu**: Nạp file nội quy, tách thành đơn vị điều/khoản/mục có cấu trúc, gắn phiên bản & ngày hiệu lực.
- **Input**: File nội quy (`.md/.docx/.pdf/text`) + metadata (tiêu đề, phiên bản, ngày hiệu lực).
- **Output**: Bản ghi `regulations` + danh sách `rule_chunks` (số điều, số khoản, tiêu đề, nội dung) truy lại theo số điều.
- **Edge**: File không rõ cấu trúc điều/khoản → tách theo heading/đoạn + gắn `needs_review`; nạp phiên bản mới → giữ lịch sử, chỉ bản hiệu lực được phục vụ.
- **Test**: Nạp 1 văn bản mẫu → truy "Điều X" trả đúng nội dung điều đó.

**Acceptance Criteria**
- **AC-1** — Given file nội quy hợp lệ, When nạp, Then sinh các `rule_chunk` gắn số điều/khoản truy lại được.
- **AC-2** — Given nạp phiên bản nội quy mới, When lưu, Then bản cũ được lưu lịch sử và chỉ bản hiệu lực được phục vụ.

### Từ BR-201 · Trích xuất khung hình video `[v1]`

#### US-201.1 — Trích frame đại diện + timestamp từ video
*Là `hệ thống`, tôi muốn tách frame kèm timestamp từ video, để lập chỉ mục và cắt clip về sau.*

- **Mục tiêu**: Mỗi video sinh ≥1 frame đại diện gắn timestamp chính xác.
- **Input**: File video trong kho.
- **Output**: Bản ghi `video_frames` (frame_path, timestamp_sec) cho mỗi keyframe.
- **Edge**: Video hỏng/codec lạ → bỏ qua, ghi log lỗi; video quá dài → lấy mẫu theo bước thời gian cấu hình (giảm tải compute).
- **Test**: Trích 1 video mẫu → số frame ≥1, timestamp tăng dần & nằm trong thời lượng video.

**Acceptance Criteria**
- **AC-1** — Given một video hợp lệ, When chạy trích frame, Then sinh ≥1 frame với timestamp chính xác (sai số ≤ 0.5s).
- **AC-2** — Given video lỗi codec, When xử lý, Then job ghi log lỗi và bỏ qua, không chặn cả batch.

### Từ BR-202 · Sinh embedding đa phương thức `[v1]`

#### US-202.1 — Sinh embedding text & ảnh trong cùng không gian
*Là `hệ thống`, tôi muốn nhúng text và ảnh vào cùng không gian vector, để so khớp chéo text↔ảnh.*

- **Mục tiêu**: Sinh vector cho ảnh, frame, và text (caption/nội dung) bằng multilingual CLIP.
- **Input**: Ảnh/frame + text (caption/OCR/nội dung bài).
- **Output**: Vector + bản ghi `embeddings` (target_type, target_id, model, dim, vector_id).
- **Edge**: Ảnh lỗi/decode fail → bỏ qua + log; model đổi version → đánh dấu cần reindex (tránh trộn không gian).
- **Test**: Query ảnh sự kiện → trả ảnh cùng sự kiện trong top-k; query text → trả ảnh liên quan.

**Acceptance Criteria**
- **AC-1** — Given ảnh và text hợp lệ, When sinh embedding, Then vector cùng số chiều, cùng model, lưu kèm tham chiếu media.
- **AC-2** — Given truy vấn text bằng tiếng Việt, When tìm, Then trả về ảnh liên quan ngữ nghĩa (không chỉ khớp từ khóa).

### Từ BR-203 · Sinh mô tả/caption tự động `[v1]`

#### US-203.1 — Caption tự động cho ảnh/frame thiếu mô tả
*Là `hệ thống`, tôi muốn sinh caption cho ảnh/frame chưa có mô tả, để tăng khả năng tìm & trình bày kết quả.*

- **Mục tiêu**: ≥80% ảnh mẫu có caption tự động hợp lý.
- **Input**: Ảnh/frame không có caption gốc.
- **Output**: Bản ghi `captions` (text, lang, generated_by=auto).
- **Edge**: Caption rỗng/độ tin cậy thấp → gắn cờ `low_confidence`, vẫn cho phép admin sửa tay.
- **Test**: Chạy trên tập mẫu → đánh giá tay tỷ lệ caption hợp lý ≥80%.

**Acceptance Criteria**
- **AC-1** — Given ảnh chưa có caption, When chạy caption, Then sinh caption gắn `generated_by=auto` + ngôn ngữ.
- **AC-2** — Given caption độ tin cậy thấp, When lưu, Then gắn `low_confidence` để review.

### Từ BR-204 · OCR văn bản trong ảnh `[v2]`

#### US-204.1 — Trích text trong poster/banner
*Là `hệ thống`, tôi muốn OCR chữ trong ảnh, để tìm được sự kiện qua chữ trên poster.*

- **Mục tiêu**: Trích text tiếng Việt trong ảnh và đưa vào index tìm kiếm.
- **Input**: Ảnh có chữ (poster, banner).
- **Output**: Bản ghi `ocr_texts` (text, lang) + embedding text bổ sung.
- **Edge**: Ảnh nhiễu/chữ nghệ thuật → độ tin cậy thấp gắn cờ; không có chữ → bỏ qua.
- **Test**: OCR poster mẫu → tìm bằng từ khóa trên poster trả đúng ảnh đó.

**Acceptance Criteria**
- **AC-1** — Given ảnh có chữ tiếng Việt, When OCR, Then text được trích và lập chỉ mục.
- **AC-2** — Given truy vấn chứa cụm từ trên poster, When tìm, Then ảnh poster đó xuất hiện trong top-k.

### Từ BR-205 · Xây vector index `[v1]`

#### US-205.1 — Lập chỉ mục vector cho tìm kiếm ngữ nghĩa
*Là `hệ thống`, tôi muốn lưu embedding vào vector store, để truy vấn top-k nhanh.*

- **Mục tiêu**: Index hỗ trợ ANN search trả top-k dưới ngưỡng độ trễ mục tiêu.
- **Input**: Tập embedding + metadata tham chiếu.
- **Output**: Collection trong vector DB (Qdrant) sẵn sàng truy vấn top-k.
- **Edge**: Index lỗi/khác chiều vector → từ chối nạp + báo; rỗng → query trả "không có dữ liệu".
- **Test**: Nạp N vector → truy vấn top-k trả đúng số kết quả, độ trễ ≤ ngưỡng (đo ở BR-603).

**Acceptance Criteria**
- **AC-1** — Given embedding đã sinh, When build index, Then truy vấn top-k trả kết quả kèm điểm tương đồng.
- **AC-2** — Given vector sai số chiều, When nạp, Then bị từ chối với lỗi rõ.

### Từ BR-206 · Cập nhật index tăng tiến `[v2]`

#### US-206.1 — Upsert dữ liệu mới không rebuild toàn bộ
*Là `Data Engineer`, tôi muốn thêm dữ liệu mới vào index ngay, để không phải dựng lại toàn bộ mỗi lần crawl.*

- **Mục tiêu**: Index hỗ trợ thêm/cập nhật batch mới (upsert) tức thời.
- **Input**: Batch embedding mới.
- **Output**: Index cập nhật; mục mới tìm được ngay mà không rebuild full.
- **Edge**: Trùng vector_id → upsert ghi đè đúng bản; nửa chừng lỗi → rollback batch, không để index nửa vời.
- **Test**: Thêm batch mới → query trả ngay kết quả mới; không cần rebuild.

**Acceptance Criteria**
- **AC-1** — Given index đang chạy, When upsert batch mới, Then dữ liệu mới truy vấn được ngay.
- **AC-2** — Given batch lỗi giữa chừng, When upsert, Then rollback và giữ index nhất quán.

### Từ BR-207 · Lập chỉ mục văn bản nội quy `[v1]`

#### US-207.1 — Chunk + embed nội quy vào collection riêng
*Là `Data/ML Engineer`, tôi muốn embed các điều/khoản nội quy vào một collection vector riêng, để truy vấn ngữ nghĩa text không trộn với index media.*

- **Mục tiêu**: Sinh embedding **text** cho từng `rule_chunk`, lưu collection vector riêng (tách khỏi embedding ảnh/CLIP).
- **Input**: `rule_chunks` (BR-107) + model embedding text.
- **Output**: Vector cho mỗi chunk trong collection nội quy + payload (số điều/khoản, `regulation_id`).
- **Edge**: Chunk quá dài → cắt theo cửa sổ có overlap; đổi model embedding → re-embed toàn collection (đồng nhất `model`+`dim`).
- **Test**: Truy vấn text mẫu → top-k chunk trả về đúng điều/khoản liên quan.

**Acceptance Criteria**
- **AC-1** — Given `rule_chunks` đã nạp, When chạy index nội quy, Then mỗi chunk có vector trong collection riêng kèm payload điều/khoản.
- **AC-2** — Given một truy vấn text, When tìm trên collection nội quy, Then trả chunk liên quan theo điểm tương đồng giảm dần.

### Từ BR-208 · Tách audio & sinh transcript (ASR) `[v1]`

#### US-208.1 — Tách audio video → transcript theo timestamp
*Là `Data/ML Engineer`, tôi muốn tách audio mỗi video rồi chạy ASR sinh transcript cắt theo timestamp, để tìm được video theo nội dung được nói.*

- **Mục tiêu**: Mỗi video tách 2 nhánh — keyframe (US-201.1) và **audio → ASR → transcript** cắt theo đoạn/timestamp; embed transcript (text) để truy xuất.
- **Input**: File video trong kho (BR-102/105).
- **Output**: File `transcript/video-<n>/transcript.json` (đoạn + start/end time + text) + bản ghi `transcripts` (Postgres) + embedding text cho từng đoạn (collection transcript).
- **Edge**: Video không có tiếng/chỉ nhạc nền → transcript rỗng, bỏ qua không lỗi; nhiều người nói/tạp âm → vẫn lưu best-effort, đánh dấu độ tin cậy thấp; ngôn ngữ lạ → ghi log, không chặn batch.
- **Test**: Chạy 1 video mẫu có lời nói → ≥1 đoạn transcript, timestamp tăng dần & trong thời lượng video; embed được vào collection transcript.

**Acceptance Criteria**
- **AC-1** — Given một video có giọng nói, When chạy tách audio + ASR, Then sinh ≥1 đoạn transcript kèm start/end timestamp và embedding text trong collection transcript.
- **AC-2** — Given video không có lời nói, When xử lý, Then transcript rỗng được ghi nhận, job không lỗi và không chặn batch.

### Từ BR-301 · Truy vấn bằng văn bản `[v1]`

#### US-301.1 — Hỏi bằng câu chữ tiếng Việt
*Là `Thành viên CLB`, tôi muốn gõ câu hỏi tự nhiên, để tìm ảnh/bài liên quan đúng chủ đề.*

- **Mục tiêu**: Truy xuất media + nội dung liên quan từ truy vấn text.
- **Input**: Câu hỏi text (tiếng Việt), tham số top-k.
- **Output**: Danh sách kết quả top-k (media + điểm + metadata).
- **Edge**: Truy vấn quá ngắn/mơ hồ → vẫn trả top-k tốt nhất + gợi ý làm rõ; ngoài miền → "không có dữ liệu" (BR-406/704).
- **Test**: Bộ câu hỏi mẫu → kết quả đúng chủ đề nằm trong top-k (đo Recall@k).

**Acceptance Criteria**
- **AC-1** — Given câu hỏi text trong miền, When gửi, Then trả top-k media liên quan kèm điểm tương đồng.
- **AC-2** — Given câu hỏi ngoài miền dữ liệu, When gửi, Then trả thông báo không có dữ liệu thay vì kết quả ngẫu nhiên.

### Từ BR-302 · Nhận diện sự kiện từ ảnh `[v1]`

#### US-302.1 — "Đây là sự kiện nào?"
*Là `Sinh viên mới / khách`, tôi muốn upload một ảnh và hỏi đây là sự kiện gì, để biết bối cảnh.*

- **Mục tiêu**: Từ ảnh truy vấn, suy ra sự kiện liên quan nhất.
- **Input**: Ảnh upload.
- **Output**: Tên/sự kiện ứng viên (top-k) + ảnh/bài cùng sự kiện + độ tin cậy.
- **Edge**: Ảnh không thuộc kho/không có sự kiện khớp → trả "không xác định được sự kiện" + ảnh tương tự gần nhất.
- **Test**: Ảnh sự kiện mẫu → sự kiện đúng nằm trong top-k.

**Acceptance Criteria**
- **AC-1** — Given ảnh thuộc một sự kiện trong kho, When hỏi "đây là sự kiện nào", Then sự kiện đúng xuất hiện trong top-k với độ tin cậy.
- **AC-2** — Given ảnh không khớp sự kiện nào, When hỏi, Then trả "không xác định" kèm ảnh tương tự nhất.

### Từ BR-303 · Tìm ảnh tương tự `[v1]`

#### US-303.1 — Tìm ảnh liên quan từ một ảnh
*Là `Cựu thành viên`, tôi muốn từ một ảnh tìm các ảnh liên quan, để xem thêm khoảnh khắc cùng dịp.*

- **Mục tiêu**: Trả về ảnh tương tự ngữ nghĩa/cùng sự kiện từ ảnh truy vấn.
- **Input**: Ảnh upload + top-k.
- **Output**: Danh sách ảnh tương tự xếp theo độ tương đồng.
- **Edge**: Kho rỗng/không đủ ảnh tương tự → trả ít hơn k + thông báo; ảnh decode fail → lỗi rõ.
- **Test**: Ảnh truy vấn → ≥k ảnh cùng sự kiện/chủ đề trong kết quả.

**Acceptance Criteria**
- **AC-1** — Given ảnh hợp lệ, When tìm ảnh tương tự, Then trả danh sách xếp theo độ tương đồng giảm dần.
- **AC-2** — Given không đủ k ảnh tương tự, When tìm, Then trả số có được và nêu rõ.

### Từ BR-304 · Tìm hoạt động theo người (face) `[v2]`

#### US-304.1 — Upload ảnh một người → hoạt động họ tham gia
*Là `Cựu thành viên`, tôi muốn upload ảnh của mình để tìm các sự kiện tôi từng tham gia, có kiểm soát quyền riêng tư.*

- **Mục tiêu**: Khớp khuôn mặt truy vấn với ảnh/sự kiện trong kho (chỉ với người đã đồng thuận).
- **Input**: Ảnh chân dung + (tùy chọn) định danh.
- **Output**: Danh sách sự kiện/ảnh người đó xuất hiện, loại trừ người opt-out.
- **Edge**: Người opt-out → ẩn khỏi kết quả (BR-702); nhiều mặt trong ảnh → hỏi chọn khuôn mặt; không khớp → "không tìm thấy".
- **Test**: Ảnh người mẫu (đã opt-in) → trả đúng sự kiện họ xuất hiện; người opt-out → không xuất hiện.

**Acceptance Criteria**
- **AC-1** — Given ảnh người đã opt-in, When tìm theo người, Then trả các sự kiện/ảnh họ xuất hiện.
- **AC-2** — Given người đã opt-out, When họ xuất hiện trong kho, Then không bị trả về trong bất kỳ kết quả face nào.

### Từ BR-305 · Lọc theo thời gian/sự kiện `[v2]`

#### US-305.1 — Lọc kết quả theo năm/sự kiện
*Là `Ban Chủ nhiệm / Admin`, tôi muốn lọc kết quả theo khoảng thời gian hoặc tên sự kiện, để tổng hợp tư liệu nhanh.*

- **Mục tiêu**: Áp bộ lọc metadata lên kết quả truy xuất.
- **Input**: Truy vấn + filter (khoảng ngày, event_id/tên).
- **Output**: Kết quả chỉ trong phạm vi lọc.
- **Edge**: Filter không có dữ liệu → trả rỗng + thông báo; filter sai định dạng ngày → lỗi xác thực.
- **Test**: Lọc "năm 2023" → mọi kết quả có `posted_at` trong 2023.

**Acceptance Criteria**
- **AC-1** — Given truy vấn kèm filter thời gian, When tìm, Then mọi kết quả nằm trong khoảng đó.
- **AC-2** — Given filter sự kiện cụ thể, When tìm, Then chỉ trả media thuộc sự kiện đó.

### Từ BR-306 · Xếp hạng độ liên quan `[v1]`

#### US-306.1 — Xếp hạng kết quả theo độ liên quan
*Là `Thành viên CLB`, tôi muốn kết quả liên quan nhất đứng đầu, để không phải lọc thủ công.*

- **Mục tiêu**: Sắp xếp kết quả theo điểm tương đồng (tùy chọn rerank).
- **Input**: Tập ứng viên top-k thô.
- **Output**: Danh sách đã xếp hạng giảm dần theo độ liên quan.
- **Edge**: Điểm bằng nhau → tie-break theo độ mới/độ rõ metadata; tất cả điểm dưới ngưỡng → coi như "không tìm thấy".
- **Test**: Trên bộ đánh giá → top-1 liên quan hơn hạng thấp (đo MRR).

**Acceptance Criteria**
- **AC-1** — Given tập ứng viên, When xếp hạng, Then thứ tự giảm dần theo điểm liên quan.
- **AC-2** — Given mọi ứng viên dưới ngưỡng liên quan, When xếp hạng, Then trả "không tìm thấy" thay vì kết quả yếu.

### Từ BR-307 · Truy vấn nội quy `[v1]`

#### US-307.1 — Tra cứu điều/khoản liên quan câu hỏi
*Là `thành viên CLB`, tôi muốn hỏi về nội quy và nhận đúng điều/khoản liên quan, để biết quy định.*

- **Mục tiêu**: Từ câu hỏi text (tra cứu/tình huống), truy xuất top-k điều/khoản nội quy liên quan nhất làm ngữ cảnh.
- **Input**: Câu hỏi text + top-k.
- **Output**: Danh sách `rule_chunk` liên quan (số điều/khoản + nội dung + điểm).
- **Edge**: Không khớp điều khoản nào trên ngưỡng → coi như "nội quy không quy định" (chuyển BR-407); câu hỏi mơ hồ → vẫn trả ứng viên gần nhất.
- **Test**: Bộ câu hỏi nội quy mẫu → điều/khoản đúng nằm trong top-k (đo ở BR-606).

**Acceptance Criteria**
- **AC-1** — Given câu hỏi về nội quy, When truy vấn, Then trả các điều/khoản liên quan kèm điểm.
- **AC-2** — Given không điều khoản nào trên ngưỡng, When truy vấn, Then báo "không có điều khoản phù hợp" thay vì trả khoản không liên quan.

### Từ BR-308 · Truy xuất theo lời nói trong video (transcript) `[v1]`

#### US-308.1 — Tìm video theo nội dung được nói
*Là `thành viên CLB`, tôi muốn hỏi về điều ai đó **nói** trong video, để tìm đúng video và đoạn chứa nội dung đó.*

- **Mục tiêu**: Truy vấn text khớp transcript; **gộp với kết quả keyframe** theo `video_id` + cửa sổ timestamp để trả về đúng video + đoạn + frame trong khoảng nói.
- **Input**: Câu hỏi text (tiếng Việt) + top-k.
- **Output**: Kết quả video kèm đoạn transcript khớp (start/end time), keyframe trong khoảng đó, điểm tương đồng — sẵn cho clip/preview (BR-403/404).
- **Edge**: Khớp transcript nhưng điểm thấp → vẫn xếp dưới keyframe khớp mạnh; transcript trùng nhiều đoạn 1 video → gộp về đoạn đại diện; chưa có transcript (video câm) → chỉ dùng nhánh keyframe.
- **Test**: Câu hỏi về nội dung nói trong video mẫu → đúng video + đoạn (timestamp) nằm trong top-k.

**Acceptance Criteria**
- **AC-1** — Given câu hỏi khớp nội dung nói, When truy vấn, Then trả đúng video + đoạn transcript (timestamp) trong top-k, gộp với keyframe cùng `video_id`.
- **AC-2** — Given cùng nội dung xuất hiện ở cả transcript và keyframe, When xếp hạng, Then kết quả của một video không bị nhân đôi mà gộp theo `video_id` + timestamp.

### Từ BR-401 · Sinh câu trả lời RAG `[v1]`

#### US-401.1 — Tổng hợp câu trả lời bám nguồn
*Là `người dùng`, tôi muốn nhận câu trả lời ngôn ngữ tự nhiên dựa trên media truy xuất, để hiểu nhanh thay vì tự đọc hết.*

- **Mục tiêu**: LLM (API free-tier) sinh câu trả lời dựa trên ngữ cảnh truy xuất, không bịa.
- **Input**: Câu hỏi + top-k media/caption/nội dung làm ngữ cảnh.
- **Output**: Câu trả lời text + tham chiếu tới media nguồn.
- **Edge**: Ngữ cảnh rỗng → không gọi sinh tự do, trả "không tìm thấy" (BR-406); LLM API lỗi/timeout → fallback trả danh sách kết quả thô + thông báo.
- **Test**: Đánh giá tay: câu trả lời bám nội dung nguồn, không thêm sự kiện không có trong ngữ cảnh.

**Acceptance Criteria**
- **AC-1** — Given có ngữ cảnh truy xuất, When sinh câu trả lời, Then câu trả lời chỉ dựa trên ngữ cảnh đó và kèm tham chiếu nguồn.
- **AC-2** — Given LLM API lỗi, When sinh, Then hệ thống fallback hiển thị kết quả truy xuất thô + báo lỗi nhẹ, không sập.

### Từ BR-402 · Trả ảnh kèm mô tả `[v1]`

#### US-402.1 — Hiển thị ảnh + mô tả
*Là `người dùng`, tôi muốn mỗi ảnh kết quả có mô tả đi kèm, để hiểu ngữ cảnh ảnh.*

- **Mục tiêu**: Mỗi kết quả ảnh hiển thị kèm caption/mô tả.
- **Input**: Ảnh kết quả + caption (gốc hoặc auto).
- **Output**: Card kết quả: ảnh + mô tả + (nếu có) tên sự kiện/ngày.
- **Edge**: Không có caption nào → hiển thị nhãn "chưa có mô tả" thay vì trống.
- **Test**: Mỗi ảnh trong kết quả render kèm phần mô tả.

**Acceptance Criteria**
- **AC-1** — Given ảnh kết quả có caption, When render, Then ảnh hiển thị kèm mô tả.
- **AC-2** — Given ảnh không có caption, When render, Then hiển thị nhãn "chưa có mô tả".

### Từ BR-403 · Trả clip video ngắn `[v1]`

#### US-403.1 — Clip 3s trước + 3s sau frame khớp
*Là `người dùng`, tôi muốn xem đoạn ~6s quanh frame video khớp, để thấy bối cảnh động thay vì ảnh tĩnh.*

- **Mục tiêu**: Khi kết quả là frame video, cắt clip 3s trước + 3s sau timestamp.
- **Input**: `media_asset` (video) + `timestamp_sec` của frame khớp.
- **Output**: Clip ~6s (cắt bằng ffmpeg) phát được trong kết quả.
- **Edge**: Frame gần đầu/cuối video → cắt theo biên (clip ngắn hơn 6s); cắt lỗi → fallback sang preview tại timestamp (BR-404).
- **Test**: Kết quả frame video → clip phát đúng đoạn, độ dài ≤6s và chứa timestamp.

**Acceptance Criteria**
- **AC-1** — Given kết quả là frame video ở giữa video, When yêu cầu clip, Then trả clip 3s trước + 3s sau, phát được.
- **AC-2** — Given frame ở sát đầu/cuối, When cắt clip, Then clip co theo biên video, không lỗi.

### Từ BR-404 · Preview video tới timestamp `[v1]`

#### US-404.1 — Tua video tới đúng thời điểm chứa frame
*Là `người dùng`, tôi muốn mở video và nhảy tới đúng giây chứa frame, để xem nguyên video từ đó.*

- **Mục tiêu**: Mở player video và seek tới `timestamp_sec`.
- **Input**: `media_asset` video + timestamp.
- **Output**: Player mở đúng tại timestamp (qua `#t=` / range request streaming).
- **Edge**: Trình duyệt không seek được (codec) → tải đoạn quanh timestamp; file mất → lỗi rõ.
- **Test**: Click kết quả → player mở đúng tại timestamp (±1s).

**Acceptance Criteria**
- **AC-1** — Given kết quả gắn timestamp, When click "xem trong video", Then player mở đúng tại thời điểm đó (±1s).
- **AC-2** — Given video không hỗ trợ seek trực tiếp, When mở, Then hệ thống tải đoạn quanh timestamp để phát.

### Từ BR-405 · Trích dẫn nguồn `[v1]`

#### US-405.1 — Gắn link bài gốc cho câu trả lời
*Là `người dùng`, tôi muốn mỗi câu trả lời có link về bài gốc, để kiểm chứng & xem thêm.*

- **Mục tiêu**: Đính kèm `source_url` (bài fanpage) vào câu trả lời/kết quả khi có.
- **Input**: Media/bài nguồn của kết quả.
- **Output**: Trích dẫn dạng link tới bài gốc.
- **Edge**: Bài không còn link/đã xóa nguồn → hiển thị "nguồn nội bộ" thay vì link gãy.
- **Test**: Kết quả có `source_url` → render link click được về bài gốc.

**Acceptance Criteria**
- **AC-1** — Given kết quả có `source_url`, When render, Then hiển thị link tới bài gốc.
- **AC-2** — Given không có `source_url`, When render, Then hiển thị nhãn "nguồn nội bộ", không link gãy.

### Từ BR-406 · Xử lý "không tìm thấy" `[v1]`

#### US-406.1 — Báo rõ khi không đủ dữ liệu liên quan
*Là `người dùng`, tôi muốn biết khi không có kết quả phù hợp, để không bị trả lời bịa.*

- **Mục tiêu**: Khi không đủ kết quả trên ngưỡng liên quan, báo rõ thay vì sinh tự do.
- **Input**: Kết quả truy xuất + điểm liên quan.
- **Output**: Thông báo "không tìm thấy dữ liệu phù hợp" + gợi ý chỉnh truy vấn.
- **Edge**: Truy vấn ngoài miền (BR-704) → cùng thông báo; kết quả yếu sát ngưỡng → trả kèm cảnh báo độ tin cậy thấp.
- **Test**: Truy vấn ngoài miền → thông báo không có dữ liệu, không có câu trả lời bịa.

**Acceptance Criteria**
- **AC-1** — Given mọi kết quả dưới ngưỡng liên quan, When trả lời, Then hiển thị "không tìm thấy" + gợi ý, không gọi LLM sinh tự do.
- **AC-2** — Given truy vấn ngoài miền dữ liệu, When gửi, Then nhận thông báo từ chối lịch sự.

### Từ BR-407 · Trả lời nội quy & tình huống `[v1]`

#### US-407.1 — Trả lời có dẫn điều khoản + disclaimer
*Là `người dùng`, tôi muốn câu trả lời nội quy/tình huống bám điều khoản và nói rõ căn cứ, để tin và kiểm chứng được.*

- **Mục tiêu**: LLM tổng hợp câu trả lời dựa trên điều/khoản truy xuất, **áp dụng nhẹ** vào tình huống; luôn trích dẫn điều/khoản + disclaimer.
- **Input**: Câu hỏi + top-k `rule_chunk` làm ngữ cảnh.
- **Output**: Câu trả lời text + trích dẫn "Điều X, Khoản Y" + disclaimer "thông tin tham khảo, BCN quyết định cuối".
- **Edge**: Ngữ cảnh rỗng/không quy định → trả "nội quy không quy định nội dung này", không bịa; tình huống cần nhiều điều → liệt kê các điều căn cứ, **không suy diễn vượt văn bản**.
- **Test**: Câu hỏi tình huống mẫu → câu trả lời neo đúng điều/khoản + có trích dẫn & disclaimer.

**Acceptance Criteria**
- **AC-1** — Given có điều/khoản liên quan, When trả lời, Then câu trả lời chỉ dựa trên các điều/khoản đó, kèm trích dẫn + disclaimer.
- **AC-2** — Given nội quy không quy định, When hỏi, Then báo rõ "không quy định", không sinh câu trả lời tự do.

### Từ BR-501 · Giao diện chatbot hội thoại `[v1]`

#### US-501.1 — Hỏi-đáp trong khung chat
*Là `người dùng`, tôi muốn tương tác qua khung chat, để hỏi tự nhiên như nhắn tin.*

- **Mục tiêu**: Khung chat gửi câu hỏi và nhận câu trả lời theo lượt.
- **Input**: Tin nhắn text của người dùng.
- **Output**: Bong bóng câu trả lời + kết quả media trong luồng chat; lưu `messages`.
- **Edge**: Mất mạng/timeout → hiển thị trạng thái lỗi + nút gửi lại; tin rỗng → chặn gửi.
- **Test**: Gửi câu hỏi → nhận trả lời hiển thị trong khung chat.

**Acceptance Criteria**
- **AC-1** — Given đã đăng nhập, When gửi tin nhắn text, Then nhận câu trả lời trong cùng luồng chat.
- **AC-2** — Given request lỗi/timeout, When gửi, Then hiển thị lỗi + cho gửi lại, không mất nội dung đã gõ.

### Từ BR-502 · Upload ảnh trong hội thoại `[v1]`

#### US-502.1 — Đính kèm ảnh ngay trong chat
*Là `người dùng`, tôi muốn đính kèm ảnh trong khung chat, để truy vấn bằng ảnh không cần rời màn hình.*

- **Mục tiêu**: Cho phép gửi ảnh kèm/không kèm text trong chat để truy vấn.
- **Input**: File ảnh (+ text tùy chọn).
- **Output**: Tin nhắn người dùng có thumbnail ảnh + kết quả truy vấn ảnh trả về.
- **Edge**: File quá lớn/sai định dạng → chặn + thông báo; ảnh + text mâu thuẫn → ưu tiên xử lý theo cấu hình (mặc định ưu tiên ảnh).
- **Test**: Upload ảnh trong chat → nhận kết quả truy vấn ảnh (sự kiện/ảnh tương tự).

**Acceptance Criteria**
- **AC-1** — Given ảnh hợp lệ, When đính kèm và gửi, Then hệ thống chạy truy vấn ảnh và trả kết quả trong chat.
- **AC-2** — Given file quá lớn/sai định dạng, When gửi, Then bị chặn với thông báo rõ.

### Từ BR-503 · Hiển thị kết quả đa phương thức `[v1]`

#### US-503.1 — Render ảnh/clip/link/mô tả inline
*Là `người dùng`, tôi muốn xem ảnh, clip, link, mô tả ngay trong chat, để không phải mở nhiều tab.*

- **Mục tiêu**: Hiển thị kết quả đa phương thức inline trong bong bóng trả lời.
- **Input**: Kết quả truy xuất (ảnh + caption + clip/timestamp + source_url).
- **Output**: Card kết quả: ảnh, player clip, mô tả, link nguồn.
- **Edge**: Media tải lỗi → placeholder + nút thử lại; nhiều kết quả → phân trang/cuộn ngang.
- **Test**: Kết quả gồm ảnh + clip → cả hai render & phát được inline.

**Acceptance Criteria**
- **AC-1** — Given kết quả gồm ảnh và clip, When render, Then cả hai hiển thị/phát được trong khung chat.
- **AC-2** — Given một media tải lỗi, When render, Then hiển thị placeholder + thử lại, không vỡ layout.

### Từ BR-504 · Ngữ cảnh hội thoại `[v2]`

#### US-504.1 — Hỏi nối tiếp theo ngữ cảnh
*Là `người dùng`, tôi muốn hỏi tiếp dựa trên lượt trước, để không phải lặp lại bối cảnh.*

- **Mục tiêu**: Hiểu câu hỏi tham chiếu lượt trước trong cùng hội thoại.
- **Input**: Tin nhắn mới + lịch sử hội thoại gần.
- **Output**: Câu trả lời có tính đến ngữ cảnh trước (vd "còn ảnh nào khác không").
- **Edge**: Ngữ cảnh quá dài → cắt cửa sổ gần nhất; tham chiếu mơ hồ → hỏi làm rõ.
- **Test**: Hỏi A rồi "còn nữa không" → trả thêm kết quả cùng chủ đề A.

**Acceptance Criteria**
- **AC-1** — Given một lượt hỏi trước về chủ đề X, When hỏi nối tiếp không nêu lại X, Then câu trả lời vẫn theo chủ đề X.
- **AC-2** — Given tham chiếu mơ hồ, When không suy được, Then hệ thống hỏi làm rõ thay vì đoán.

### Từ BR-505 · Web app truy cập đa nhóm `[v1]`

#### US-505.1 — Web app chạy trên trình duyệt phổ thông
*Là `khách/thành viên/alumni/admin`, tôi muốn truy cập qua một URL trên trình duyệt, để dùng không cần cài đặt.*

- **Mục tiêu**: Một web app đáp ứng (responsive) cho mọi nhóm người dùng.
- **Input**: URL ứng dụng.
- **Output**: Giao diện chat tải được trên Chrome/Firefox/Safari, desktop & mobile.
- **Edge**: Trình duyệt cũ không hỗ trợ → thông báo nâng cấp; màn hình nhỏ → layout co lại dùng được.
- **Test**: Mở URL trên 2–3 trình duyệt + mobile → giao diện chat dùng được.

**Acceptance Criteria**
- **AC-1** — Given một URL, When mở trên trình duyệt phổ thông, Then web app chat tải và dùng được.
- **AC-2** — Given màn hình mobile, When mở, Then layout responsive, thao tác chat được.

### Từ BR-507 · Bộ định tuyến đa nguồn (Router RAG) `[v1]`

#### US-507.1 — Tự định tuyến câu hỏi tới nguồn phù hợp
*Là `người dùng`, tôi muốn hỏi tự nhiên trong một khung chat và hệ tự biết tra media hay nội quy, để không phải chọn chế độ thủ công.*

- **Mục tiêu**: Một **router** phân loại ý định mỗi câu hỏi → chọn công cụ (truy vấn media / truy vấn nội quy / cả hai); ưu tiên luật khi tín hiệu rõ (có ảnh / từ khóa nội quy), fallback classifier khi mơ hồ. *(Không thực hiện phân rã/multi-hop ở v1 — đó là Agentic RAG v2.)*
- **Input**: Tin nhắn người dùng (text ± ảnh) + lịch sử hội thoại gần.
- **Output**: Câu trả lời từ (các) nguồn được định tuyến + nhãn nguồn đã dùng; có ảnh → ưu tiên tool media.
- **Edge**: Ý định mơ hồ giữa hai nguồn → hỏi làm rõ hoặc chạy cả hai rồi gộp; định tuyến sai → cho người dùng **ép chế độ thủ công** (override).
- **Test**: Câu hỏi nội quy → gọi tool nội quy; câu hỏi tư liệu/ảnh → gọi tool media; đo routing accuracy (BR-606).

**Acceptance Criteria**
- **AC-1** — Given câu hỏi về nội quy, When gửi, Then router định tuyến tới tool nội quy và trả lời từ điều/khoản.
- **AC-2** — Given câu hỏi về tư liệu/sự kiện (hoặc kèm ảnh), When gửi, Then router định tuyến tới tool media và trả ảnh/clip liên quan.

### Từ BR-601 · Bộ truy vấn đánh giá `[v1]`

#### US-601.1 — Xây tập đánh giá có nhãn
*Là `Data/ML Engineer`, tôi muốn một tập truy vấn mẫu có đáp án đúng, để đo chất lượng khách quan.*

- **Mục tiêu**: Tạo bộ `eval_queries` (text & ảnh) kèm media kỳ vọng.
- **Input**: Câu hỏi/ảnh mẫu + nhãn media đúng.
- **Output**: Bảng `eval_queries` ≥ N mục dùng để chạy đánh giá.
- **Edge**: Nhãn mơ hồ/nhiều đáp án đúng → cho phép nhiều `expected_ids`; mục thiếu nhãn → loại khỏi tính metric.
- **Test**: Tạo tập ≥N mục → chạy đánh giá đọc được tập này.

**Acceptance Criteria**
- **AC-1** — Given các truy vấn mẫu, When lưu, Then mỗi mục có ≥1 media kỳ vọng và loại truy vấn (text/ảnh).
- **AC-2** — Given mục thiếu nhãn, When chạy metric, Then mục đó bị loại, có cảnh báo.

### Từ BR-602 · Đo độ chính xác truy xuất `[v1]`

#### US-602.1 — Báo cáo Recall@k & MRR
*Là `Data/ML Engineer`, tôi muốn đo Recall@k và MRR trên tập đánh giá, để biết hệ thống đạt mục tiêu chưa.*

- **Mục tiêu**: Chạy đánh giá tự động và xuất Recall@k, MRR.
- **Input**: `eval_queries` + hệ thống truy xuất hiện tại.
- **Output**: Báo cáo số liệu Recall@5, MRR (+ so với mục tiêu 0.80 / 0.60).
- **Edge**: Tập rỗng → báo lỗi cấu hình; kết quả dưới mục tiêu → đánh dấu đỏ trong báo cáo.
- **Test**: Chạy `POST /eval/run` → trả Recall@k, MRR khớp tính tay trên tập nhỏ.

**Acceptance Criteria**
- **AC-1** — Given tập đánh giá, When chạy đánh giá, Then xuất Recall@k và MRR.
- **AC-2** — Given chỉ số dưới mục tiêu, When xem báo cáo, Then mục đó được đánh dấu chưa đạt.

### Từ BR-603 · Đo độ trễ `[v1]`

#### US-603.1 — Đo thời gian phản hồi
*Là `Data/ML Engineer`, tôi muốn đo độ trễ trung bình/percentile, để kiểm chứng mục tiêu ≤5s.*

- **Mục tiêu**: Ghi & báo cáo latency truy vấn (avg, p95).
- **Input**: Log thời gian xử lý mỗi truy vấn.
- **Output**: Báo cáo latency trung bình + p95 (so ngưỡng ≤5s).
- **Edge**: Truy vấn lỗi không tính vào latency thành công; outlier → báo riêng.
- **Test**: Chạy tập truy vấn → latency trung bình ≤ ngưỡng mục tiêu, có số p95.

**Acceptance Criteria**
- **AC-1** — Given các truy vấn đã chạy, When xem báo cáo, Then có latency trung bình và p95.
- **AC-2** — Given latency vượt ngưỡng, When xem báo cáo, Then được đánh dấu cảnh báo.

### Từ BR-604 · Đánh giá chất lượng câu trả lời `[v1]`

#### US-604.1 — Chấm tay chất lượng & độ bám nguồn
*Là `Data/ML Engineer`, tôi muốn chấm tay câu trả lời theo tiêu chí đúng/bám nguồn, để kiểm soát hallucination.*

- **Mục tiêu**: Quy trình chấm tay tỷ lệ câu trả lời đạt (đúng + bám nguồn) ≥ ngưỡng (vd 80%).
- **Input**: Mẫu câu trả lời + ngữ cảnh nguồn.
- **Output**: Bảng chấm (đạt/không + lý do) + tỷ lệ tổng.
- **Edge**: Bất đồng giữa người chấm → chấm đôi & lấy đồng thuận; câu trả lời "không tìm thấy" đúng lúc → tính là đạt.
- **Test**: Chấm tập mẫu → tỷ lệ đạt tính được & ghi nhận.

**Acceptance Criteria**
- **AC-1** — Given mẫu câu trả lời + nguồn, When chấm, Then mỗi câu được gán đạt/không kèm lý do.
- **AC-2** — Given hoàn tất chấm, When tổng hợp, Then xuất tỷ lệ đạt so với ngưỡng mục tiêu.

### Từ BR-605 · Độ phủ demo end-to-end `[v1]`

#### US-605.1 — Kịch bản demo phủ mọi luồng lõi
*Là `nhóm capstone`, tôi muốn một kịch bản demo chạy trọn mọi luồng lõi, để chứng minh end-to-end khi bảo vệ.*

- **Mục tiêu**: Demo phủ: text→ảnh+mô tả; ảnh→sự kiện; ảnh→ảnh tương tự; kết quả frame video→clip/preview.
- **Input**: Tập dữ liệu mẫu + danh mục kịch bản demo.
- **Output**: Checklist demo có trạng thái pass cho từng luồng.
- **Edge**: Một luồng fail → đánh dấu rõ luồng nào, không che bằng luồng khác.
- **Test**: Chạy checklist → tất cả luồng lõi pass end-to-end.

**Acceptance Criteria**
- **AC-1** — Given dữ liệu mẫu, When chạy kịch bản demo, Then mỗi luồng lõi có kết quả pass quan sát được.
- **AC-2** — Given một luồng fail, When chạy, Then checklist chỉ rõ luồng fail.

### Từ BR-606 · Đánh giá hỏi-đáp nội quy `[v1]`

#### US-606.1 — Đo độ đúng điều khoản, groundedness & định tuyến
*Là `Data/ML Engineer`, tôi muốn bộ đánh giá riêng cho luồng nội quy, để đo khách quan chất lượng tra cứu, trả lời và định tuyến.*

- **Mục tiêu**: Tập câu hỏi nội quy có nhãn điều/khoản đúng + nhãn nguồn kỳ vọng; đo accuracy điều khoản, groundedness, routing accuracy.
- **Input**: Bộ câu hỏi nội quy (tra cứu + tình huống) + nhãn.
- **Output**: Báo cáo: % điều khoản đúng trong top-k, % câu trả lời bám đúng điều khoản, % định tuyến đúng.
- **Edge**: Câu "không quy định" cũng là nhãn hợp lệ (đo việc từ chối đúng); tình huống nhiều điều → chấp nhận tập điều khoản.
- **Test**: Chạy đánh giá → xuất 3 chỉ số trên bộ câu hỏi nội quy.

**Acceptance Criteria**
- **AC-1** — Given bộ câu hỏi nội quy có nhãn, When chạy đánh giá, Then xuất accuracy điều khoản + groundedness + routing accuracy.
- **AC-2** — Given câu hỏi "không quy định", When đánh giá, Then tính đúng trường hợp hệ thống từ chối hợp lệ.

### Từ BR-701 · Đồng thuận sử dụng dữ liệu `[v1]`

#### US-701.1 — Ràng buộc phạm vi sử dụng theo consent
*Là `Quản trị viên`, tôi muốn hệ thống chỉ dùng dữ liệu trong phạm vi được CLB cho phép, để tuân thủ thỏa thuận.*

- **Mục tiêu**: Mọi truy vấn/thu thập chỉ trên dữ liệu thuộc phạm vi consent đã lưu (liên kết US-101.1).
- **Input**: Bản ghi consent (phạm vi, thời hạn).
- **Output**: Hệ thống gắn & kiểm tra phạm vi consent cho dữ liệu được phục vụ.
- **Edge**: Consent hết hạn → khóa phục vụ dữ liệu liên quan + cảnh báo admin.
- **Test**: Vô hiệu consent → dữ liệu thuộc phạm vi đó không còn được truy vấn.

**Acceptance Criteria**
- **AC-1** — Given consent còn hiệu lực, When phục vụ truy vấn, Then chỉ dùng dữ liệu trong phạm vi consent.
- **AC-2** — Given consent hết hạn/bị thu hồi, When truy vấn, Then dữ liệu liên quan bị loại + admin được cảnh báo.

### Từ BR-704 · Giới hạn phạm vi miền dữ liệu `[v1]`

#### US-704.1 — Chỉ phục vụ miền fanpage HIT (5 năm)
*Là `người dùng`, tôi muốn hệ thống chỉ trả lời trong phạm vi dữ liệu CLB, để câu trả lời đáng tin và không lạc đề.*

- **Mục tiêu**: Từ chối/định hướng lại truy vấn ngoài miền dữ liệu.
- **Input**: Truy vấn của người dùng.
- **Output**: Truy vấn ngoài miền → thông báo phạm vi + gợi ý hỏi trong phạm vi.
- **Edge**: Truy vấn nửa trong nửa ngoài → trả phần trong miền + nêu giới hạn.
- **Test**: Hỏi chủ đề ngoài CLB → thông báo ngoài phạm vi, không bịa.

**Acceptance Criteria**
- **AC-1** — Given truy vấn ngoài miền, When gửi, Then nhận thông báo giới hạn phạm vi, không có câu trả lời bịa.
- **AC-2** — Given truy vấn trong miền, When gửi, Then xử lý bình thường.

### Từ BR-702 · Bảo vệ dữ liệu cá nhân `[v2]`

#### US-702.1 — Cơ chế opt-out cho nhận diện khuôn mặt
*Là `cá nhân xuất hiện trong ảnh`, tôi muốn chọn không bị nhận diện, để bảo vệ quyền riêng tư.*

- **Mục tiêu**: Người opt-out không bị đưa vào/không xuất hiện trong kết quả face search.
- **Input**: Yêu cầu opt-out (định danh người).
- **Output**: Trạng thái `consent_status=opt_out`; loại khỏi index khuôn mặt & kết quả.
- **Edge**: Opt-out sau khi đã index → xóa face embedding tương ứng; ảnh nhóm có cả người opt-in & opt-out → vẫn trả nhưng làm mờ/loại người opt-out.
- **Test**: Đặt 1 người opt-out → face search không trả người đó dù họ có trong kho.

**Acceptance Criteria**
- **AC-1** — Given một người đã opt-out, When chạy face search, Then người đó không xuất hiện trong kết quả.
- **AC-2** — Given opt-out sau khi đã index, When cập nhật, Then face embedding của họ bị gỡ.

### Từ BR-703 · Gỡ bỏ theo yêu cầu `[v2]`

#### US-703.1 — Gỡ media/dữ liệu cá nhân theo yêu cầu
*Là `cá nhân/Ban CN`, tôi muốn yêu cầu gỡ một ảnh/dữ liệu, để hệ thống tôn trọng quyền gỡ bỏ.*

- **Mục tiêu**: Yêu cầu gỡ → media bị loại khỏi index & kết quả.
- **Input**: Yêu cầu gỡ (media_id hoặc người), lý do.
- **Output**: Bản ghi `removal_requests`; media bị ẩn/loại khỏi index, không còn trả về.
- **Edge**: Media đã embed nhiều nơi (ảnh + frame + caption) → gỡ đồng bộ mọi tham chiếu; yêu cầu trùng → idempotent.
- **Test**: Gửi yêu cầu gỡ 1 media → sau xử lý, truy vấn không còn trả media đó.

**Acceptance Criteria**
- **AC-1** — Given một yêu cầu gỡ hợp lệ, When xử lý, Then media bị loại khỏi index và mọi kết quả.
- **AC-2** — Given media có nhiều embedding liên quan, When gỡ, Then tất cả tham chiếu bị gỡ đồng bộ.

### Từ BR-705 · Nhật ký truy vấn ẩn danh `[v2]`

#### US-705.1 — Ghi log truy vấn ẩn danh
*Là `Data/ML Engineer`, tôi muốn log truy vấn không chứa định danh, để gỡ lỗi & cải thiện mà vẫn riêng tư.*

- **Mục tiêu**: Lưu `query_logs` ẩn danh (loại truy vấn, biểu diễn, kết quả, latency).
- **Input**: Sự kiện truy vấn.
- **Output**: Bản ghi log không chứa thông tin định danh người dùng.
- **Edge**: Truy vấn ảnh chứa khuôn mặt → không lưu ảnh gốc, chỉ lưu hash/biểu diễn; tắt log được qua cấu hình.
- **Test**: Kiểm tra bản ghi log → không có trường định danh; ảnh truy vấn không bị lưu nguyên bản.

**Acceptance Criteria**
- **AC-1** — Given một truy vấn, When ghi log, Then bản ghi không chứa thông tin định danh người dùng.
- **AC-2** — Given truy vấn bằng ảnh, When ghi log, Then không lưu ảnh gốc, chỉ lưu biểu diễn ẩn danh.

## 4. Non-Functional Requirements (NFR)

| Loại | Yêu cầu | Đo lường |
|---|---|---|
| Hiệu năng | Truy vấn top-k trả về với độ trễ trung bình ≤ 5s (đề xuất), p95 hợp lý | Báo cáo latency (BR-603) |
| Hiệu năng | Trích frame/cắt clip không chặn luồng chat (xử lý bất đồng bộ/cache) | Clip sẵn sàng ≤ vài giây sau khi chọn |
| Hiệu năng | **ASR transcript (BR-208) là tác vụ nặng → chạy offline ở worker**, không nằm trên đường request; truy vấn online chỉ đọc transcript đã embed | Transcript sinh sẵn trước truy vấn; latency online không tăng |
| Bảo mật | Đăng nhập đơn giản: hash mật khẩu (bcrypt/argon2), session/JWT, rate-limit | Không lưu mật khẩu thô; endpoint ingestion chỉ cho admin |
| Bảo mật | Endpoint nạp/crawl/đánh giá chỉ truy cập bởi role admin | 403 với non-admin |
| Quyền riêng tư | Log ẩn danh (BR-705); opt-out & gỡ bỏ (BR-702/703); phục vụ trong phạm vi consent (BR-701/704) | Kiểm thử không rò định danh |
| i18n / Locale | Giao diện & truy vấn tiếng Việt (dấu) là chính; chuẩn hóa Unicode/dấu | Truy vấn có/không dấu cho kết quả tương đương |
| Accessibility | Bàn phím điều hướng được; ảnh có alt từ caption; tương phản đạt | Kiểm thử a11y cơ bản |
| Độ tin cậy | LLM API lỗi → fallback kết quả thô; pipeline lỗi 1 mục không chặn cả batch | Không sập khi phụ thuộc ngoài lỗi |
| Router (định tuyến) | Định tuyến thêm tối đa 1 bước phân loại ý định; khi tín hiệu rõ (có ảnh / từ khóa nội quy) định tuyến bằng luật, không tốn thêm lượt LLM; định tuyến phải xác định (deterministic) & log được | Routing accuracy (BR-606); độ trễ tổng vẫn ≤ ngưỡng (BR-603) |
| Khả năng mở rộng | Index hỗ trợ upsert tăng tiến (BR-206, v2) | Thêm batch không rebuild full |
| Quan sát được | Log job thu thập/đánh giá + latency | Có dashboard/log tra cứu được |

## 5. Data Model

| Bảng / Entity | Trường chính | Quan hệ | BR liên quan |
|---|---|---|---|
| consents | id, scope, granted_at, expires_at, signed_by, evidence_file, enabled | 1–n posts (qua lô nạp) | BR-101, BR-701 |
| users | id, name, email, role[admin\|member\|alumni\|guest], password_hash, created_at | 1–n conversations | BR-505 (login), NFR bảo mật |
| posts | id, fb_post_id, posted_at, text_content, source_url, event_id?, collection_method[manual\|crawl], consent_id, checksum, needs_review | n–1 events; 1–n media_assets | BR-102,103,104,105,106 |
| media_assets | id, post_id, type[image\|video], file_path, duration?, width, height, checksum, status | n–1 posts; 1–n video_frames; 1–n captions | BR-102,105,106,201 |
| video_frames | id, media_asset_id, timestamp_sec, frame_path | n–1 media_assets | BR-201, BR-403, BR-404 |
| transcripts | id, media_asset_id, segment_idx, start_sec, end_sec, text, lang, confidence, low_confidence | n–1 media_assets; 1–1 embedding (text) | BR-208, BR-308, BR-403, BR-404 |
| captions | id, target_type[image\|frame], target_id, caption_text, lang, generated_by[auto\|manual], low_confidence | n–1 media/frame | BR-203, BR-402 |
| ocr_texts | id, target_type, target_id, text, lang, confidence | n–1 media/frame | BR-204 (v2) |
| events | id, name, year, description, aliases[] | 1–n posts | BR-302, BR-305 |
| regulations | id, title, version, effective_date, source_file, status[active\|archived], created_at | 1–n rule_chunks | BR-107 |
| rule_chunks | id, regulation_id, article_no (điều), clause_no (khoản), heading, text, order, needs_review | n–1 regulations; 1–1 embedding (text) | BR-107, BR-207, BR-307, BR-407 |
| embeddings | id, target_type[text\|image\|frame\|rule_chunk\|transcript], target_id, vector_id, collection, model, dim | tham chiếu media/transcript/nội quy; vector_id ↔ Qdrant — **3 collection riêng**: media (CLIP) / transcript (text) / nội quy (text) | BR-202, BR-205, BR-206, BR-207, BR-208 |
| persons | id, display_name, consent_status[opt_in\|opt_out\|unknown] | 1–n face_embeddings, person_appearances | BR-304, BR-702 (v2) |
| face_embeddings | id, person_id?, media_asset_id\|frame_id, vector_id | n–1 persons | BR-304, BR-702 (v2) |
| person_appearances | id, person_id, media_asset_id\|frame_id, event_id? | n–1 persons/events | BR-304 (v2) |
| conversations | id, user_id, created_at | 1–n messages | BR-501, BR-504 |
| messages | id, conversation_id, role[user\|assistant], content, attachments[], result_refs[], created_at | n–1 conversations | BR-501,502,503,504 |
| eval_queries | id, query_type[text\|image], query_text?, query_image_path?, expected_media_ids[], note | — | BR-601, BR-602 |
| query_logs | id, anon_session, query_type, query_repr, top_result_ids[], latency_ms, created_at | — | BR-603, BR-705 (v2) |
| removal_requests | id, target_type[media\|person], target_id, requester, reason, status, created_at | — | BR-703 (v2) |

## 6. API Endpoints

| Method | Path | Mục đích | Auth | BR |
|---|---|---|---|---|
| POST | /auth/login | Đăng nhập | No | BR-505 |
| POST | /auth/logout | Đăng xuất | User | BR-505 |
| GET | /auth/me | Thông tin phiên | User | BR-505 |
| POST | /consents | Lưu hồ sơ cấp quyền | Admin | BR-101, BR-701 |
| POST | /ingest/upload | Tải tay media + metadata | Admin | BR-102, BR-104, BR-105 |
| POST | /ingest/crawl | Kích hoạt job crawl | Admin | BR-103 (v2) |
| POST | /ingest/regulations | Nạp & tách văn bản nội quy theo điều/khoản | Admin | BR-107 |
| POST | /index/regulations | Chunk + embed nội quy vào collection vector riêng | Admin | BR-207 |
| GET | /media/{id} | Lấy media + metadata | User | BR-105 |
| GET | /media/{id}/clip?t= | Cắt & trả clip 3s trước/sau | User | BR-403 |
| GET | /media/{id}/stream?t= | Stream/seek video tới timestamp | User | BR-404 |
| POST | /pipeline/process | Trích frame + **tách audio/ASR transcript** + caption + OCR + embedding | Admin | BR-201,208,203,204,202 |
| POST | /index/build | Xây vector index | Admin | BR-205 |
| POST | /index/upsert | Cập nhật index tăng tiến | Admin | BR-206 (v2) |
| POST | /query/text | Truy vấn bằng văn bản (media + **transcript video**, gộp theo video_id+timestamp) | User | BR-301, BR-306, BR-308 |
| POST | /query/image | Truy vấn ảnh (sự kiện / tương tự) | User | BR-302, BR-303 |
| POST | /query/regulations | Truy vấn điều/khoản nội quy (tool nội quy) | User | BR-307 |
| POST | /query/person | Tìm theo khuôn mặt | User | BR-304 (v2) |
| POST | /chat/message | Hội thoại có định tuyến: text/ảnh → **router** chọn tool media/nội quy → câu trả lời + kết quả (media gộp cả transcript video) | User | BR-307,308,401,402,405,406,407,501,502,503,504,507 |
| GET | /events, /events/{id} | Danh mục/sự kiện để lọc | User | BR-305 |
| POST | /eval/run | Chạy đánh giá Recall@k/MRR/latency + đánh giá nội quy (điều khoản/groundedness/routing) | Admin | BR-601,602,603,606 |
| GET | /eval/report | Xem báo cáo đánh giá | Admin | BR-602,603,604,605,606 |
| GET | /admin/logs | Xem log truy vấn ẩn danh | Admin | BR-705 (v2) |
| POST | /privacy/opt-out | Đăng ký opt-out khuôn mặt | Admin/User | BR-702 (v2) |
| POST | /privacy/removal-request | Yêu cầu gỡ media/dữ liệu | Admin/User | BR-703 (v2) |

## 7. Screens

| Màn hình | Mô tả | US liên quan |
|---|---|---|
| Đăng nhập | Form đăng nhập đơn giản (admin vs người dùng) | US-505.1 |
| Chat (chính) | Khung hội thoại có định tuyến: nhập text, upload ảnh, **router tự định tuyến** media/nội quy, render kết quả đa phương thức + câu trả lời nội quy (dẫn điều khoản) inline; tìm video theo **lời nói** (transcript); có thể ép chế độ thủ công | US-501.1, US-502.1, US-503.1, US-507.1, US-301.1, US-302.1, US-303.1, US-307.1, US-308.1, US-401.1, US-407.1, US-406.1 |
| Trình xem media | Lightbox ảnh + player clip/seek timestamp + link nguồn | US-402.1, US-403.1, US-404.1, US-405.1 |
| Tìm theo người (v2) | Upload chân dung → sự kiện người đó xuất hiện (có opt-out) | US-304.1, US-702.1 |
| Lọc kết quả (v2) | Bộ lọc thời gian/sự kiện trên kết quả | US-305.1 |
| Admin · Nạp dữ liệu | Lưu consent, tải tay, kích hoạt crawl, sửa metadata, **nạp văn bản nội quy** | US-101.1, US-102.1, US-103.1, US-104.1, US-106.1, US-107.1 |
| Admin · Pipeline & Index | Chạy xử lý, build/upsert index media + **index transcript** + **index nội quy**, trạng thái job | US-201.1, US-208.1, US-203.1, US-204.1, US-202.1, US-205.1, US-206.1, US-207.1 |
| Admin · Đánh giá | Dashboard Recall@k/MRR/latency + chấm chất lượng + **đánh giá nội quy (điều khoản/groundedness/routing)** + checklist demo | US-601.1, US-602.1, US-603.1, US-604.1, US-605.1, US-606.1 |
| Admin · Quyền riêng tư (v2) | Quản lý opt-out, yêu cầu gỡ bỏ, log ẩn danh | US-702.1, US-703.1, US-705.1 |

## 8. Traceability Matrix

> Kiểm tra độ phủ: không story mồ côi, không BR (trong phạm vi) bị bỏ quên. BR-506 (v3) ngoài phạm vi PRD này.

| BR | User Story | Screen | Table | Test hook |
|---|---|---|---|---|
| BR-101 | US-101.1 | Admin · Nạp dữ liệu | consents | TC-101: ingest bị chặn khi thiếu consent |
| BR-102 | US-102.1 | Admin · Nạp dữ liệu | posts, media_assets | TC-102: upload batch + chặn thiếu metadata |
| BR-103 | US-103.1 | Admin · Nạp dữ liệu | posts, media_assets | TC-103: crawl incremental không nhân đôi |
| BR-104 | US-104.1 | Admin · Nạp dữ liệu | posts | TC-104: % mục đủ trường + needs_review |
| BR-105 | US-105.1 | Trình xem media | media_assets | TC-105: GET /media/{id} đúng + 404 |
| BR-106 | US-106.1 | Admin · Nạp dữ liệu | media_assets | TC-106: dedup checksum + perceptual |
| BR-107 | US-107.1 | Admin · Nạp dữ liệu | regulations, rule_chunks | TC-107: nạp nội quy → truy "Điều X" + phiên bản |
| BR-201 | US-201.1 | Admin · Pipeline & Index | video_frames | TC-201: frame + timestamp chính xác |
| BR-202 | US-202.1 | Admin · Pipeline & Index | embeddings | TC-202: text↔ảnh cross-modal |
| BR-203 | US-203.1 | Admin · Pipeline & Index | captions | TC-203: ≥80% caption hợp lý |
| BR-204 | US-204.1 | Admin · Pipeline & Index | ocr_texts | TC-204: tìm theo chữ poster |
| BR-205 | US-205.1 | Admin · Pipeline & Index | embeddings | TC-205: top-k + reject sai chiều |
| BR-206 | US-206.1 | Admin · Pipeline & Index | embeddings | TC-206: upsert không rebuild |
| BR-207 | US-207.1 | Admin · Pipeline & Index | rule_chunks, embeddings | TC-207: chunk nội quy → top-k đúng điều/khoản (collection riêng) |
| BR-208 | US-208.1 | Admin · Pipeline & Index | transcripts, embeddings | TC-208: video có tiếng → ≥1 đoạn transcript + timestamp + embed collection transcript |
| BR-301 | US-301.1 | Chat | embeddings | TC-301: Recall@k câu hỏi mẫu |
| BR-302 | US-302.1 | Chat | events, embeddings | TC-302: sự kiện đúng trong top-k |
| BR-303 | US-303.1 | Chat | embeddings | TC-303: ảnh tương tự cùng sự kiện |
| BR-304 | US-304.1 | Tìm theo người | persons, face_embeddings | TC-304: opt-in trả đúng, opt-out ẩn |
| BR-305 | US-305.1 | Lọc kết quả | events, posts | TC-305: lọc năm 2023 |
| BR-306 | US-306.1 | Chat | embeddings | TC-306: MRR + ngưỡng "không tìm thấy" |
| BR-307 | US-307.1 | Chat | rule_chunks, embeddings | TC-307: câu hỏi nội quy → điều/khoản đúng top-k |
| BR-308 | US-308.1 | Chat | transcripts, embeddings | TC-308: hỏi nội dung nói → đúng video + đoạn (timestamp) top-k, gộp keyframe theo video_id |
| BR-401 | US-401.1 | Chat | messages | TC-401: bám nguồn + fallback LLM lỗi |
| BR-402 | US-402.1 | Trình xem media | captions | TC-402: ảnh kèm mô tả/nhãn trống |
| BR-403 | US-403.1 | Trình xem media | video_frames | TC-403: clip ~6s đúng đoạn + biên |
| BR-404 | US-404.1 | Trình xem media | video_frames | TC-404: seek đúng timestamp ±1s |
| BR-405 | US-405.1 | Trình xem media | posts | TC-405: link nguồn / nhãn nội bộ |
| BR-406 | US-406.1 | Chat | — | TC-406: ngoài miền → không bịa |
| BR-407 | US-407.1 | Chat | rule_chunks, messages | TC-407: tình huống → dẫn điều/khoản + disclaimer; không quy định → báo rõ |
| BR-501 | US-501.1 | Chat | conversations, messages | TC-501: gửi/nhận + lỗi gửi lại |
| BR-502 | US-502.1 | Chat | messages | TC-502: upload ảnh trong chat |
| BR-503 | US-503.1 | Chat | messages | TC-503: render ảnh+clip inline |
| BR-504 | US-504.1 | Chat | conversations, messages | TC-504: hỏi nối tiếp theo ngữ cảnh |
| BR-505 | US-505.1 | Đăng nhập, Chat | users | TC-505: responsive đa trình duyệt |
| BR-507 | US-507.1 | Chat | conversations, messages | TC-507: định tuyến đúng media/nội quy + override thủ công |
| BR-601 | US-601.1 | Admin · Đánh giá | eval_queries | TC-601: tập ≥N có nhãn |
| BR-602 | US-602.1 | Admin · Đánh giá | eval_queries | TC-602: Recall@k/MRR khớp tính tay |
| BR-603 | US-603.1 | Admin · Đánh giá | query_logs | TC-603: avg + p95 latency |
| BR-604 | US-604.1 | Admin · Đánh giá | — | TC-604: tỷ lệ đạt ≥ ngưỡng |
| BR-605 | US-605.1 | Admin · Đánh giá | — | TC-605: checklist mọi luồng pass |
| BR-606 | US-606.1 | Admin · Đánh giá | eval_queries, rule_chunks | TC-606: accuracy điều khoản + groundedness + routing accuracy |
| BR-701 | US-701.1 | Admin · Nạp dữ liệu | consents | TC-701: hết hạn consent → khóa dữ liệu |
| BR-702 | US-702.1 | Admin · Quyền riêng tư | persons, face_embeddings | TC-702: opt-out ẩn khỏi face search |
| BR-703 | US-703.1 | Admin · Quyền riêng tư | removal_requests | TC-703: gỡ đồng bộ mọi tham chiếu |
| BR-704 | US-704.1 | Chat | — | TC-704: truy vấn ngoài miền bị giới hạn |
| BR-705 | US-705.1 | Admin · Quyền riêng tư | query_logs | TC-705: log không định danh |

---

> Bước tiếp theo trong pipeline: `/write-hld` (vẽ kiến trúc/pipeline: ingestion → embedding/index → retrieval/RAG → chat web) → `/deep-research` (đào sâu lựa chọn model embedding tiếng Việt, video moment retrieval, ngưỡng metric) → **QC spec thủ công** → implement.
