# BRD — Trợ lý ảo Multimodal RAG cho Fanpage CLB Tin học HIT

> Business Requirements Document · Version tài liệu: 1.0 · Ngày: 2026-06-10 · Mode: capstone
> Trạng thái: Draft · Người soạn: Nhóm capstone (Đạt phụ trách Data) · Người duyệt: [GVHD] / [Ban Chủ nhiệm CLB HIT]
> Stakeholders cần ký duyệt: Giảng viên hướng dẫn, Ban Chủ nhiệm CLB Tin học HIT (chủ sở hữu dữ liệu)

## 1. Tổng quan

**Problem statement.** Sau ~16 năm hoạt động, Fanpage CLB Tin học HIT tích lũy một kho tư liệu lớn (bài viết, ảnh, video sự kiện) nhưng **rời rạc và khó truy xuất**: muốn tìm "sự kiện X năm nào", "ảnh của hoạt động Y", hay "các hoạt động một thành viên từng tham gia" thì phải cuộn tay qua hàng nghìn bài. Người chịu pain chính là Ban Chủ nhiệm/Admin (tổng hợp truyền thông, báo cáo), thành viên và cựu thành viên (tìm lại kỷ niệm), và sinh viên mới (tìm hiểu CLB). Dự án tập trung vào **tư liệu 5 năm gần nhất** — phần dày đặc và còn giá trị tra cứu cao nhất. Nếu không giải, kho tư liệu này gần như "chết" trong kho — không tái sử dụng được.

**Mục tiêu kinh doanh.** Xây một **trợ lý ảo hội thoại (chatbot trên web app)** cho phép hỏi-đáp và truy xuất chuyên sâu trong dữ liệu multimedia của fanpage bằng cả **văn bản và hình ảnh**, trả về **ảnh kèm mô tả + đoạn video ngắn** đúng ngữ cảnh — biến kho tư liệu thụ động thành nguồn tri thức tra cứu được. Trợ lý còn **hỏi-đáp nội quy/quy chế CLB** (tra cứu điều khoản + trả lời câu hỏi tình huống có dẫn căn cứ), và được thiết kế theo hướng **Router RAG** — một bộ định tuyến (router) tự chọn nguồn tri thức phù hợp (kho media fanpage ↔ văn bản nội quy) cho từng câu hỏi. *(Agentic RAG đầy đủ — phân rã truy vấn, multi-hop, tổng hợp đa nguồn — để dành v2; xem BR-507.)*

**Success metrics.** Capstone đặt tiêu chí **cân bằng cả ba trục** (chốt số khi có tập dữ liệu thật):
- *Độ chính xác truy xuất*: Recall@5 ≥ 0.80, MRR ≥ 0.60 trên bộ truy vấn đánh giá *(mục tiêu đề xuất — [cần xác minh])*.
- *Độ phủ demo end-to-end*: 100% các luồng lõi chạy trọn vẹn. **Đầu vào** chỉ gồm text & ảnh: (1) text → ảnh+mô tả; (2) ảnh → nhận diện sự kiện; (3) ảnh → ảnh tương tự. **Đầu ra** phủ đủ định dạng: ảnh kèm mô tả, và — khi kết quả khớp một frame trích từ video trong kho — clip 3s trước/sau frame hoặc preview video tới đúng timestamp. *(Lưu ý: video là dữ liệu trong kho, KHÔNG phải đầu vào truy vấn.)*
- *Trải nghiệm & độ trễ*: thời gian phản hồi trung bình ≤ 5s cho truy vấn top-k *(đề xuất — [cần xác minh])*.

> **Ghi chú giảng dạy.** Mục tiêu + metric đo được là thứ hội đồng dùng để chấm: mọi BR bên dưới được viết để *quy về một tiêu chí nghiệm thu duy nhất*, nhờ đó dễ phân vai (đặc biệt vai **Data Engineer** ở module BR-100/BR-200) và dễ chứng minh "đã xong".

## 2. Bối cảnh thị trường
<!-- Capstone: giữ ngắn. -->

- **Bối cảnh.** Không phải sản phẩm thương mại; giá trị nằm ở **tái sử dụng tài sản dữ liệu nội bộ** của CLB và ở **năng lực kỹ thuật multimodal RAG** mà nhóm chứng minh được.
- **Tham chiếu công nghệ.** Cùng họ với các hệ multimodal search/RAG (CLIP-based image-text retrieval, video moment retrieval). Khoảng trống nhóm khai thác: **miền dữ liệu tiếng Việt, đặc thù một fanpage cụ thể**, nơi các giải pháp chung chung không hiểu ngữ cảnh sự kiện CLB.
- **Lợi thế.** Data moat nội bộ (dữ liệu được cấp quyền, không công khai) + hiểu ngữ cảnh sự kiện CLB.

## 3. Personas

| Persona | Đặc điểm | Nhu cầu chính | Sẵn sàng chi trả |
|---|---|---|---|
| Thành viên CLB hiện tại | Sinh viên đang sinh hoạt CLB | Tra cứu hoạt động/sự kiện; tìm lại ảnh mình tham gia | N/A (capstone) |
| Ban Chủ nhiệm / Admin | Quản lý fanpage, làm truyền thông/báo cáo | Tổng hợp tư liệu, thống kê hoạt động, tìm nhanh ảnh sự kiện | N/A |
| Cựu thành viên / Alumni | Đã ra trường, gắn bó kỷ niệm | Tìm lại ảnh & sự kiện đã tham gia trước đây | N/A |
| Sinh viên mới / khách quan tâm | Chưa biết nhiều về CLB | Tìm hiểu CLB qua tư liệu, ảnh, sự kiện đã tổ chức | N/A |

## 4. Phạm vi (Scope)

**Trong phạm vi (In scope — trọng tâm v1).**
- Thu thập dữ liệu fanpage **thủ công + tập mẫu** (sau khi có quyền), chuẩn hóa metadata, lưu kho thô.
- Pipeline xử lý multimodal: trích frame video, **tách audio + sinh transcript (ASR) cho video**, sinh embedding text/ảnh, caption tự động, xây vector index.
- Truy vấn **văn bản** và **hình ảnh** (nhận diện sự kiện từ ảnh, tìm ảnh tương tự).
- Sinh câu trả lời RAG: **ảnh kèm mô tả**, **clip video 3s trước + 3s sau frame**, **preview video tới timestamp**, trích dẫn nguồn.
- **Trợ lý ảo hội thoại** dạng chatbot trên web app (upload ảnh ngay trong chat).
- **Hỏi-đáp nội quy CLB**: nạp văn bản nội quy, tra cứu điều/khoản và trả lời **câu hỏi tình huống** có dẫn căn cứ + disclaimer (text RAG riêng).
- **Router RAG (định tuyến)**: bộ định tuyến chọn nguồn phù hợp cho mỗi câu hỏi (media fanpage / nội quy / cả hai) — ưu tiên luật khi tín hiệu rõ (có ảnh / từ khóa nội quy), fallback classifier khi mơ hồ.
- Đánh giá: Recall@k, MRR, độ trễ; demo end-to-end các loại truy vấn lõi (gồm cả hỏi-đáp nội quy).

**Ngoài phạm vi (Out of scope — chống scope creep).**
- **Nhận diện khuôn mặt / tìm hoạt động theo người** (face search) → để **v2/v3** vì lý do quyền riêng tư & độ phức tạp.
- **Crawl tự động đầy đủ 5 năm** → v2 (v1 dùng tải tay + tập mẫu).
- **Triển khai chatbot lên Messenger của fanpage** → v3.
- Dữ liệu **ngoài fanpage CLB Tin học HIT** (page khác, web ngoài).
- **Tạo/chỉnh sửa nội dung mới** (chỉ truy xuất & tổng hợp, không sinh ảnh/video mới).
- Phân tích livestream/realtime.
- Đa ngôn ngữ ngoài tiếng Việt *(giả định focus tiếng Việt — sửa nếu sai)*.

## 5. Lộ trình theo Version

| Version | Trọng tâm | Vấn đề giải quyết | Ghi chú |
|---|---|---|---|
| v1 POC | Core multimodal RAG engine + router nội quy | Tìm được ảnh/sự kiện/clip từ truy vấn text & ảnh trên tập mẫu; **video tách keyframe + transcript (ASR)** để tìm theo lời nói; **hỏi-đáp nội quy** (tra cứu + tình huống); **router tự định tuyến** media ↔ nội quy — qua chatbot web | Tải tay + tập mẫu; không face search; router định tuyến + 2 nguồn tri thức |
| v2 MVP | Phủ dữ liệu + tìm theo người + Agentic RAG | Crawl tự động đủ 5 năm; face/person search có kiểm soát quyền riêng tư; OCR; index tăng tiến; ngữ cảnh hội thoại; **nâng router → Agentic RAG** (phân rã truy vấn, multi-hop, tổng hợp đa nguồn) | Cần quyền Admin & cơ chế đồng thuận |
| v3 Beta | Đưa vào dùng thật | Triển khai chatbot lên Messenger fanpage; scale; polish UX | Phụ thuộc nghiệm thu v2 |

## 6. Business Requirements (BR)

> Quy ước: `BR-<module><nn>`. Mỗi BR atomic, có Priority (MoSCoW), Version và tiêu chí nghiệm thu sơ bộ.

### BR-100 · Thu thập & Quản lý dữ liệu nguồn *(vai Data Engineer)*

| ID | Tên | Mô tả (nghiệp vụ) | Priority | Version | Tiêu chí nghiệm thu sơ bộ |
|---|---|---|---|---|---|
| BR-101 | Xin quyền truy cập dữ liệu | Chỉ thu thập dữ liệu fanpage sau khi có sự đồng ý/cấp quyền Admin từ Ban Chủ nhiệm CLB | Must | v1 | Có email/văn bản chấp thuận phạm vi sử dụng được lưu trữ trước khi thu thập |
| BR-102 | Thu thập thủ công | Tải tay bài viết/ảnh/video từ fanpage vào kho dữ liệu thô | Must | v1 | Nạp thành công ≥1 batch dữ liệu mẫu (vd các sự kiện tiêu biểu) |
| BR-103 | Thu thập tự động (crawl) | Tự động crawl bài/ảnh/video kèm metadata từ fanpage qua quyền Admin | Should | v2 | Crawl được dữ liệu 5 năm; chạy lại cập nhật bài mới |
| BR-104 | Chuẩn hóa metadata | Mỗi mục dữ liệu gắn metadata: ngày đăng, caption, link bài, loại media, sự kiện (nếu có) | Must | v1 | 100% mục trong kho có đủ trường metadata bắt buộc |
| BR-105 | Lưu trữ kho dữ liệu thô | Lưu media gốc + metadata có tổ chức, truy lại được theo ID | Must | v1 | Truy xuất 1 media bất kỳ theo ID trả về đúng file + metadata |
| BR-106 | Khử trùng lặp | Phát hiện & loại bài/ảnh trùng khi thu thập nhiều lần | Could | v2 | Ảnh/bài trùng không tạo bản ghi nhân đôi trong index |
| BR-107 | Nạp & cấu trúc văn bản nội quy | Nạp văn bản nội quy/quy chế CLB, tách theo điều/khoản/mục làm nguồn tri thức **text** riêng cho luồng hỏi-đáp nội quy | Must | v1 | Văn bản nội quy được nạp & tách thành đơn vị điều/khoản, truy lại được theo số điều |

### BR-200 · Pipeline xử lý & lập chỉ mục Multimodal *(vai Data/ML Engineer)*

| ID | Tên | Mô tả (nghiệp vụ) | Priority | Version | Tiêu chí nghiệm thu sơ bộ |
|---|---|---|---|---|---|
| BR-201 | Trích xuất khung hình video | Tách frame đại diện + timestamp từ video để lập chỉ mục | Must | v1 | Mỗi video sinh ≥1 frame kèm timestamp chính xác |
| BR-202 | Sinh embedding đa phương thức | Sinh vector cho text và ảnh trong cùng không gian để so khớp chéo | Must | v1 | Query ảnh trả về ảnh tương tự; query text trả về ảnh liên quan |
| BR-203 | Sinh mô tả/caption tự động | Tạo mô tả nội dung cho ảnh/frame chưa có caption | Should | v1 | ≥80% ảnh mẫu có caption sinh tự động hợp lý (đánh giá tay) |
| BR-204 | OCR văn bản trong ảnh | Trích text trong ảnh (poster, banner sự kiện) để tăng khả năng tìm | Could | v2 | Text trên poster mẫu được trích & tìm được qua truy vấn text |
| BR-205 | Xây vector index | Lưu embedding vào vector store cho tìm kiếm ngữ nghĩa nhanh | Must | v1 | Truy vấn top-k trả về dưới ngưỡng độ trễ mục tiêu |
| BR-206 | Cập nhật index tăng tiến | Nạp dữ liệu mới vào index mà không dựng lại toàn bộ | Should | v2 | Thêm batch mới, tìm được ngay, không rebuild full |
| BR-207 | Lập chỉ mục văn bản nội quy | Chunk nội quy theo điều/khoản, sinh embedding text, lưu **collection vector riêng** cho nội quy (tách khỏi index media) | Must | v1 | Truy vấn text trả đúng điều/khoản liên quan từ collection nội quy |
| BR-208 | Tách audio & sinh transcript (ASR) | Mỗi video tách audio → **ASR** (speech-to-text) sinh transcript cắt theo timestamp; embed transcript (text) vào index để truy xuất video theo **nội dung nói** (tách 2 nhánh khỏi frame: keyframe ↔ audio→transcript) | Must | v1 | Mỗi video có giọng nói sinh ≥1 đoạn transcript kèm timestamp; truy vấn text khớp lời nói trả về đúng video/đoạn |

### BR-300 · Truy vấn & Truy xuất Multimodal

| ID | Tên | Mô tả (nghiệp vụ) | Priority | Version | Tiêu chí nghiệm thu sơ bộ |
|---|---|---|---|---|---|
| BR-301 | Truy vấn bằng văn bản | Nhập câu hỏi text, hệ thống truy xuất media + nội dung liên quan | Must | v1 | Câu hỏi mẫu trả về ảnh/bài đúng chủ đề trong top-k |
| BR-302 | Nhận diện sự kiện từ ảnh | Tải ảnh hỏi "đây là sự kiện nào", hệ thống trả tên/sự kiện liên quan | Must | v1 | Ảnh sự kiện mẫu được gán đúng sự kiện trong top-k |
| BR-303 | Tìm ảnh tương tự | Từ 1 ảnh, trả về các ảnh liên quan trong kho | Must | v1 | Ảnh truy vấn trả về ≥k ảnh cùng sự kiện/chủ đề |
| BR-304 | Tìm hoạt động theo người (face) | Tải ảnh 1 cá nhân → trả về hoạt động/ảnh người đó tham gia | Should | v2 | Ảnh người mẫu trả về đúng sự kiện họ xuất hiện (có đồng thuận) |
| BR-305 | Lọc theo thời gian/sự kiện | Lọc kết quả theo khoảng thời gian hoặc tên sự kiện | Could | v2 | Lọc "năm 2023" chỉ trả kết quả trong năm đó |
| BR-306 | Xếp hạng độ liên quan | Sắp xếp kết quả theo độ liên quan ngữ nghĩa | Must | v1 | Kết quả top-1 liên quan hơn kết quả hạng thấp (metric/đánh giá tay) |
| BR-307 | Truy vấn nội quy | Từ câu hỏi text (tra cứu hoặc tình huống), truy xuất các điều/khoản nội quy liên quan nhất | Must | v1 | Câu hỏi nội quy mẫu trả về điều/khoản đúng trong top-k |
| BR-308 | Truy xuất theo lời nói trong video (transcript) | Truy vấn text khớp transcript của video; gộp với kết quả keyframe theo `video_id` + timestamp để trả về đúng video + đoạn + frame trong khoảng nói | Must | v1 | Câu hỏi về nội dung được nói trong video trả về đúng video + đoạn (timestamp) trong top-k |

### BR-400 · Sinh câu trả lời & Trình bày kết quả

| ID | Tên | Mô tả (nghiệp vụ) | Priority | Version | Tiêu chí nghiệm thu sơ bộ |
|---|---|---|---|---|---|
| BR-401 | Sinh câu trả lời RAG | Tổng hợp câu trả lời ngôn ngữ tự nhiên dựa trên media/ngữ cảnh truy xuất | Must | v1 | Câu trả lời bám nội dung nguồn, không bịa (đánh giá tay) |
| BR-402 | Trả ảnh kèm mô tả | Hiển thị ảnh kết quả kèm mô tả/câu trả lời | Must | v1 | Mỗi kết quả ảnh có mô tả đi kèm |
| BR-403 | Trả clip video ngắn | Với frame video, trả clip 3s trước + 3s sau frame | Should | v1 | Clip ~6s quanh frame phát được, đúng đoạn |
| BR-404 | Preview video tới timestamp | Cho phép tua/preview video tới đúng thời điểm chứa frame | Should | v1 | Click kết quả mở video đúng tại timestamp |
| BR-405 | Trích dẫn nguồn | Mỗi câu trả lời gắn link/nguồn bài gốc trên fanpage khi có | Should | v1 | Kết quả có link về bài gốc khi tồn tại |
| BR-406 | Xử lý "không tìm thấy" | Khi không đủ dữ liệu liên quan, báo rõ thay vì bịa | Must | v1 | Truy vấn ngoài miền trả về thông báo không có dữ liệu |
| BR-407 | Trả lời nội quy & tình huống | Sinh câu trả lời dựa trên điều/khoản truy xuất, áp dụng nhẹ vào tình huống; **bắt buộc trích dẫn điều/khoản căn cứ + disclaimer** ("tham khảo, BCN quyết định cuối"); nội quy không quy định → nói rõ | Must | v1 | Câu hỏi tình huống mẫu neo đúng điều/khoản + có trích dẫn + disclaimer; câu ngoài nội quy → báo "không quy định" |

### BR-500 · Trợ lý ảo hội thoại (Web app + Chatbot)

| ID | Tên | Mô tả (nghiệp vụ) | Priority | Version | Tiêu chí nghiệm thu sơ bộ |
|---|---|---|---|---|---|
| BR-501 | Giao diện chatbot hội thoại | Người dùng tương tác chính qua chatbot trên web app | Must | v1 | Gửi câu hỏi và nhận trả lời trong khung chat |
| BR-502 | Upload ảnh trong hội thoại | Đính kèm ảnh để truy vấn ngay trong chat | Must | v1 | Tải ảnh trong chat và nhận kết quả |
| BR-503 | Hiển thị kết quả đa phương thức | Render ảnh, clip, link, mô tả inline trong khung chat | Must | v1 | Kết quả ảnh + clip hiển thị trong chat |
| BR-504 | Ngữ cảnh hội thoại | Hỗ trợ hỏi nối tiếp dựa trên lượt trước | Should | v2 | "Còn ảnh nào khác không" hiểu theo ngữ cảnh trước |
| BR-505 | Web app truy cập đa nhóm | Web app dùng được cho thành viên, alumni, admin, khách | Must | v1 | 1 URL chạy được trên trình duyệt phổ thông |
| BR-506 | Tích hợp Messenger fanpage | Đưa chatbot lên Messenger của fanpage | Won't (v1) | v3 | Hỏi-đáp được trong inbox fanpage |
| BR-507 | Bộ định tuyến đa nguồn (Router RAG) | Trợ lý dùng một **router**: với mỗi truy vấn chọn công cụ phù hợp (tìm media fanpage / tra nội quy / cả hai) — ưu tiên luật khi tín hiệu rõ, fallback classifier khi mơ hồ; cho phép **ép chế độ thủ công**. *(Nâng lên Agentic RAG đầy đủ — phân rã/multi-hop/tổng hợp đa nguồn — thuộc v2.)* | Must | v1 | Câu hỏi nội quy → định tuyến tool nội quy; câu hỏi tư liệu/ảnh → tool media; định tuyến đúng ≥ ngưỡng trên bộ thử |

### BR-600 · Đánh giá & Chất lượng

| ID | Tên | Mô tả (nghiệp vụ) | Priority | Version | Tiêu chí nghiệm thu sơ bộ |
|---|---|---|---|---|---|
| BR-601 | Bộ truy vấn đánh giá | Xây tập câu hỏi/ảnh mẫu có nhãn đúng để đo | Must | v1 | Có ≥N truy vấn mẫu kèm đáp án kỳ vọng |
| BR-602 | Đo độ chính xác truy xuất | Báo cáo Recall@k / MRR trên tập đánh giá | Must | v1 | Có số liệu Recall@k, MRR trong báo cáo |
| BR-603 | Đo độ trễ | Đo thời gian phản hồi trung bình/percentile | Should | v1 | Báo cáo độ trễ trung bình ≤ ngưỡng mục tiêu |
| BR-604 | Đánh giá chất lượng câu trả lời | Đánh giá tay tính đúng/bám nguồn của câu trả lời | Should | v1 | Tỷ lệ câu trả lời đạt ≥ ngưỡng (vd 80%) |
| BR-605 | Độ phủ demo end-to-end | Chứng minh chạy trọn luồng cho các loại truy vấn chính | Must | v1 | Demo đủ: text, ảnh→sự kiện, ảnh tương tự, video clip, **hỏi-đáp nội quy** |
| BR-606 | Đánh giá hỏi-đáp nội quy | Bộ câu hỏi nội quy (tra cứu + tình huống) có nhãn điều/khoản đúng; đo độ đúng điều khoản, độ bám nguồn (groundedness) & độ chính xác định tuyến của router | Should | v1 | Báo cáo: accuracy điều khoản, groundedness, routing accuracy |

### BR-700 · Quyền riêng tư, Pháp lý & Vận hành

| ID | Tên | Mô tả (nghiệp vụ) | Priority | Version | Tiêu chí nghiệm thu sơ bộ |
|---|---|---|---|---|---|
| BR-701 | Đồng thuận sử dụng dữ liệu | Chỉ dùng dữ liệu fanpage trong phạm vi được CLB cho phép | Must | v1 | Có thỏa thuận phạm vi sử dụng được lưu trữ |
| BR-702 | Bảo vệ dữ liệu cá nhân | Hạn chế/ẩn dữ liệu nhận diện cá nhân, đặc biệt với face search | Must | v2 | Face search có cơ chế opt-out/ẩn người không đồng thuận |
| BR-703 | Gỡ bỏ theo yêu cầu | Cho phép gỡ ảnh/dữ liệu cá nhân khi được yêu cầu | Should | v2 | Yêu cầu gỡ → media bị loại khỏi index |
| BR-704 | Giới hạn phạm vi miền dữ liệu | Chỉ phục vụ truy vấn trong miền fanpage HIT (5 năm) | Must | v1 | Truy vấn ngoài miền được từ chối/thông báo |
| BR-705 | Nhật ký truy vấn ẩn danh | Ghi log truy vấn phục vụ gỡ lỗi & đánh giá | Could | v2 | Có log truy vấn không chứa thông tin định danh |

## 7. Ràng buộc, Giả định, Phụ thuộc

- **Constraints.**
  - Thời gian: theo lịch capstone (giả định ~1 học kỳ — [cần xác minh]).
  - Compute: GPU server của dự án → ưu tiên model open-source; lấy mẫu frame và giảm độ phân giải khi cần để tối ưu tài nguyên.
  - Ngân sách: gần như $0 → dùng model/hạ tầng miễn phí hoặc free-tier *(giả định)*.
- **Assumptions.**
  - Dữ liệu fanpage được Ban Chủ nhiệm cấp quyền sử dụng cho mục đích học thuật (gắn với BR-101).
  - Tiếng Việt là ngôn ngữ truy vấn chính.
  - Nhóm có vai **Data Engineer** (Đạt phụ trách xin quyền & thu thập dữ liệu — theo brief).
  - Quy mô dữ liệu ~vài nghìn bài / hàng chục nghìn ảnh / hàng trăm video trong 5 năm *([cần xác minh])*.
- **Dependencies.**
  - Quyền Admin page từ Ban Chủ nhiệm CLB (chặn BR-101 → BR-103).
  - Facebook Graph API / công cụ crawl cho thu thập tự động.
  - Model embedding đa phương thức (CLIP / đa ngôn ngữ) + LLM sinh câu trả lời + vector DB *(lựa chọn cụ thể để PRD/HLD chốt)*.

## 8. Rủi ro

| Rủi ro | Mức độ | Giảm thiểu |
|---|---|---|
| Không được cấp quyền Admin / Facebook hạn chế crawl | Cao | Xin văn bản đồng ý sớm; phương án tải tay + tập mẫu cho v1 (BR-102) |
| Quyền riêng tư cá nhân khi tìm theo người (face search) | Cao | Hoãn sang v2; cơ chế opt-out/ẩn người không đồng thuận (BR-702/703) |
| Embedding/caption tiếng Việt chất lượng thấp | Trung bình | Chọn model đa ngôn ngữ; đánh giá sớm bằng bộ truy vấn (BR-601/602) |
| Chất lượng ASR transcript tiếng Việt thấp (tạp âm sự kiện, nhiều người nói) | Trung bình | Dùng model ASR mạnh tiếng Việt (vd WhisperX/faster-whisper); transcript chỉ **bổ trợ recall** cạnh keyframe, không phải nguồn duy nhất; chạy offline ở worker (BR-208) |
| Dữ liệu/video lớn vượt khả năng compute | Trung bình | Lấy mẫu frame, giảm độ phân giải, index tăng tiến (BR-201/206) |
| Bài cũ thiếu/nhiễu metadata | Trung bình | Chuẩn hóa metadata; bỏ qua mục thiếu trường bắt buộc (BR-104) |
| Mô hình "bịa" câu trả lời (hallucination) | Trung bình | Bắt buộc trích dẫn nguồn & xử lý "không tìm thấy" (BR-405/406) |
| Router định tuyến sai nguồn (media ↔ nội quy) | Trung bình | Bộ thử định tuyến + đo routing accuracy (BR-606); fallback hỏi làm rõ; cho phép ép chế độ thủ công |
| Trả lời nội quy/tình huống sai hoặc "phán xử" vượt thẩm quyền | Cao | Chỉ áp dụng nhẹ, bắt buộc neo điều/khoản + disclaimer "BCN quyết định cuối"; nội quy không quy định → báo rõ (BR-407) |

---

## Phụ lục — Traceability (điền dần)

| BR | Liên kết PRD (User Story) | Trạng thái |
|---|---|---|
| BR-101 | US-101.1 | Spec'd (v1) |
| BR-102 | US-102.1 | Spec'd (v1) |
| BR-103 | US-103.1 | Spec'd (v2) |
| BR-104 | US-104.1 | Spec'd (v1) |
| BR-105 | US-105.1 | Spec'd (v1) |
| BR-106 | US-106.1 | Spec'd (v2) |
| BR-107 | US-107.1 | Spec'd (v1) |
| BR-201 | US-201.1 | Spec'd (v1) |
| BR-202 | US-202.1 | Spec'd (v1) |
| BR-203 | US-203.1 | Spec'd (v1) |
| BR-204 | US-204.1 | Spec'd (v2) |
| BR-205 | US-205.1 | Spec'd (v1) |
| BR-206 | US-206.1 | Spec'd (v2) |
| BR-207 | US-207.1 | Spec'd (v1) |
| BR-208 | US-208.1 | Spec'd (v1) |
| BR-301 | US-301.1 | Spec'd (v1) |
| BR-302 | US-302.1 | Spec'd (v1) |
| BR-303 | US-303.1 | Spec'd (v1) |
| BR-304 | US-304.1 | Spec'd (v2) |
| BR-305 | US-305.1 | Spec'd (v2) |
| BR-306 | US-306.1 | Spec'd (v1) |
| BR-307 | US-307.1 | Spec'd (v1) |
| BR-308 | US-308.1 | Spec'd (v1) |
| BR-401 | US-401.1 | Spec'd (v1) |
| BR-402 | US-402.1 | Spec'd (v1) |
| BR-403 | US-403.1 | Spec'd (v1) |
| BR-404 | US-404.1 | Spec'd (v1) |
| BR-405 | US-405.1 | Spec'd (v1) |
| BR-406 | US-406.1 | Spec'd (v1) |
| BR-407 | US-407.1 | Spec'd (v1) |
| BR-501 | US-501.1 | Spec'd (v1) |
| BR-502 | US-502.1 | Spec'd (v1) |
| BR-503 | US-503.1 | Spec'd (v1) |
| BR-504 | US-504.1 | Spec'd (v2) |
| BR-505 | US-505.1 | Spec'd (v1) |
| BR-506 | — | v3 — ngoài phạm vi PRD hiện tại |
| BR-507 | US-507.1 | Spec'd (v1) |
| BR-601 | US-601.1 | Spec'd (v1) |
| BR-602 | US-602.1 | Spec'd (v1) |
| BR-603 | US-603.1 | Spec'd (v1) |
| BR-604 | US-604.1 | Spec'd (v1) |
| BR-605 | US-605.1 | Spec'd (v1) |
| BR-606 | US-606.1 | Spec'd (v1) |
| BR-701 | US-701.1 | Spec'd (v1) |
| BR-702 | US-702.1 | Spec'd (v2) |
| BR-703 | US-703.1 | Spec'd (v2) |
| BR-704 | US-704.1 | Spec'd (v1) |
| BR-705 | US-705.1 | Spec'd (v2) |

> Bước tiếp theo: chạy `/write-prd` để bung mỗi BR thành User Story + Acceptance Criteria + Edge cases + Test hooks.
