# Báo cáo đánh giá truy xuất — HIT-MIRA v1

> Đo ngày **02/08/2026** trên kho thật: 4.056 điểm `media_clip`, 248 điểm
> `video_transcript`, 4 điểm `rag_documents`. Nguồn nhúng: pod GPU tự host
> `jina-clip-v2`. Chạy lại: `python scripts/run_eval.py`.
>
> Bám BR-601..BR-603, BR-605 · US-601.1, US-602.1, US-603.1, US-605.1 ·
> TC-601, TC-602, TC-603, TC-605.

## 1. Kết quả

| Chỉ số | Đo được | Mục tiêu | |
|---|---|---|---|
| **Recall@5** | **0,642** | ≥ 0,80 | **CHƯA ĐẠT** |
| **MRR** | **0,776** | ≥ 0,60 | **ĐẠT** |
| **Latency trung bình** | **0,85s** (p95 1,71s · cao nhất 2,35s) | ≤ 5s | **ĐẠT** |

Trên 38 truy vấn trong miền, trong tổng số 50 truy vấn có nhãn.

### Theo từng nhóm

| Nhóm | n | Recall@5 | MRR | Ghi chú |
|---|---|---|---|---|
| `media_image` (ảnh, tìm theo nội dung nhìn thấy) | 20 | 0,600 | 0,770 | |
| `media_transcript` (lời thoại video) | 10 | 0,440 | 0,608 | yếu nhất |
| `regulation` (nội quy) | 8 | 1,000 | 1,000 | **xem cảnh báo §3** |
| `media_image_ocr` (chữ trong ảnh) | 6 | 0,367 | 0,500 | phạm vi v2, không tính vào chỉ số chính |

**MRR 0,776 nói lên điều quan trọng nhất: kết quả đúng thường nằm ở hạng 1–2.**
Với một chatbot chỉ hiện vài kết quả đầu, đây là chỉ số sát trải nghiệm hơn
Recall@5. Recall@5 thấp hơn vì nó đòi lấp đủ cả 5 chỗ bằng kết quả đúng.

## 2. Cách gán nhãn

Nhãn **không** lấy từ đầu ra của hệ thống — làm vậy thì Recall@k luôn bằng 1,0.
Mỗi câu hỏi mang một luật từ khoá, và đáp án đúng là mọi đơn vị chứa đủ từ khoá
đó trong caption / lời thoại / văn bản nội quy: một kênh khớp chuỗi, độc lập
hoàn toàn với không gian vector đang được đo.

Câu hỏi viết bằng lời người dùng thật, cố ý không chép câu chữ trong caption —
chép lại thì phép đo chỉ còn là đo khớp chuỗi.

**Nhãn phải có nhóm đồng nghĩa.** Bản đo đầu tiên dùng một từ khoá duy nhất cho
mỗi câu và ra Recall@5 = 0,568. Soi tay thì thấy câu "ảnh các bạn đá bóng" chấm
trượt cả *"Các cầu thủ đang đứng trên sân cỏ nhân tạo"* — ảnh đúng, chỉ là
caption dùng từ khác. Sau khi cho mỗi câu một nhóm đồng nghĩa
(`["bóng đá", "cầu thủ", "sân cỏ"]`), chỉ số lên **0,642** và MRR lên **0,776**.
Không có gì trong hệ thống thay đổi — chỉ là phép đo thôi hết chấm trượt oan.

> Bài học: nhãn hẹp đo cách viết caption, không đo chất lượng truy xuất.

## 3. Ba cảnh báo phải đọc trước khi trích số

**a) `regulation` Recall@5 = 1,000 là con số rỗng.** Kho nội quy chỉ có **4
chunk**, mà k = 5 — mọi truy vấn đều trả về **toàn bộ** kho, nên Recall@5 không
thể khác 1,0. Chỉ **MRR = 1,000 mới có nghĩa**: điều khoản đúng luôn đứng hạng 1.
Số Recall của nhánh này chỉ có giá trị trở lại khi corpus vượt k một cách đáng
kể — trùng với ngưỡng ~50 chunk mà `CLAUDE.md` đã đặt.

**b) Recall thật có thể cao hơn 0,642.** Nhãn từ khoá vẫn bỏ sót: một ảnh đúng
nhưng caption dùng từ ngoài nhóm đồng nghĩa vẫn bị chấm trượt. Con số này là
**cận dưới**, không phải giá trị đúng.

**c) Latency 0,85s là cận dưới.** Đo trong tiến trình, gọi thẳng service như
`src/routers/` gọi, nên chưa gồm chi phí HTTP. Đo qua HTTP thật trước đó:
`/api/media/search` 1,7s và `/api/retrieval/search` 5,5s — con số sau gồm cả
bước viết lại truy vấn bằng LLM, thứ bộ đánh giá cố ý tắt để kết quả tái lập
được (bật bằng `--rewrite`).

## 4. Ngưỡng "không tìm thấy" — T-33 có số để quyết

Đo điểm cao nhất của 6 câu ngoài miền (giá vàng, nấu phở, visa du học…) so với
38 câu trong miền:

| | Điểm |
|---|---|
| Ngoài miền — cao nhất | **0,389** |
| Ngoài miền — trung bình | 0,315 |
| Trong miền — thấp nhất | **0,254** |
| Trong miền — trung bình | 0,412 |

**Hai khoảng chồng lấn nhau.** Câu ngoài miền cao nhất (0,389) vượt hẳn câu
trong miền thấp nhất (0,254), nên **không một ngưỡng điểm nào tách được hai
nhóm**: đặt ngưỡng 0,3 là giết luôn những câu hỏi hợp lệ điểm thấp; đặt 0,25 là
để lọt gần hết câu ngoài miền.

**Kết luận cho T-33: không chốt "không tìm thấy" bằng ngưỡng cosine.** Đây là hệ
quả trực tiếp của quyết định một-model trong `CLAUDE.md` — CLIP cho điểm nén
trong dải hẹp. Việc từ chối phải nằm ở tầng trả lời, và hiện `supervisor_prompt.md`
đang làm đúng vậy (luật "tool trả về trống hoặc không liên quan → báo rõ chưa tìm
thấy, tuyệt đối không bịa"). Checklist demo L5 xác nhận tầng truy xuất không tự
dựng link giả.

## 5. Vì sao tách `media_image_ocr` ra khỏi chỉ số chính

Đếm trên kho thật: tên sự kiện (**Open Day, hackathon, seminar, gala, chuyển
giao ban chủ nhiệm, online judge**) xuất hiện **0 lần trong caption**, chỉ nằm
trong OCR. Caption tả cảnh chung ("một nhóm người trong phòng họp"), còn danh
tính sự kiện nằm ở chữ trên banner.

Vector trong `media_clip` là vector của **hình ảnh**, không phải của chuỗi OCR.
Nên "cho tôi xem ảnh HIT Open Day" đòi hệ thống đọc chữ trong ảnh — đúng thứ
`CLAUDE.md` đã hoãn sang v2 (OCR + hybrid keyword search).

Đo được **0,367 / 0,500** — thấp hơn hẳn nhóm ảnh thường, nhưng **khác 0 rõ
rệt**: jina-clip-v2 có đọc được phần nào chữ lớn trên poster. Đây là số liệu để
quyết định có làm hybrid search ở v2 hay không, chứ không phải một lỗi cần sửa
ở v1.

## 6. Muốn đạt mục tiêu Recall@5 ≥ 0,80 thì làm gì

Theo thứ tự đáng-công-sức:

1. **Hybrid search (BM25 trên caption + OCR, trộn với điểm vector).** Nhắm thẳng
   vào nhóm yếu nhất và vào cả `media_image_ocr`. Đây cũng là hạng mục v2 đã ghi
   trong backlog.
2. **Model text riêng cho nhánh nội quy và lời thoại.** `media_transcript` là
   nhóm thấp nhất (0,440) đúng như dự đoán trong `CLAUDE.md`: CLIP yếu ở truy
   hồi văn bản thuần. Đổi được mà không phá tính chất một-vector-cho-mọi-thứ
   **chỉ với nhánh nội quy**; tách lời thoại ra sẽ làm mất tính chất đó.
3. **Bơm thêm nội quy.** 4 chunk là quá ít để nhánh này có ý nghĩa thống kê.
4. **Caption giàu hơn.** Nếu caption nêu được tên sự kiện đọc từ banner thì phần
   lớn khoảng trống OCR biến mất mà không cần hybrid search.

## 7. Checklist demo — TC-605

`python scripts/demo_checklist.py` — kết quả ngày 02/08/2026: **6 PASS · 0 FAIL
· 1 KHÔNG HỖ TRỢ**.

| | Luồng | US | Kết quả |
|---|---|---|---|
| L1 | text → ảnh + mô tả + link bài gốc | US-301.1 | PASS |
| L2 | text → lời thoại có mốc thời gian + link | US-303.1 | PASS |
| L3 | text → điều khoản nội quy | US-407.1 | PASS |
| L4 | ảnh tĩnh không mang mốc giây | US-405.1 | PASS |
| L5 | câu ngoài miền không dựng link giả | US-704.1 | PASS |
| L6 | ảnh làm truy vấn → sự kiện / ảnh tương tự | US-605.1 | **KHÔNG HỖ TRỢ** |
| L7 | xem ảnh + phát video đúng khoảnh khắc | US-605.1 | PASS |

"KHÔNG HỖ TRỢ" cố ý khác "FAIL": đó là luồng PRD có nêu mà v1 chưa làm, không
phải thứ đang hỏng.

- **L6** — `/api/media/search` chỉ nhận truy vấn văn bản; `embed_query(query: str)`
  không có nhánh ảnh. Model thì nhúng ảnh được (`embed_images`), nên đây là việc
  thiếu đường ống chứ không thiếu năng lực.
- **L7** — đã làm được sau khi thêm `API/src/routers/media_files.py`: ảnh và
  video tải thật qua presigned MinIO, và **Range trả 206** nên `<video ...#t=125>`
  tua đúng giây 125. **Cắt clip ~6s (TC-403) vẫn chưa** — cần ffmpeg lúc chạy;
  đổi lại là trình duyệt tải video đầy đủ.

## 8. Nhánh nội quy — TC-606 (T-73)

`python scripts/run_eval_noiquy.py` — đo ở **tầng trả lời**, gọi LangGraph thật
chứ không gọi service trong tiến trình. Kết quả 02/08/2026 trên 18 truy vấn
(8 nội quy + 6 ngoài miền + 4 media):

| Chỉ số | Đo được | Cỡ mẫu |
|---|---|---|
| **Routing accuracy** | **1,000** | 12 truy vấn có nhãn công cụ |
| **Groundedness** | **1,000** | 3 câu trả lời có dữ kiện kiểm được |
| **Disclaimer nhánh nội quy** | **1,000** | 8 câu nội quy |

*Accuracy điều khoản* đã có ở §1: MRR nhánh `regulation` = **1,000**, tức điều
khoản đúng luôn đứng hạng 1.

**Cách đo groundedness:** rút các **dữ kiện kiểm được** trong câu trả lời (số
tiền có dấu chấm nghìn, số điện thoại, email) rồi soi ngược vào ngữ cảnh mà công
cụ trả về. Một con số xuất hiện trong câu trả lời mà không có trong ngữ cảnh
chính là dấu hiệu bịa. Cố ý không bắt mọi con số — "1 ngày", "lần 2" là cách
diễn đạt lại, còn "100.000" hay "0823 644 212" mà sai thì người đọc bị dẫn sai
thật sự. Cỡ mẫu nhỏ (3) vì phần lớn câu trả lời không nêu con số nào.

> **Lần đo đầu ra groundedness = 0,000 và cả 5 dữ kiện bị gắn cờ "nghi bịa" —
> đó là lỗi của phép đo, không phải của hệ thống.** Message tool của LangChain
> có `content` là **danh sách phần tử** `[{"type":"text","text":...}]`, mà parser
> chỉ nhận chuỗi nên vứt sạch ngữ cảnh. Nay script ghi luôn `do_dai_ngu_canh`
> vào báo cáo và loại các mục không thu được ngữ cảnh khỏi phép tính kèm cảnh
> báo — một phép đo hỏng trông y hệt một hệ thống hỏng, phải phân biệt được.

### Lỗ hổng phép đo này tìm ra: trả lời câu ngoài miền bằng kiến thức chung

Cả 6 câu ngoài miền đều **không gọi công cụ nào** — nghe thì tốt, nhưng agent
lại **tự trả lời bằng kiến thức của mô hình**: công thức phở bò 864 ký tự, quy
trình thay dầu nhớt Honda Wave kèm số liệu. Các luật chống bịa trong
`supervisor_prompt.md` đều gắn với "sau khi tool trả về", nên không chạm tới
trường hợp này.

Đây là vi phạm BR-704/TC-704, và cũng là một dạng bịa: người đọc tưởng đó là
thông tin của CLB, mà không có nguồn nào để kiểm lại.

**Đã vá** bằng mục `II.1. PHẠM VI` trong `supervisor_prompt.md` — chặn **trước
cả việc gọi tool**, và nói rõ phải từ chối *kể cả khi mô hình biết câu trả lời*.
Đo lại sau khi vá:

| Câu hỏi | Trước | Sau |
|---|---|---|
| cách nấu phở bò Nam Định | 864 ký tự công thức | 207 ký tự từ chối |
| thay dầu nhớt Honda Wave | 778 ký tự hướng dẫn | 232 ký tự từ chối |
| giá vàng SJC hôm nay | 243 ký tự | 195 ký tự từ chối |

Có test canh: `ChatBot/tests/test_prompt_supervisor.py::test_co_luat_tu_choi_cau_ngoai_pham_vi`.

## 9. Tái lập

```bash
python scripts/build_eval_labels.py            # xem trước nhãn
python scripts/build_eval_labels.py --apply    # ghi data/eval/eval_queries.json
python scripts/run_eval.py                     # chạy, ghi data/eval/report.json
python scripts/run_eval_noiquy.py              # TC-606, cần LangGraph + MCP + API chạy
python scripts/demo_checklist.py               # checklist demo, exit != 0 nếu có FAIL
cd API && pytest tests/test_eval_metrics.py    # 14 test, chốt TC-602 "khớp tính tay"
```

Cần một nguồn nhúng đang sống (pod GPU hoặc API Jina) — xem
`embedding_server/README.md`. Nhãn cố định trong `eval_queries.json`, nên chỉ
phải chạy lại `build_eval_labels.py` khi kho thay đổi.
