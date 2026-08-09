# Jina-CLIP v2 với Azure text-embedding-3-small — đo trên cùng một tập

**Ngày chạy:** 06/08/2026
**Corpus:** 505 message có nội dung trên tổng 514 post (`facebook_post_id` làm document id)
**Bộ câu hỏi:** `data/eval/text_embedding_benchmark_v1.json` — 30 ca dương + 5 ca ngoài phạm vi, 76 qrel
**Kết quả thô:** `data/eval/results/text-embedding-33f94cb3f7313847.json`

> Trạng thái: **tạm thời**, đúng như baseline Azure. 76 qrel hiện có đều đúng
> nhưng chưa đầy đủ, nên cả hai cột số dưới đây là **cận dưới**. Điều đã ngã ngũ
> là *thứ hạng giữa hai model*, không phải giá trị tuyệt đối.

## Kết quả chính

| Tập | n | Azure Recall@5 | Jina Recall@5 | Azure MRR@10 | Jina MRR@10 | Azure nDCG@10 | Jina nDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Vector-text chính | 29 | 0,565 | **0,688** | 0,626 | **0,728** | 0,594 | **0,693** |
| Bỏ 2 câu title-overlap cao | 27 | 0,532 | **0,665** | 0,599 | **0,708** | 0,565 | **0,674** |
| TXT-18 metadata-filter | 1 | 0,000 | 0,000 | 0,000 | 0,143 | 0,000 | 0,333 |

Jina hơn Azure ở cả ba chỉ số, và khoảng cách **không** đến từ hai câu trùng
tiêu đề: bỏ chúng đi thì cách biệt Recall@5 còn rộng hơn (+0,133 so với +0,123).

Theo độ phủ token, phần thắng nằm đúng ở nhóm khó nhất chứ không ở nhóm dễ:

| Coverage | n | Azure Recall@5 | Jina Recall@5 |
|---|---:|---:|---:|
| all | 21 | 0,591 | **0,730** |
| partial | 5 | **0,592** | 0,525 |
| unparsed | 3 | 0,333 | **0,667** |

`unparsed` là nhóm câu hỏi không dùng lại token đặc trưng nào của bài — đúng chỗ
mà so khớp từ khoá bó tay và chỉ ngữ nghĩa mới cứu được. Azure thắng ở `partial`
với n = 5, quá nhỏ để kết luận gì.

## Khe từ chối câu ngoài phạm vi

| Model | min top-1 của câu trong phạm vi | max top-1 của câu ngoài phạm vi | Khe | Câu trong phạm vi bị lọt xuống dưới |
|---|---:|---:|---:|---:|
| Azure | 0,5252 | 0,5103 | +0,0149 | 0/30 |
| Jina | 0,5647 | 0,5228 | **+0,0419** | 0/30 |

Cả hai đều tách sạch 35 câu, nhưng khe của Jina rộng gần gấp ba. Với hệ thống
mà "không được bịa" là ràng buộc cứng, đây là con số quan trọng ngang metric xếp
hạng: khe càng rộng thì ngưỡng "không tìm thấy" càng ít gãy khi corpus lớn lên.

**Vẫn không được chốt ngưỡng từ chính 5 câu này.** Chọn trên đây rồi báo điểm
trên đây là tự chấm bài mình.

## Độ trễ — đọc kỹ trước khi trích

| | Azure | Jina qua RunPod Serverless |
|---|---:|---:|
| Query đầu tiên | 5,79s | 2,48s |
| Query nóng p50 | 0,33s | 1,18s |
| Query nóng p95 | 0,48s | 2,39s |
| Nhúng 505 document | 23,4s (21,5 doc/s) | 54,7s (9,2 doc/s) |

Hai cảnh báo, cái thứ nhất quan trọng hơn cả bảng:

1. **2,48s KHÔNG phải cold start.** Worker vẫn còn ấm từ smoke test chạy ngay
   trước đó. Cold start thật đo được khoảng **190 giây** (xem
   [runpod-serverless-embedding.md](runpod-serverless-embedding.md)). Nói cách
   khác, bảng này đo model, không đo hạ tầng.
2. Số của Jina là **thời gian một job hàng đợi**, gồm cả một vòng hỏi trạng thái
   của adapter, không phải một lời gọi HTTP. So p50 1,18s với 0,33s của Azure là
   so hai kiến trúc triển khai khác nhau, không phải hai model.

Kết luận về đường online vì vậy **không đổi**: không cho truy vấn của người dùng
đi qua endpoint `active workers = 0`.

## Việc này thay đổi điều gì

`CLAUDE.md` đang ghi rằng "một model" và "text retrieval tốt nhất" là hai thứ
loại trừ nhau, vì họ CLIP vốn yếu nhất ở text-to-text. **Trên corpus này giả
định đó sai.** Model đa phương thức — thứ duy nhất nhúng được ảnh — đồng thời
cũng là model nhúng chữ tốt hơn.

Nếu kết quả đứng vững sau adjudication, phương án một không gian vector không
còn là đánh đổi "chấp nhận text yếu hơn để giữ một model", mà là lựa chọn tốt
hơn ở cả hai mặt. Quyết định D-07/§7 nên được chốt lại theo hướng đó.

## Việc này KHÔNG kết luận điều gì

- **Chỉ đo message của post.** Không đo caption, OCR, transcript, và không đo
  text→ảnh. Kết luận cho transcript cần một tập nhãn riêng sau khi ingest.
- **Chưa đo trên nội quy.** Nhánh `rag_documents` là corpus khác, ngưỡng xét lại
  ở `CLAUDE.md` (trên 50 chunk) vẫn còn nguyên hiệu lực.
- **qrel chưa đầy đủ.** Cả hai điểm đều là cận dưới. Baseline Azure đã nêu ví dụ
  TXT-03: trả đúng một bài GEN 15 ở hạng 1 nhưng bị chấm 0 vì ID chưa có trong
  qrel.
- **Pool adjudication chưa chấm.** Pool hợp cả hai model đã dựng ngày 06/08:
  `data/eval/review_pools/text-retrieval-pool-v2.json` — 30 câu hỏi, **451 lượt
  chấm**, chưa ai chấm. Bản v1 (05/08) chỉ có Azure và chính nó ghi "do not
  finalize from Azure-only pool"; con số dưới đây cho thấy vì sao.

  **Top-10 của hai model chỉ trùng nhau 35,7%.** 30,8% ứng viên chỉ Azure tìm
  ra, 30,8% chỉ Jina tìm ra, và 2,7% là nhãn cũ mà không model nào lôi lên
  top-10. Nghĩa là pool chỉ-Azure bỏ sót khoảng một phần ba số ứng viên cần xét,
  và toàn bộ phần bỏ sót nghiêng về phía model kia — đúng kiểu thiên vị mà
  pooling sinh ra để tránh.
- **n = 29.** Đủ để thấy một chênh lệch lớn, không đủ để tin vào chữ số thứ ba.

## Tái lập

```bash
cd /home/ubuntu/HIT-MIRA-Multimodal-RAG
python scripts/benchmark_text_embeddings.py --providers azure jina-runpod          # dry-run
python scripts/benchmark_text_embeddings.py --providers azure jina-runpod --apply  # gọi thật
```

Cache vector nằm ở `data/eval/cache/text-<provider>-24eec9999fb45712.npz`; còn
cache thì lần chạy sau không gọi lại provider và không tốn tiền. Fingerprint chỉ
băm đầu vào của provider, nên sửa nhãn không làm hỏng cache đã trả tiền.

## Artifact

- Vector cache: `data/eval/cache/text-azure-24eec9999fb45712.npz`,
  `data/eval/cache/text-jina-runpod-24eec9999fb45712.npz`
- Kết quả thô: `data/eval/results/text-embedding-33f94cb3f7313847.json`
- Baseline một provider: [azure-text-embedding-baseline.md](azure-text-embedding-baseline.md)
