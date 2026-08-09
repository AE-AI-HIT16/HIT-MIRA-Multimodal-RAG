# Azure text embedding baseline — provisional

**Ngày chạy:** 05/08/2026  
**Provider:** Azure OpenAI `text-embedding-3-small`, 1.536 chiều  
**Corpus:** 505 message có nội dung trên tổng 514 post  
**Benchmark:** 30 ca dương + 5 ca ngoài phạm vi; TXT-18 tách sang metadata filter  

> Trạng thái: **tạm thời**. 76 qrel hiện tại đều đúng nhưng chưa đầy đủ. Recall/MRR/nDCG chỉ được coi là chính thức sau khi hợp pool Azure + Jina và blind adjudication.

## Kết quả chính

| Tập | n | Recall@5 | MRR@10 | nDCG@10 |
|---|---:|---:|---:|---:|
| Vector-text chính | 29 | 0.565 | 0.626 | 0.594 |
| Bỏ 2 câu title-overlap cao | 27 | 0.532 | 0.599 | 0.565 |
| TXT-18 metadata-filter | 1 | 0.000 | 0.000 | 0.000 |

Ngưỡng tài liệu: Recall@5 ≥ 0,80; MRR ≥ 0,60. Baseline tạm thời có Recall chưa đạt và MRR vừa đạt, nhưng chưa được dùng để chốt topology do qrel chưa exhaustive.

## Theo độ phủ token

| Coverage | n | Recall@5 | MRR@10 | nDCG@10 |
|---|---:|---:|---:|---:|
| all | 21 | 0.591 | 0.574 | 0.572 |
| partial | 5 | 0.592 | 0.800 | 0.727 |
| unparsed | 3 | 0.333 | 0.700 | 0.520 |

## Độ trễ

- Request đầu: 5.791s
- Query nóng p50: 0.332s
- Query nóng p95: 0.475s
- Nhúng 505 document: 23.446s (21.54 document/s)

## Năm câu ngoài phạm vi

| ID | Top cosine | Top post ID |
|---|---:|---|
| OOD-01 | 0.510319 | `213347192060163_1210859354195881` |
| OOD-02 | 0.400674 | `213347192060163_882494550365698` |
| OOD-03 | 0.509746 | `213347192060163_782938820321272` |
| OOD-04 | 0.353061 | `213347192060163_1155940409687776` |
| OOD-05 | 0.449806 | `213347192060163_865566558725164` |

Phân bố top-score: min 0.353061; p50 0.449806; p95/max 0.510319. Không chọn threshold từ năm câu này.

## Vì sao metric hiện chỉ là provisional

- TXT-03 trả một bài GEN 15 đúng ở hạng 1 nhưng ID đó chưa có trong qrel, nên bị chấm Recall@5 = 0.
- TXT-04 cũng trả nhiều bài HIT-16 đúng nhưng chưa nằm trong ba qrel ban đầu.
- Đây là lỗi thiếu độ phủ nhãn, không phải nhãn hiện tại sai.
- Không bổ sung nhãn theo riêng Azure vì sẽ thiên vị model. Cần hợp top-10 Azure + Jina, xáo thứ tự và review mù.

## Artifact

- Vector cache: `data/eval/cache/text-azure-24eec9999fb45712.npz`
- Raw result: `data/eval/results/text-embedding-b6761da66aed3621.json`
- Review pool: `data/eval/review_pools/text-retrieval-pool-v1.json`
