# Specs theo Phase

Spec chi tiết từng chức năng của mỗi Phase (API contract · chữ ký hàm · hành vi & edge · DoD/test).
Bản đồ tổng quan: [../phases.md](../phases.md) · Task board (theo dõi tiến độ ở đây): [../tasks.md](../tasks.md) · AC gốc: [../prd.md](../prd.md)

| Phase | File | Nội dung | Exit gate |
|---|---|---|---|
| P0 | [phase-0-foundation.md](phase-0-foundation.md) | docker · Alembic · 19 bảng ORM · CI · 4 provider · DI wiring | compose up healthy + CI xanh + provider chạy thật |
| P1 | [phase-1-ingest-pipeline.md](phase-1-ingest-pipeline.md) | consent · storage · ingest · nội quy · keyframe · caption · embed · index · ASR · orchestrator | 1 video thật → vector trong 3 collection; "Điều X" truy đúng |
| P2 | [phase-2-retrieval-router.md](phase-2-retrieval-router.md) | 3 nhánh retrieve · rank ngưỡng · tools/registry · router luật + classifier | route đúng nguồn; top-k đúng payload; ngưỡng cắt "không tìm thấy" |
| P3 | [phase-3-answer-chat.md](phase-3-answer-chat.md) | synthesize · answer_regulation · not_found · **handle_message** · auth · clip/stream · lưu hội thoại | `/chat/message` end-to-end, có citation/disclaimer, không bịa |
| P4 | [phase-4-frontend-eval.md](phase-4-frontend-eval.md) | web chat · render đa phương thức · admin · eval set · Recall/MRR · latency · eval routing · demo | web chạy + báo cáo số đo + 4 luồng demo pass |

**Đường găng:** T-05 → T-14 → T-30 → T-41 → T-50 → T-60 → T-74 — trễ 1 mắt xích là trễ cả dự án; ưu tiên khi phân việc sprint.
