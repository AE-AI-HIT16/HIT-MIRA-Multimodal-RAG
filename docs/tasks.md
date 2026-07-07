# Task Board — HIT-MIRA Multimodal RAG (Capstone)

> Thiết kế lại từ "1 card = 1 BR" → **task làm được**. Mỗi task: actionable · 1 owner ·
> có test nghiệm thu (DoD) · có phụ thuộc. Bám đúng code base (`api/`) + PRD (`US-xxx`, `TC-xxx`).
> Team: **3 người** — `DE` (Data Engineer) · `AIE-1` (Retrieval/Search) · `AIE-2` (RAG/App).

## 0. Nguyên tắc task
- **Task ≠ BR.** BR là mục tiêu; task là 1 đơn vị code ~0.5–2 ngày, kéo được sang Done.
- **DoD = test xanh.** Mỗi task chốt bằng 1 `TC-xxx` (hoặc test viết mới) trong `api/tests/`.
- **Trạng thái:** Todo → In Progress → Done. Chỉ sang Done khi test pass + PR merge.
- **Ưu tiên:** `Must` (v1 lõi) · `Should` (v1 phụ) · `v2` (backlog).

## 1. Phân vai (3 người)
| Vai | Phụ trách | Vùng code |
|---|---|---|
| **DE** | Dữ liệu & pipeline offline | `domains/{consent,ingest}` · `shared/db` · `pipeline/*` · infra |
| **AIE-1** | Truy xuất & đánh giá | `providers/embeddings` · `domains/retrieval` · `tools/*` · `domains/eval` |
| **AIE-2** | Sinh câu trả lời & app | `providers/{asr,captioner,llm}` · `routing` · `domains/{answer,chat,auth,media}` · `web/` |

> Frontend (`web/`) do AIE-2 chủ trì, DE/AIE-1 hỗ trợ màn admin.

## 2. Bảng task theo Epic

### E0 · Infra & Setup
| ID | Task | Owner | File | Depends | DoD / Test | Prio | Sprint |
|---|---|---|---|---|---|---|---|
| T-01 | docker-compose: postgres + qdrant + volumes | DE | `docker-compose.yml` | — | `docker compose up` → 2 service healthy, app `/health` OK | Must | 1 |
| T-02 | Alembic init + tạo bảng từ models | DE | `shared/db/migrations/` | T-05 | `alembic upgrade head` dựng đủ bảng | Must | 2 |
| T-03 | CI: ruff + pytest (GitHub Actions) | AIE-2 | `.github/workflows/` | — | PR chạy lint + test tự động | Should | 1 |

### E1 · Data foundation (DE)
| ID | Task | Owner | File | Depends | DoD / Test | Prio | Sprint |
|---|---|---|---|---|---|---|---|
| ✅ T-04 | Domain mẫu `consent` + gate | DE | `domains/consent/*` | — | `test_consent.py` (đã xanh) | Must | ✔ done |
| T-05 | 18 bảng ORM còn lại (PRD §5) | DE | `shared/db/models.py` | — | `create_all` chạy, FK/relationship đúng | Must | 1 |
| T-06 | Storage layer lưu/đọc media theo id | DE | `domains/media/service.py::get_media` | T-05 | `TC-105`: GET đúng file+metadata, id lạ→404 | Must | 2 |
| T-07 | Ingest upload + validate metadata + gate consent | DE | `domains/ingest/service.py::save_upload` | T-05 | `TC-102`: batch vào kho, thiếu trường→chặn | Must | 2 |
| T-08 | Nạp nội quy → tách điều/khoản | DE | `domains/ingest/service.py::load_regulations` | T-05 | `TC-107`: truy "Điều X" đúng nội dung | Must | 2 |

### E2 · Offline pipeline (DE, dùng provider của AIE)
| ID | Task | Owner | File | Depends | DoD / Test | Prio | Sprint |
|---|---|---|---|---|---|---|---|
| T-10 | Trích keyframe + timestamp (ffmpeg) | DE | `pipeline/frames.py` | — | `TC-201` `test_frames_have_timestamp` | Must | 2 |
| T-11 | ASR transcript theo timestamp | DE | `pipeline/asr.py` | T-21 | `TC-208` `test_transcript_segments` | Must | 2 |
| T-12 | Caption cho ảnh/frame | DE | `pipeline/caption.py` | T-22 | `TC-203`: ≥80% caption hợp lý | Should | 2 |
| T-13 | Sinh embedding ảnh/text | DE | `pipeline/embed.py` | T-20 | `test_embed_same_dim` | Must | 2 |
| T-14 | Build/upsert Qdrant (3 collection) | DE | `pipeline/index.py` | T-13 | `test_index_upsert`; sai chiều→từ chối | Must | 2 |
| T-15 | Orchestrator `run.py` ghép cả pipeline | DE | `pipeline/run.py` | T-10..14 | 1 video → frame+transcript+embed+index; lỗi 1 mục không chặn batch | Must | 3 |

### E3 · Providers / models (AIE)
| ID | Task | Owner | File | Depends | DoD / Test | Prio | Sprint |
|---|---|---|---|---|---|---|---|
| T-20 | Embeddings: CLIP ảnh + text VN | AIE-1 | `providers/embeddings.py` | — | embed cùng dim; query text→ảnh liên quan | Must | 1 |
| T-21 | ASR: faster-whisper (VN) | AIE-2 | `providers/asr.py` | — | audio mẫu → segment có text+timestamp | Must | 1 |
| T-22 | Captioner: BLIP-2/VLM | AIE-2 | `providers/captioner.py` | — | ảnh mẫu → caption tiếng Việt | Should | 2 |
| T-23 | LLM: Gemini free-tier | AIE-2 | `providers/llm.py` | — | `generate(prompt)` trả text ổn định | Must | 1 |

### E4 · Retrieval (AIE-1)
| ID | Task | Owner | File | Depends | DoD / Test | Prio | Sprint |
|---|---|---|---|---|---|---|---|
| T-30 | `retrieve_media` (embed→search) | AIE-1 | `domains/retrieval/service.py` | T-14,T-20 | `TC-301` `test_recall_at_k` | Must | 3 |
| T-31 | `retrieve_regulations` | AIE-1 | `domains/retrieval/service.py` | T-08,T-14 | `TC-307`: điều/khoản đúng top-k | Must | 3 |
| T-32 | `retrieve_by_transcript` + gộp keyframe theo video_id | AIE-1 | `domains/retrieval/service.py` | T-11,T-14 | `TC-308`: đúng video+đoạn, không nhân đôi | Must | 3 |
| T-33 | `rank` + ngưỡng "không tìm thấy" | AIE-1 | `domains/retrieval/service.py` | T-30 | `TC-306` `test_ranking_and_threshold` | Must | 3 |
| T-34 | MediaTool + RegulationTool + register | AIE-1 | `tools/*` | T-30,T-31 | tool.run trả `ToolResult` đúng nhãn nguồn | Must | 3 |

### E5 · Router & Answer (AIE-2)
| ID | Task | Owner | File | Depends | DoD / Test | Prio | Sprint |
|---|---|---|---|---|---|---|---|
| T-40 | Classifier fallback cho router | AIE-2 | `routing/intent.py` | — | `test_ambiguous_uses_classifier` | Must | 3 |
| T-41 | `synthesize_answer` (RAG + trích dẫn) | AIE-2 | `domains/answer/service.py` | T-23,T-30 | bám nguồn; LLM lỗi→fallback thô | Must | 4 |
| T-42 | `answer_regulation` (neo điều/khoản + disclaimer) | AIE-2 | `domains/answer/service.py` | T-23,T-31 | `TC-407` citation+disclaimer | Must | 4 |
| T-43 | `handle_not_found` | AIE-2 | `domains/answer/service.py` | — | `TC-406`: ngoài miền→không bịa | Must | 4 |

### E6 · Chat & App (AIE-2)
| ID | Task | Owner | File | Depends | DoD / Test | Prio | Sprint |
|---|---|---|---|---|---|---|---|
| T-50 | `handle_message` orchestrate + `/chat/message` | AIE-2 | `domains/chat/*` | T-34,T-40,T-41,T-42 | `TC-507` định tuyến đúng + override | Must | 4 |
| T-51 | Auth: login + JWT + role | AIE-2 | `domains/auth/*` | T-05 | `TC-505`; non-admin→403 endpoint admin | Must | 1 |
| T-52 | Media serving: clip + stream/seek | AIE-2 | `domains/media/service.py` | T-06,T-10 | `TC-403`/`TC-404` clip ~6s + seek ±1s | Should | 4 |
| T-53 | Lưu conversations/messages | AIE-2 | `domains/chat/*` | T-05 | `TC-501`: gửi/nhận lưu message | Should | 4 |

### E7 · Frontend (AIE-2 + shared)
| ID | Task | Owner | File | Depends | DoD / Test | Prio | Sprint |
|---|---|---|---|---|---|---|---|
| T-60 | Init Next.js + khung chat (text + upload ảnh) | AIE-2 | `web/` | T-50 | `TC-501/502/505` chat gửi/nhận, upload ảnh | Must | 5 |
| T-61 | Render kết quả đa phương thức inline | shared | `web/` | T-52,T-60 | `TC-503`: ảnh+clip+link render/phát | Must | 5 |
| T-62 | Màn admin (nạp dữ liệu · pipeline · đánh giá) | DE | `web/` | T-07,T-71 | thao tác nạp/chạy pipeline/xem eval | Should | 5 |

### E8 · Eval & QA (AIE-1)
| ID | Task | Owner | File | Depends | DoD / Test | Prio | Sprint |
|---|---|---|---|---|---|---|---|
| T-70 | Bộ `eval_queries` có nhãn | AIE-1 | `data/eval/` + `models` | T-05 | `TC-601`: ≥N truy vấn có đáp án | Must | 5 |
| T-71 | Recall@k + MRR | AIE-1 | `domains/eval/service.py` | T-30,T-70 | `TC-602` `test_recall_and_mrr_match_by_hand` | Must | 5 |
| T-72 | Đo latency (avg, p95) | AIE-1 | `domains/eval/service.py` | T-50 | `TC-603`: avg ≤5s có p95 | Should | 5 |
| T-73 | Eval nội quy (điều khoản/groundedness/routing) | AIE-1 | `domains/eval/service.py` | T-40,T-42 | `TC-606`: 3 chỉ số | Should | 5 |
| T-74 | Checklist demo end-to-end | all | `docs/` | E4–E7 | `TC-605`: 4 luồng lõi pass | Must | 5 |

## 3. Phụ thuộc & Kế hoạch Sprint (5 sprint · ~2 tuần)

```
Sprint 1  Nền móng (song song 3 vai)
  DE : T-01 infra · T-05 models
  AIE-1: T-20 embeddings
  AIE-2: T-21 ASR · T-23 LLM · T-51 auth
Sprint 2  Pipeline dữ liệu (DE nặng)
  DE : T-06 storage · T-07 ingest · T-08 nội quy · T-10..14 pipeline · T-02 alembic
  AIE: T-22 caption · hoàn thiện provider
Sprint 3  Truy xuất + Router
  DE : T-15 orchestrator
  AIE-1: T-30..34 retrieval + tools
  AIE-2: T-40 classifier
Sprint 4  RAG + Chat
  AIE-2: T-41..43 answer · T-50 chat · T-52 media · T-53 messages
Sprint 5  Frontend + Đánh giá + Demo
  AIE-2/shared: T-60..62 web
  AIE-1: T-70..73 eval
  all : T-74 demo checklist
```

Đường găng (critical path): `T-05 → T-14 → T-30 → T-41 → T-50 → T-60 → T-74`.

## 4. Set up GitHub Project
- **Cột:** Todo · In Progress · Done (giữ nguyên).
- **Custom fields nên thêm:** `Owner` (DE/AIE-1/AIE-2) · `Epic` (E0–E8) · `Sprint` (1–5) · `Priority` (Must/Should/v2).
- **Labels:** `role/*`, `epic/*`, `prio/*` — script `scripts/create_issues.sh` tạo sẵn.
- Xoá/lưu trữ 35 card BR cũ; thay bằng issue task từ script (mỗi issue link về BR/US trong body).

## 5. Backlog v2 (chưa tạo issue)
Crawl tự động (BR-103) · Khử trùng lặp (BR-106) · OCR (BR-204) · Index tăng tiến (BR-206) ·
Face search (BR-304) · Lọc thời gian/sự kiện (BR-305) · Ngữ cảnh hội thoại (BR-504) ·
Privacy opt-out/removal/log (BR-702/703/705) · **Nâng Router → Agentic RAG** (phân rã/multi-hop).
