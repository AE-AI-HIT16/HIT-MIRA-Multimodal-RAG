#!/usr/bin/env bash
# Tạo issues task lên GitHub + đẩy vào Project HIT16.
# An toàn: mặc định DRY_RUN=1 (chỉ in, không tạo gì). Xem OK rồi chạy thật:
#     DRY_RUN=0 REPO=AE-AI-HIT16/<repo> bash scripts/create_issues.sh
# Yêu cầu: gh đã `gh auth login` với quyền repo + project.
set -euo pipefail

# ===== CẤU HÌNH — SỬA CHO ĐÚNG =====
REPO="${REPO:-AE-AI-HIT16/HIT-MIRA-Multimodal-RAG}"   # owner/repo chứa issue
PROJECT_OWNER="${PROJECT_OWNER:-AE-AI-HIT16}"          # org sở hữu Project
PROJECT_NUMBER="${PROJECT_NUMBER:-1}"                  # số Project (URL .../projects/1)
DRY_RUN="${DRY_RUN:-1}"                                # 1=chỉ in · 0=chạy thật

echo "REPO=$REPO  PROJECT=$PROJECT_OWNER/$PROJECT_NUMBER  DRY_RUN=$DRY_RUN"
echo

# ===== LABELS =====
create_label() {
  if [ "$DRY_RUN" = "1" ]; then echo "DRY label: $1"; return; fi
  gh label create "$1" --repo "$REPO" --color "$2" --force >/dev/null 2>&1 || true
}
while read -r name color; do
  [ -z "${name// }" ] && continue
  create_label "$name" "$color"
done <<'LABELS'
role/DE          1f6feb
role/AIE-1       0e8a16
role/AIE-2       8250df
role/shared      6e7681
epic/infra       d4c5f9
epic/data        d4c5f9
epic/pipeline    d4c5f9
epic/providers   d4c5f9
epic/retrieval   d4c5f9
epic/router-answer d4c5f9
epic/chat-app    d4c5f9
epic/frontend    d4c5f9
epic/eval        d4c5f9
prio/must        b60205
prio/should      fbca04
sprint/1         ededed
sprint/2         ededed
sprint/3         ededed
sprint/4         ededed
sprint/5         ededed
LABELS
echo

# ===== ISSUES ===== (định dạng: ID|Title|labels|body)
while IFS='|' read -r id title labels body; do
  [ -z "${id// }" ] && continue
  case "$id" in \#*) continue ;; esac
  labelargs=()
  IFS=',' read -ra L <<< "$labels"
  for l in "${L[@]}"; do labelargs+=(--label "$l"); done
  full="$id · $title"
  if [ "$DRY_RUN" = "1" ]; then
    echo "DRY issue: $full   [${labels}]"
    continue
  fi
  url=$(gh issue create --repo "$REPO" --title "$full" --body "$body" "${labelargs[@]}" </dev/null)
  echo "created: $url"
  if gh project item-add "$PROJECT_NUMBER" --owner "$PROJECT_OWNER" --url "$url" </dev/null >/dev/null 2>&1; then
    echo "  → added to project"
  else
    echo "  (chưa thêm vào project — cần scope 'project': gh auth refresh -s project)"
  fi
done <<'TASKS'
T-01|docker-compose postgres+qdrant|role/DE,epic/infra,prio/must,sprint/1|Owner DE · docker-compose.yml · DoD: `docker compose up` 2 service healthy + app /health OK
T-02|Alembic init + migrate models|role/DE,epic/infra,prio/must,sprint/2|Owner DE · shared/db/migrations · depends T-05 · DoD: alembic upgrade head dựng đủ bảng
T-03|CI ruff + pytest|role/AIE-2,epic/infra,prio/should,sprint/1|Owner AIE-2 · .github/workflows · DoD: PR chạy lint+test tự động
T-05|18 bảng ORM còn lại (PRD §5)|role/DE,epic/data,prio/must,sprint/1|Owner DE · shared/db/models.py · DoD: create_all + FK/relationship đúng · ref PRD §5
T-06|Storage lưu/đọc media theo id|role/DE,epic/data,prio/must,sprint/2|Owner DE · domains/media get_media · depends T-05 · DoD: TC-105 (id lạ→404)
T-07|Ingest upload + validate + gate consent|role/DE,epic/data,prio/must,sprint/2|Owner DE · domains/ingest save_upload · depends T-05 · DoD: TC-102
T-08|Nạp nội quy tách điều/khoản|role/DE,epic/data,prio/must,sprint/2|Owner DE · domains/ingest load_regulations · depends T-05 · DoD: TC-107
T-10|Trích keyframe + timestamp (ffmpeg)|role/DE,epic/pipeline,prio/must,sprint/2|Owner DE · pipeline/frames.py · DoD: TC-201 test_frames_have_timestamp
T-11|ASR transcript theo timestamp|role/DE,epic/pipeline,prio/must,sprint/2|Owner DE · pipeline/asr.py · depends T-21 · DoD: TC-208
T-12|Caption ảnh/frame|role/DE,epic/pipeline,prio/should,sprint/2|Owner DE · pipeline/caption.py · depends T-22 · DoD: TC-203
T-13|Sinh embedding ảnh/text|role/DE,epic/pipeline,prio/must,sprint/2|Owner DE · pipeline/embed.py · depends T-20 · DoD: test_embed_same_dim
T-14|Build/upsert Qdrant 3 collection|role/DE,epic/pipeline,prio/must,sprint/2|Owner DE · pipeline/index.py · depends T-13 · DoD: test_index_upsert (sai chiều→từ chối)
T-15|Orchestrator run.py|role/DE,epic/pipeline,prio/must,sprint/3|Owner DE · pipeline/run.py · depends T-10..14 · DoD: 1 video chạy trọn, lỗi 1 mục không chặn batch
T-20|Embeddings CLIP ảnh + text VN|role/AIE-1,epic/providers,prio/must,sprint/1|Owner AIE-1 · providers/embeddings.py · DoD: embed cùng dim, text→ảnh liên quan
T-21|ASR faster-whisper VN|role/AIE-2,epic/providers,prio/must,sprint/1|Owner AIE-2 · providers/asr.py · DoD: audio→segment text+timestamp
T-22|Captioner BLIP-2/VLM|role/AIE-2,epic/providers,prio/should,sprint/2|Owner AIE-2 · providers/captioner.py · DoD: ảnh→caption tiếng Việt
T-23|LLM Gemini free-tier|role/AIE-2,epic/providers,prio/must,sprint/1|Owner AIE-2 · providers/llm.py · DoD: generate(prompt) trả text ổn định
T-30|retrieve_media (embed→search)|role/AIE-1,epic/retrieval,prio/must,sprint/3|Owner AIE-1 · domains/retrieval · depends T-14,T-20 · DoD: TC-301 test_recall_at_k
T-31|retrieve_regulations|role/AIE-1,epic/retrieval,prio/must,sprint/3|Owner AIE-1 · domains/retrieval · depends T-08,T-14 · DoD: TC-307
T-32|retrieve_by_transcript + gộp keyframe|role/AIE-1,epic/retrieval,prio/must,sprint/3|Owner AIE-1 · domains/retrieval · depends T-11,T-14 · DoD: TC-308 (không nhân đôi video)
T-33|rank + ngưỡng không tìm thấy|role/AIE-1,epic/retrieval,prio/must,sprint/3|Owner AIE-1 · domains/retrieval · depends T-30 · DoD: TC-306
T-34|MediaTool + RegulationTool register|role/AIE-1,epic/retrieval,prio/must,sprint/3|Owner AIE-1 · tools/* · depends T-30,T-31 · DoD: tool.run trả ToolResult đúng nguồn
T-40|Router classifier fallback|role/AIE-2,epic/router-answer,prio/must,sprint/3|Owner AIE-2 · routing/intent.py · DoD: test_ambiguous_uses_classifier
T-41|synthesize_answer RAG + trích dẫn|role/AIE-2,epic/router-answer,prio/must,sprint/4|Owner AIE-2 · domains/answer · depends T-23,T-30 · DoD: bám nguồn; LLM lỗi→fallback thô
T-42|answer_regulation + disclaimer|role/AIE-2,epic/router-answer,prio/must,sprint/4|Owner AIE-2 · domains/answer · depends T-23,T-31 · DoD: TC-407 citation+disclaimer
T-43|handle_not_found|role/AIE-2,epic/router-answer,prio/must,sprint/4|Owner AIE-2 · domains/answer · DoD: TC-406 (ngoài miền→không bịa)
T-50|chat handle_message + /chat/message|role/AIE-2,epic/chat-app,prio/must,sprint/4|Owner AIE-2 · domains/chat · depends T-34,T-40,T-41,T-42 · DoD: TC-507 định tuyến+override
T-51|Auth login + JWT + role|role/AIE-2,epic/chat-app,prio/must,sprint/1|Owner AIE-2 · domains/auth · depends T-05 · DoD: TC-505, non-admin→403
T-52|Media clip + stream/seek|role/AIE-2,epic/chat-app,prio/should,sprint/4|Owner AIE-2 · domains/media · depends T-06,T-10 · DoD: TC-403/404
T-53|Lưu conversations/messages|role/AIE-2,epic/chat-app,prio/should,sprint/4|Owner AIE-2 · domains/chat · depends T-05 · DoD: TC-501
T-60|Init Next.js + khung chat|role/AIE-2,epic/frontend,prio/must,sprint/5|Owner AIE-2 · web/ · depends T-50 · DoD: TC-501/502/505
T-61|Render kết quả đa phương thức inline|role/shared,epic/frontend,prio/must,sprint/5|Owner shared · web/ · depends T-52,T-60 · DoD: TC-503 (ảnh+clip+link)
T-62|Màn admin (nạp/pipeline/eval)|role/DE,epic/frontend,prio/should,sprint/5|Owner DE · web/ · depends T-07,T-71 · DoD: thao tác nạp/chạy pipeline/xem eval
T-70|Bộ eval_queries có nhãn|role/AIE-1,epic/eval,prio/must,sprint/5|Owner AIE-1 · data/eval + models · depends T-05 · DoD: TC-601 (≥N truy vấn có đáp án)
T-71|Recall@k + MRR|role/AIE-1,epic/eval,prio/must,sprint/5|Owner AIE-1 · domains/eval · depends T-30,T-70 · DoD: TC-602 test_recall_and_mrr_match_by_hand
T-72|Đo latency avg/p95|role/AIE-1,epic/eval,prio/should,sprint/5|Owner AIE-1 · domains/eval · depends T-50 · DoD: TC-603 (avg ≤5s + p95)
T-73|Eval nội quy 3 chỉ số|role/AIE-1,epic/eval,prio/should,sprint/5|Owner AIE-1 · domains/eval · depends T-40,T-42 · DoD: TC-606
T-74|Checklist demo end-to-end|role/shared,epic/eval,prio/must,sprint/5|Owner all · docs/ · depends E4-E7 · DoD: TC-605 (4 luồng lõi pass)
TASKS
echo
echo "Xong. Nếu DRY_RUN=1: chưa tạo gì. Chạy thật: DRY_RUN=0 REPO=<owner/repo> bash scripts/create_issues.sh"
