#!/usr/bin/env bash
# Đưa các issue T-xx đã tạo vào GitHub Project #1.
# Chạy SAU khi cấp scope project:   gh auth refresh -s project
# Rồi:                              bash scripts/add_issues_to_project.sh
# An toàn: item-add idempotent (issue đã ở project sẽ không bị nhân đôi).
set -euo pipefail

REPO="${REPO:-AE-AI-HIT16/HIT-MIRA-Multimodal-RAG}"
PROJECT_OWNER="${PROJECT_OWNER:-AE-AI-HIT16}"
PROJECT_NUMBER="${PROJECT_NUMBER:-1}"

gh issue list --repo "$REPO" --state open --limit 200 --json title,url \
  --jq '.[] | select(.title|test("^T-[0-9]")) | .url' | while read -r url; do
    if gh project item-add "$PROJECT_NUMBER" --owner "$PROJECT_OWNER" --url "$url" </dev/null >/dev/null 2>&1; then
      echo "added: $url"
    else
      echo "LỖI (kiểm tra scope 'project'): $url"
    fi
done
