#!/usr/bin/env bash
# Archive các card BR-* cũ (Draft item) trong GitHub Project #1.
# Chỉ đụng Draft có title bắt đầu "BR-" → KHÔNG ảnh hưởng 38 issue T-xx.
# Archive có thể khôi phục lại trong Project (không phải xoá vĩnh viễn).
#
# Cần scope project (một lần):   gh auth refresh -s project
# Xem trước (an toàn):           bash scripts/archive_br_cards.sh
# Archive thật:                  DRY_RUN=0 bash scripts/archive_br_cards.sh
set -euo pipefail

PROJECT_OWNER="${PROJECT_OWNER:-AE-AI-HIT16}"
PROJECT_NUMBER="${PROJECT_NUMBER:-1}"
DRY_RUN="${DRY_RUN:-1}"

echo "PROJECT=$PROJECT_OWNER/$PROJECT_NUMBER  DRY_RUN=$DRY_RUN"
echo

count=0
while IFS=$'\t' read -r id title; do
  [ -z "${id// }" ] && continue
  count=$((count + 1))
  if [ "$DRY_RUN" = "1" ]; then
    echo "DRY archive: $title   ($id)"
    continue
  fi
  if gh project item-archive "$PROJECT_NUMBER" --owner "$PROJECT_OWNER" --id "$id" </dev/null >/dev/null 2>&1; then
    echo "archived: $title"
  else
    echo "LỖI (kiểm tra scope 'project'): $title"
  fi
done < <(
  gh project item-list "$PROJECT_NUMBER" --owner "$PROJECT_OWNER" --limit 200 --format json \
    --jq '.items[] | select(.content.type=="DraftIssue") | select(.title|test("^BR-")) | "\(.id)\t\(.title)"'
)

echo
echo "Tổng card BR khớp: $count"
[ "$DRY_RUN" = "1" ] && echo "DRY_RUN=1 → chưa archive gì. Chạy thật: DRY_RUN=0 bash scripts/archive_br_cards.sh"
