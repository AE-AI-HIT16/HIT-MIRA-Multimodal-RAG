#!/usr/bin/env bash
# Quay lại một bản backup do scripts/backup_stack.sh tạo ra.
#
# XOÁ SẠCH dữ liệu hiện tại của postgres + qdrant + minio rồi ghi đè bằng bản
# backup. Theo quy ước repo: mặc định chạy khô, phải có --apply mới làm thật.
#
# Xem trước:  bash scripts/restore_stack.sh /home/ubuntu/backups/hit-mira/<ten>
# Làm thật:   bash scripts/restore_stack.sh /home/ubuntu/backups/hit-mira/<ten> --apply
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT="hit-mira-multimodal-rag"

DICH="${1:-}"
APPLY=0
[ "${2:-}" = "--apply" ] && APPLY=1
if [ -z "$DICH" ]; then
  echo "Dùng: bash scripts/restore_stack.sh <thư-mục-backup> [--apply]" >&2
  echo "Các bản có sẵn:" >&2
  ls -1 "${BACKUP_ROOT:-/home/ubuntu/backups/hit-mira}" 2>/dev/null | sed 's/^/  /' >&2
  exit 2
fi
DICH="$(cd "$DICH" && pwd)"

DOCKER="docker"
if ! docker ps >/dev/null 2>&1; then DOCKER="sudo docker"; fi
SUDO=""
if [ "$DOCKER" = "sudo docker" ]; then SUDO="sudo"; fi

echo "=== Phục hồi từ $DICH ==="
[ -f "$DICH/manifest.txt" ] && cat "$DICH/manifest.txt"
echo

# Kiểm tra tar đọc được TRƯỚC khi xoá gì cả. Phát hiện tar hỏng sau khi đã xoá
# volume thì không còn đường lui nào nữa.
echo "--- kiểm tra bản backup ---"
for v in pgdata qdrant_storage minio_data; do
  f="$DICH/${v}.tar.gz"; [ -e "$f" ] || f="$DICH/${v}.tar"
  if [ ! -e "$f" ]; then echo "THIẾU: ${v}.tar[.gz]" >&2; exit 1; fi
  tar -tf "$f" >/dev/null || { echo "HỎNG: $f" >&2; exit 1; }
  echo "  OK  $(basename "$f")  ($(du -h "$f" | cut -f1))"
done

if [ "$APPLY" = "0" ]; then
  echo
  echo "[CHẠY KHÔ] Sẽ làm những việc sau nếu thêm --apply:"
  echo "  1. dừng postgres, qdrant, minio"
  echo "  2. với mỗi volume: XOÁ SẠCH rồi giải nén bản backup vào đó"
  for v in pgdata qdrant_storage minio_data; do
    d="/var/lib/docker/volumes/${PROJECT}_${v}/_data"
    echo "       $d  ($($SUDO du -sh "$d" 2>/dev/null | cut -f1) hiện tại)  ←  ${v}.tar*"
  done
  echo "  3. bật lại dịch vụ và đếm lại để đối chiếu với manifest"
  exit 0
fi

echo
echo "--- dừng dịch vụ ---"
$DOCKER compose -f "$REPO_DIR/docker-compose.yml" stop postgres qdrant minio

for v in pgdata qdrant_storage minio_data; do
  d="/var/lib/docker/volumes/${PROJECT}_${v}/_data"
  f="$DICH/${v}.tar.gz"; [ -e "$f" ] || f="$DICH/${v}.tar"
  echo "--- $v: xoá + giải nén ---"
  # -mindepth 1 để giữ lại chính thư mục _data (docker đang mount nó), và -delete
  # quét cả file ẩn — `rm -rf $d/*` bỏ sót .minio.sys, phục hồi xong minio mất
  # toàn bộ metadata bucket.
  $SUDO find "$d" -mindepth 1 -delete
  $SUDO tar -xf "$f" -C "$d"
done

echo "--- bật lại dịch vụ ---"
$DOCKER compose -f "$REPO_DIR/docker-compose.yml" start postgres qdrant minio

echo "--- chờ healthy ---"
for i in $(seq 1 60); do
  trang_thai=$($DOCKER compose -f "$REPO_DIR/docker-compose.yml" ps --format '{{.Service}} {{.Health}}' 2>/dev/null | grep -cE '(postgres|qdrant|minio) healthy' || true)
  [ "$trang_thai" = "3" ] && break
  sleep 2
done

doc_env() { sed -nE "s/^$1=(.*)$/\1/p" "$REPO_DIR/.env" | tail -1; }
QURL="$(doc_env QDRANT_URL)"; QKEY="$(doc_env QDRANT_API_KEY)"

echo
echo "=== Sau phục hồi (đối chiếu với manifest ở trên) ==="
echo "[qdrant]"
for c in $(curl -sf -H "api-key: $QKEY" "$QURL/collections" | jq -r '.result.collections[].name'); do
  n=$(curl -sf -H "api-key: $QKEY" "$QURL/collections/$c" | jq -r '.result.points_count')
  echo "  $c: $n"
done
echo "[postgres]"
$DOCKER compose -f "$REPO_DIR/docker-compose.yml" exec -T postgres \
  psql -U "$(doc_env POSTGRES_USER)" -d "$(doc_env POSTGRES_DB)" -At -c "
    select relname || ':' || (xpath('/row/c/text()',
             query_to_xml(format('select count(*) as c from %I.%I', schemaname, relname),
                          false, true, '')))[1]::text
    from pg_stat_user_tables order by relname;
  " 2>/dev/null | sed 's/^/  /'
