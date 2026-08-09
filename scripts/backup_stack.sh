#!/usr/bin/env bash
# Sao lưu NGUỘI toàn bộ dữ liệu hạ tầng: postgres + qdrant + minio.
#
# Vì sao nguội (dừng container rồi mới tar) chứ không sao lưu nóng: ba dịch vụ
# này tham chiếu lẫn nhau — hàng `media` trong postgres trỏ tới object trong
# minio, point trong qdrant mang `unit_id` của hàng đó. Sao lưu nóng từng cái
# một cho ba mốc thời gian khác nhau, phục hồi xong là point trỏ vào object
# chưa kịp ghi. Dừng ~1-2 phút đổi lấy một mốc thời gian nhất quán cho cả ba.
#
# Chạy:      bash scripts/backup_stack.sh
#            bash scripts/backup_stack.sh --nhan truoc-ingest-thang-8
# Phục hồi:  bash scripts/restore_stack.sh <thư-mục-backup> --apply
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKUP_ROOT="${BACKUP_ROOT:-/home/ubuntu/backups/hit-mira}"
PROJECT="hit-mira-multimodal-rag"
VOLUMES=(pgdata qdrant_storage minio_data)

NHAN=""
while [ $# -gt 0 ]; do
  case "$1" in
    --nhan) NHAN="$2"; shift 2 ;;
    *) echo "Tham số lạ: $1" >&2; exit 2 ;;
  esac
done

# Docker ở máy này cần sudo; nơi khác thì không. Dò một lần thay vì bắt người
# chạy phải nhớ.
DOCKER="docker"
if ! docker ps >/dev/null 2>&1; then DOCKER="sudo docker"; fi
SUDO=""
if [ "$DOCKER" = "sudo docker" ]; then SUDO="sudo"; fi

TEN="$(date +%Y%m%d-%H%M%S)"
[ -n "$NHAN" ] && TEN="${TEN}-${NHAN}"
DICH="$BACKUP_ROOT/$TEN"
mkdir -p "$DICH"

echo "=== Sao lưu vào $DICH ==="

# ---------------------------------------------------------------------------
# 1. Ghi số liệu TRƯỚC khi dừng — đây là thứ đối chiếu sau khi phục hồi.
#    Không có nó thì "phục hồi xong" chỉ là niềm tin.
# ---------------------------------------------------------------------------
# Đọc từng khoá thay vì `source .env`: trong đó có giá trị chứa ký tự đặc biệt,
# sourcing sẽ để shell diễn giải nó thay vì đọc nguyên văn.
doc_env() { sed -nE "s/^$1=(.*)$/\1/p" "$REPO_DIR/.env" | tail -1; }
QDRANT_URL="$(doc_env QDRANT_URL)"
QDRANT_API_KEY="$(doc_env QDRANT_API_KEY)"
POSTGRES_USER="$(doc_env POSTGRES_USER)"
POSTGRES_DB="$(doc_env POSTGRES_DB)"

MANIFEST="$DICH/manifest.txt"
{
  echo "thoi_diem: $(date -Iseconds)"
  echo "git_commit: $(git -C "$REPO_DIR" rev-parse HEAD 2>/dev/null || echo '?')"
  echo "git_branch: $(git -C "$REPO_DIR" rev-parse --abbrev-ref HEAD 2>/dev/null || echo '?')"
  echo "nhan: ${NHAN:-(khong)}"
  echo
  echo "[qdrant] collection: points_count"
} > "$MANIFEST"

QURL="${QDRANT_URL:-http://localhost:6333}"
for c in $(curl -sf -H "api-key: ${QDRANT_API_KEY:-}" "$QURL/collections" | jq -r '.result.collections[].name'); do
  n=$(curl -sf -H "api-key: ${QDRANT_API_KEY:-}" "$QURL/collections/$c" | jq -r '.result.points_count')
  echo "  $c: $n" >> "$MANIFEST"
done

echo >> "$MANIFEST"
echo "[postgres] bảng: số hàng" >> "$MANIFEST"
# count(*) thật, không dùng n_live_tup: đó là số ước lượng của bộ thu thập thống
# kê, reset khi container khởi động lại — đối chiếu bằng nó thì một bản phục hồi
# thiếu hàng vẫn trông khớp.
DEM_HANG_SQL="
  select relname || ':' || (xpath('/row/c/text()',
           query_to_xml(format('select count(*) as c from %I.%I', schemaname, relname),
                        false, true, '')))[1]::text
  from pg_stat_user_tables order by relname;
"
$DOCKER compose -f "$REPO_DIR/docker-compose.yml" exec -T postgres \
  psql -U "${POSTGRES_USER:-hit}" -d "${POSTGRES_DB:-hit_mira}" -At -c "$DEM_HANG_SQL" \
  2>/dev/null | sed 's/^/  /' >> "$MANIFEST"

# pg_dump nóng là bản dự phòng thứ hai, đọc được bằng mắt và chuyển máy được.
# Bản phục hồi chính vẫn là tar volume ở dưới.
echo "--- pg_dump (bản phụ) ---"
$DOCKER compose -f "$REPO_DIR/docker-compose.yml" exec -T postgres \
  pg_dump -U "${POSTGRES_USER:-hit}" -d "${POSTGRES_DB:-hit_mira}" --clean --if-exists \
  | gzip > "$DICH/pg_dump.sql.gz"

# ---------------------------------------------------------------------------
# 2. Dừng — từ đây không ai ghi vào volume nữa.
# ---------------------------------------------------------------------------
echo "--- dừng postgres/qdrant/minio ---"
$DOCKER compose -f "$REPO_DIR/docker-compose.yml" stop postgres qdrant minio

khoi_phuc_dich_vu() {
  echo "--- bật lại dịch vụ ---"
  $DOCKER compose -f "$REPO_DIR/docker-compose.yml" start postgres qdrant minio || true
}
trap khoi_phuc_dich_vu EXIT

# ---------------------------------------------------------------------------
# 3. Tar từng volume.
#    minio_data không nén: bên trong là ảnh/video đã nén sẵn, gzip chỉ tốn thời
#    gian mà gần như không giảm dung lượng.
# ---------------------------------------------------------------------------
for v in "${VOLUMES[@]}"; do
  src="/var/lib/docker/volumes/${PROJECT}_${v}/_data"
  if [ "$v" = "minio_data" ]; then
    out="$DICH/${v}.tar"; nen=""
  else
    out="$DICH/${v}.tar.gz"; nen="z"
  fi
  echo "--- tar $v → $(basename "$out") ---"
  $SUDO tar -c${nen}f "$out" -C "$src" .
  echo "  $(du -h "$out" | cut -f1)"
done

$SUDO chown -R "$(id -u):$(id -g)" "$DICH"

# ---------------------------------------------------------------------------
# 4. Kiểm tra tar đọc được. Một file tar hỏng chỉ lộ ra lúc phục hồi — tức là
#    đúng lúc không còn đường lui.
# ---------------------------------------------------------------------------
echo "--- kiểm tra tính toàn vẹn ---"
for f in "$DICH"/*.tar "$DICH"/*.tar.gz; do
  [ -e "$f" ] || continue
  tar -tf "$f" >/dev/null && echo "  OK  $(basename "$f")"
done
gzip -t "$DICH/pg_dump.sql.gz" && echo "  OK  pg_dump.sql.gz"

# Không băm chính manifest.txt: nó đang được ghi thêm ngay lúc này.
{
  echo
  echo "[file]"
  ( cd "$DICH" && sha256sum ./*.tar ./*.tar.gz ./pg_dump.sql.gz 2>/dev/null | sed 's/^/  /' )
} >> "$MANIFEST"

trap - EXIT
khoi_phuc_dich_vu

echo
echo "=== Xong: $DICH ==="
cat "$MANIFEST"
