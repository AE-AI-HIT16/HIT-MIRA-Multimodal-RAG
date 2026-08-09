"""Đẩy dữ liệu crawl lên MinIO theo chuẩn key events/<post_id>/...

Object key giữ đúng quy ước cũ nên các hàng `media` trong PostgreSQL và payload
trong Qdrant vẫn trỏ đúng sau khi đẩy lại.

MẶC ĐỊNH LÀ CHẠY THỬ. Muốn ghi thật phải truyền --apply:

    python scripts/upload_crawled_data_to_minio.py data/crawled_full
    python scripts/upload_crawled_data_to_minio.py data/crawled_full --apply
    python scripts/upload_crawled_data_to_minio.py data/crawled_full --apply --purge

--purge xóa toàn bộ object cũ dưới prefix TRƯỚC khi đẩy bản mới. Dùng khi bộ
crawl mới thay thế hẳn bộ cũ; không có cờ này thì script chỉ bù các file còn
thiếu và bỏ qua file đã trùng kích thước.

Hai prefix `frames/` và `runpod-artifacts/` là artifact do pipeline sinh ra và
đang được DB tham chiếu, nên script từ chối xóa chúng.
"""

from __future__ import annotations

import argparse
import mimetypes
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
API_ROOT = PROJECT_ROOT / "API"
sys.path.insert(0, str(API_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

from minio.deleteobjects import DeleteObject  # noqa: E402

from src.rag_video_anh.pipeline.minio_storage import MinioStorage  # noqa: E402

# Artifact do pipeline sinh ra, có hàng trong DB trỏ tới -> không được xóa nhầm.
PROTECTED_PREFIXES = {"frames", "runpod-artifacts"}
# File mô tả cả bộ dữ liệu (crawl.log, posts.jsonl, manifest.json) nằm ở gốc.
DATASET_SUBDIR = "_dataset"
PROGRESS_EVERY = 100


def collect_local_files(data_root: Path, prefix: str, with_dataset_files: bool) -> list[tuple[Path, str]]:
    """Ghép từng file local với object key tương ứng trên MinIO."""
    if not data_root.is_dir():
        raise NotADirectoryError(f"Không tìm thấy thư mục dữ liệu: {data_root}")

    pairs: list[tuple[Path, str]] = []

    # Trường hợp trỏ thẳng vào một thư mục bài.
    if (data_root / "post.json").is_file():
        for path in sorted(p for p in data_root.rglob("*") if p.is_file()):
            relative = path.relative_to(data_root).as_posix()
            pairs.append((path, f"{prefix}/{data_root.name}/{relative}"))
        return pairs

    for event_dir in sorted(p for p in data_root.iterdir() if p.is_dir()):
        if not (event_dir / "post.json").is_file():
            print(f"  ! bỏ qua '{event_dir.name}' vì không có post.json")
            continue
        for path in sorted(p for p in event_dir.rglob("*") if p.is_file()):
            relative = path.relative_to(event_dir).as_posix()
            pairs.append((path, f"{prefix}/{event_dir.name}/{relative}"))

    if with_dataset_files:
        for path in sorted(p for p in data_root.iterdir() if p.is_file()):
            pairs.append((path, f"{prefix}/{DATASET_SUBDIR}/{path.name}"))

    return pairs


def remote_sizes(storage: MinioStorage, prefix: str) -> dict[str, int]:
    """Đọc một lượt kích thước mọi object dưới prefix để so sánh nhanh."""
    return {
        obj.object_name: obj.size
        for obj in storage.client.list_objects(storage.bucket_name, prefix=f"{prefix}/", recursive=True)
    }


def purge_prefix(storage: MinioStorage, prefix: str, keys: list[str]) -> tuple[int, list[str]]:
    """Xóa hàng loạt object cũ; trả về số xóa được và danh sách lỗi."""
    if prefix in PROTECTED_PREFIXES:
        raise ValueError(f"Từ chối xóa prefix được bảo vệ: '{prefix}'")

    errors: list[str] = []
    deleted = 0
    for start in range(0, len(keys), 1000):
        batch = keys[start : start + 1000]
        results = storage.client.remove_objects(
            storage.bucket_name,
            (DeleteObject(key) for key in batch),
        )
        batch_errors = [f"{err.name}: {err.message}" for err in results]
        errors.extend(batch_errors)
        deleted += len(batch) - len(batch_errors)
    return deleted, errors


def upload_pairs(storage: MinioStorage, pairs: list[tuple[Path, str]], workers: int) -> tuple[int, list[str]]:
    """Đẩy song song; gọi thẳng fput_object để không ngập log INFO mỗi file."""
    done = 0
    sent_bytes = 0
    errors: list[str] = []
    lock = threading.Lock()
    total = len(pairs)

    def send(pair: tuple[Path, str]) -> int:
        path, key = pair
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        storage.client.fput_object(
            bucket_name=storage.bucket_name,
            object_name=MinioStorage.normalize_object_key(key),
            file_path=str(path),
            content_type=content_type,
        )
        return path.stat().st_size

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(send, pair): pair for pair in pairs}
        for future in as_completed(futures):
            path, key = futures[future]
            try:
                size = future.result()
            except Exception as exc:
                with lock:
                    done += 1
                    errors.append(f"{key}: {exc.__class__.__name__}: {exc}")
                continue
            with lock:
                done += 1
                sent_bytes += size
                if done % PROGRESS_EVERY == 0 or done == total:
                    print(f"  ... {done}/{total} file, {sent_bytes / 1e9:.2f} GB", flush=True)

    return total - len(errors), errors


def run(
    data_root: Path,
    prefix: str,
    apply: bool,
    purge: bool,
    workers: int,
    with_dataset_files: bool,
) -> dict:
    prefix = MinioStorage.normalize_object_key(prefix)
    if purge and prefix in PROTECTED_PREFIXES:
        raise ValueError(f"Từ chối xóa prefix được bảo vệ: '{prefix}'")

    storage = MinioStorage()
    pairs = collect_local_files(data_root, prefix, with_dataset_files)
    local_bytes = sum(path.stat().st_size for path, _ in pairs)
    existing = remote_sizes(storage, prefix)

    if purge:
        to_upload = pairs
        to_delete = sorted(existing)
    else:
        # Không purge thì chỉ bù file thiếu hoặc lệch kích thước.
        to_upload = [(path, key) for path, key in pairs if existing.get(key) != path.stat().st_size]
        to_delete = []

    summary = {
        "bucket": storage.bucket_name,
        "prefix": prefix,
        "local_files": len(pairs),
        "local_bytes": local_bytes,
        "remote_before": len(existing),
        "to_delete": len(to_delete),
        "to_upload": len(to_upload),
        "skipped_same_size": len(pairs) - len(to_upload),
        "applied": apply,
    }

    if not apply:
        return summary

    if to_delete:
        print(f"Xóa {len(to_delete)} object cũ dưới '{prefix}/' ...", flush=True)
        deleted, delete_errors = purge_prefix(storage, prefix, to_delete)
        summary["deleted"] = deleted
        summary["delete_errors"] = delete_errors[:10]
        if delete_errors:
            print(f"  ! {len(delete_errors)} lỗi khi xóa")

    if to_upload:
        print(f"Đẩy {len(to_upload)} file lên '{prefix}/' bằng {workers} luồng ...", flush=True)
        uploaded, upload_errors = upload_pairs(storage, to_upload, workers)
        summary["uploaded"] = uploaded
        summary["upload_errors"] = upload_errors[:10]
        summary["upload_error_count"] = len(upload_errors)

    after = remote_sizes(storage, prefix)
    summary["remote_after"] = len(after)
    summary["remote_bytes_after"] = sum(after.values())
    # Đối chiếu lại từng key để chắc chắn không thiếu file nào.
    summary["missing_after"] = sum(1 for path, key in pairs if after.get(key) != path.stat().st_size)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Đẩy dữ liệu crawl lên MinIO theo chuẩn events/<post_id>/...")
    parser.add_argument("data_root", help="Thư mục chứa các thư mục bài, hoặc một thư mục bài có post.json.")
    parser.add_argument("--prefix", default="events", help="Prefix object key trên MinIO. Mặc định: events")
    parser.add_argument("--apply", action="store_true", help="Ghi thật lên MinIO. Không có cờ này thì chỉ xem trước.")
    parser.add_argument(
        "--purge",
        action="store_true",
        help="Xóa sạch object cũ dưới prefix trước khi đẩy bản mới.",
    )
    parser.add_argument("--workers", type=int, default=8, help="Số luồng đẩy song song. Mặc định: 8")
    parser.add_argument(
        "--skip-dataset-files",
        action="store_true",
        help=f"Không đẩy crawl.log/posts.jsonl/manifest.json lên <prefix>/{DATASET_SUBDIR}/.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = run(
        data_root=Path(args.data_root).resolve(),
        prefix=args.prefix,
        apply=args.apply,
        purge=args.purge,
        workers=max(1, args.workers),
        with_dataset_files=not args.skip_dataset_files,
    )

    print()
    print(f"Bucket '{summary['bucket']}', prefix '{summary['prefix']}/'")
    print(f"  Local:        {summary['local_files']} file, {summary['local_bytes'] / 1e9:.2f} GB")
    print(f"  Trên MinIO:   {summary['remote_before']} object trước khi chạy")
    print(f"  Sẽ xóa:       {summary['to_delete']}")
    print(f"  Sẽ đẩy:       {summary['to_upload']} (bỏ qua {summary['skipped_same_size']} file đã trùng kích thước)")

    if not summary["applied"]:
        print("\nChưa đổi gì trên MinIO. Thêm --apply để thực hiện.")
        return

    print(f"  Đã xóa:       {summary.get('deleted', 0)}")
    print(f"  Đã đẩy:       {summary.get('uploaded', 0)}")
    print(f"  Sau khi chạy: {summary['remote_after']} object, {summary['remote_bytes_after'] / 1e9:.2f} GB")

    for label, key in (("lỗi xóa", "delete_errors"), ("lỗi đẩy", "upload_errors")):
        for message in summary.get(key, []):
            print(f"  ! {label}: {message}")

    if summary["missing_after"]:
        print(f"\nCẢNH BÁO: {summary['missing_after']} file chưa lên đúng. Chạy lại script để bù.")
    else:
        print("\nĐối chiếu xong: mọi file local đều có trên MinIO đúng kích thước.")


if __name__ == "__main__":
    main()
