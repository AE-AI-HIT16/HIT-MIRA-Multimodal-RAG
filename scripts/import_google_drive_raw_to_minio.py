"""Tải một ZIP dữ liệu raw từ Google Drive rồi đẩy an toàn lên MinIO.

Mặc định chỉ lập kế hoạch upload. Thêm ``--apply`` mới ghi object vào MinIO.
Archive và thư mục đã giải nén được giữ lại để lần chạy lại không phải tải lại
2GB từ Google Drive.

Ví dụ (folder hoặc file Drive phải chia sẻ "Anyone with the link: Viewer")::

    python -m pip install gdown minio python-dotenv
    python scripts/import_google_drive_raw_to_minio.py \\
      --drive-url 'https://drive.google.com/drive/folders/FOLDER_ID' \\
      --archive-name data.zip --prefix raw/google-drive --apply

Nếu đã tải ZIP thủ công lên server, bỏ qua gdown bằng ``--archive``::

    python scripts/import_google_drive_raw_to_minio.py \\
      --archive /path/to/data.zip --prefix raw/google-drive --apply

Không dùng ``--purge``: script chỉ upload object thiếu hoặc khác kích thước,
vì raw data là nguồn gốc cần được giữ lại.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import shutil
import stat
import sys
import threading
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]

try:
    from dotenv import load_dotenv
    from minio import Minio
except ImportError as exc:  # pragma: no cover - depends on operator environment
    raise SystemExit(
        "Thiếu dependency. Chạy: python -m pip install gdown minio python-dotenv"
    ) from exc

load_dotenv(PROJECT_ROOT / ".env")


GIB = 1024**3


def _normalize_key(key: str) -> str:
    normalized = key.replace("\\", "/").strip("/")
    if not normalized:
        raise ValueError("MinIO object prefix cannot be empty.")
    return normalized


def _download_drive_archive(drive_url: str, destination: Path, archive_name: str | None) -> Path:
    """Download one file or find exactly one ZIP downloaded from a Drive folder."""
    try:
        import gdown
    except ImportError as exc:  # pragma: no cover - depends on operator environment
        raise SystemExit("Thiếu gdown. Chạy: python -m pip install gdown") from exc

    destination.mkdir(parents=True, exist_ok=True)
    is_folder = "/folders/" in drive_url
    if is_folder:
        downloaded = gdown.download_folder(
            url=drive_url,
            output=str(destination),
            quiet=False,
            use_cookies=True,
        )
        paths = [Path(item) for item in (downloaded or []) if Path(item).is_file()]
        if archive_name:
            paths = [item for item in paths if item.name == archive_name]
        else:
            paths = [item for item in paths if item.suffix.lower() == ".zip"]
        if len(paths) != 1:
            hint = "đúng tên --archive-name" if archive_name else "đuôi .zip duy nhất"
            raise RuntimeError(f"Google Drive phải trả đúng một archive {hint}; thấy {len(paths)} file.")
        return paths[0]

    target_name = archive_name or "google-drive-data.zip"
    target = destination / target_name
    result = gdown.download(url=drive_url, output=str(target), quiet=False, fuzzy=True)
    if not result or not target.is_file():
        raise RuntimeError("Google Drive không trả file. Kiểm tra quyền chia sẻ 'Anyone with the link: Viewer'.")
    return target


def _assert_safe_zip(archive: Path, max_unpacked_bytes: int) -> int:
    """Validate paths and total expanded size before extracting untrusted ZIP input."""
    total_size = 0
    with zipfile.ZipFile(archive) as bundle:
        for info in bundle.infolist():
            target = Path(info.filename)
            if target.is_absolute() or ".." in target.parts:
                raise ValueError(f"ZIP contains unsafe path: {info.filename!r}")
            mode = info.external_attr >> 16
            if stat.S_ISLNK(mode):
                raise ValueError(f"ZIP contains symlink, not allowed: {info.filename!r}")
            total_size += info.file_size
    if total_size > max_unpacked_bytes:
        raise ValueError(
            f"ZIP expands to {total_size / GIB:.2f} GiB, above --max-unpacked-gb limit "
            f"({max_unpacked_bytes / GIB:.2f} GiB)."
        )
    return total_size


def _extract_archive(archive: Path, destination: Path, max_unpacked_bytes: int) -> Path:
    """Extract atomically; an interrupted extraction is never reused as complete."""
    if destination.is_dir():
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    expanded_size = _assert_safe_zip(archive, max_unpacked_bytes)
    free_bytes = shutil.disk_usage(destination.parent).free
    if free_bytes < expanded_size:
        raise RuntimeError(
            f"Không đủ disk để giải nén: cần {expanded_size / GIB:.2f} GiB, "
            f"còn {free_bytes / GIB:.2f} GiB."
        )

    temporary = destination.with_name(f".{destination.name}.extracting")
    if temporary.exists():
        raise RuntimeError(f"Tìm thấy extraction dở dang: {temporary}. Xem và xoá thủ công trước khi chạy lại.")
    temporary.mkdir(parents=True)
    try:
        with zipfile.ZipFile(archive) as bundle:
            bundle.extractall(temporary)
        temporary.rename(destination)
    except Exception:
        # Giữ thư mục lỗi để operator có thể điều tra, không xoá dữ liệu ngầm.
        raise
    return destination


def _files_with_keys(root: Path, prefix: str) -> list[tuple[Path, str]]:
    normalized_prefix = _normalize_key(prefix)
    return [
        (path, _normalize_key(f"{normalized_prefix}/{path.relative_to(root).as_posix()}"))
        for path in sorted(root.rglob("*"))
        if path.is_file()
    ]


def _remote_sizes(client: Minio, bucket_name: str, prefix: str) -> dict[str, int]:
    return {
        item.object_name: item.size
        for item in client.list_objects(bucket_name, prefix=f"{prefix}/", recursive=True)
    }


def _upload_pairs(
    client: Minio,
    bucket_name: str,
    pairs: Iterable[tuple[Path, str]],
    workers: int,
) -> tuple[int, list[str]]:
    all_pairs = list(pairs)
    completed = 0
    uploaded_bytes = 0
    errors: list[str] = []
    lock = threading.Lock()

    def upload(pair: tuple[Path, str]) -> int:
        path, key = pair
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        client.fput_object(bucket_name, key, str(path), content_type=content_type)
        return path.stat().st_size

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(upload, pair): pair for pair in all_pairs}
        for future in as_completed(futures):
            path, key = futures[future]
            with lock:
                completed += 1
            try:
                size = future.result()
            except Exception as exc:
                with lock:
                    errors.append(f"{key}: {exc.__class__.__name__}: {exc}")
            else:
                with lock:
                    uploaded_bytes += size
                    if completed % 100 == 0 or completed == len(all_pairs):
                        print(f"  ... {completed}/{len(all_pairs)} file, {uploaded_bytes / GIB:.2f} GiB", flush=True)
    return len(all_pairs) - len(errors), errors


def run(args: argparse.Namespace) -> dict[str, object]:
    staging = args.staging_dir.resolve()
    if args.archive:
        archive = args.archive.resolve()
        if not archive.is_file():
            raise FileNotFoundError(f"Không thấy ZIP local: {archive}")
    else:
        expected = staging / args.archive_name
        archive = expected if expected.is_file() else _download_drive_archive(args.drive_url, staging, args.archive_name)

    if not zipfile.is_zipfile(archive):
        raise ValueError(f"Không phải ZIP hợp lệ: {archive}")
    extract_dir = staging / "extracted" / archive.stem
    root = _extract_archive(archive, extract_dir, args.max_unpacked_gb * GIB)
    pairs = _files_with_keys(root, args.prefix)
    if not pairs:
        raise RuntimeError(f"ZIP đã giải nén nhưng không có file: {root}")

    endpoint = os.getenv("MINIO_ENDPOINT", "").strip()
    access_key = os.getenv("MINIO_ACCESS_KEY", "").strip()
    secret_key = os.getenv("MINIO_SECRET_KEY", "").strip()
    bucket_name = os.getenv("MINIO_BUCKET", "").strip()
    secure = os.getenv("MINIO_SECURE", "false").strip().lower() in {"1", "true", "yes", "on"}
    if not all((endpoint, access_key, secret_key, bucket_name)):
        raise RuntimeError("Thiếu MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY hoặc MINIO_BUCKET trong .env.")
    client = Minio(
        endpoint=endpoint,
        access_key=access_key,
        secret_key=secret_key,
        secure=secure,
    )
    if not client.bucket_exists(bucket_name):
        client.make_bucket(bucket_name)

    prefix = _normalize_key(args.prefix)
    remote_before = _remote_sizes(client, bucket_name, prefix)
    to_upload = [(path, key) for path, key in pairs if remote_before.get(key) != path.stat().st_size]
    summary: dict[str, object] = {
        "archive": str(archive),
        "extracted_to": str(root),
        "bucket": bucket_name,
        "prefix": prefix,
        "local_files": len(pairs),
        "local_bytes": sum(path.stat().st_size for path, _ in pairs),
        "to_upload": len(to_upload),
        "skipped_same_size": len(pairs) - len(to_upload),
        "applied": args.apply,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not args.apply:
        print("Chưa upload object nào. Thêm --apply để ghi lên MinIO.")
        return summary

    uploaded, errors = _upload_pairs(client, bucket_name, to_upload, args.workers)
    remote_after = _remote_sizes(client, bucket_name, prefix)
    missing = [key for path, key in pairs if remote_after.get(key) != path.stat().st_size]
    summary.update(
        {
            "uploaded": uploaded,
            "upload_errors": errors[:10],
            "upload_error_count": len(errors),
            "remote_files": len(remote_after),
            "remote_bytes": sum(remote_after.values()),
            "missing_after": len(missing),
        }
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if errors or missing:
        raise RuntimeError(f"Upload chưa hoàn tất: {len(errors)} lỗi, {len(missing)} object thiếu.")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--drive-url", help="Google Drive file/folder URL công khai.")
    source.add_argument("--archive", type=Path, help="ZIP đã có sẵn trên server.")
    parser.add_argument("--archive-name", default="data.zip", help="Tên ZIP trong folder Google Drive.")
    parser.add_argument("--staging-dir", type=Path, default=PROJECT_ROOT / "data" / "staging" / "google-drive")
    parser.add_argument("--prefix", default="raw/google-drive", help="Prefix object key trên MinIO.")
    parser.add_argument("--workers", type=int, default=4, help="Số upload song song; 4 là mức ổn định cho file lớn.")
    parser.add_argument("--max-unpacked-gb", type=int, default=40, help="Chặn ZIP giải nén quá lớn.")
    parser.add_argument("--apply", action="store_true", help="Ghi thật object lên MinIO.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.workers < 1:
        raise SystemExit("--workers phải >= 1")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
