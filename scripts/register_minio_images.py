"""Đăng ký ảnh đã có trên MinIO thành hàng media trong PostgreSQL.

Song song với `register_minio_videos.py`, khác mỗi chỗ tạo hàng
media_type='image'. Quy ước key, cách đọc `<prefix>/<post>/post.json` để lấy
nội dung bài thật (message, link, ngày đăng) nằm trong `minio_registration.py`.

MẶC ĐỊNH LÀ CHẠY THỬ. Muốn ghi thật phải truyền --apply:

    python scripts/register_minio_images.py                              # bộ events/, xem trước
    python scripts/register_minio_images.py --prefix raw/google-drive/data   # bộ Google Drive
    python scripts/register_minio_images.py --prefix raw/google-drive/data --apply

Truyền sai --prefix thì script dừng ngay với danh sách key sai, không ghi dòng
nào — xem giải thích trong `minio_registration.py`.

Sau khi đăng ký, chạy phân tích và index bằng:

    python -m src.rag_video_anh.pipeline.image_processing_worker <media_id>
    python scripts/index_image_units.py --apply
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import select

PROJECT_ROOT = Path(__file__).resolve().parents[1]
API_ROOT = PROJECT_ROOT / "API"
sys.path.insert(0, str(API_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
load_dotenv(PROJECT_ROOT / ".env")

from minio_registration import (  # noqa: E402
    DEFAULT_DATASET_PREFIX,
    IMAGE_EXTENSIONS,
    ObjectKeyError,
    PostMetadataResolver,
    build_dataset_create,
    list_objects_by_extension,
    map_keys_to_event_ids,
    normalize_prefix,
)

from src.rag_video_anh.pipeline.minio_storage import MinioStorage  # noqa: E402
from src.rag_video_anh.repository import MediaCreate, MediaType, RepositoryUnitOfWork  # noqa: E402
from src.rag_video_anh.repository.models import MediaModel  # noqa: E402


def register_images(
    prefix: str = DEFAULT_DATASET_PREFIX,
    apply: bool = False,
    *,
    allow_missing_metadata: bool = False,
    allow_empty: bool = False,
    storage: MinioStorage | None = None,
    uow_factory=None,
) -> dict:
    storage = storage or MinioStorage()
    object_keys = list_objects_by_extension(storage, prefix, IMAGE_EXTENSIONS)
    # Soi key trước khi mở kết nối DB: prefix sai thì không được ghi gì.
    pairs = map_keys_to_event_ids(object_keys, prefix, allow_empty=allow_empty)

    created = 0
    skipped = 0
    skipped_missing_metadata = 0
    posts_touched: set[str] = set()
    missing_metadata: set[str] = set()

    with (uow_factory or RepositoryUnitOfWork)() as uow:
        if uow.session is None or uow.datasets is None or uow.posts is None or uow.media is None:
            raise RuntimeError("RepositoryUnitOfWork did not expose session/datasets/posts/media")

        dataset = uow.datasets.upsert_by_location(build_dataset_create(storage, prefix)) if apply else None

        resolver = PostMetadataResolver(
            storage,
            uow,
            prefix=prefix,
            allow_missing_metadata=allow_missing_metadata,
            dataset_id=dataset.dataset_id if dataset else None,
        )

        for object_key, event_id in pairs:
            exists = uow.session.scalar(
                select(MediaModel)
                .where(
                    MediaModel.media_type == MediaType.IMAGE.value,
                    MediaModel.bucket_name == storage.bucket_name,
                    MediaModel.object_key == object_key,
                )
                .limit(1)
            )
            # Kiểm tra post.json ngay cả khi chạy thử, và cả khi hàng media đã
            # tồn tại: thiếu provenance thì trích dẫn sau này không truy được về
            # bài gốc, phải biết trước khi --apply. Đặt sau truy vấn `exists` chỉ
            # để đếm cho đúng, không phải để bỏ qua việc kiểm.
            if not resolver.has_metadata(event_id):
                missing_metadata.add(event_id)
                if not allow_missing_metadata:
                    # Ảnh đã đăng ký rồi thì vẫn là "bỏ qua vì đã có"; chỉ ảnh
                    # chưa vào DB mới thật sự bị chặn vì thiếu metadata.
                    if exists is None:
                        skipped_missing_metadata += 1
                    else:
                        skipped += 1
                    continue

            if exists is not None:
                skipped += 1
                continue

            posts_touched.add(event_id)
            created += 1
            if not apply:
                continue

            post = resolver.upsert(event_id)
            uow.media.create_media(
                MediaCreate(
                    post_id=post.post_id,
                    media_type=MediaType.IMAGE.value,
                    bucket_name=storage.bucket_name,
                    object_key=object_key,
                )
            )

    return {
        "found": len(object_keys),
        "created": created,
        "skipped": skipped,
        "skipped_missing_metadata": skipped_missing_metadata,
        "posts": len(posts_touched),
        "missing_metadata": sorted(missing_metadata),
        "prefix": normalize_prefix(prefix),
        "applied": apply,
        "allow_missing_metadata": allow_missing_metadata,
        "allow_empty": allow_empty,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Đăng ký ảnh trên MinIO vào bảng media của PostgreSQL.")
    parser.add_argument(
        "--prefix",
        default=DEFAULT_DATASET_PREFIX,
        help=(
            "Thư mục gốc của bộ dữ liệu, theo quy ước <prefix>/<post_id>/media/<file>. "
            f"Mặc định: {DEFAULT_DATASET_PREFIX}. Bộ Google Drive: raw/google-drive/data"
        ),
    )
    parser.add_argument(
        "--allow-missing-metadata",
        action="store_true",
        help="Vẫn đăng ký media của bài không đọc được post.json (post sẽ không có nội dung/permalink).",
    )
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="Chấp nhận prefix không có ảnh nào. Không có cờ này thì listing rỗng bị coi là sai --prefix.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Ghi thật vào DB. Không truyền cờ này thì chỉ xem trước, không đổi gì.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        summary = register_images(
            prefix=args.prefix,
            apply=args.apply,
            allow_missing_metadata=args.allow_missing_metadata,
            allow_empty=args.allow_empty,
        )
    except ObjectKeyError as exc:
        print(f"DỪNG: {exc}", file=sys.stderr)
        return 2

    action = "Đã tạo" if summary["applied"] else "SẼ tạo (chạy thử)"
    print(f"Tìm thấy {summary['found']} ảnh dưới '{summary['prefix']}/' thuộc {summary['posts']} bài.")
    print(f"{action} {summary['created']} hàng media; bỏ qua {summary['skipped']} hàng đã có.")
    if summary["missing_metadata"]:
        print(f"  ! {len(summary['missing_metadata'])} bài không đọc được post.json: {summary['missing_metadata'][:10]}")
    if summary["skipped_missing_metadata"]:
        print(
            f"  ! Bỏ qua {summary['skipped_missing_metadata']} ảnh vì thiếu post.json — "
            "không tạo post giữ chỗ. Thêm --allow-missing-metadata nếu chấp nhận mất provenance."
        )
    if not summary["applied"]:
        print("Chưa ghi gì vào DB. Thêm --apply để thực hiện.")
    return 1 if summary["skipped_missing_metadata"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
