#!/usr/bin/env python3
"""Index caption/OCR/transcript của media vào collection văn bản 1.536 chiều.

KHÔNG chạm `media_clip` (Jina, 1.024 chiều) và không nhúng ảnh: ảnh/keyframe chỉ
còn là metadata cha để trích dẫn trỏ đúng chỗ và giao diện hiển thị được.

Dry-run là mặc định — nó dựng unit thật và đếm, nhưng không gọi nhà cung cấp
nhúng và không ghi Qdrant. Nhờ vậy xem trước được đúng số point sẽ sinh ra, và
biết ngay có bao nhiêu media chưa có caption.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "API"))

from src.configuration import AppConfig  # noqa: E402
from src.rag_video_anh.repository import RepositoryUnitOfWork  # noqa: E402
from src.rag_video_anh.retrieval.retrieval_units import (  # noqa: E402
    ImageRetrievalUnitBuilder,
    VideoRetrievalUnitBuilder,
)
from src.rag_video_anh.retrieval.text_indexing_service import (  # noqa: E402
    DEFAULT_MEDIA_TEXT_COLLECTION,
    MediaTextIndexingService,
)


def liet_ke(media_type: str, post_id: str | None) -> list[str]:
    with RepositoryUnitOfWork() as uow:
        if uow.media is None:
            raise RuntimeError("RepositoryUnitOfWork không mở repository media")
        rows = (uow.media.list_by_post(post_id, media_type=media_type) if post_id
                else uow.media.list_by_type(media_type))
        return [str(r.media_id) for r in rows]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--media-type", choices=("image", "video"), default="image")
    p.add_argument("--post-id", default=None, help="Chỉ xử lý media của một bài.")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--collection", default=DEFAULT_MEDIA_TEXT_COLLECTION)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument(
        "--max-consecutive-errors", type=int, default=10,
        help="Dừng sớm sau ngần này media hỏng liên tiếp, thay vì đánh hỏng cả nghìn cái trong bốn phút.",
    )
    p.add_argument("--apply", action="store_true", help="Gọi API nhúng thật và ghi Qdrant.")
    args = p.parse_args()

    media_ids = liet_ke(args.media_type, args.post_id)
    if args.limit is not None:
        media_ids = media_ids[: args.limit]

    cfg = AppConfig().embedding
    print(json.dumps({
        "media_type": args.media_type,
        "so_media": len(media_ids),
        "collection": args.collection,
        "model": cfg.model,
        "dimensions": cfg.dimensions,
        "applied": args.apply,
    }, ensure_ascii=False, indent=2))
    if not media_ids:
        print("Không có media nào để xử lý.")
        return 0

    # Dry-run vẫn dựng unit thật: đó là cách duy nhất biết trước số point và
    # biết bao nhiêu media chưa có caption/OCR.
    svc = MediaTextIndexingService(
        image_builder=ImageRetrievalUnitBuilder(check_image_exists=False),
        video_builder=VideoRetrievalUnitBuilder(check_frame_exists=False),
        collection_name=args.collection,
        batch_size=args.batch_size,
        **({} if args.apply else {"text_embedder": _KhongNhung(), "vector_store": _KhongGhi()}),
    )
    dung = svc.build_image_group if args.media_type == "image" else svc.build_video_group

    theo_loai: Counter = Counter()
    thong_ke = {"khong_co_text": 0, "tong_point": 0, "loi_lien_tiep": 0}

    def sinh_units():
        """Sinh từng nhóm unit; `index_groups` gom lại thành lô đủ lớn rồi mới gửi."""
        for i, media_id in enumerate(media_ids, start=1):
            try:
                group = dung(media_id)
            except Exception as exc:
                thong_ke["loi_lien_tiep"] += 1
                print(f"  ! {media_id}: {exc.__class__.__name__}: {exc}", flush=True)
                if thong_ke["loi_lien_tiep"] >= args.max_consecutive_errors:
                    raise SystemExit(
                        f"\nDỪNG SỚM: {thong_ke['loi_lien_tiep']} media hỏng liên tiếp."
                    ) from exc
                continue
            thong_ke["loi_lien_tiep"] = 0
            if not group.units:
                thong_ke["khong_co_text"] += 1
            theo_loai.update(u.source_type for u in group.units)
            thong_ke["tong_point"] += len(group.units)
            if i % 100 == 0 or i == len(media_ids):
                print(f"  ... {i}/{len(media_ids)} media | {thong_ke['tong_point']} point", flush=True)
            # Yield cả nhóm RỖNG: nó vẫn mang scope, và đó là thứ duy nhất cho
            # biết phải dọn point cũ của media nào khi caption/OCR bị xoá hết.
            yield group

    if args.apply:
        # Gom qua nhiều media rồi mới gọi API: gọi từng ảnh thì --batch-size vô
        # nghĩa và caption trùng nhau giữa các ảnh không bao giờ gộp được.
        ket = svc.index_groups(sinh_units()).to_dict()
    else:
        for _ in sinh_units():
            pass
        ket = {}

    print(json.dumps({
        "media_da_duyet": len(media_ids),
        "media_khong_co_caption_ocr": thong_ke["khong_co_text"],
        "point_se_sinh" if not args.apply else "point_da_xu_ly": thong_ke["tong_point"],
        "theo_source_type": dict(sorted(theo_loai.items())),
        **ket,
    }, ensure_ascii=False, indent=2))
    if not args.apply:
        print("\nDry-run: chưa gọi API nhúng và chưa ghi Qdrant. Thêm --apply để thực hiện.")
    return 0


class _KhongNhung:
    """Chỗ giữ chỗ cho dry-run: dựng unit được, nhưng gọi nhúng là nổ."""

    model = "(dry-run)"

    def embed_documents(self, texts: list[str]):
        raise RuntimeError("dry-run không được gọi nhà cung cấp nhúng")


class _KhongGhi:
    """Đọc thì cho, ghi và xoá thì nổ — dry-run không được đụng vào kho."""

    def upsert_points(self, **_):
        raise RuntimeError("dry-run không được ghi Qdrant")

    def delete_points(self, **_):
        raise RuntimeError("dry-run không được xoá point")

    def get_payloads(self, **_):
        return {}

    def list_unit_ids_by_field(self, **_):
        return []


if __name__ == "__main__":
    raise SystemExit(main())
