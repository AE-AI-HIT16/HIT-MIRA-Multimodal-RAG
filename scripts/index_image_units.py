"""Nhúng và index ảnh tĩnh vào collection media_clip của Qdrant.

Ảnh nằm CHUNG collection với keyframe video vì dùng chung không gian vector
Jina-CLIP v2 — một truy vấn text xếp hạng được cả hai loại.

MẶC ĐỊNH LÀ CHẠY THỬ. Muốn index thật phải truyền --apply:

    python scripts/index_image_units.py                    # chỉ đếm, không gọi API
    python scripts/index_image_units.py --apply            # nhúng + ghi Qdrant
    python scripts/index_image_units.py --apply --limit 5  # thử vài ảnh trước

Lưu ý hạn mức: Jina tính 4.000 token cho mỗi ảnh 512px và giới hạn 100.000
token/phút, tức khoảng 25 ảnh/phút. 166 ảnh sẽ mất chừng 7 phút.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
API_ROOT = PROJECT_ROOT / "API"
sys.path.insert(0, str(API_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

from src.rag_video_anh.retrieval.indexing_service import (  # noqa: E402
    VideoRetrievalIndexingService,
)
from src.rag_video_anh.retrieval.retrieval_units import (  # noqa: E402
    ImageRetrievalUnitBuilder,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Index ảnh tĩnh vào Qdrant collection media_clip.")
    parser.add_argument("--post-id", default=None, help="Chỉ index ảnh của một bài (post_id trong DB).")
    parser.add_argument("--limit", type=int, default=None, help="Chỉ xử lý N ảnh đầu tiên.")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Gọi API nhúng và ghi vào Qdrant. Không có cờ này thì chỉ đếm.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    media_ids = ImageRetrievalUnitBuilder().list_image_media_ids(post_id=args.post_id)
    if args.limit is not None:
        media_ids = media_ids[: args.limit]

    if not args.apply:
        print(f"Có {len(media_ids)} ảnh sẽ được index vào 'media_clip'.")
        print("Chưa gọi API nhúng, chưa ghi Qdrant. Thêm --apply để thực hiện.")
        return

    summary = VideoRetrievalIndexingService().index_images(media_ids)
    print(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
