"""Nhúng và index ảnh tĩnh vào collection media_clip của Qdrant.

Ảnh nằm CHUNG collection với keyframe video vì dùng chung không gian vector
Jina-CLIP v2 — một truy vấn text xếp hạng được cả hai loại.

MẶC ĐỊNH LÀ CHẠY THỬ. Muốn index thật phải truyền --apply:

    python scripts/index_image_units.py                    # chỉ đếm, không gọi API
    python scripts/index_image_units.py --apply            # nhúng + ghi Qdrant
    python scripts/index_image_units.py --apply --limit 5  # thử vài ảnh trước

Chạy theo lô để ghi Qdrant dần: đứt giữa chừng thì phần đã xong vẫn còn, chạy
lại sẽ bỏ qua ảnh đã có điểm (trừ khi truyền --reindex). Point id suy tất định
từ media_id nên index lại là ghi đè chứ không nhân bản điểm.

Lưu ý hạn mức: Jina tính 4.000 token cho mỗi ảnh 512px và giới hạn 100.000
token/phút, tức khoảng 25 ảnh/phút. 1.628 ảnh sẽ mất chừng 65 phút.
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

DEFAULT_CHUNK_SIZE = 200
# Qdrant nhận danh sách id dài, nhưng hỏi theo lô cho nhẹ bộ nhớ.
LOOKUP_BATCH = 500


def already_indexed(service: VideoRetrievalIndexingService, media_ids: list[str]) -> set[str]:
    """Hỏi Qdrant xem media_id nào đã có điểm, để chạy lại không nhúng thừa."""
    store = service.vector_store
    by_point_id = {store.point_id(media_id): media_id for media_id in media_ids}
    point_ids = list(by_point_id)
    found: set[str] = set()

    for start in range(0, len(point_ids), LOOKUP_BATCH):
        batch = point_ids[start : start + LOOKUP_BATCH]
        try:
            records = store.client.retrieve(
                collection_name=service.media_clip_collection,
                ids=batch,
                with_payload=False,
                with_vectors=False,
            )
        except Exception as exc:
            # Collection chưa tồn tại thì coi như chưa có gì được index.
            print(f"  ! không hỏi được Qdrant ({exc.__class__.__name__}), coi như chưa index gì")
            return set()
        found.update(by_point_id[str(record.id)] for record in records if str(record.id) in by_point_id)
    return found


def merge_summary(total: dict, part: dict) -> dict:
    for key in ("media_clip_units_received", "media_clip_indexed"):
        total[key] = total.get(key, 0) + part.get(key, 0)
    skips = total.setdefault("skipped_by_reason", {})
    for reason, count in part.get("skipped_by_reason", {}).items():
        skips[reason] = skips.get(reason, 0) + count
    total["collections"] = part.get("collections", total.get("collections", {}))
    return total


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Index ảnh tĩnh vào Qdrant collection media_clip.")
    parser.add_argument("--post-id", default=None, help="Chỉ index ảnh của một bài (post_id trong DB).")
    parser.add_argument("--limit", type=int, default=None, help="Chỉ xử lý N ảnh đầu tiên.")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=f"Số ảnh mỗi lô ghi Qdrant. Mặc định: {DEFAULT_CHUNK_SIZE}",
    )
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="Nhúng lại cả ảnh đã có điểm trong Qdrant.",
    )
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

    service = VideoRetrievalIndexingService()

    skipped_done = 0
    if not args.reindex:
        done = already_indexed(service, media_ids)
        skipped_done = len(done)
        media_ids = [media_id for media_id in media_ids if media_id not in done]

    print(f"Tổng {len(media_ids) + skipped_done} ảnh; đã index sẵn {skipped_done}; còn lại {len(media_ids)}.")

    if not args.apply:
        print("Chưa gọi API nhúng, chưa ghi Qdrant. Thêm --apply để thực hiện.")
        return
    if not media_ids:
        print("Không còn ảnh nào cần index.")
        return

    chunk_size = max(1, args.chunk_size)
    total: dict = {}
    for start in range(0, len(media_ids), chunk_size):
        chunk = media_ids[start : start + chunk_size]
        part = service.index_images(chunk).to_dict()
        merge_summary(total, part)
        done_count = min(start + chunk_size, len(media_ids))
        print(
            f"  lô {start // chunk_size + 1}: +{part.get('media_clip_indexed', 0)} điểm "
            f"({done_count}/{len(media_ids)} ảnh)",
            flush=True,
        )

    total["skipped_already_indexed"] = skipped_done
    print(json.dumps(total, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
