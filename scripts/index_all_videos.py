"""Index MỌI video đã parse vào Qdrant — nguồn của nút "Index video" ở màn admin.

`index_video_retrieval_units.py` chỉ nhận đúng một `video_media_id`, nên muốn
chạy cả kho thì trước đây phải tự viết truy vấn PostgreSQL rồi gọi từng cái một
(xem `docs/handoff.md` §3). Script này làm đúng vòng lặp đó, có bỏ qua việc đã
xong và có ngưỡng dừng khi hỏng liên tiếp.

MẶC ĐỊNH LÀ CHẠY THỬ. Muốn index thật phải truyền --apply:

    python scripts/index_all_videos.py                 # chỉ đếm
    python scripts/index_all_videos.py --apply         # index thật
    python scripts/index_all_videos.py --apply --limit 2

Lưu ý hạn mức: mỗi keyframe tốn 4.000 token và Jina cho 100.000 token/phút, nên
một video vài trăm keyframe mất vài phút. Đứt giữa chừng vẫn an toàn — point id
suy tất định nên chạy lại là ghi đè, không nhân bản.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import sqlalchemy as sa
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
API_ROOT = PROJECT_ROOT / "API"
sys.path.insert(0, str(API_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

from src.rag_video_anh.repository.database import get_session_manager  # noqa: E402
from src.rag_video_anh.retrieval.indexing_service import (  # noqa: E402
    VideoRetrievalIndexingService,
)

# Hỏng liên tiếp tới ngưỡng này thì dừng hẳn. Một nguồn nhúng chết làm mọi video
# hỏng y hệt nhau; chạy tiếp chỉ để đánh dấu cả kho là lỗi trong vài phút.
MAC_DINH_HONG_LIEN_TIEP = 3


def danh_sach_video() -> list[tuple[str, str]]:
    """(video_media_id, video_id) của mọi video đã parse xong.

    Lấy từ bảng `videos` chứ không phải `media`: có dòng trong `videos` nghĩa là
    video đã qua pipeline và có metadata, còn `media.media_type='video'` mới chỉ
    là "file đã nằm trong MinIO".
    """
    with get_session_manager().session() as session:
        dong = session.execute(sa.text("select media_id, video_id from videos")).all()
    return [(str(media_id), str(video_id)) for media_id, video_id in dong]


def video_da_co_diem(service: VideoRetrievalIndexingService, collection: str) -> set[str]:
    """Tập video_id đã có ít nhất một điểm trong collection."""
    ids: set[str] = set()
    offset = None
    try:
        while True:
            diem, offset = service.vector_store.client.scroll(
                collection, limit=1000, offset=offset, with_payload=["video_id"], with_vectors=False
            )
            ids.update(
                str((p.payload or {})["video_id"]) for p in diem if (p.payload or {}).get("video_id")
            )
            if offset is None:
                break
    except Exception as exc:  # noqa: BLE001 — collection chưa có = chưa index gì
        print(f"  ! không cuộn được '{collection}' ({exc.__class__.__name__}), coi như chưa index")
        return set()
    return ids


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Index toàn bộ video đã parse vào Qdrant.")
    parser.add_argument("--limit", type=int, default=None, help="Chỉ xử lý N video đầu tiên.")
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="Index lại cả video đã có điểm trong Qdrant (tốn quota, dùng khi payload đổi).",
    )
    parser.add_argument(
        "--transcript-only",
        action="store_true",
        help="Chỉ index nhánh lời thoại, không đụng keyframe.",
    )
    parser.add_argument(
        "--max-consecutive-failures",
        type=int,
        default=MAC_DINH_HONG_LIEN_TIEP,
        help=f"Dừng sau ngần này video hỏng liên tiếp. Mặc định: {MAC_DINH_HONG_LIEN_TIEP}",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Gọi API nhúng và ghi vào Qdrant. Không có cờ này thì chỉ đếm.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    videos = danh_sach_video()
    service = VideoRetrievalIndexingService()

    bo_qua_da_xong = 0
    if not args.reindex:
        # "Có điểm ở MỘT trong hai nhánh" là đã qua index. Đòi đủ cả hai thì
        # video hỏng frame (chỉ có transcript) sẽ bị nhúng lại mỗi lần chạy —
        # đúng thứ CLAUDE.md gọi là hai nhánh độc lập.
        da_xong = video_da_co_diem(service, service.video_transcript_collection) | video_da_co_diem(
            service, service.media_clip_collection
        )
        con_lai = [cap for cap in videos if cap[1] not in da_xong]
        bo_qua_da_xong = len(videos) - len(con_lai)
        videos = con_lai

    if args.limit is not None:
        videos = videos[: args.limit]

    print(f"Tổng {len(videos) + bo_qua_da_xong} video; đã index sẵn {bo_qua_da_xong}; còn lại {len(videos)}.")
    if not args.apply:
        print("Chưa gọi API nhúng, chưa ghi Qdrant. Thêm --apply để thực hiện.")
        return 0

    thanh_cong: list[str] = []
    that_bai: list[dict[str, str]] = []
    hong_lien_tiep = 0

    for thu_tu, (video_media_id, video_id) in enumerate(videos, start=1):
        print(f"[{thu_tu}/{len(videos)}] {video_media_id}", flush=True)
        try:
            tom_tat = service.index_video(video_media_id, transcript_only=args.transcript_only)
        except Exception as exc:  # noqa: BLE001 — một video hỏng không được giết cả mẻ
            hong_lien_tiep += 1
            that_bai.append({"video_media_id": video_media_id, "loi": f"{exc.__class__.__name__}: {exc}"})
            print(f"  ! hỏng ({exc.__class__.__name__}: {exc})", flush=True)
            if hong_lien_tiep >= args.max_consecutive_failures:
                print(
                    f"\nDỪNG: {hong_lien_tiep} video hỏng liên tiếp. "
                    "Nhiều khả năng nguồn nhúng hoặc Qdrant đang chết, không phải dữ liệu.",
                    flush=True,
                )
                break
            continue

        hong_lien_tiep = 0
        thanh_cong.append(video_media_id)
        so = tom_tat.to_dict()
        print(
            f"  ok — keyframe {so.get('media_clip_indexed', 0)}, "
            f"transcript {so.get('video_transcript_indexed', 0)}",
            flush=True,
        )

    print(
        json.dumps(
            {
                "da_index_san": bo_qua_da_xong,
                "thanh_cong": len(thanh_cong),
                "that_bai": that_bai,
                "con_lai_chua_chay": len(videos) - len(thanh_cong) - len(that_bai),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    # Có video hỏng thì mã thoát phải khác 0, nếu không màn admin sẽ báo "xong".
    return 1 if that_bai else 0


if __name__ == "__main__":
    raise SystemExit(main())
