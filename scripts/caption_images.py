"""Chạy OCR + caption cho ảnh tĩnh đã đăng ký, ghi kết quả vào PostgreSQL.

`image_processing_worker` chỉ nhận một media_id mỗi lần; với hơn một nghìn ảnh
thì mỗi ảnh một tiến trình là quá phí, nên script này dựng worker một lần rồi
chạy nhiều luồng.

Qwen-VL trả OCR và caption trong cùng một lời gọi, nên không tách hai lượt.

MẶC ĐỊNH LÀ CHẠY THỬ. Muốn gọi API thật phải truyền --apply:

    python scripts/caption_images.py                     # chỉ đếm
    python scripts/caption_images.py --apply --limit 5   # thử vài ảnh trước
    python scripts/caption_images.py --apply             # chạy hết

Chạy lại chỉ làm phần còn thiếu: ảnh nào đã có hàng caption thì bỏ qua. Ảnh
lỗi cũng không sinh hàng caption, nên lượt sau tự thử lại.

Khâu detection mặc định TẮT: nó cần ultralytics chạy trên GPU, còn máy này chỉ
có CPU. Muốn bật thì truyền --with-detection.
"""

from __future__ import annotations

import argparse
import sys
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import select

PROJECT_ROOT = Path(__file__).resolve().parents[1]
API_ROOT = PROJECT_ROOT / "API"
sys.path.insert(0, str(API_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

from src.configuration import AppConfig  # noqa: E402
from src.rag_video_anh.pipeline.image_processing_worker import ImageProcessingWorker  # noqa: E402
from src.rag_video_anh.repository import MediaType, RepositoryUnitOfWork  # noqa: E402
from src.rag_video_anh.repository.models import CaptionResultModel, MediaModel  # noqa: E402

DEFAULT_WORKERS = 4
PROGRESS_EVERY = 25


def pending_media_ids(post_id: str | None, redo: bool) -> tuple[list[str], int]:
    """Trả về ảnh cần chạy và số ảnh đã có caption từ trước."""
    with RepositoryUnitOfWork() as uow:
        if uow.session is None:
            raise RuntimeError("RepositoryUnitOfWork did not expose session")

        query = select(MediaModel.media_id).where(MediaModel.media_type == MediaType.IMAGE.value)
        if post_id:
            query = query.where(MediaModel.post_id == post_id)
        media_ids = [str(row) for row in uow.session.scalars(query)]

        if redo:
            return media_ids, 0

        done = {str(row) for row in uow.session.scalars(select(CaptionResultModel.media_id))}
    pending = [media_id for media_id in media_ids if media_id not in done]
    return pending, len(media_ids) - len(pending)


def build_worker(with_detection: bool) -> ImageProcessingWorker:
    config = AppConfig()
    if not with_detection:
        # Không có GPU trên máy này; detection bằng CPU chậm mà caption không cần.
        config.media_pipeline.enable_detection = False
    return ImageProcessingWorker(config=config)


def run(media_ids: list[str], workers: int, with_detection: bool) -> Counter:
    worker = build_worker(with_detection)
    statuses: Counter = Counter()
    lock = threading.Lock()
    done = 0
    total = len(media_ids)

    def process(media_id: str) -> str:
        result = worker.process_media(media_id)
        return result.status.value

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(process, media_id): media_id for media_id in media_ids}
        for future in as_completed(futures):
            media_id = futures[future]
            try:
                status = future.result()
            except Exception as exc:
                status = f"crashed:{exc.__class__.__name__}"
                print(f"  ! {media_id}: {exc.__class__.__name__}: {exc}", flush=True)
            with lock:
                done += 1
                statuses[status] += 1
                if done % PROGRESS_EVERY == 0 or done == total:
                    print(f"  ... {done}/{total} ảnh | {dict(statuses)}", flush=True)
    return statuses


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Chạy OCR + caption cho ảnh tĩnh đã đăng ký.")
    parser.add_argument("--post-id", default=None, help="Chỉ xử lý ảnh của một bài.")
    parser.add_argument("--limit", type=int, default=None, help="Chỉ xử lý N ảnh đầu tiên.")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help=f"Số luồng. Mặc định: {DEFAULT_WORKERS}")
    parser.add_argument("--redo", action="store_true", help="Chạy lại cả ảnh đã có caption.")
    parser.add_argument(
        "--with-detection",
        action="store_true",
        help="Bật khâu detection (cần ultralytics; rất chậm nếu không có GPU).",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Gọi API thật và ghi DB. Không có cờ này thì chỉ đếm.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    media_ids, already_done = pending_media_ids(args.post_id, args.redo)
    if args.limit is not None:
        media_ids = media_ids[: args.limit]

    print(f"Đã có caption sẵn: {already_done} ảnh. Cần xử lý: {len(media_ids)} ảnh.")
    if not args.apply:
        print("Chưa gọi API, chưa ghi DB. Thêm --apply để thực hiện.")
        return
    if not media_ids:
        print("Không còn ảnh nào cần xử lý.")
        return

    print(f"Detection: {'BẬT' if args.with_detection else 'tắt'} | {args.workers} luồng")
    statuses = run(media_ids, max(1, args.workers), args.with_detection)
    print()
    for status, count in statuses.most_common():
        print(f"  {status}: {count}")


if __name__ == "__main__":
    main()
