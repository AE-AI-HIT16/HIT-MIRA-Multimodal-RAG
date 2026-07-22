"""Create keyframe jobs for every existing video media row.

This is the server-side bootstrap step before running MediaTaskWorker loops.
It is idempotent for active/completed keyframe jobs: existing PENDING,
PROCESSING, or DONE keyframe jobs are not duplicated.
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
load_dotenv(PROJECT_ROOT / ".env")

from src.rag_video_anh.repository.database import DatabaseSessionManager  # noqa: E402
from src.rag_video_anh.repository.models import MediaModel, ProcessingJobModel  # noqa: E402
from src.rag_video_anh.repository.schemas import ProcessingStatus, TaskType  # noqa: E402


ACTIVE_STATUSES = {
    ProcessingStatus.PENDING.value,
    ProcessingStatus.PROCESSING.value,
    ProcessingStatus.DONE.value,
}


def enqueue_existing_videos(*, limit: int | None = None) -> tuple[int, int]:
    manager = DatabaseSessionManager()
    created = 0
    skipped = 0
    with manager.session() as session:
        stmt = select(MediaModel).where(MediaModel.media_type == "video").order_by(MediaModel.created_at, MediaModel.media_id)
        if limit is not None:
            stmt = stmt.limit(limit)

        for media in session.scalars(stmt).all():
            existing = session.scalar(
                select(ProcessingJobModel)
                .where(
                    ProcessingJobModel.media_id == media.media_id,
                    ProcessingJobModel.task_type == TaskType.KEYFRAME.value,
                    ProcessingJobModel.status.in_(ACTIVE_STATUSES),
                )
                .limit(1)
            )
            if existing is not None:
                skipped += 1
                continue

            session.add(ProcessingJobModel(media_id=media.media_id, task_type=TaskType.KEYFRAME.value))
            created += 1
    return created, skipped


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Enqueue keyframe jobs for existing video media rows.")
    parser.add_argument("--limit", type=int, help="Only inspect/enqueue up to this many video rows.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    created, skipped = enqueue_existing_videos(limit=args.limit)
    print(f"Created {created} keyframe job(s); skipped {skipped} existing active/completed job(s).")


if __name__ == "__main__":
    main()
