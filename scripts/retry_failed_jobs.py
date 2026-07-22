"""Reset failed processing jobs so workers can retry them.

Example:
    PYTHONPATH=API python scripts/retry_failed_jobs.py --task-type ocr
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import update


PROJECT_ROOT = Path(__file__).resolve().parents[1]
API_ROOT = PROJECT_ROOT / "API"
sys.path.insert(0, str(API_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

from src.rag_video_anh.repository.database import DatabaseSessionManager  # noqa: E402
from src.rag_video_anh.repository.models import ProcessingJobModel  # noqa: E402
from src.rag_video_anh.repository.schemas import ProcessingStatus, TaskType  # noqa: E402


def retry_failed_jobs(task_type: str | None = None) -> int:
    manager = DatabaseSessionManager()
    with manager.session() as session:
        stmt = (
            update(ProcessingJobModel)
            .where(ProcessingJobModel.status == ProcessingStatus.FAILED.value)
            .values(
                status=ProcessingStatus.PENDING.value,
                error_message=None,
                started_at=None,
                finished_at=None,
            )
        )
        if task_type:
            stmt = stmt.where(ProcessingJobModel.task_type == task_type)
        result = session.execute(stmt)
        return int(result.rowcount or 0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reset failed processing jobs to PENDING.")
    parser.add_argument(
        "--task-type",
        choices=[task.value for task in TaskType],
        help="Only retry failed jobs for one task type. Omit to retry all failed jobs.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    updated = retry_failed_jobs(task_type=args.task_type)
    scope = args.task_type or "all"
    print(f"Reset {updated} failed {scope} job(s) to PENDING.")


if __name__ == "__main__":
    main()
