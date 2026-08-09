"""Submit a registered MinIO video to the HIT-MIRA Runpod endpoint.

The trusted server creates short-lived presigned URLs, while the remote worker
only sees those URLs.  On completion this command downloads the ZIP artifact,
extracts it safely, and reuses the existing importer to write MinIO/PostgreSQL.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
import uuid
import zipfile
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from sqlalchemy import select


PROJECT_ROOT = Path(__file__).resolve().parents[1]
API_ROOT = PROJECT_ROOT / "API"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(API_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

from src.rag_video_anh.pipeline.minio_storage import MinioStorage  # noqa: E402
from src.rag_video_anh.pipeline.runpod_client import RunpodClient  # noqa: E402
from src.rag_video_anh.repository.models import MediaModel  # noqa: E402
from src.rag_video_anh.repository import RepositoryUnitOfWork  # noqa: E402
from scripts.import_media_outputs import ArtifactImporter  # noqa: E402


TERMINAL_STATUSES = {"COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"}


def _video_media(media_id: str):
    with RepositoryUnitOfWork() as uow:
        assert uow.media is not None
        media = uow.media.get_media(media_id)
    if media is None:
        raise ValueError(f"media_id does not exist: {media_id}")
    if media.media_type != "video":
        raise ValueError(f"media_id must be a video, got {media.media_type!r}")
    return media


def submit(
    media_id: str,
    *,
    language: str,
    expires_seconds: int,
    disable_stages: tuple[str, ...] = (),
) -> dict[str, Any]:
    media = _video_media(media_id)
    if media.media_id is None:
        raise ValueError("media row does not have media_id")
    storage = MinioStorage()
    run_id = uuid.uuid4().hex
    artifact_key = f"runpod-artifacts/{media.media_id}/{run_id}.zip"
    payload = {
        "media_id": str(media.media_id),
        "video_url": storage.presigned_download_url(
            media.object_key,
            bucket_name=media.bucket_name,
            expires_seconds=expires_seconds,
        ),
        "artifact_upload_url": storage.presigned_upload_url(
            artifact_key,
            bucket_name=media.bucket_name,
            expires_seconds=expires_seconds,
        ),
        "bucket_name": media.bucket_name,
        "source_object_key": media.object_key,
        "language": language,
    }
    # Caption/OCR chỉ là lời gọi API, GPU không giúp gì; tắt ở đây để làm tại
    # chỗ bằng model ta đang cấu hình, thay vì model nướng cứng trong image.
    for stage in disable_stages:
        payload[f"disable_{stage}"] = True
    response = RunpodClient().submit(payload)
    return {
        "runpod_job_id": response.get("id"),
        "runpod_status": response.get("status"),
        "media_id": str(media.media_id),
        "artifact_key": artifact_key,
    }


def list_videos(limit: int) -> list[dict[str, str]]:
    with RepositoryUnitOfWork() as uow:
        assert uow.session is not None
        rows = uow.session.scalars(
            select(MediaModel)
            .where(MediaModel.media_type == "video")
            .order_by(MediaModel.created_at.desc(), MediaModel.media_id)
            .limit(limit)
        ).all()
    return [
        {
            "media_id": str(row.media_id),
            "bucket_name": str(row.bucket_name),
            "object_key": row.object_key,
        }
        for row in rows
    ]


def _safe_extract(archive: Path, destination: Path) -> Path:
    with zipfile.ZipFile(archive) as bundle:
        root = destination.resolve()
        for member in bundle.infolist():
            target = (destination / member.filename).resolve()
            if target != root and root not in target.parents:
                raise ValueError("artifact ZIP contains an unsafe path")
        bundle.extractall(destination)
    manifests = list(destination.rglob("manifest.json"))
    if len(manifests) != 1:
        raise ValueError(f"expected exactly one manifest.json in artifact, found {len(manifests)}")
    return manifests[0].parent


def import_artifact(artifact_key: str) -> None:
    storage = MinioStorage()
    with tempfile.TemporaryDirectory(prefix="hit-mira-runpod-import-") as temp_dir:
        temporary = Path(temp_dir)
        archive = storage.download_file(artifact_key, temporary / "artifact.zip")
        run_dir = _safe_extract(archive, temporary / "artifact")
        ArtifactImporter(run_dir).import_all()


def wait_and_import(runpod_job_id: str, artifact_key: str, *, poll_seconds: int) -> dict[str, Any]:
    client = RunpodClient()
    while True:
        result = client.status(runpod_job_id)
        status = str(result.get("status") or "").upper()
        print(json.dumps({"runpod_job_id": runpod_job_id, "status": status}, ensure_ascii=False), flush=True)
        if status in TERMINAL_STATUSES:
            break
        time.sleep(poll_seconds)
    if status != "COMPLETED":
        raise RuntimeError(f"Runpod job {runpod_job_id} ended with status {status}: {result.get('error') or result.get('output')}")
    output = result.get("output") or {}
    if not output.get("artifact_uploaded"):
        raise RuntimeError(f"Runpod job completed without artifact upload: {output}")
    import_artifact(artifact_key)
    return result


def process_media_ids(
    media_ids: list[str],
    *,
    language: str,
    expires_seconds: int,
    poll_seconds: int,
    continue_on_error: bool,
    disable_stages: tuple[str, ...] = (),
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for media_id in media_ids:
        summary: dict[str, Any] = {"media_id": media_id, "status": "SUBMITTING"}
        print(json.dumps(summary, ensure_ascii=False), flush=True)
        try:
            submitted = submit(
                media_id,
                language=language,
                expires_seconds=expires_seconds,
                disable_stages=disable_stages,
            )
            summary.update(submitted)
            summary["status"] = str(submitted.get("runpod_status") or "SUBMITTED")
            print(json.dumps(summary, ensure_ascii=False), flush=True)
            wait_and_import(
                str(submitted["runpod_job_id"]),
                str(submitted["artifact_key"]),
                poll_seconds=poll_seconds,
            )
            summary["status"] = "IMPORTED"
        except Exception as exc:
            summary["status"] = "FAILED"
            summary["error"] = f"{exc.__class__.__name__}: {exc}"
            print(json.dumps(summary, ensure_ascii=False), flush=True)
            if not continue_on_error:
                raise
        summaries.append(summary)
    return summaries


STAGES = ("ocr", "caption", "detection", "asr")


def _add_disable_flags(parser: argparse.ArgumentParser) -> None:
    """Cho phép tắt từng khâu trên worker GPU."""
    for stage in STAGES:
        parser.add_argument(
            f"--disable-{stage}",
            action="store_true",
            help=f"Bỏ khâu {stage} trên worker GPU (làm tại chỗ sau).",
        )


def _disable_stages(args: argparse.Namespace) -> tuple[str, ...]:
    return tuple(stage for stage in STAGES if getattr(args, f"disable_{stage}", False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Submit, monitor, and import Runpod video-processing jobs.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    cancel_parser = subparsers.add_parser("cancel", help="Cancel a queued or in-progress Runpod job.")
    cancel_parser.add_argument("runpod_job_id")

    health_parser = subparsers.add_parser("health", help="Show Runpod endpoint health.")
    health_parser.set_defaults(command="health")

    status_parser = subparsers.add_parser("status", help="Show one Runpod job status.")
    status_parser.add_argument("runpod_job_id")

    list_parser = subparsers.add_parser("list", help="List registered video media IDs.")
    list_parser.add_argument("--limit", type=int, default=20)

    submit_parser = subparsers.add_parser("submit", help="Submit one registered video to Runpod.")
    submit_parser.add_argument("media_id")
    submit_parser.add_argument("--language", default="vi")
    submit_parser.add_argument("--expires-seconds", type=int, default=7200)
    _add_disable_flags(submit_parser)

    wait_parser = subparsers.add_parser("wait", help="Wait for one Runpod job, then import its artifact.")
    wait_parser.add_argument("runpod_job_id")
    wait_parser.add_argument("artifact_key")
    wait_parser.add_argument("--poll-seconds", type=int, default=10)

    process_parser = subparsers.add_parser("process", help="Submit, wait, and import one or more registered videos sequentially.")
    process_parser.add_argument("media_ids", nargs="*", help="Video media_id values to process. Use --all to process registered videos.")
    process_parser.add_argument("--all", action="store_true", help="Process registered video media rows from newest to oldest.")
    process_parser.add_argument("--limit", type=int, default=20, help="Maximum videos to process when --all is used.")
    process_parser.add_argument("--language", default="vi")
    process_parser.add_argument("--expires-seconds", type=int, default=7200)
    process_parser.add_argument("--poll-seconds", type=int, default=10)
    process_parser.add_argument("--continue-on-error", action="store_true", help="Continue with remaining videos after one failure.")
    _add_disable_flags(process_parser)

    args = parser.parse_args()
    if args.command == "cancel":
        print(json.dumps(RunpodClient().cancel(args.runpod_job_id), ensure_ascii=False))
        return
    if args.command == "health":
        print(json.dumps(RunpodClient().health(), ensure_ascii=False, indent=2))
        return
    if args.command == "status":
        print(json.dumps(RunpodClient().status(args.runpod_job_id), ensure_ascii=False, indent=2))
        return
    if args.command == "list":
        print(json.dumps(list_videos(args.limit), ensure_ascii=False, indent=2))
        return
    if args.command == "submit":
        print(json.dumps(submit(args.media_id, language=args.language,
                                expires_seconds=args.expires_seconds,
                                disable_stages=_disable_stages(args)), ensure_ascii=False))
        return
    if args.command == "wait":
        result = wait_and_import(args.runpod_job_id, args.artifact_key, poll_seconds=args.poll_seconds)
        print(json.dumps(result, ensure_ascii=False))
        return

    media_ids = list(args.media_ids)
    if args.all:
        media_ids.extend(item["media_id"] for item in list_videos(args.limit))
    media_ids = list(dict.fromkeys(media_ids))
    if not media_ids:
        raise SystemExit("provide at least one media_id or use --all")
    summaries = process_media_ids(
        media_ids,
        disable_stages=_disable_stages(args),
        language=args.language,
        expires_seconds=args.expires_seconds,
        poll_seconds=args.poll_seconds,
        continue_on_error=args.continue_on_error,
    )
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
