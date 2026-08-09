"""Runpod queue handler for offline video processing.

The worker is deliberately artifact-only: it receives a short-lived HTTPS URL
for an input video, creates the repository.s portable
artifact folder, and optionally uploads one ZIP archive through another
short-lived HTTPS URL.  It never needs PostgreSQL, MinIO credentials, or the
Runpod API key.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import tempfile
import traceback
from importlib import metadata
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx
import runpod


MAX_DOWNLOAD_BYTES = int(os.getenv("MAX_DOWNLOAD_BYTES", str(2 * 1024 * 1024 * 1024)))
DEFAULT_TIMEOUT_SECONDS = int(os.getenv("PIPELINE_TIMEOUT_SECONDS", "3600"))


def _runpod_version() -> str:
    try:
        return metadata.version("runpod")
    except metadata.PackageNotFoundError:
        return "unknown"


def _log_startup() -> None:
    env_keys = sorted(
        key
        for key in os.environ
        if key.startswith(("RUNPOD", "RP_", "CONFIG_", "MODELS_", "PROMPTS_"))
    )
    print(
        "[runpod-worker] startup "
        f"python={platform.python_version()} "
        f"runpod={_runpod_version()} "
        f"cwd={Path.cwd()} "
        f"env_keys={env_keys}",
        flush=True,
    )


def _require_text(payload: dict[str, Any], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"input.{field} is required")
    return value.strip()


def _safe_suffix(url: str) -> str:
    suffix = Path(urlparse(url).path).suffix.lower()
    return suffix if suffix in {".mp4", ".mov", ".mkv", ".avi", ".webm"} else ".mp4"


def _download(url: str, target: Path) -> None:
    total = 0
    with httpx.stream("GET", url, follow_redirects=True, timeout=120.0) as response:
        response.raise_for_status()
        with target.open("wb") as output:
            for chunk in response.iter_bytes():
                total += len(chunk)
                if total > MAX_DOWNLOAD_BYTES:
                    raise ValueError(f"input video exceeds MAX_DOWNLOAD_BYTES ({MAX_DOWNLOAD_BYTES})")
                output.write(chunk)


def _upload(url: str, archive: Path) -> None:
    with archive.open("rb") as content:
        response = httpx.put(
            url,
            content=content,
            headers={"content-type": "application/zip"},
            follow_redirects=True,
            timeout=300.0,
        )
    response.raise_for_status()


def _export_artifacts(payload: dict[str, Any], work_dir: Path) -> Path:
    media_id = _require_text(payload, "media_id")
    video_url = _require_text(payload, "video_url")
    source_object_key = str(payload.get("source_object_key") or Path(urlparse(video_url).path).name)
    bucket_name = str(payload.get("bucket_name") or "hit-mira-media")
    language = str(payload.get("language") or "vi")

    video_path = work_dir / f"input{_safe_suffix(video_url)}"
    _download(video_url, video_path)

    command = [
        "python",
        "scripts/export_video_artifacts.py",
        "--video-path",
        str(video_path),
        "--media-id",
        media_id,
        "--output-root",
        str(work_dir / "outputs"),
        "--bucket-name",
        bucket_name,
        "--source-object-key",
        source_object_key,
        "--language",
        language,
    ]
    for input_name, option in (
        ("disable_ocr", "--disable-ocr"),
        ("disable_caption", "--disable-caption"),
        ("disable_detection", "--disable-detection"),
        ("disable_asr", "--disable-asr"),
    ):
        if payload.get(input_name) is True:
            command.append(option)

    completed = subprocess.run(
        command,
        cwd="/app",
        text=True,
        timeout=int(payload.get("pipeline_timeout_seconds") or DEFAULT_TIMEOUT_SECONDS),
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"pipeline exited with code {completed.returncode}")

    artifact_dir = work_dir / "outputs" / media_id
    if not (artifact_dir / "manifest.json").is_file():
        raise RuntimeError("pipeline did not create manifest.json")
    return artifact_dir


def handler(job: dict[str, Any]) -> dict[str, Any]:
    """Process one Runpod job and return an artifact reference, never video bytes."""
    print(
        "[runpod-worker] received job "
        f"id={job.get('id')} "
        f"input_keys={sorted((job.get('input') or {}).keys()) if isinstance(job.get('input'), dict) else 'invalid'}",
        flush=True,
    )
    payload = job.get("input") or {}
    if not isinstance(payload, dict):
        return {"error": "input must be an object"}

    try:
        media_id = _require_text(payload, "media_id")
        with tempfile.TemporaryDirectory(prefix="hit-mira-runpod-") as temporary_dir:
            artifact_dir = _export_artifacts(payload, Path(temporary_dir))
            archive_path = Path(temporary_dir) / f"{media_id}.zip"
            shutil.make_archive(str(archive_path.with_suffix("")), "zip", artifact_dir.parent, artifact_dir.name)

            upload_url = payload.get("artifact_upload_url")
            if upload_url:
                if not isinstance(upload_url, str) or not upload_url.startswith(("https://", "http://")):
                    raise ValueError("input.artifact_upload_url must be an HTTP(S) URL")
                _upload(upload_url, archive_path)

            return {
                "media_id": media_id,
                "status": "completed",
                "artifact_uploaded": bool(upload_url),
                "artifact_filename": archive_path.name,
                "artifact_size_bytes": archive_path.stat().st_size,
            }
    except Exception as exc:  # Runpod receives a structured, non-secret error.
        print(traceback.format_exc(), flush=True)
        return {"error": f"{exc.__class__.__name__}: {exc}"}


if __name__ == "__main__":
    _log_startup()
    runpod.serverless.start({"handler": handler})
