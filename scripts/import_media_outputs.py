"""Import video-processing artifacts into MinIO and PostgreSQL.

Expected artifact layout:

    <run-dir>/
      manifest.json
      frames/
        frame_000123.jpg
      ocr.json
      captions.json
      detections.json
      transcript.json

The script owns
the trusted writes to MinIO and PostgreSQL.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import sys
from pathlib import Path
from typing import Any
from uuid import UUID


PROJECT_ROOT = Path(__file__).resolve().parents[1]
API_ROOT = PROJECT_ROOT / "API"
sys.path.insert(0, str(API_ROOT))

from src.configuration import AppConfig  # noqa: E402
from src.rag_video_anh.pipeline.minio_storage import MinioStorage  # noqa: E402
from src.rag_video_anh.repository import (  # noqa: E402
    DetectedObjectCreate,
    FrameCreate,
    MediaRecord,
    OcrBoxCreate,
    ProcessingStatus,
    RepositoryUnitOfWork,
    TranscriptSegmentCreate,
    VideoMetadataCreate,
)


def load_json(path: Path, default: Any | None = None) -> Any:
    if not path.exists():
        if default is not None:
            return default
        raise FileNotFoundError(f"required artifact not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def status(value: str | None, default: str = ProcessingStatus.DONE.value) -> str:
    if not value:
        return default
    normalized = str(value).strip().upper()
    aliases = {
        "DONE": ProcessingStatus.DONE.value,
        "SUCCESS": ProcessingStatus.DONE.value,
        "PENDING": ProcessingStatus.PENDING.value,
        "PROCESSING": ProcessingStatus.PROCESSING.value,
        "ERROR": ProcessingStatus.FAILED.value,
        "FAILED": ProcessingStatus.FAILED.value,
        "FAILURE": ProcessingStatus.FAILED.value,
        "SKIPPED": ProcessingStatus.DONE.value,
        "NOT_FOUND": ProcessingStatus.DONE.value,
    }
    return aliases.get(normalized, normalized)


def frame_object_key(manifest: dict[str, Any], frame: dict[str, Any]) -> str:
    if frame.get("object_key"):
        return str(frame["object_key"])
    media_id = str(manifest["media_id"])
    source_key = str(manifest.get("source_object_key") or manifest.get("object_key") or manifest.get("video_file") or "video")
    stem = Path(source_key).stem or "video"
    frame_index = int(frame["frame_index"])
    return f"frames/{media_id}/{stem}_frame_{frame_index:06d}.jpg"


def frame_local_path(run_dir: Path, frame: dict[str, Any]) -> Path:
    relative_file = frame.get("file") or frame.get("image_path")
    if not relative_file:
        raise ValueError(f"frame artifact does not include file/image_path: {frame}")
    path = run_dir / str(relative_file)
    if not path.is_file():
        raise FileNotFoundError(f"frame image artifact not found: {path}")
    return path


def upload_frame(storage: MinioStorage, bucket_name: str, local_path: Path, object_key: str) -> None:
    content_type = mimetypes.guess_type(local_path.name)[0] or "image/jpeg"
    storage.client.fput_object(
        bucket_name=bucket_name,
        object_name=MinioStorage.normalize_object_key(object_key),
        file_path=str(local_path),
        content_type=content_type,
    )


def bbox_from_polygon(bbox: Any) -> dict[str, float | None]:
    if not bbox:
        return {"x1": None, "y1": None, "x2": None, "y2": None}
    if isinstance(bbox, list) and len(bbox) >= 4 and all(isinstance(item, (int, float)) for item in bbox[:4]):
        return {"x1": float(bbox[0]), "y1": float(bbox[1]), "x2": float(bbox[2]), "y2": float(bbox[3])}
    points = []
    for point in bbox:
        if isinstance(point, (list, tuple)) and len(point) >= 2:
            points.append((float(point[0]), float(point[1])))
    if not points:
        return {"x1": None, "y1": None, "x2": None, "y2": None}
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return {"x1": min(xs), "y1": min(ys), "x2": max(xs), "y2": max(ys)}


def result_by_frame(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(item["frame_id"]): item for item in payload.get("results", []) if item.get("frame_id")}


class ArtifactImporter:
    def __init__(self, run_dir: Path, config: AppConfig | None = None, storage: MinioStorage | None = None) -> None:
        self.run_dir = run_dir
        self.config = config or AppConfig()
        self.storage = storage or MinioStorage(config=self.config)
        self.manifest = load_json(run_dir / "manifest.json")
        self.media_id = UUID(str(self.manifest["media_id"]))
        self.bucket_name = str(self.manifest.get("bucket_name") or self.storage.bucket_name)

    def import_all(self) -> None:
        with RepositoryUnitOfWork() as uow:
            assert uow.media is not None
            assert uow.results is not None

            media = uow.media.get_media(self.media_id)
            if media is None:
                raise ValueError(f"video media row does not exist: {self.media_id}")
            if media.media_type != "video":
                raise ValueError(f"expected media_type='video' for {self.media_id}, got {media.media_type!r}")

            self._upsert_video_metadata(uow, media)
            frame_media_ids = self._import_frames(uow, media)
            self._import_ocr(uow, frame_media_ids)
            self._import_captions(uow, frame_media_ids)
            self._import_detections(uow, frame_media_ids)
            self._import_transcript(uow, media)

    def _upsert_video_metadata(self, uow: RepositoryUnitOfWork, media: MediaRecord) -> None:
        video = self.manifest.get("video", {}) or {}
        uow.media.upsert_video_metadata(
            VideoMetadataCreate(
                media_id=media.media_id,
                duration=video.get("duration_sec") or video.get("duration"),
                fps=video.get("fps"),
                raw_frames=video.get("frame_count") or video.get("raw_frames"),
                selected_keyframes=len(self.manifest.get("frames", [])),
            )
        )

    def _import_frames(self, uow: RepositoryUnitOfWork, media: MediaRecord) -> dict[str, UUID]:
        assert uow.media is not None
        frame_media_ids: dict[str, UUID] = {}
        existing_frames = uow.media.list_frames_for_video(media.media_id)
        existing_by_index = {frame.frame_index: frame for frame in existing_frames if frame.frame_index is not None}
        existing_by_key = {frame.object_key: frame for frame in existing_frames}

        for frame in self.manifest.get("frames", []):
            frame_id = str(frame["frame_id"])
            frame_index = int(frame["frame_index"])
            timestamp = float(frame.get("timestamp_sec") or 0.0)
            local_path = frame_local_path(self.run_dir, frame)
            object_key = frame_object_key(self.manifest, frame)

            upload_frame(self.storage, self.bucket_name, local_path, object_key)

            existing = existing_by_index.get(frame_index) or existing_by_key.get(object_key)
            if existing is not None and existing.media_id is not None:
                frame_media_ids[frame_id] = existing.media_id
                continue

            frame_record = uow.media.register_frame(
                FrameCreate(
                    video_media_id=media.media_id,
                    bucket_name=self.bucket_name,
                    object_key=object_key,
                    timestamp=timestamp,
                    frame_index=frame_index,
                )
            )
            frame_media_ids[frame_id] = frame_record.media_id
            existing_by_index[frame_index] = frame_record
            existing_by_key[object_key] = frame_record
        return frame_media_ids

    def _import_ocr(self, uow: RepositoryUnitOfWork, frame_media_ids: dict[str, UUID]) -> None:
        payload = load_json(self.run_dir / "ocr.json", default={"results": []})
        results = result_by_frame(payload)
        for frame_id, frame_media_id in frame_media_ids.items():
            item = results.get(frame_id, {})
            spans = item.get("text_spans") or item.get("spans") or []
            boxes = [
                OcrBoxCreate(
                    text=span.get("text"),
                    confidence=span.get("confidence"),
                    **bbox_from_polygon(span.get("bbox")),
                )
                for span in spans
            ]
            uow.results.upsert_ocr_result(
                frame_media_id,
                status=status(item.get("status"), status(payload.get("status"))),
                text=item.get("full_text") or item.get("text"),
                avg_confidence=item.get("confidence") or item.get("avg_confidence"),
                model=item.get("model") or payload.get("model") or "unknown-ocr",
                boxes=boxes,
            )

    def _import_captions(self, uow: RepositoryUnitOfWork, frame_media_ids: dict[str, UUID]) -> None:
        payload = load_json(self.run_dir / "captions.json", default={"results": []})
        results = result_by_frame(payload)
        for frame_id, frame_media_id in frame_media_ids.items():
            item = results.get(frame_id, {})
            generation_meta = item.get("generation_meta") or {}
            uow.results.upsert_caption_result(
                frame_media_id,
                status=status(item.get("status"), status(payload.get("status"))),
                caption_text=item.get("caption_text") or item.get("caption") or item.get("reason") or payload.get("reason"),
                model=item.get("model") or generation_meta.get("model") or payload.get("model") or "unknown-caption",
            )

    def _import_detections(self, uow: RepositoryUnitOfWork, frame_media_ids: dict[str, UUID]) -> None:
        payload = load_json(self.run_dir / "detections.json", default={"results": []})
        results = result_by_frame(payload)
        for frame_id, frame_media_id in frame_media_ids.items():
            item = results.get(frame_id, {})
            inference_meta = item.get("inference_meta") or {}
            objects = [
                DetectedObjectCreate(
                    label=detected.get("label"),
                    confidence=detected.get("confidence"),
                    **bbox_from_polygon(detected.get("bbox")),
                )
                for detected in item.get("detections", [])
            ]
            uow.results.upsert_object_result(
                frame_media_id,
                status=status(item.get("status"), status(payload.get("status"))),
                model=item.get("model") or inference_meta.get("model") or payload.get("model") or "unknown-detection",
                objects=objects,
            )

    def _import_transcript(self, uow: RepositoryUnitOfWork, media: MediaRecord) -> None:
        payload = load_json(self.run_dir / "transcript.json", default={"segments": []})
        transcription_meta = payload.get("transcription_meta") or {}
        segments = [
            TranscriptSegmentCreate(
                start_time=segment.get("start_sec") or segment.get("start_time"),
                end_time=segment.get("end_sec") or segment.get("end_time"),
                text=segment.get("text"),
            )
            for segment in payload.get("segments", [])
        ]
        full_text = payload.get("full_text") or " ".join(segment.text or "" for segment in segments).strip() or None
        uow.results.upsert_transcript_for_video_media(
            media.media_id,
            status=status(payload.get("status")),
            language=payload.get("language"),
            model=payload.get("model") or transcription_meta.get("model") or "unknown-asr",
            full_text=full_text,
            segments=segments,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Import video-processing artifacts into MinIO/PostgreSQL.")
    parser.add_argument("run_dir", help="Folder containing manifest.json and video-processing artifact JSON files.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    importer = ArtifactImporter(Path(args.run_dir).expanduser().resolve())
    importer.import_all()
    print(f"Imported artifacts for media_id={importer.media_id}")


if __name__ == "__main__":
    main()
