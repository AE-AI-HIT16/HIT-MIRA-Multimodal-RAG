"""Embed and index clean video retrieval units into Qdrant."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from minio.error import S3Error

from src.log.logger import logger
from src.rag_video_anh.embedding.embedding_service import ImageEmbeddingService
from src.rag_video_anh.pipeline.minio_storage import MinioStorage
from src.rag_video_anh.retrieval.retrieval_units import (
    ImageRetrievalUnitBuilder,
    VideoRetrievalUnitBuilder,
)
from src.rag_video_anh.vector_store.vector_store import QdrantVideoVectorStore


DEFAULT_MEDIA_CLIP_COLLECTION = "media_clip"
DEFAULT_VIDEO_TRANSCRIPT_COLLECTION = "video_transcript"


@dataclass
class VideoRetrievalIndexSummary:
    video_id: str | None
    video_media_id: str
    media_clip_units_received: int = 0
    video_transcript_units_received: int = 0
    media_clip_indexed: int = 0
    video_transcript_indexed: int = 0
    skipped_by_reason: dict[str, int] = field(default_factory=dict)
    collections: dict[str, str] = field(default_factory=dict)

    def add_skip(self, reason: str, count: int = 1) -> None:
        self.skipped_by_reason[reason] = self.skipped_by_reason.get(reason, 0) + count

    def to_dict(self) -> dict[str, Any]:
        return {
            "video_id": self.video_id,
            "video_media_id": self.video_media_id,
            "media_clip_units_received": self.media_clip_units_received,
            "video_transcript_units_received": self.video_transcript_units_received,
            "media_clip_indexed": self.media_clip_indexed,
            "video_transcript_indexed": self.video_transcript_indexed,
            "skipped_by_reason": dict(sorted(self.skipped_by_reason.items())),
            "collections": self.collections,
        }


class VideoRetrievalIndexingService:
    """Build clean units, embed keyframes/text, and upsert video retrieval points."""

    def __init__(
        self,
        *,
        builder: Any | None = None,
        image_builder: Any | None = None,
        storage: Any | None = None,
        image_embedder: Any | None = None,
        text_embedder: Any | None = None,
        vector_store: Any | None = None,
        media_clip_collection: str = DEFAULT_MEDIA_CLIP_COLLECTION,
        video_transcript_collection: str = DEFAULT_VIDEO_TRANSCRIPT_COLLECTION,
    ) -> None:
        self.storage = storage
        if builder is None:
            self.storage = self.storage or MinioStorage()
            builder = VideoRetrievalUnitBuilder(storage=self.storage)
        self.builder = builder
        if image_builder is None:
            self.storage = self.storage or MinioStorage()
            image_builder = ImageRetrievalUnitBuilder(storage=self.storage)
        self.image_builder = image_builder
        self.image_embedder = image_embedder or ImageEmbeddingService()
        self.text_embedder = text_embedder or self.image_embedder
        self.vector_store = vector_store or QdrantVideoVectorStore()
        self.media_clip_collection = media_clip_collection
        self.video_transcript_collection = video_transcript_collection

    def index_video(self, video_media_id: str) -> VideoRetrievalIndexSummary:
        build_result = self.builder.build(video_media_id)
        builder_summary = build_result.summary.to_dict()
        summary = VideoRetrievalIndexSummary(
            video_id=builder_summary.get("video_id"),
            video_media_id=str(video_media_id),
            media_clip_units_received=len(build_result.media_clip_units),
            video_transcript_units_received=len(build_result.video_transcript_units),
            collections={
                "media_clip": self.media_clip_collection,
                "video_transcript": self.video_transcript_collection,
            },
        )
        for reason, count in (builder_summary.get("skipped_by_reason") or {}).items():
            if count:
                summary.add_skip(reason, int(count))

        # Hai nhánh độc lập: keyframe hỏng vẫn phải index được lời thoại, và ngược lại.
        try:
            with tempfile.TemporaryDirectory(prefix="hit-mira-video-index-") as temp_dir:
                self._index_media_clip_units(
                    [unit.to_dict() for unit in build_result.media_clip_units],
                    temp_dir=Path(temp_dir),
                    summary=summary,
                )
        except Exception as exc:
            summary.add_skip("media_clip_indexing_failed", summary.media_clip_units_received)
            logger.warning(
                f"Media clip indexing failed for video '{video_media_id}'; "
                f"continuing with transcripts: {exc.__class__.__name__}: {exc}"
            )

        try:
            self._index_video_transcript_units(
                [unit.to_dict() for unit in build_result.video_transcript_units],
                summary=summary,
            )
        except Exception as exc:
            summary.add_skip("video_transcript_indexing_failed", summary.video_transcript_units_received)
            logger.warning(
                f"Transcript indexing failed for video '{video_media_id}': "
                f"{exc.__class__.__name__}: {exc}"
            )
        logger.info(f"Indexed video retrieval units: {summary.to_dict()}")
        return summary

    def _index_media_clip_units(
        self,
        units: list[dict[str, Any]],
        *,
        temp_dir: Path,
        summary: VideoRetrievalIndexSummary,
    ) -> None:
        jobs: list[tuple[dict[str, Any], Path]] = []
        for unit in units:
            image_path = self._download_frame(unit, temp_dir, summary)
            if image_path is not None:
                jobs.append((unit, image_path))
        if not jobs:
            return

        vectors = self.image_embedder.embed_images([path for _, path in jobs])
        self._validate_vector_payload_alignment(vectors, jobs, label="media_clip")
        point_ids = [str(unit.get("frame_media_id") or unit.get("unit_id")) for unit, _ in jobs]
        payloads = [self._media_clip_payload(unit) for unit, _ in jobs]
        result = self.vector_store.upsert_points(
            collection_name=self.media_clip_collection,
            point_ids=point_ids,
            vectors=vectors,
            payloads=payloads,
        )
        summary.media_clip_indexed += int(result.get("upserted", len(payloads)))

    def _index_video_transcript_units(
        self,
        units: list[dict[str, Any]],
        *,
        summary: VideoRetrievalIndexSummary,
    ) -> None:
        valid_units = []
        for unit in units:
            text = str(unit.get("text") or "").strip()
            if not text:
                summary.add_skip("empty_transcript")
                continue
            copied = dict(unit)
            copied["text"] = text
            valid_units.append(copied)
        if not valid_units:
            return

        try:
            vectors = self.text_embedder.embed_texts([unit["text"] for unit in valid_units])
        except Exception as exc:
            logger.warning(
                f"Transcript batch embedding failed; retrying item-by-item: {exc.__class__.__name__}"
            )
            valid_units, vectors = self._embed_transcript_units_individually(valid_units, summary)
        if not valid_units:
            return

        self._validate_vector_payload_alignment(vectors, valid_units, label="video_transcript")
        payloads = [self._video_transcript_payload(unit) for unit in valid_units]
        point_ids = [str(unit.get("unit_id")) for unit in valid_units]
        result = self.vector_store.upsert_points(
            collection_name=self.video_transcript_collection,
            point_ids=point_ids,
            vectors=vectors,
            payloads=payloads,
        )
        summary.video_transcript_indexed += int(result.get("upserted", len(payloads)))

    def _embed_transcript_units_individually(
        self,
        units: list[dict[str, Any]],
        summary: VideoRetrievalIndexSummary,
    ) -> tuple[list[dict[str, Any]], list[list[float]]]:
        valid_units: list[dict[str, Any]] = []
        vectors: list[list[float]] = []
        for unit in units:
            try:
                unit_vectors = self.text_embedder.embed_texts([unit["text"]])
            except Exception as exc:
                summary.add_skip("transcript_embedding_failed")
                logger.warning(
                    f"Skipping transcript unit '{unit.get('unit_id')}' because embedding failed: "
                    f"{exc.__class__.__name__}"
                )
                continue
            if not unit_vectors:
                summary.add_skip("transcript_embedding_failed")
                continue
            valid_units.append(unit)
            vectors.append(unit_vectors[0])
        return valid_units, vectors

    def _download_frame(
        self,
        unit: dict[str, Any],
        temp_dir: Path,
        summary: VideoRetrievalIndexSummary,
    ) -> Path | None:
        bucket_name = str(unit.get("bucket_name") or "").strip()
        object_key = str(unit.get("frame_object_key") or "").strip()
        frame_media_id = str(unit.get("frame_media_id") or unit.get("unit_id") or "frame")
        if not bucket_name or not object_key:
            summary.add_skip("missing_frame_image")
            return None

        try:
            normalized_key = MinioStorage.normalize_object_key(object_key)
        except ValueError:
            summary.add_skip("missing_frame_image")
            return None

        suffix = Path(normalized_key).suffix or ".jpg"
        local_path = temp_dir / f"{frame_media_id}{suffix}"
        try:
            self._download_object(bucket_name, normalized_key, local_path)
        except S3Error as exc:
            if exc.code in {"NoSuchKey", "NoSuchObject", "NotFound", "NoSuchBucket"}:
                summary.add_skip("missing_frame_image")
                logger.warning(f"Skipping missing keyframe image '{bucket_name}/{normalized_key}'")
                return None
            raise
        except (FileNotFoundError, OSError, RuntimeError) as exc:
            summary.add_skip("frame_download_failed")
            logger.warning(
                f"Skipping keyframe image '{bucket_name}/{normalized_key}' "
                f"because download failed: {exc.__class__.__name__}"
            )
            return None

        if not local_path.is_file():
            summary.add_skip("frame_download_failed")
            return None
        return local_path

    def _download_object(self, bucket_name: str, object_key: str, local_path: Path) -> None:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        if self.storage is None:
            self.storage = MinioStorage()
        if hasattr(self.storage, "download_object"):
            self.storage.download_object(bucket_name, object_key, local_path)
            return
        if hasattr(self.storage, "client"):
            self.storage.client.fget_object(bucket_name, object_key, str(local_path))
            return
        if getattr(self.storage, "bucket_name", None) == bucket_name and hasattr(self.storage, "download_file"):
            self.storage.download_file(object_key, local_path)
            return
        raise RuntimeError("storage does not support bucket-specific object download")

    def index_images(self, image_media_ids: list[str]) -> VideoRetrievalIndexSummary:
        """Nhúng và index ảnh tĩnh vào CÙNG collection media_clip với keyframe.

        Cùng một không gian vector (Jina-CLIP v2) nên một truy vấn text duy nhất
        xếp hạng chung được cả ảnh lẫn keyframe video — đúng yêu cầu 'trả về ảnh
        kèm mô tả + clip video' trong PRD.
        """
        summary = VideoRetrievalIndexSummary(
            video_id=None,
            video_media_id="",
            collections={"media_clip": self.media_clip_collection},
        )
        units: list[dict[str, Any]] = []
        for media_id in image_media_ids:
            try:
                unit = self.image_builder.build(media_id)
            except Exception as exc:
                summary.add_skip("image_unit_build_failed")
                logger.warning(f"Could not build image unit '{media_id}': {exc.__class__.__name__}: {exc}")
                continue
            if unit is None:
                summary.add_skip("missing_image_object")
                continue
            units.append(unit.to_dict())
        summary.media_clip_units_received = len(units)
        if not units:
            return summary

        with tempfile.TemporaryDirectory(prefix="hit-mira-image-index-") as temp_dir:
            jobs: list[tuple[dict[str, Any], Path]] = []
            for unit in units:
                image_path = self._download_image(unit, Path(temp_dir), summary)
                if image_path is not None:
                    jobs.append((unit, image_path))
            if not jobs:
                return summary

            vectors = self.image_embedder.embed_images([path for _, path in jobs])
            self._validate_vector_payload_alignment(vectors, jobs, label="image")
            result = self.vector_store.upsert_points(
                collection_name=self.media_clip_collection,
                point_ids=[str(unit["image_media_id"]) for unit, _ in jobs],
                vectors=vectors,
                payloads=[self._image_payload(unit) for unit, _ in jobs],
            )
            summary.media_clip_indexed += int(result.get("upserted", len(jobs)))
        logger.info(f"Indexed image retrieval units: {summary.to_dict()}")
        return summary

    def _download_image(
        self,
        unit: dict[str, Any],
        temp_dir: Path,
        summary: VideoRetrievalIndexSummary,
    ) -> Path | None:
        # Dùng lại nguyên bộ xử lý lỗi của keyframe: khác tên khoá object thôi.
        return self._download_frame(
            {
                "bucket_name": unit.get("bucket_name"),
                "frame_object_key": unit.get("object_key"),
                "frame_media_id": unit.get("image_media_id"),
                "unit_id": unit.get("unit_id"),
            },
            temp_dir,
            summary,
        )

    @staticmethod
    def _image_payload(unit: dict[str, Any]) -> dict[str, Any]:
        return {
            "media_kind": "image",
            "unit_id": unit.get("unit_id"),
            "image_media_id": unit.get("image_media_id"),
            "post_id": unit.get("post_id"),
            "bucket_name": unit.get("bucket_name"),
            "frame_object_key": unit.get("object_key"),
            "caption": unit.get("caption") or "",
            "ocr_text": unit.get("ocr_text") or "",
            "vision_metadata": unit.get("vision_metadata") or {},
            "detected_objects": unit.get("detected_objects") or [],
            "object_counts": unit.get("object_counts") or {},
        }

    @staticmethod
    def _media_clip_payload(unit: dict[str, Any]) -> dict[str, Any]:
        context = unit.get("transcript_context") if isinstance(unit.get("transcript_context"), dict) else {}
        segments = context.get("segments") if isinstance(context.get("segments"), list) else []
        return {
            "media_kind": "video_frame",
            "unit_id": unit.get("unit_id"),
            "video_id": unit.get("video_id"),
            "post_id": unit.get("post_id"),
            "frame_media_id": unit.get("frame_media_id"),
            "frame_index": unit.get("frame_index"),
            "timestamp_sec": unit.get("timestamp_sec"),
            "bucket_name": unit.get("bucket_name"),
            "frame_object_key": unit.get("frame_object_key"),
            "caption": unit.get("caption") or "",
            "ocr_text": unit.get("ocr_text") or "",
            "vision_metadata": unit.get("vision_metadata") or {},
            "detected_objects": unit.get("detected_objects") or [],
            "object_counts": unit.get("object_counts") or {},
            "transcript_context": context,
            "transcript_context_start_sec": segments[0].get("start_sec") if segments else None,
            "transcript_context_end_sec": segments[-1].get("end_sec") if segments else None,
        }

    @staticmethod
    def _video_transcript_payload(unit: dict[str, Any]) -> dict[str, Any]:
        return {
            "unit_id": unit.get("unit_id"),
            "video_id": unit.get("video_id"),
            "post_id": unit.get("post_id"),
            "start_sec": unit.get("start_sec"),
            "end_sec": unit.get("end_sec"),
            "text": unit.get("text"),
            "language": unit.get("language"),
            "source_segment_ids": unit.get("source_segment_ids") or [],
        }

    @staticmethod
    def _validate_vector_payload_alignment(vectors: list[list[float]], items: list[Any], *, label: str) -> None:
        if len(vectors) != len(items):
            raise ValueError(f"{label} vector/payload length mismatch: {len(vectors)} != {len(items)}")
        if not vectors:
            raise ValueError(f"{label} vectors must not be empty")


VideoRetrievalIndexer = VideoRetrievalIndexingService
