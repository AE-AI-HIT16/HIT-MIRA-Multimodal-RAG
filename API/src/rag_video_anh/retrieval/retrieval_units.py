"""Build clean retrieval units from parsed video artifacts in storage.

This module is intentionally pre-indexing only: it does not embed content,
write to Qdrant, or depend on debug JSON artifacts.  The builder returns
in-memory units that a later embedding/indexing stage can consume directly.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from datetime import timezone
from typing import Any
from uuid import UUID

from minio.error import S3Error

from src.log.logger import logger
from src.rag_video_anh.pipeline.minio_storage import MinioStorage
from src.rag_video_anh.repository import (
    CaptionResultRecord,
    DetectedObjectRecord,
    FrameRecord,
    MediaType,
    ObjectResultRecord,
    OcrResultRecord,
    RepositoryUnitOfWork,
    TranscriptRecord,
)

ERROR_TEXT_VALUES = {
    "[]",
    "{}",
    "error",
    "failed",
    "failure",
    "null",
    "none",
    "n/a",
    "na",
    "not_found",
    "not found",
    "pending",
    "processing",
    "skipped",
    "no caption",
    "no caption generated",
    "no objects",
    "no speech",
    "no speech detected",
    "no text",
    "no text detected",
    "no transcript",
    "khong co chu",
    "khong co phu de",
    "khong co van ban",
    "khong phat hien van ban",
    "không có chữ",
    "không có phụ đề",
    "không có văn bản",
    "không phát hiện văn bản",
}

ERROR_TEXT_PATTERNS = (
    re.compile(r"^(?:api\s+)?(?:error|failed|failure|exception|traceback|timeout|timed out)\b", re.IGNORECASE),
    re.compile(r"\b(?:traceback \(most recent call last\)|exception:|api error|connection error|rate limit)\b", re.IGNORECASE),
    re.compile(r"^(?:lỗi|loi)\b", re.IGNORECASE),
    re.compile(
        r"^(?:không thể|khong the).*(?:xử lý|xu ly|phân tích|phan tich|trích xuất|trich xuat|nhận diện|nhan dien)",
        re.IGNORECASE,
    ),
)

SUMMARY_REASON_KEYS = (
    "missing_frame_image",
    "empty_transcript",
    "invalid_timestamp",
    "invalid_frame_index",
    "invalid_detected_object",
)


@dataclass(frozen=True)
class TranscriptContextSegment:
    segment_id: str
    start_sec: float
    end_sec: float
    text: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "segment_id": self.segment_id,
            "start_sec": self.start_sec,
            "end_sec": self.end_sec,
            "text": self.text,
        }


@dataclass(frozen=True)
class TranscriptContext:
    text: str
    source_segment_ids: list[str]
    segments: list[TranscriptContextSegment]

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "source_segment_ids": self.source_segment_ids,
            "segments": [segment.to_dict() for segment in self.segments],
        }


@dataclass(frozen=True)
class ImageUnit:
    """Một ảnh tĩnh của bài đăng, sẵn sàng để nhúng và index.

    Khác `MediaClipUnit` ở chỗ KHÔNG có video_id/timestamp/transcript: ảnh
    không nằm trên trục thời gian nào cả, nên bịa mốc thời gian cho nó sẽ tạo
    ra trích dẫn sai.
    """

    unit_id: str
    image_media_id: str
    post_id: str
    bucket_name: str
    object_key: str
    caption: str
    ocr_text: str
    vision_metadata: dict[str, Any]
    detected_objects: list[dict[str, Any]]
    object_counts: dict[str, int]
    # Bản OCR còn nguyên xuống dòng, CHỈ dùng cho nhánh nhúng văn bản. Cố ý
    # không đưa vào `to_dict()`: payload của collection Jina là hợp đồng, thêm
    # khoá vào đó là đổi hợp đồng mà không ai yêu cầu.
    ocr_text_layout: str = ""
    # US-405.1: link bài gốc trên fanpage. Để None khi bài không có link —
    # tầng trả lời hiển thị "nguồn nội bộ" chứ không dựng một link gãy.
    source_url: str | None = None
    post_created_at: str | None = None
    event_key: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "image_media_id": self.image_media_id,
            "post_id": self.post_id,
            "source_url": self.source_url,
            "post_created_at": self.post_created_at,
            "event_key": self.event_key,
            "bucket_name": self.bucket_name,
            "object_key": self.object_key,
            "caption": self.caption,
            "ocr_text": self.ocr_text,
            "vision_metadata": self.vision_metadata,
            "detected_objects": self.detected_objects,
            "object_counts": self.object_counts,
        }


@dataclass(frozen=True)
class MediaClipUnit:
    video_id: str
    post_id: str
    frame_media_id: str
    frame_index: int
    timestamp_sec: float
    bucket_name: str
    frame_object_key: str
    caption: str
    ocr_text: str
    vision_metadata: dict[str, Any]
    detected_objects: list[dict[str, Any]]
    object_counts: dict[str, int]
    transcript_context: TranscriptContext
    unit_id: str
    video_media_id: str
    ocr_text_layout: str = ""
    source_url: str | None = None
    post_created_at: str | None = None
    event_key: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "video_id": self.video_id,
            "video_media_id": self.video_media_id,
            "post_id": self.post_id,
            "source_url": self.source_url,
            "post_created_at": self.post_created_at,
            "event_key": self.event_key,
            "frame_media_id": self.frame_media_id,
            "frame_index": self.frame_index,
            "timestamp_sec": self.timestamp_sec,
            "bucket_name": self.bucket_name,
            "frame_object_key": self.frame_object_key,
            "caption": self.caption,
            "ocr_text": self.ocr_text,
            "vision_metadata": self.vision_metadata,
            "detected_objects": self.detected_objects,
            "object_counts": self.object_counts,
            "transcript_context": self.transcript_context.to_dict(),
        }


@dataclass(frozen=True)
class VideoTranscriptUnit:
    unit_id: str
    video_id: str
    post_id: str
    start_sec: float
    end_sec: float
    text: str
    language: str
    source_segment_ids: list[str]
    source_url: str | None = None
    post_created_at: str | None = None
    event_key: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "video_id": self.video_id,
            "post_id": self.post_id,
            "source_url": self.source_url,
            "post_created_at": self.post_created_at,
            "event_key": self.event_key,
            "start_sec": self.start_sec,
            "end_sec": self.end_sec,
            "text": self.text,
            "language": self.language,
            "source_segment_ids": self.source_segment_ids,
        }


@dataclass
class RetrievalUnitBuildSummary:
    video_id: str | None
    video_media_id: str
    post_id: str | None
    media_clip_units_created: int = 0
    video_transcript_units_created: int = 0
    skipped_by_reason: dict[str, int] = field(default_factory=dict)
    skipped_items: list[dict[str, str]] = field(default_factory=list)

    def add_skip(
        self,
        reason: str,
        *,
        item_type: str,
        item_id: str | None = None,
        message: str | None = None,
    ) -> None:
        self.skipped_by_reason[reason] = self.skipped_by_reason.get(reason, 0) + 1
        item = {"reason": reason, "item_type": item_type}
        if item_id:
            item["item_id"] = item_id
        if message:
            item["message"] = message
        self.skipped_items.append(item)

    def to_dict(self) -> dict[str, Any]:
        skipped = {
            reason: self.skipped_by_reason.get(reason, 0)
            for reason in SUMMARY_REASON_KEYS
        }
        for reason, count in sorted(self.skipped_by_reason.items()):
            skipped.setdefault(reason, count)
        return {
            "video_id": self.video_id,
            "video_media_id": self.video_media_id,
            "post_id": self.post_id,
            "media_clip_units_created": self.media_clip_units_created,
            "video_transcript_units_created": self.video_transcript_units_created,
            "skipped_by_reason": skipped,
            "skipped_items": self.skipped_items,
        }


@dataclass(frozen=True)
class RetrievalUnitBuildResult:
    media_clip_units: list[MediaClipUnit]
    video_transcript_units: list[VideoTranscriptUnit]
    summary: RetrievalUnitBuildSummary

    def to_dict(self) -> dict[str, Any]:
        return {
            "media_clip_units": [unit.to_dict() for unit in self.media_clip_units],
            "video_transcript_units": [unit.to_dict() for unit in self.video_transcript_units],
            "summary": self.summary.to_dict(),
        }


@dataclass(frozen=True)
class CleanTranscriptSegment:
    segment_id: str
    start_sec: float
    end_sec: float
    text: str


class VideoRetrievalUnitBuilder:
    """Assemble reusable retrieval units from PostgreSQL rows and MinIO keys."""

    def __init__(
        self,
        *,
        uow_factory: Any | None = None,
        storage: Any | None = None,
        check_frame_exists: bool = True,
    ) -> None:
        self.uow_factory = uow_factory or RepositoryUnitOfWork
        self.storage = storage
        self.check_frame_exists = check_frame_exists

    def build(self, video_media_id: str | UUID) -> RetrievalUnitBuildResult:
        with self.uow_factory() as uow:
            if uow.media is None or uow.results is None:
                raise RuntimeError("RepositoryUnitOfWork did not expose media/results repositories")

            media = uow.media.get_media(video_media_id)
            if media is None:
                raise ValueError(f"video media row does not exist: {video_media_id}")
            if media.media_type != MediaType.VIDEO.value:
                raise ValueError(f"expected media_type='video' for {video_media_id}, got {media.media_type!r}")

            video = uow.media.get_video_by_media_id(media.media_id)
            if video is None:
                raise ValueError(f"video metadata does not exist for media_id: {video_media_id}")

            summary = RetrievalUnitBuildSummary(
                video_id=_str_id(video.video_id),
                video_media_id=_str_id(media.media_id),
                post_id=_str_id(media.post_id),
            )
            # Đọc một lần cho cả video: mọi keyframe và mọi đoạn lời thoại của
            # video này đều thuộc đúng một bài đăng.
            post_meta = _post_index_metadata(uow, media.post_id)
            transcript = uow.results.get_transcript_by_video_media_id(media.media_id)
            transcript_segments = _clean_transcript_segments(transcript, summary)
            transcript_units = _build_transcript_units(
                transcript=transcript,
                transcript_segments=transcript_segments,
                video_id=_str_id(video.video_id),
                post_id=_str_id(media.post_id),
                post_meta=post_meta,
            )

            media_clip_units: list[MediaClipUnit] = []
            for frame in uow.media.list_frames_for_video(media.media_id):
                unit = self._build_media_clip_unit(
                    frame=frame,
                    video_id=_str_id(video.video_id),
                    video_media_id=_str_id(media.media_id),
                    post_id=_str_id(media.post_id),
                    post_meta=post_meta,
                    transcript_segments=transcript_segments,
                    summary=summary,
                    results_repo=uow.results,
                )
                if unit is not None:
                    media_clip_units.append(unit)

            summary.media_clip_units_created = len(media_clip_units)
            summary.video_transcript_units_created = len(transcript_units)
            return RetrievalUnitBuildResult(
                media_clip_units=media_clip_units,
                video_transcript_units=transcript_units,
                summary=summary,
            )

    def _build_media_clip_unit(
        self,
        *,
        frame: FrameRecord,
        video_id: str,
        video_media_id: str,
        post_id: str,
        transcript_segments: list[CleanTranscriptSegment],
        summary: RetrievalUnitBuildSummary,
        results_repo: Any,
        post_meta: PostIndexMetadata | None = None,
    ) -> MediaClipUnit | None:
        post_meta = post_meta or PostIndexMetadata()
        frame_media_id = _str_id(frame.media_id)
        frame_index = _finite_int(frame.frame_index)
        if frame_index is None:
            summary.add_skip(
                "invalid_frame_index",
                item_type="media_clip",
                item_id=frame_media_id,
                message="frame_index is missing or invalid",
            )
            return None

        timestamp_sec = _finite_float(frame.timestamp)
        if timestamp_sec is None:
            summary.add_skip(
                "invalid_timestamp",
                item_type="media_clip",
                item_id=frame_media_id,
                message="frame timestamp is missing or invalid",
            )
            return None

        frame_object_key = _clean_object_key(frame.object_key)
        bucket_name = clean_text(frame.bucket_name) or "mira-data"
        if not frame_object_key or not self._frame_exists(bucket_name, frame_object_key):
            summary.add_skip(
                "missing_frame_image",
                item_type="media_clip",
                item_id=frame_media_id,
                message=f"{bucket_name}/{frame_object_key or '<empty object key>'}",
            )
            return None

        ocr_result = results_repo.get_ocr_result(frame.media_id)
        caption_result = results_repo.get_caption_result(frame.media_id)
        object_result = results_repo.get_object_result(frame.media_id)
        detected_objects = clean_detected_objects(object_result, summary, frame_media_id)

        return MediaClipUnit(
            unit_id=f"media_clip:{frame_media_id}",
            video_id=video_id,
            video_media_id=video_media_id,
            post_id=post_id,
            source_url=post_meta.source_url,
            post_created_at=post_meta.post_created_at,
            event_key=post_meta.event_key,
            frame_media_id=frame_media_id,
            frame_index=frame_index,
            timestamp_sec=timestamp_sec,
            bucket_name=bucket_name,
            frame_object_key=frame_object_key,
            caption=clean_caption(caption_result),
            ocr_text=clean_ocr(ocr_result),
            ocr_text_layout=clean_ocr_layout(ocr_result),
            vision_metadata=clean_vision_metadata(caption_result.vision_metadata if caption_result else None),
            detected_objects=detected_objects,
            object_counts=dict(sorted(Counter(item["label"] for item in detected_objects).items())),
            transcript_context=_transcript_context(timestamp_sec, transcript_segments),
        )

    def _frame_exists(self, bucket_name: str, object_key: str) -> bool:
        if not self.check_frame_exists:
            return True

        if self.storage is None:
            self.storage = MinioStorage()
        storage = self.storage
        if hasattr(storage, "object_exists"):
            return bool(storage.object_exists(bucket_name, object_key))

        normalized_key = MinioStorage.normalize_object_key(object_key)
        try:
            storage.client.stat_object(bucket_name, normalized_key)
            return True
        except S3Error as exc:
            if exc.code in {"NoSuchKey", "NoSuchObject", "NotFound", "NoSuchBucket"}:
                return False
            raise


class ImageRetrievalUnitBuilder:
    """Dựng đơn vị truy hồi từ một hàng media ảnh.

    Dùng chung toàn bộ hàm làm sạch với nhánh video (caption/OCR lỗi hoặc rỗng
    bị loại y hệt) để ảnh và keyframe có cùng chuẩn chất lượng.
    """

    def __init__(
        self,
        uow_factory: Any | None = None,
        storage: Any | None = None,
        check_image_exists: bool = True,
    ) -> None:
        self.uow_factory = uow_factory or RepositoryUnitOfWork
        self.storage = storage
        self.check_image_exists = check_image_exists
        # Builder ảnh được gọi RIÊNG cho từng ảnh, nên một bài 10 ảnh sẽ đọc
        # bảng posts 10 lần cho đúng một hàng. Nhánh video không cần cache vì nó
        # đọc một lần rồi dùng chung cho mọi keyframe.
        self._cache_post: dict[str, PostIndexMetadata] = {}

    def build(self, image_media_id: str | UUID) -> ImageUnit | None:
        """Trả về unit, hoặc None nếu ảnh không dùng được (kèm log lý do)."""
        with self.uow_factory() as uow:
            if uow.media is None or uow.results is None:
                raise RuntimeError("RepositoryUnitOfWork did not expose media/results repositories")

            media = uow.media.get_media(image_media_id)
            if media is None:
                raise ValueError(f"image media row does not exist: {image_media_id}")
            if media.media_type != MediaType.IMAGE.value:
                raise ValueError(f"expected media_type='image' for {image_media_id}, got {media.media_type!r}")

            media_id = _str_id(media.media_id)
            object_key = _clean_object_key(media.object_key)
            bucket_name = clean_text(media.bucket_name) or "mira-data"
            if not object_key or not self._image_exists(bucket_name, object_key):
                logger.warning(
                    f"Skipping image '{media_id}': object "
                    f"{bucket_name}/{object_key or '<empty object key>'} does not exist"
                )
                return None

            caption_result = uow.results.get_caption_result(media.media_id)
            object_result = uow.results.get_object_result(media.media_id)
            caption = clean_caption(caption_result)
            ocr_result = uow.results.get_ocr_result(media.media_id)
            ocr_text = clean_ocr(ocr_result)
            if not caption and not ocr_text:
                # Không caption cũng không chữ -> vector vẫn dựng được từ ảnh thô,
                # nhưng không có gì để trích dẫn. Index tiếp, chỉ cảnh báo.
                logger.warning(f"Image '{media_id}' has neither caption nor OCR text")

            summary = RetrievalUnitBuildSummary(video_id=None, video_media_id=media_id, post_id=_str_id(media.post_id))
            post_meta = self._metadata_bai(uow, media.post_id)
            detected_objects = clean_detected_objects(object_result, summary, media_id)

            return ImageUnit(
                unit_id=f"image:{media_id}",
                image_media_id=media_id,
                post_id=_str_id(media.post_id),
                source_url=post_meta.source_url,
                post_created_at=post_meta.post_created_at,
                event_key=post_meta.event_key,
                bucket_name=bucket_name,
                object_key=object_key,
                caption=caption,
                ocr_text=ocr_text,
                ocr_text_layout=clean_ocr_layout(ocr_result),
                vision_metadata=clean_vision_metadata(caption_result.vision_metadata if caption_result else None),
                detected_objects=detected_objects,
                object_counts=dict(sorted(Counter(item["label"] for item in detected_objects).items())),
            )

    def list_image_media_ids(self, post_id: str | UUID | None = None) -> list[str]:
        """Liệt kê media_id của ảnh, để tầng gọi lặp qua mà index."""
        with self.uow_factory() as uow:
            if uow.media is None:
                raise RuntimeError("RepositoryUnitOfWork did not expose the media repository")
            if post_id is not None:
                rows = uow.media.list_by_post(post_id, media_type=MediaType.IMAGE.value)
            else:
                rows = uow.media.list_by_type(MediaType.IMAGE.value)
            return [_str_id(row.media_id) for row in rows]

    def _metadata_bai(self, uow: Any, post_id: Any) -> PostIndexMetadata:
        khoa = _str_id(post_id) if post_id is not None else ""
        if khoa not in self._cache_post:
            self._cache_post[khoa] = _post_index_metadata(uow, post_id)
        return self._cache_post[khoa]

    def _image_exists(self, bucket_name: str, object_key: str) -> bool:
        if not self.check_image_exists:
            return True
        if self.storage is None:
            self.storage = MinioStorage()
        storage = self.storage
        if hasattr(storage, "object_exists"):
            return bool(storage.object_exists(bucket_name, object_key))
        try:
            storage.client.stat_object(bucket_name, MinioStorage.normalize_object_key(object_key))
            return True
        except S3Error as exc:
            if exc.code in {"NoSuchKey", "NoSuchObject", "NotFound", "NoSuchBucket"}:
                return False
            raise


def clean_caption(result: CaptionResultRecord | None) -> str:
    if result is None or not _is_done_status(result.caption_status):
        return ""
    return clean_text(result.caption_text)


def clean_ocr(result: OcrResultRecord | None) -> str:
    if result is None or not _is_done_status(result.ocr_status):
        return ""
    return clean_text(result.ocr_text)


def clean_ocr_layout(result: OcrResultRecord | None) -> str:
    """Như `clean_ocr` nhưng **giữ ranh giới dòng**, dành cho nhánh nhúng văn bản.

    `clean_text` gộp mọi khoảng trắng thành một dấu cách. Với payload thì không
    sao, nhưng đem đi nhúng thì "12h30 - 20/04/2024" dính luôn vào dòng kế tiếp
    và thành một câu không ai viết bao giờ. Bố cục của poster là thông tin, nên
    nhánh text cần bản còn nguyên xuống dòng.
    """
    if result is None or not _is_done_status(result.ocr_status):
        return ""
    value = result.ocr_text
    if not isinstance(value, str):
        return ""
    text = unicodedata.normalize("NFC", value)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]+", " ", text)
    dong = [re.sub(r"[ \t ]+", " ", d).strip() for d in text.split("\n")]
    ket = "\n".join(d for d in dong if d)
    # Dùng lại đúng bộ lọc "trông như thông báo lỗi" của `clean_text`, để một
    # nhánh không nhận vào thứ mà nhánh kia đã loại.
    return "" if not ket or _is_error_like_text(ket) else ket


def clean_text(value: Any) -> str:
    """Clean text while preserving Vietnamese accents and real content."""
    if value is None or not isinstance(value, str):
        return ""
    text = unicodedata.normalize("NFC", value)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text or _is_error_like_text(text):
        return ""
    return text


def clean_vision_metadata(value: Any) -> dict[str, Any]:
    cleaned = _clean_json_value(value)
    return cleaned if isinstance(cleaned, dict) else {}


def clean_detected_objects(
    result: ObjectResultRecord | None,
    summary: RetrievalUnitBuildSummary | None = None,
    frame_media_id: str | None = None,
) -> list[dict[str, Any]]:
    objects = result.objects if result and _is_done_status(result.status) else []
    cleaned_objects: list[dict[str, Any]] = []
    for detected in objects:
        cleaned = _clean_detected_object(detected)
        if cleaned is None:
            if summary is not None:
                summary.add_skip(
                    "invalid_detected_object",
                    item_type="detected_object",
                    item_id=_str_id(detected.object_id),
                    message=f"frame_media_id={frame_media_id}" if frame_media_id else None,
                )
            continue
        cleaned_objects.append(cleaned)
    return cleaned_objects


def _clean_transcript_segments(
    transcript: TranscriptRecord | None,
    summary: RetrievalUnitBuildSummary,
) -> list[CleanTranscriptSegment]:
    cleaned_segments: list[CleanTranscriptSegment] = []
    for index, segment in enumerate(transcript.segments if transcript else []):
        segment_id = _str_id(segment.segment_id) or f"segment-{index}"
        start_sec = _finite_float(segment.start_time)
        end_sec = _finite_float(segment.end_time)
        text = clean_text(segment.text)

        if start_sec is None or end_sec is None or end_sec < start_sec:
            summary.add_skip(
                "invalid_timestamp",
                item_type="video_transcript",
                item_id=segment_id,
                message="segment start/end timestamp is missing or invalid",
            )
            continue
        if not text:
            summary.add_skip(
                "empty_transcript",
                item_type="video_transcript",
                item_id=segment_id,
                message="segment text is empty or error-like after cleaning",
            )
            continue

        cleaned_segments.append(
            CleanTranscriptSegment(
                segment_id=segment_id,
                start_sec=start_sec,
                end_sec=end_sec,
                text=text,
            )
        )
    return cleaned_segments


def _build_transcript_units(
    *,
    transcript: TranscriptRecord | None,
    transcript_segments: list[CleanTranscriptSegment],
    video_id: str,
    post_id: str,
    post_meta: "PostIndexMetadata | None" = None,
) -> list[VideoTranscriptUnit]:
    language = clean_text(transcript.language if transcript else None) or "unknown"
    meta = post_meta or PostIndexMetadata()
    return [
        VideoTranscriptUnit(
            unit_id=f"video_transcript:{video_id}:{segment.segment_id}",
            video_id=video_id,
            post_id=post_id,
            source_url=meta.source_url,
            post_created_at=meta.post_created_at,
            event_key=meta.event_key,
            start_sec=segment.start_sec,
            end_sec=segment.end_sec,
            text=segment.text,
            language=language,
            source_segment_ids=[segment.segment_id],
        )
        for segment in transcript_segments
    ]


@dataclass(frozen=True)
class PostIndexMetadata:
    """Metadata cấp BÀI, đọc một lần rồi truyền xuống mọi ảnh/keyframe/lời thoại.

    Đọc lại theo từng point thì một video 200 keyframe là 200 lượt truy vấn
    posts cho đúng một hàng.
    """

    source_url: str | None = None
    post_created_at: str | None = None
    event_key: str | None = None


def _rfc3339(value: Any) -> str | None:
    """Chuỗi RFC3339 **có múi giờ**, để Qdrant đánh index datetime đọc được.

    `parse_created_time()` trong `minio_registration.py` gỡ timezone trước khi
    ghi, nên `posts.created_time` trong PostgreSQL là datetime *naïve*. Gọi
    thẳng `isoformat()` sẽ ra `2024-04-11T12:59:01` — thiếu `Z` hoặc offset, tức
    KHÔNG đúng RFC3339, và Qdrant có quyền từ chối hoặc hiểu theo múi giờ khác.

    `created_time` của Facebook luôn là `+0000`, và đó đúng là thứ đã bị gỡ, nên
    coi datetime naïve là UTC là khôi phục thông tin gốc chứ không phải đoán bừa.
    """
    if value is None:
        return None
    if isinstance(value, str):
        return clean_text(value) or None
    try:
        moc = value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)
    except AttributeError:
        return None
    return moc.isoformat().replace("+00:00", "Z")


def _post_index_metadata(uow: Any, post_id: Any) -> PostIndexMetadata:
    """Đọc link, thời điểm đăng và khoá sự kiện của một bài.

    Nuốt mọi lỗi và trả None: thiếu metadata chỉ làm trích dẫn kém đẹp và mất
    một chiều lọc, không đáng để chặn cả mẻ index. Cũng nhờ vậy mà các Unit of
    Work giả trong test — vốn chỉ dựng `media`/`results` — vẫn chạy được như cũ.
    """
    if post_id is None:
        return PostIndexMetadata()

    source_url = post_created_at = None
    posts_repo = getattr(uow, "posts", None)
    if posts_repo is not None:
        try:
            post = posts_repo.get(post_id)
        except Exception as exc:  # pragma: no cover - phụ thuộc hạ tầng
            logger.warning(f"Could not read post '{post_id}': {exc.__class__.__name__}: {exc}")
            post = None
        if post is not None:
            source_url = clean_text(getattr(post, "post_url", None)) or None
            post_created_at = _rfc3339(getattr(post, "created_time", None))

    # Bài chưa gán sự kiện, hoặc gán nhiều sự kiện mà không cái nào là chính,
    # đều trả None chứ KHÔNG ghi "unknown" — một khoá sự kiện bịa ra sẽ gom nhầm
    # mọi bài chưa nhận diện được vào cùng một nhóm. Vẫn dùng `getattr` vì các
    # Unit of Work giả trong test chỉ dựng `media`/`results`.
    event_key = None
    events_repo = getattr(uow, "events", None)
    if events_repo is not None:
        try:
            event_key = clean_text(events_repo.primary_series_slug(post_id)) or None
        except Exception as exc:  # pragma: no cover
            logger.warning(f"Could not read event of post '{post_id}': {exc.__class__.__name__}: {exc}")

    return PostIndexMetadata(source_url=source_url, post_created_at=post_created_at, event_key=event_key)


def _transcript_context(
    timestamp_sec: float,
    transcript_segments: list[CleanTranscriptSegment],
) -> TranscriptContext:
    current_index = None
    for index, segment in enumerate(transcript_segments):
        if segment.start_sec <= timestamp_sec <= segment.end_sec:
            current_index = index
            break

    if current_index is None:
        return TranscriptContext(text="", source_segment_ids=[], segments=[])

    start_index = max(0, current_index - 1)
    end_index = min(len(transcript_segments), current_index + 2)
    context_segments = [
        TranscriptContextSegment(
            segment_id=segment.segment_id,
            start_sec=segment.start_sec,
            end_sec=segment.end_sec,
            text=segment.text,
        )
        for segment in transcript_segments[start_index:end_index]
    ]
    return TranscriptContext(
        text=" ".join(segment.text for segment in context_segments).strip(),
        source_segment_ids=[segment.segment_id for segment in context_segments],
        segments=context_segments,
    )


def _clean_json_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        return clean_text(value) or None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value if _finite_float(value) is not None else None
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, dict):
        cleaned_dict: dict[str, Any] = {}
        for key, raw_item in value.items():
            clean_key = clean_text(str(key))
            if not clean_key:
                continue
            clean_item = _clean_json_value(raw_item)
            if not _empty_cleaned_value(clean_item):
                cleaned_dict[clean_key] = clean_item
        return cleaned_dict or None
    if isinstance(value, (list, tuple, set)):
        cleaned_list = []
        for raw_item in value:
            clean_item = _clean_json_value(raw_item)
            if not _empty_cleaned_value(clean_item):
                cleaned_list.append(clean_item)
        return cleaned_list or None
    return clean_text(str(value)) or None


def _clean_detected_object(detected: DetectedObjectRecord) -> dict[str, Any] | None:
    label = clean_text(detected.label).lower()
    if not label:
        return None

    item: dict[str, Any] = {
        "object_id": _str_id(detected.object_id),
        "label": label,
    }
    confidence = _finite_float(detected.confidence)
    if confidence is not None:
        item["confidence"] = confidence

    bbox = {
        "x1": _finite_float(detected.x1),
        "y1": _finite_float(detected.y1),
        "x2": _finite_float(detected.x2),
        "y2": _finite_float(detected.y2),
    }
    clean_bbox = {key: value for key, value in bbox.items() if value is not None}
    if clean_bbox:
        item["bbox"] = clean_bbox
    return item


def _clean_object_key(value: Any) -> str:
    if value is None:
        return ""
    return str(value).replace("\\", "/").strip("/")


def _empty_cleaned_value(value: Any) -> bool:
    return value is None or value == "" or value == [] or value == {}


def _is_done_status(status: Any) -> bool:
    return str(status or "").strip().upper() in {"DONE", "SUCCESS"}


def _is_error_like_text(text: str) -> bool:
    normalized = text.strip().lower()
    if normalized in ERROR_TEXT_VALUES:
        return True
    ascii_normalized = unicodedata.normalize("NFKD", normalized).encode("ascii", "ignore").decode("ascii")
    if ascii_normalized in ERROR_TEXT_VALUES:
        return True
    return any(pattern.search(text) for pattern in ERROR_TEXT_PATTERNS)


def _finite_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def _finite_int(value: Any) -> int | None:
    parsed = _finite_float(value)
    if parsed is None:
        return None
    return int(parsed)


def _str_id(value: Any) -> str:
    return "" if value is None else str(value)
