"""Điều phối truy hồi media: nhúng câu hỏi rồi tìm keyframe + transcript.

Không viết lại truy vấn bằng LLM ở đây (khác `rag_noiquy`): mỗi request sẽ
thêm một lượt gọi LLM, mà quota free-tier là giới hạn thật. `query_rewriter`
của nhánh media vẫn để trống có chủ ý.
"""

from __future__ import annotations

from typing import Any

from src.configuration import AppConfig
from src.log.logger import logger
from src.rag_video_anh.embedding.embedding_service import ImageEmbeddingService
from src.rag_video_anh.retrieval.retriever import (
    MediaClipHit,
    TranscriptVideoHit,
    VideoRetriever,
)
from src.rag_video_anh.vector_store.vector_store import QdrantVideoVectorStore

SOURCE_CLIP = "clip"
SOURCE_TRANSCRIPT = "transcript"
SOURCE_BOTH = "both"
ALLOWED_SOURCES = (SOURCE_CLIP, SOURCE_TRANSCRIPT, SOURCE_BOTH)


class VideoRetrievalService:
    """Trả về keyframe và video khớp truy vấn, kèm context để tổng hợp câu trả lời."""

    def __init__(self, retriever: VideoRetriever) -> None:
        self.retriever = retriever

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        video_ids: list[str] | None = None,
        source: str = SOURCE_BOTH,
    ) -> dict[str, Any]:
        normalized_query = VideoRetriever.normalize_query(query)
        normalized_source = self._normalize_source(source)

        # Nhúng MỘT lần, dùng chung cho cả hai collection (không gian vector chung).
        query_vector = self.retriever.embed_query(normalized_query)

        clips: list[MediaClipHit] = []
        videos: list[TranscriptVideoHit] = []
        errors: list[str] = []

        if normalized_source in (SOURCE_CLIP, SOURCE_BOTH):
            try:
                clips = self.retriever.retrieve_clips(
                    normalized_query,
                    top_k=top_k,
                    video_ids=video_ids,
                    query_vector=query_vector,
                )
            except Exception as exc:
                # Một nhánh lỗi không được làm chết nhánh còn lại.
                errors.append(f"media_clip: {exc.__class__.__name__}")
                logger.warning(f"Media clip retrieval failed: {exc.__class__.__name__}: {exc}")

        if normalized_source in (SOURCE_TRANSCRIPT, SOURCE_BOTH):
            try:
                videos = self.retriever.retrieve_by_transcript(
                    normalized_query,
                    top_k=top_k,
                    video_ids=video_ids,
                    query_vector=query_vector,
                )
            except Exception as exc:
                errors.append(f"video_transcript: {exc.__class__.__name__}")
                logger.warning(f"Transcript retrieval failed: {exc.__class__.__name__}: {exc}")

        total = len(clips) + len(videos)
        logger.info(
            f"Media retrieval flow completed: {len(clips)} keyframe(s), {len(videos)} video(s)"
        )
        return {
            "query": normalized_query,
            "source": normalized_source,
            "clips": [clip.as_dict() for clip in clips],
            "videos": [video.as_dict() for video in videos],
            "context": self._format_context(clips, videos),
            "total": total,
            "found": total > 0,
            "errors": errors,
        }

    @classmethod
    def _format_context(
        cls,
        clips: list[MediaClipHit],
        videos: list[TranscriptVideoHit],
    ) -> str:
        """Ghép context có nhãn trích dẫn kèm mốc thời gian cho tầng trả lời."""
        parts: list[str] = []
        index = 0
        for clip in clips:
            index += 1
            if clip.is_image:
                # Ảnh không nằm trên trục thời gian: trích dẫn theo bài đăng.
                post_id = clip.payload.get("post_id")
                label = f"[{index}] ảnh trong bài {post_id}" if post_id else f"[{index}] ảnh"
            else:
                label = f"[{index}] video {clip.video_id or 'không rõ'}"
                timestamp = cls._format_timestamp(clip.timestamp_sec)
                if timestamp:
                    label = f"{label} tại {timestamp}"
            body = clip.caption or clip.ocr_text or "(không có caption)"
            if clip.caption and clip.ocr_text:
                body = f"{clip.caption}\nChữ trong hình: {clip.ocr_text}"
            parts.append(f"{label}\n{body}")

        for video in videos:
            index += 1
            label = f"[{index}] video {video.video_id or 'không rõ'} (lời thoại)"
            lines: list[str] = []
            for moment in video.moments:
                span = cls._format_span(moment.start_sec, moment.end_sec)
                text = moment.text.strip()
                if not text:
                    continue
                lines.append(f"{span} {text}" if span else text)
            parts.append("\n".join([label, *lines]))

        return "\n\n".join(parts)

    @classmethod
    def _format_span(cls, start_sec: float | None, end_sec: float | None) -> str:
        start = cls._format_timestamp(start_sec)
        end = cls._format_timestamp(end_sec)
        if start and end:
            return f"[{start}-{end}]"
        if start:
            return f"[{start}]"
        return ""

    @staticmethod
    def _format_timestamp(value: float | None) -> str:
        if value is None:
            return ""
        total_seconds = int(value)
        if total_seconds < 0:
            return ""
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours:
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        return f"{minutes:02d}:{seconds:02d}"

    @staticmethod
    def _normalize_source(source: str) -> str:
        normalized = str(source or SOURCE_BOTH).strip().lower()
        if normalized not in ALLOWED_SOURCES:
            raise ValueError(f"source phải thuộc {ALLOWED_SOURCES}, nhận được '{source}'")
        return normalized


def build_video_retrieval_service(config: AppConfig | None = None) -> VideoRetrievalService:
    app_config = config or AppConfig()
    return VideoRetrievalService(
        retriever=VideoRetriever(
            embedding_service=ImageEmbeddingService(config=app_config),
            vector_store=QdrantVideoVectorStore(config=app_config),
            config=app_config,
        )
    )
