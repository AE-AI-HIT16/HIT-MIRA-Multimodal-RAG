"""Truy hồi keyframe và transcript video từ Qdrant.

Jina-CLIP v2 nhúng ảnh và text vào CÙNG một không gian vector, nên chỉ cần
nhúng câu hỏi (dạng text) MỘT lần rồi dùng chung vector đó để tìm trong cả
`media_clip` (vector ảnh keyframe) và `video_transcript` (vector text) —
tiết kiệm một lượt gọi API embedding cho mỗi request.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.configuration import AppConfig
from src.log.logger import logger
from src.rag_video_anh.embedding.provider import build_media_embedder
from src.rag_video_anh.retrieval.indexing_service import (
    DEFAULT_MEDIA_CLIP_COLLECTION,
    DEFAULT_VIDEO_TRANSCRIPT_COLLECTION,
)
from src.rag_video_anh.vector_store.vector_store import QdrantVideoVectorStore

DEFAULT_TOP_K = 5
MAX_TOP_K = 50

# Ảnh tĩnh và keyframe video nằm chung collection media_clip, phân biệt bằng khoá này.
MEDIA_KIND_IMAGE = "image"
MEDIA_KIND_VIDEO_FRAME = "video_frame"


@dataclass(frozen=True)
class MediaClipHit:
    """Một hình ảnh khớp truy vấn: keyframe của video, hoặc ảnh tĩnh của bài đăng.

    Cả hai nằm chung collection `media_clip` vì dùng chung không gian vector
    Jina-CLIP v2. `media_kind` cho biết đang là loại nào — ảnh tĩnh không có
    `video_id` lẫn `timestamp_sec`, nên tầng trên không được trích dẫn mốc
    thời gian cho nó.
    """

    score: float
    video_id: str | None
    unit_id: str | None
    timestamp_sec: float | None
    caption: str
    ocr_text: str
    bucket_name: str | None
    frame_object_key: str | None
    media_kind: str = MEDIA_KIND_VIDEO_FRAME
    payload: dict[str, Any] = field(default_factory=dict)

    @property
    def is_image(self) -> bool:
        return self.media_kind == MEDIA_KIND_IMAGE

    def as_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "media_kind": self.media_kind,
            "video_id": self.video_id,
            "unit_id": self.unit_id,
            "timestamp_sec": self.timestamp_sec,
            "caption": self.caption,
            "ocr_text": self.ocr_text,
            "bucket_name": self.bucket_name,
            "frame_object_key": self.frame_object_key,
            "post_id": self.payload.get("post_id"),
            # US-405.1: None khi bài không có link — tầng trả lời hiển thị
            # "nguồn nội bộ" thay vì dựng một link gãy.
            "source_url": self.payload.get("source_url"),
            # Thời điểm của bài gốc, để tầng trả lời nói được "ảnh từ HIT Open
            # Day 2025" thay vì chỉ đưa ra một tấm ảnh không mốc thời gian.
            # None với các điểm index trước khi hai khoá này tồn tại.
            "post_created_at": self.payload.get("post_created_at"),
            "event_key": self.payload.get("event_key"),
            "image_media_id": self.payload.get("image_media_id"),
            "frame_index": self.payload.get("frame_index"),
            "detected_objects": self.payload.get("detected_objects") or [],
        }


@dataclass(frozen=True)
class TranscriptMoment:
    """Một đoạn transcript khớp truy vấn trong một video."""

    score: float
    start_sec: float | None
    end_sec: float | None
    text: str
    unit_id: str | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "start_sec": self.start_sec,
            "end_sec": self.end_sec,
            "text": self.text,
            "unit_id": self.unit_id,
        }


@dataclass(frozen=True)
class TranscriptVideoHit:
    """Các đoạn transcript của CÙNG một video đã gộp lại thành 1 kết quả."""

    video_id: str | None
    score: float
    moments: list[TranscriptMoment]
    post_id: str | None = None
    language: str | None = None
    source_url: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "video_id": self.video_id,
            "score": self.score,
            "post_id": self.post_id,
            "source_url": self.source_url,
            "language": self.language,
            "moments": [moment.as_dict() for moment in self.moments],
        }


class VideoRetriever:
    """Nhúng câu hỏi rồi tìm trong hai collection video."""

    def __init__(
        self,
        embedding_service: Any | None = None,
        vector_store: Any | None = None,
        top_k: int | None = None,
        media_clip_collection: str = DEFAULT_MEDIA_CLIP_COLLECTION,
        video_transcript_collection: str = DEFAULT_VIDEO_TRANSCRIPT_COLLECTION,
        config: AppConfig | None = None,
    ) -> None:
        app_config = config or AppConfig()
        retrieval_config = getattr(app_config, "retrieval", None)
        self.embedding_service = embedding_service or build_media_embedder(
            app_config, for_online_queries=True
        )
        self.vector_store = vector_store or QdrantVideoVectorStore(config=app_config)
        self.top_k = int(top_k or getattr(retrieval_config, "top_k", None) or DEFAULT_TOP_K)
        self.media_clip_collection = media_clip_collection
        self.video_transcript_collection = video_transcript_collection

    def embed_query(self, query: str) -> list[float]:
        normalized_query = self.normalize_query(query)
        vectors = self.embedding_service.embed_texts([normalized_query])
        if not vectors or not vectors[0]:
            raise ValueError("embedding service returned no query vector")
        return list(vectors[0])

    def embed_image_query(self, image: bytes) -> list[float]:
        """Nhúng ảnh truy vấn vào ĐÚNG không gian vector của `media_clip`.

        Điểm mấu chốt của US-303.1: `media_clip` chứa vector ẢNH, nên so ảnh
        với ảnh là so cùng loại vector — không còn phải đi vòng qua việc tả ảnh
        bằng lời rồi tìm bằng text.
        """
        vectors = self.embedding_service.embed_image_blobs([image])
        if not vectors or not vectors[0]:
            raise ValueError("embedding service returned no query vector")
        return list(vectors[0])

    def retrieve_clips(
        self,
        query: str | None,
        top_k: int | None = None,
        video_ids: list[str] | None = None,
        query_vector: list[float] | None = None,
        years: list[int] | None = None,
    ) -> list[MediaClipHit]:
        vector = query_vector if query_vector is not None else self.embed_query(query)
        points = self.vector_store.search_points(
            collection_name=self.media_clip_collection,
            vector=vector,
            limit=self._resolve_top_k(top_k),
            query_filter=self._search_filter(video_ids, years),
        )
        hits = [self._as_clip_hit(point) for point in points]
        logger.info(f"Media clip retrieval returned {len(hits)} keyframe(s)")
        return hits

    def retrieve_by_transcript(
        self,
        query: str | None,
        top_k: int | None = None,
        video_ids: list[str] | None = None,
        query_vector: list[float] | None = None,
        years: list[int] | None = None,
    ) -> list[TranscriptVideoHit]:
        """Tìm transcript rồi GỘP theo video_id để một video không bị đếm hai lần.

        Điểm của video = điểm cao nhất trong các đoạn khớp; các đoạn còn lại
        được giữ trong `moments` để tầng trả lời trích dẫn mốc thời gian.
        """
        requested_k = self._resolve_top_k(top_k)
        vector = query_vector if query_vector is not None else self.embed_query(query)
        # Lấy dư ứng viên vì nhiều đoạn có thể thuộc cùng một video, gộp lại
        # sẽ còn ít hơn requested_k.
        candidate_k = min(requested_k * 3, MAX_TOP_K * 3)
        points = self.vector_store.search_points(
            collection_name=self.video_transcript_collection,
            vector=vector,
            limit=candidate_k,
            query_filter=self._search_filter(video_ids, years),
        )
        merged = self._merge_transcript_points(points)
        logger.info(
            f"Transcript retrieval merged {len(points)} segment(s) into {len(merged)} video(s)"
        )
        return merged[:requested_k]

    @classmethod
    def _merge_transcript_points(cls, points: list[dict[str, Any]]) -> list[TranscriptVideoHit]:
        grouped: dict[str, dict[str, Any]] = {}
        order: list[str] = []
        for point in points:
            payload = point.get("payload") or {}
            score = float(point.get("score") or 0.0)
            video_id = payload.get("video_id")
            # Đoạn không có video_id thì không gộp được -> tự thành một nhóm
            # riêng theo unit_id để không bị trộn lẫn với video khác.
            group_key = str(video_id) if video_id else f"__unit__{payload.get('unit_id')}"
            moment = TranscriptMoment(
                score=score,
                start_sec=cls._as_float(payload.get("start_sec")),
                end_sec=cls._as_float(payload.get("end_sec")),
                text=str(payload.get("text") or ""),
                unit_id=cls._as_optional_str(payload.get("unit_id")),
            )
            if group_key not in grouped:
                grouped[group_key] = {
                    "video_id": cls._as_optional_str(video_id),
                    "post_id": cls._as_optional_str(payload.get("post_id")),
                    "source_url": cls._as_optional_str(payload.get("source_url")),
                    "language": cls._as_optional_str(payload.get("language")),
                    "score": score,
                    "moments": [moment],
                }
                order.append(group_key)
                continue
            group = grouped[group_key]
            group["score"] = max(float(group["score"]), score)
            group["moments"].append(moment)
            # Điểm cũ (index trước US-405.1) không có source_url; đoạn nào có thì lấy.
            if not group.get("source_url"):
                group["source_url"] = cls._as_optional_str(payload.get("source_url"))

        hits = [
            TranscriptVideoHit(
                video_id=grouped[key]["video_id"],
                score=float(grouped[key]["score"]),
                post_id=grouped[key]["post_id"],
                source_url=grouped[key]["source_url"],
                language=grouped[key]["language"],
                moments=sorted(
                    grouped[key]["moments"],
                    key=lambda moment: moment.score,
                    reverse=True,
                ),
            )
            for key in order
        ]
        hits.sort(key=lambda hit: hit.score, reverse=True)
        return hits

    @classmethod
    def _as_clip_hit(cls, point: dict[str, Any]) -> MediaClipHit:
        payload = point.get("payload") or {}
        return MediaClipHit(
            score=float(point.get("score") or 0.0),
            video_id=cls._as_optional_str(payload.get("video_id")),
            unit_id=cls._as_optional_str(payload.get("unit_id")),
            timestamp_sec=cls._as_float(payload.get("timestamp_sec")),
            caption=str(payload.get("caption") or ""),
            ocr_text=str(payload.get("ocr_text") or ""),
            bucket_name=cls._as_optional_str(payload.get("bucket_name")),
            frame_object_key=cls._as_optional_str(payload.get("frame_object_key")),
            media_kind=cls._as_media_kind(payload),
            payload=dict(payload),
        )

    @staticmethod
    def _as_media_kind(payload: dict[str, Any]) -> str:
        """Điểm cũ được index trước khi có `media_kind` vẫn phải đọc đúng.

        Không có khoá này thì suy ra từ dữ liệu: có video_id là keyframe video.
        """
        raw = payload.get("media_kind")
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
        return MEDIA_KIND_VIDEO_FRAME if payload.get("video_id") else MEDIA_KIND_IMAGE

    def _search_filter(
        self,
        video_ids: list[str] | None,
        years: list[int] | None = None,
    ) -> Any | None:
        if not video_ids and not years:
            return None
        if not hasattr(self.vector_store, "search_filter"):
            # Không im lặng bỏ qua: người dùng lọc "năm 2024" mà nhận về mọi năm
            # thì kết quả trông vẫn hợp lý — sai kiểu này không ai báo lỗi.
            logger.warning("vector store does not support filtering; ignoring video_ids/years")
            return None
        return self.vector_store.search_filter(video_ids=video_ids, years=years)

    def _resolve_top_k(self, top_k: int | None) -> int:
        resolved = int(top_k or self.top_k)
        if resolved <= 0:
            raise ValueError("top_k must be a positive integer")
        return min(resolved, MAX_TOP_K)

    @staticmethod
    def normalize_query(query: str) -> str:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        return " ".join(query.split())

    @staticmethod
    def _as_float(value: Any) -> float | None:
        if value is None or isinstance(value, bool):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _as_optional_str(value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None
