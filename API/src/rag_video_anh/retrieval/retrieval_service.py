"""Điều phối truy hồi media: nhúng câu hỏi rồi tìm keyframe + transcript.

Không viết lại truy vấn bằng LLM ở đây (khác `rag_noiquy`): mỗi request sẽ
thêm một lượt gọi LLM, mà quota free-tier là giới hạn thật. `query_rewriter`
của nhánh media vẫn để trống có chủ ý.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from src.configuration import AppConfig
from src.log.logger import logger
from src.rag_video_anh.embedding.provider import build_media_embedder
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

    def __init__(
        self,
        retriever: VideoRetriever,
        event_resolver: Callable[[str], str | None] | None = None,
    ) -> None:
        self.retriever = retriever
        # Không có bộ giải tên thì hành vi y hệt trước: chỉ slug hoá. Tiêm vào
        # được để test khỏi cần PostgreSQL.
        self.event_resolver = event_resolver

    def retrieve(
        self,
        query: str | None = None,
        top_k: int | None = None,
        video_ids: list[str] | None = None,
        source: str = SOURCE_BOTH,
        image: bytes | None = None,
        years: list[int] | None = None,
        events: list[str] | None = None,
    ) -> dict[str, Any]:
        """Truy hồi bằng chữ, bằng ảnh, hoặc cả hai.

        US-502.1 chốt "ảnh + text mâu thuẫn → ưu tiên ảnh", nên khi có ảnh thì
        nhánh `media_clip` luôn tìm bằng vector ẢNH. Phần chữ không bị vứt đi:
        nó lo nhánh lời thoại — nơi vector ảnh không dùng được.

        `years` lọc theo mốc thời gian của BÀI ĐĂNG, `events` lọc theo chuỗi sự
        kiện của bài; cả hai áp cho cả hai nhánh và ghép với nhau bằng AND.
        """
        normalized_query = self._normalize_optional_query(query, image_present=bool(image))
        normalized_source = self._normalize_source(source)
        normalized_years = QdrantVideoVectorStore.normalize_years(years)
        # Nhận cả "HIT Contest Series" lẫn "hit-contest-series": payload chỉ có
        # slug, nên tên người gõ phải được quy về đúng dạng đó trước khi lọc.
        # Trước đó còn một bước tra cách gọi tắt ("Open Day" → "hit-open-day").
        normalized_events = QdrantVideoVectorStore.normalize_event_keys(
            self._resolve_event_names(events)
        )

        clip_vector, text_vector = self._embed_query_vectors(normalized_query, image)

        clips: list[MediaClipHit] = []
        videos: list[TranscriptVideoHit] = []
        errors: list[str] = []
        notes: list[str] = []

        if normalized_source in (SOURCE_CLIP, SOURCE_BOTH):
            try:
                clips = self.retriever.retrieve_clips(
                    normalized_query,
                    top_k=top_k,
                    video_ids=video_ids,
                    query_vector=clip_vector,
                    years=normalized_years,
                    events=normalized_events,
                )
            except Exception as exc:
                # Một nhánh lỗi không được làm chết nhánh còn lại.
                errors.append(f"media_clip: {exc.__class__.__name__}")
                logger.warning(f"Media clip retrieval failed: {exc.__class__.__name__}: {exc}")

        if normalized_source in (SOURCE_TRANSCRIPT, SOURCE_BOTH):
            if text_vector is None:
                # Bỏ qua vì định tuyến, không phải vì lỗi — phải nói ra, nếu
                # không người dùng đọc "0 video" thành "CLB không nói gì về
                # chuyện này" thay vì "hệ thống không tìm lời thoại bằng ảnh".
                notes.append(
                    "video_transcript: bỏ qua vì truy vấn chỉ có ảnh — "
                    "lời thoại chỉ tìm được bằng chữ. Gõ thêm mô tả để tìm cả lời thoại."
                )
            else:
                try:
                    videos = self.retriever.retrieve_by_transcript(
                        normalized_query,
                        top_k=top_k,
                        video_ids=video_ids,
                        query_vector=text_vector,
                        years=normalized_years,
                        events=normalized_events,
                    )
                except Exception as exc:
                    errors.append(f"video_transcript: {exc.__class__.__name__}")
                    logger.warning(f"Transcript retrieval failed: {exc.__class__.__name__}: {exc}")

        total = len(clips) + len(videos)
        if total == 0 and not errors:
            # Rỗng vì bộ lọc chứ không phải vì kho không có gì: nói ra để người
            # dùng biết nên bỏ lọc, thay vì kết luận "CLB không có ảnh này".
            if normalized_years:
                notes.append(
                    f"Không có kết quả nào thuộc năm {', '.join(str(nam) for nam in normalized_years)}. "
                    "Bỏ bộ lọc năm để tìm trong toàn bộ kho."
                )
            if normalized_events:
                notes.append(
                    f"Không có kết quả nào thuộc sự kiện {', '.join(normalized_events)}. "
                    "Tên sự kiện có thể chưa được gán cho bài nào — bỏ bộ lọc để tìm toàn kho."
                )
        logger.info(
            f"Media retrieval flow completed: {len(clips)} keyframe(s), {len(videos)} video(s)"
        )
        return {
            "query": normalized_query or "",
            "query_kind": self._query_kind(normalized_query, image),
            "source": normalized_source,
            "years": normalized_years,
            "events": normalized_events,
            "clips": [clip.as_dict() for clip in clips],
            "videos": [video.as_dict() for video in videos],
            "context": self._format_context(clips, videos),
            "total": total,
            "found": total > 0,
            "errors": errors,
            "notes": notes,
        }

    def _resolve_event_names(self, events: list[str] | None) -> list[str] | None:
        """Quy cách gọi tắt về đúng series trước khi slug hoá.

        Người hỏi nói "Open Day", payload ghi "hit-open-day". Chỉ `slugify` thì
        ra "open-day" và bộ lọc trả về RỖNG cho một sự kiện có thật — không lỗi,
        không cảnh báo. Bảng `event_aliases` đã có sẵn cách gọi đó (được ghi lúc
        trích xuất sự kiện), nên ở đây chỉ là đọc lại thứ đã biết.

        Tra hụt thì trả về nguyên tên để `normalize_event_keys` slug hoá như cũ:
        một cái tên bịa vẫn phải khớp-không-ra-gì chứ không được thành lỗi.
        """
        if not events or self.event_resolver is None:
            return events
        ket_qua: list[str] = []
        for ten in events:
            if not isinstance(ten, str) or not ten.strip():
                continue
            try:
                slug = self.event_resolver(ten)
            except Exception as exc:
                # DB sập thì bộ lọc kém chính xác đi, chứ không được kéo sập cả
                # câu trả lời — cùng lý do với chỗ kho nội quy rỗng ở retriever.
                logger.warning(f"Không tra được alias sự kiện '{ten}': {exc}")
                slug = None
            if slug and slug != ten:
                logger.info(f"Tên sự kiện '{ten}' được quy về series '{slug}'")
            ket_qua.append(slug or ten)
        return ket_qua

    def _embed_query_vectors(
        self,
        normalized_query: str | None,
        image: bytes | None,
    ) -> tuple[list[float], list[float] | None]:
        """Trả về (vector cho nhánh ảnh, vector cho nhánh lời thoại).

        Không có ảnh thì hai vector là MỘT — đúng tính chất một-lần-nhúng của
        không gian chung Jina-CLIP v2. Chỉ khi có ảnh mới phải nhúng hai lần,
        và lúc đó lượt thứ hai là thứ đáng tiền: không có nó thì đính kèm ảnh
        đồng nghĩa với mất hẳn nhánh lời thoại.
        """
        text_vector = (
            self.retriever.embed_query(normalized_query) if normalized_query else None
        )
        if not image:
            # `_normalize_optional_query` đã chặn trường hợp không có cả hai.
            return text_vector, text_vector  # type: ignore[return-value]
        return self.retriever.embed_image_query(image), text_vector

    @staticmethod
    def _normalize_optional_query(query: str | None, *, image_present: bool) -> str | None:
        if not image_present:
            # Không ảnh thì chữ là bắt buộc — giữ nguyên lỗi cũ cho đường text.
            return VideoRetriever.normalize_query(query)
        if query is None or not str(query).strip():
            return None
        return VideoRetriever.normalize_query(query)

    @staticmethod
    def _query_kind(normalized_query: str | None, image: bytes | None) -> str:
        if not image:
            return "text"
        return "image+text" if normalized_query else "image"

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
            label = f"{label}{cls._format_source(clip.payload.get('source_url'))}"
            body = clip.caption or clip.ocr_text or "(không có caption)"
            if clip.caption and clip.ocr_text:
                body = f"{clip.caption}\nChữ trong hình: {clip.ocr_text}"
            parts.append(f"{label}\n{body}")

        for video in videos:
            index += 1
            label = f"[{index}] video {video.video_id or 'không rõ'} (lời thoại)"
            label = f"{label}{cls._format_source(video.source_url)}"
            lines: list[str] = []
            for moment in video.moments:
                span = cls._format_span(moment.start_sec, moment.end_sec)
                text = moment.text.strip()
                if not text:
                    continue
                lines.append(f"{span} {text}" if span else text)
            parts.append("\n".join([label, *lines]))

        return "\n\n".join(parts)

    @staticmethod
    def _format_source(source_url: str | None) -> str:
        """US-405.1: gắn link bài gốc, hoặc nói rõ là nguồn nội bộ.

        Ghi thẳng vào chuỗi context vì đây là thứ tầng trả lời thật sự đọc.
        Không có link thì phải nói "nguồn nội bộ" (AC-2) — im lặng sẽ khiến LLM
        tự dựng một URL trông hợp lý.
        """
        url = str(source_url or "").strip()
        return f" — nguồn: {url}" if url else " — nguồn nội bộ"

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


def tra_series_theo_ten(ten: str) -> str | None:
    """Tra một cách gọi sự kiện về slug series, `None` nếu chưa từng thấy tên đó.

    Import nằm trong hàm để tầng truy hồi không kéo theo SQLAlchemy khi test
    dựng service bằng fake — đường media vốn chạy được mà không cần PostgreSQL.
    """
    from src.rag_video_anh.repository.unit_of_work import RepositoryUnitOfWork

    with RepositoryUnitOfWork() as uow:
        record = uow.events.series_by_alias(ten)
        return record.slug if record else None


def build_video_retrieval_service(config: AppConfig | None = None) -> VideoRetrievalService:
    app_config = config or AppConfig()
    return VideoRetrievalService(
        retriever=VideoRetriever(
            embedding_service=build_media_embedder(app_config, for_online_queries=True),
            vector_store=QdrantVideoVectorStore(config=app_config),
            config=app_config,
        ),
        event_resolver=tra_series_theo_ten,
    )
