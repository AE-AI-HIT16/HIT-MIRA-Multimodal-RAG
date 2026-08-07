"""Truy cập bốn bảng sự kiện: series, occurrence, alias, và liên kết post–occurrence.

Đường ĐỌC chỉ có đúng một hàm — `primary_series_slug()` — vì đó là tất cả những
gì tầng index cần: một khoá `event_key` để nhét vào payload Qdrant. Phần còn lại
là đường GHI, dùng bởi `scripts/extract_post_events.py`.

Vì sao khoá payload là slug của SERIES chứ không phải của OCCURRENCE: người hỏi
"ảnh Open Day" muốn cả mọi kỳ, còn muốn một kỳ cụ thể thì đã có `post_created_at`
để lọc năm. Gắn occurrence vào payload sẽ chia nhỏ kho ra thành những nhóm
một-hai bài, mà lọc theo đó thì gần như luôn rỗng.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import delete, select, update
from sqlalchemy.orm import Session

# Dùng chung với `vector_store.py`: khoá ghi vào DB và khoá dùng để lọc phải do
# đúng một hàm sinh ra, nếu không thì bộ lọc trả rỗng mà không báo lỗi gì.
from src.common_utils.text_keys import normalize_alias, slugify
from src.rag_video_anh.repository.models import (
    EventAliasModel,
    EventOccurrenceModel,
    EventSeriesModel,
    PostEventOccurrenceModel,
)
from src.rag_video_anh.repository.schemas import (
    EventOccurrenceRecord,
    EventSeriesRecord,
    PostEventLinkRecord,
)

__all__ = ["EventRepository", "normalize_alias", "slugify"]


class EventRepository:
    """Đọc/ghi cụm bảng sự kiện."""

    def __init__(self, session: Session) -> None:
        self.session = session

    # ── Đường đọc (tầng index gọi) ───────────────────────────────────────────

    def primary_series_slug(self, post_id: Any) -> str | None:
        """Slug series của sự kiện CHÍNH của một bài, hoặc None.

        Bài gắn nhiều sự kiện mà không đánh dấu cái nào là chính thì trả None
        chứ không chọn bừa: `event_key` chỉ có một chỗ trong payload, đoán sai
        sẽ gom bài vào một sự kiện nó không thuộc về. Ngoại lệ duy nhất là khi
        bài chỉ có đúng một liên kết — lúc đó không có gì để đoán cả.
        """
        cau_lenh = (
            select(EventSeriesModel.slug, PostEventOccurrenceModel.is_primary)
            .join(
                EventOccurrenceModel,
                EventOccurrenceModel.series_id == EventSeriesModel.series_id,
            )
            .join(
                PostEventOccurrenceModel,
                PostEventOccurrenceModel.occurrence_id == EventOccurrenceModel.occurrence_id,
            )
            .where(PostEventOccurrenceModel.post_id == post_id)
        )
        hang = self.session.execute(cau_lenh).all()
        if not hang:
            return None
        chinh = [slug for slug, is_primary in hang if is_primary]
        if chinh:
            return chinh[0]
        # Không đánh dấu chính, nhưng mọi liên kết cùng trỏ về một series thì
        # vẫn không có gì mơ hồ (một bài có thể gắn hai kỳ của cùng một chuỗi).
        rieng_biet = {slug for slug, _ in hang}
        return rieng_biet.pop() if len(rieng_biet) == 1 else None

    def series_by_slug(self, slug: str) -> EventSeriesRecord | None:
        row = self.session.scalar(select(EventSeriesModel).where(EventSeriesModel.slug == slug))
        return self._series_record(row) if row else None

    def series_by_alias(self, alias: str) -> EventSeriesRecord | None:
        """Tra một cách gọi về series — kể cả khi alias được gắn vào occurrence.

        Đây là thứ khiến lượt trích xuất thứ hai không đẻ thêm series cho cùng
        một sự kiện đã thấy ở lượt đầu.
        """
        chuan = normalize_alias(alias)
        if not chuan:
            return None
        row = self.session.scalar(
            select(EventAliasModel).where(EventAliasModel.normalized_alias == chuan)
        )
        if row is None:
            return None
        if row.series_id is not None:
            return self.series_by_id(row.series_id)
        occurrence = self.session.get(EventOccurrenceModel, row.occurrence_id)
        return self.series_by_id(occurrence.series_id) if occurrence else None

    def series_by_id(self, series_id: Any) -> EventSeriesRecord | None:
        row = self.session.get(EventSeriesModel, series_id)
        return self._series_record(row) if row else None

    def occurrence_by_label(self, series_id: Any, label: str) -> EventOccurrenceRecord | None:
        row = self.session.scalar(
            select(EventOccurrenceModel).where(
                EventOccurrenceModel.series_id == series_id,
                EventOccurrenceModel.label == label,
            )
        )
        return self._occurrence_record(row) if row else None

    def counts(self) -> dict[str, int]:
        """Số hàng của cả bốn bảng — để script trích xuất báo cáo được kết quả."""
        return {
            "event_series": self._count(EventSeriesModel),
            "event_occurrences": self._count(EventOccurrenceModel),
            "event_aliases": self._count(EventAliasModel),
            "post_event_occurrences": self._count(PostEventOccurrenceModel),
        }

    # ── Đường ghi (script trích xuất gọi) ────────────────────────────────────

    def upsert_series(
        self,
        slug: str,
        canonical_name: str,
        description: str | None = None,
    ) -> EventSeriesRecord:
        """Tạo series, hoặc trả về series đã có cùng slug.

        Chạy lại script trích xuất KHÔNG được đẻ thêm series, nên đây là upsert
        chứ không phải insert. Tên hiển thị được cập nhật (lượt sau có thể đọc
        được tên đầy đủ hơn), còn slug là bất biến — nó đã nằm trong payload
        Qdrant rồi.
        """
        row = self.session.scalar(select(EventSeriesModel).where(EventSeriesModel.slug == slug))
        if row is None:
            row = EventSeriesModel(slug=slug, canonical_name=canonical_name, description=description)
            self.session.add(row)
        else:
            row.canonical_name = canonical_name
            if description:
                row.description = description
        self.session.flush()
        return self._series_record(row)

    def upsert_occurrence(
        self,
        series_id: Any,
        label: str,
        *,
        display_name: str | None = None,
        event_year: int | None = None,
        starts_at: datetime | None = None,
        ends_at: datetime | None = None,
        metadata: dict | None = None,
    ) -> EventOccurrenceRecord:
        """Tạo/cập nhật một kỳ. Định danh là `(series_id, label)`, không phải năm.

        Một chuỗi có thể tổ chức hai kỳ trong cùng một năm (`event_year` chỉ là
        thuộc tính lọc), nên lấy năm làm khoá sẽ gộp nhầm hai kỳ khác nhau.
        """
        row = self.session.scalar(
            select(EventOccurrenceModel).where(
                EventOccurrenceModel.series_id == series_id,
                EventOccurrenceModel.label == label,
            )
        )
        if row is None:
            row = EventOccurrenceModel(
                series_id=series_id,
                label=label,
                display_name=display_name,
                event_year=event_year,
                starts_at=starts_at,
                ends_at=ends_at,
                occurrence_metadata=metadata or {},
            )
            self.session.add(row)
        else:
            if display_name:
                row.display_name = display_name
            if event_year is not None:
                row.event_year = event_year
            if starts_at is not None:
                row.starts_at = starts_at
            if ends_at is not None:
                row.ends_at = ends_at
            if metadata:
                row.occurrence_metadata = {**(row.occurrence_metadata or {}), **metadata}
        self.session.flush()
        return self._occurrence_record(row)

    def add_alias(
        self,
        alias: str,
        *,
        series_id: Any | None = None,
        occurrence_id: Any | None = None,
    ) -> bool:
        """Gắn một cách gọi vào đúng một series hoặc một occurrence.

        Trả về False khi alias đã trỏ vào chỗ khác — KHÔNG cướp alias, vì như vậy
        lần chạy sau sẽ lặng lẽ đổi ý nghĩa của một cách gọi mà không ai thấy.
        Người gọi tự quyết định coi đó là xung đột cần in ra hay chuyện bình thường.
        """
        if (series_id is None) == (occurrence_id is None):
            raise ValueError("alias phải trỏ vào đúng một trong hai: series_id hoặc occurrence_id")
        chuan = normalize_alias(alias)
        if not chuan:
            return False
        dang_co = self.session.scalar(
            select(EventAliasModel).where(EventAliasModel.normalized_alias == chuan)
        )
        if dang_co is not None:
            return dang_co.series_id == series_id and dang_co.occurrence_id == occurrence_id
        self.session.add(
            EventAliasModel(
                alias=alias.strip(),
                normalized_alias=chuan,
                series_id=series_id,
                occurrence_id=occurrence_id,
            )
        )
        self.session.flush()
        return True

    def link_post(
        self,
        post_id: Any,
        occurrence_id: Any,
        *,
        confidence: float | None = None,
        assigned_by: str = "rule",
        evidence: dict | None = None,
        is_primary: bool = False,
    ) -> PostEventLinkRecord:
        """Gắn bài vào một kỳ. Đặt `is_primary` sẽ gỡ cờ chính của liên kết cũ.

        Có index unique một-phần trên `(post_id) WHERE is_primary`, nên ghi cờ
        mới trước khi gỡ cờ cũ là chết ngay ở tầng DB. Thứ tự dưới đây là bắt buộc.
        """
        if is_primary:
            self.session.execute(
                update(PostEventOccurrenceModel)
                .where(
                    PostEventOccurrenceModel.post_id == post_id,
                    PostEventOccurrenceModel.occurrence_id != occurrence_id,
                    PostEventOccurrenceModel.is_primary.is_(True),
                )
                .values(is_primary=False)
            )
            self.session.flush()

        row = self.session.get(PostEventOccurrenceModel, (post_id, occurrence_id))
        if row is None:
            row = PostEventOccurrenceModel(
                post_id=post_id,
                occurrence_id=occurrence_id,
                confidence=confidence,
                assigned_by=assigned_by,
                evidence=evidence or {},
                is_primary=is_primary,
            )
            self.session.add(row)
        else:
            row.confidence = confidence
            row.assigned_by = assigned_by
            row.evidence = evidence or {}
            row.is_primary = is_primary
        self.session.flush()
        return self._link_record(row)

    def unlink_post(self, post_id: Any) -> int:
        """Gỡ mọi liên kết sự kiện của một bài, trả về số liên kết đã gỡ.

        Cần cho `--redo`: trích xuất lại mà chỉ ghi đè thì sự kiện gán sai ở lượt
        trước vẫn nằm nguyên đó.
        """
        ket_qua = self.session.execute(
            delete(PostEventOccurrenceModel).where(PostEventOccurrenceModel.post_id == post_id)
        )
        self.session.flush()
        return int(ket_qua.rowcount or 0)

    def linked_post_ids(self) -> set[UUID]:
        """Các bài đã gán sự kiện — để chạy lại chỉ làm phần còn thiếu."""
        return set(self.session.scalars(select(PostEventOccurrenceModel.post_id)).all())

    # ── Chuyển hàng ORM sang DTO ─────────────────────────────────────────────

    def _count(self, model: type) -> int:
        return len(self.session.scalars(select(model)).all())

    @staticmethod
    def _series_record(row: EventSeriesModel) -> EventSeriesRecord:
        return EventSeriesRecord(
            series_id=row.series_id,
            slug=row.slug,
            canonical_name=row.canonical_name,
            description=row.description,
            created_at=row.created_at,
        )

    @staticmethod
    def _occurrence_record(row: EventOccurrenceModel) -> EventOccurrenceRecord:
        return EventOccurrenceRecord(
            occurrence_id=row.occurrence_id,
            series_id=row.series_id,
            label=row.label,
            display_name=row.display_name,
            event_year=row.event_year,
            starts_at=row.starts_at,
            ends_at=row.ends_at,
            occurrence_metadata=dict(row.occurrence_metadata or {}),
            created_at=row.created_at,
        )

    @staticmethod
    def _link_record(row: PostEventOccurrenceModel) -> PostEventLinkRecord:
        return PostEventLinkRecord(
            post_id=row.post_id,
            occurrence_id=row.occurrence_id,
            confidence=row.confidence,
            assigned_by=row.assigned_by,
            evidence=dict(row.evidence or {}),
            is_primary=row.is_primary,
            created_at=row.created_at,
        )
