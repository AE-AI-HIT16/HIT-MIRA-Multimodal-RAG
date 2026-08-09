"""Repository for MinIO object metadata and video/frame metadata."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.rag_video_anh.repository.models import FrameModel, MediaModel, VideoModel
from src.rag_video_anh.repository.schemas import (
    FrameCreate,
    FrameRecord,
    MediaCreate,
    MediaRecord,
    MediaType,
    VideoMetadataCreate,
    VideoMetadataRecord,
)


class MediaRepository:
    """Read/write access for media, videos, and frames.

    The repository stores only MinIO metadata: bucket name and object key. It
    never downloads, uploads, or inspects the binary object itself.
    """

    def __init__(self, session: Session) -> None:
        self.session = session

    def create_media(self, media: MediaCreate) -> MediaRecord:
        row = MediaModel(**media.__dict__)
        self.session.add(row)
        self.session.flush()
        return self._media_record(row)

    def get_media(self, media_id) -> MediaRecord | None:
        row = self.session.get(MediaModel, media_id)
        return self._media_record(row) if row else None

    def list_by_post(self, post_id, media_type: str | None = None) -> list[MediaRecord]:
        stmt = select(MediaModel).where(MediaModel.post_id == post_id).order_by(MediaModel.created_at, MediaModel.media_id)
        if media_type:
            stmt = stmt.where(MediaModel.media_type == media_type)
        return [self._media_record(row) for row in self.session.scalars(stmt).all()]

    def list_by_type(self, media_type: str) -> list[MediaRecord]:
        """Liệt kê toàn bộ media theo loại, để worker/indexer duyệt tuần tự."""
        stmt = (
            select(MediaModel)
            .where(MediaModel.media_type == media_type)
            .order_by(MediaModel.created_at, MediaModel.media_id)
        )
        return [self._media_record(row) for row in self.session.scalars(stmt).all()]

    def list_children(self, parent_media_id, media_type: str | None = None) -> list[MediaRecord]:
        stmt = select(MediaModel).where(MediaModel.parent_media_id == parent_media_id).order_by(MediaModel.created_at, MediaModel.media_id)
        if media_type:
            stmt = stmt.where(MediaModel.media_type == media_type)
        return [self._media_record(row) for row in self.session.scalars(stmt).all()]

    def upsert_video_metadata(self, metadata: VideoMetadataCreate) -> VideoMetadataRecord:
        row = self.session.scalar(select(VideoModel).where(VideoModel.media_id == metadata.media_id))
        if row is None:
            row = VideoModel(**metadata.__dict__)
            self.session.add(row)
        else:
            row.duration = metadata.duration
            row.fps = metadata.fps
            row.raw_frames = metadata.raw_frames
            row.selected_keyframes = metadata.selected_keyframes
        self.session.flush()
        return self._video_record(row)

    def get_video_by_media_id(self, media_id) -> VideoMetadataRecord | None:
        row = self.session.scalar(select(VideoModel).where(VideoModel.media_id == media_id))
        return self._video_record(row) if row else None

    def register_frame(self, frame: FrameCreate) -> FrameRecord:
        video_media = self.session.get(MediaModel, frame.video_media_id)
        if video_media is None:
            raise ValueError(f"video media does not exist: {frame.video_media_id}")

        media_row = MediaModel(
            post_id=video_media.post_id,
            media_type=MediaType.FRAME.value,
            bucket_name=frame.bucket_name,
            object_key=frame.object_key,
            parent_media_id=frame.video_media_id,
        )
        self.session.add(media_row)
        self.session.flush()

        frame_row = FrameModel(
            media_id=media_row.media_id,
            video_media_id=frame.video_media_id,
            timestamp=frame.timestamp,
            frame_index=frame.frame_index,
        )
        self.session.add(frame_row)
        self.session.flush()
        return self._frame_record(frame_row, media_row)

    def get_frame_by_media_id(self, media_id) -> FrameRecord | None:
        row = self.session.scalar(select(FrameModel).where(FrameModel.media_id == media_id))
        return self._frame_record(row, row.media) if row else None

    def list_frames_for_video(self, video_media_id) -> list[FrameRecord]:
        stmt = select(FrameModel).where(FrameModel.video_media_id == video_media_id).order_by(FrameModel.frame_index, FrameModel.timestamp)
        return [self._frame_record(row, row.media) for row in self.session.scalars(stmt).all()]

    @staticmethod
    def _media_record(row: MediaModel) -> MediaRecord:
        return MediaRecord(
            media_id=row.media_id,
            post_id=row.post_id,
            media_type=row.media_type,
            bucket_name=row.bucket_name or "mira-data",
            object_key=row.object_key,
            parent_media_id=row.parent_media_id,
            created_at=row.created_at,
        )

    @staticmethod
    def _video_record(row: VideoModel) -> VideoMetadataRecord:
        return VideoMetadataRecord(
            video_id=row.video_id,
            media_id=row.media_id,
            duration=row.duration,
            fps=row.fps,
            raw_frames=row.raw_frames,
            selected_keyframes=row.selected_keyframes,
            created_at=row.created_at,
        )

    @staticmethod
    def _frame_record(row: FrameModel, media_row: MediaModel) -> FrameRecord:
        return FrameRecord(
            frame_id=row.frame_id,
            media_id=row.media_id,
            post_id=media_row.post_id,
            video_media_id=row.video_media_id,
            parent_media_id=media_row.parent_media_id,
            bucket_name=media_row.bucket_name or "mira-data",
            object_key=media_row.object_key,
            timestamp=row.timestamp,
            frame_index=row.frame_index,
            created_at=media_row.created_at,
        )
