from __future__ import annotations

from uuid import uuid4

from src.rag_video_anh.repository.schemas import (
    CaptionResultRecord,
    DetectedObjectRecord,
    FrameRecord,
    MediaRecord,
    MediaType,
    ObjectResultRecord,
    OcrResultRecord,
    TranscriptRecord,
    TranscriptSegmentRecord,
    VideoMetadataRecord,
)
from src.rag_video_anh.retrieval.retrieval_units import VideoRetrievalUnitBuilder, clean_text


class FakeStorage:
    def __init__(self, existing: set[tuple[str, str]]) -> None:
        self.existing = existing

    def object_exists(self, bucket_name: str, object_key: str) -> bool:
        return (bucket_name, object_key) in self.existing


class FakeMediaRepository:
    def __init__(self, media: MediaRecord, video: VideoMetadataRecord, frames: list[FrameRecord]) -> None:
        self.media = media
        self.video = video
        self.frames = frames

    def get_media(self, media_id):
        return self.media if str(media_id) == str(self.media.media_id) else None

    def get_video_by_media_id(self, media_id):
        return self.video if str(media_id) == str(self.media.media_id) else None

    def list_frames_for_video(self, media_id):
        return self.frames if str(media_id) == str(self.media.media_id) else []


class FakeResultRepository:
    def __init__(
        self,
        *,
        transcript: TranscriptRecord,
        ocr_by_media_id: dict[str, OcrResultRecord],
        caption_by_media_id: dict[str, CaptionResultRecord],
        objects_by_media_id: dict[str, ObjectResultRecord],
    ) -> None:
        self.transcript = transcript
        self.ocr_by_media_id = ocr_by_media_id
        self.caption_by_media_id = caption_by_media_id
        self.objects_by_media_id = objects_by_media_id

    def get_transcript_by_video_media_id(self, media_id):
        return self.transcript

    def get_ocr_result(self, media_id):
        return self.ocr_by_media_id.get(str(media_id))

    def get_caption_result(self, media_id):
        return self.caption_by_media_id.get(str(media_id))

    def get_object_result(self, media_id):
        return self.objects_by_media_id.get(str(media_id))


class FakeUnitOfWork:
    def __init__(self, media_repo: FakeMediaRepository, result_repo: FakeResultRepository) -> None:
        self.media = media_repo
        self.results = result_repo

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return None


def test_clean_text_preserves_vietnamese_and_drops_errors() -> None:
    assert clean_text("  Tiếng Việt có dấu\nrất rõ  ") == "Tiếng Việt có dấu rất rõ"
    assert clean_text("Lỗi: timeout khi phân tích ảnh") == ""
    assert clean_text(None) == ""


def test_builder_creates_clean_units_and_summary() -> None:
    post_id = uuid4()
    video_media_id = uuid4()
    video_id = uuid4()
    transcript_id = uuid4()
    good_frame_media_id = uuid4()
    missing_frame_media_id = uuid4()
    invalid_frame_media_id = uuid4()

    media = MediaRecord(
        media_id=video_media_id,
        post_id=post_id,
        media_type=MediaType.VIDEO.value,
        bucket_name="mira-data",
        object_key="videos/source.mp4",
    )
    video = VideoMetadataRecord(video_id=video_id, media_id=video_media_id)
    frames = [
        FrameRecord(
            frame_id=uuid4(),
            media_id=good_frame_media_id,
            post_id=post_id,
            video_media_id=video_media_id,
            bucket_name="mira-data",
            object_key="frames/good.jpg",
            timestamp=2.5,
            frame_index=10,
        ),
        FrameRecord(
            frame_id=uuid4(),
            media_id=missing_frame_media_id,
            post_id=post_id,
            video_media_id=video_media_id,
            bucket_name="mira-data",
            object_key="frames/missing.jpg",
            timestamp=6.0,
            frame_index=20,
        ),
        FrameRecord(
            frame_id=uuid4(),
            media_id=invalid_frame_media_id,
            post_id=post_id,
            video_media_id=video_media_id,
            bucket_name="mira-data",
            object_key="frames/invalid.jpg",
            timestamp=None,
            frame_index=30,
        ),
    ]
    segments = [
        TranscriptSegmentRecord(segment_id=uuid4(), transcript_id=transcript_id, start_time=0.0, end_time=1.0, text="Xin chào"),
        TranscriptSegmentRecord(segment_id=uuid4(), transcript_id=transcript_id, start_time=1.0, end_time=4.0, text="đây là hoạt động"),
        TranscriptSegmentRecord(segment_id=uuid4(), transcript_id=transcript_id, start_time=4.0, end_time=8.0, text="kết thúc"),
        TranscriptSegmentRecord(segment_id=uuid4(), transcript_id=transcript_id, start_time=8.0, end_time=9.0, text="error"),
        TranscriptSegmentRecord(segment_id=uuid4(), transcript_id=transcript_id, start_time=None, end_time=10.0, text="timestamp lỗi"),
    ]
    transcript = TranscriptRecord(
        transcript_id=transcript_id,
        video_id=video_id,
        status="DONE",
        language="vi",
        segments=segments,
    )
    object_result_id = uuid4()
    object_result = ObjectResultRecord(
        media_id=good_frame_media_id,
        status="DONE",
        object_result_id=object_result_id,
        objects=[
            DetectedObjectRecord(object_id=uuid4(), object_result_id=object_result_id, label="Person", confidence=0.9),
            DetectedObjectRecord(object_id=uuid4(), object_result_id=object_result_id, label="person", confidence=0.8),
            DetectedObjectRecord(object_id=uuid4(), object_result_id=object_result_id, label="error", confidence=0.7),
        ],
    )
    result_repo = FakeResultRepository(
        transcript=transcript,
        ocr_by_media_id={
            str(good_frame_media_id): OcrResultRecord(media_id=good_frame_media_id, ocr_status="DONE", ocr_text="  HIT MIRA  ")
        },
        caption_by_media_id={
            str(good_frame_media_id): CaptionResultRecord(
                media_id=good_frame_media_id,
                caption_status="DONE",
                caption_text="Sinh viên đang thuyết trình.",
                vision_metadata={"scene": "hội trường", "error": "timeout", "objects": ["person", "error"]},
            )
        },
        objects_by_media_id={str(good_frame_media_id): object_result},
    )
    media_repo = FakeMediaRepository(media, video, frames)
    uow = FakeUnitOfWork(media_repo, result_repo)

    builder = VideoRetrievalUnitBuilder(
        uow_factory=lambda: uow,
        storage=FakeStorage({("mira-data", "frames/good.jpg"), ("mira-data", "frames/invalid.jpg")}),
    )
    result = builder.build(video_media_id)

    assert len(result.media_clip_units) == 1
    assert len(result.video_transcript_units) == 3

    unit = result.media_clip_units[0].to_dict()
    assert unit["video_id"] == str(video_id)
    assert unit["post_id"] == str(post_id)
    assert unit["frame_media_id"] == str(good_frame_media_id)
    assert unit["frame_index"] == 10
    assert unit["timestamp_sec"] == 2.5
    assert unit["bucket_name"] == "mira-data"
    assert unit["frame_object_key"] == "frames/good.jpg"
    assert unit["caption"] == "Sinh viên đang thuyết trình."
    assert unit["ocr_text"] == "HIT MIRA"
    assert unit["vision_metadata"] == {"scene": "hội trường", "objects": ["person"]}
    assert unit["object_counts"] == {"person": 2}
    assert unit["transcript_context"]["text"] == "Xin chào đây là hoạt động kết thúc"

    summary = result.summary.to_dict()
    assert summary["media_clip_units_created"] == 1
    assert summary["video_transcript_units_created"] == 3
    assert summary["skipped_by_reason"]["missing_frame_image"] == 1
    assert summary["skipped_by_reason"]["empty_transcript"] == 1
    assert summary["skipped_by_reason"]["invalid_timestamp"] == 2
    assert summary["skipped_by_reason"]["invalid_detected_object"] == 1
