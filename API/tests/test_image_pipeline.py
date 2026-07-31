"""Test cho luồng ảnh tĩnh: validate -> route -> khung đơn -> unit -> index -> truy hồi.

Ảnh là phần lớn tư liệu của CLB (166 ảnh so với 4 video) nhưng trước đây bị
`media_validator` chặn cứng. Bộ test này khoá lại hành vi mới.
"""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest

from src.rag_video_anh.pipeline.media_router import MediaRouterService
from src.rag_video_anh.pipeline.media_validator import MediaValidatorService
from src.rag_video_anh.pipeline.pipeline_service import PipelineService
from src.rag_video_anh.repository.schemas import (
    CaptionResultRecord,
    MediaRecord,
    MediaType,
    OcrResultRecord,
    ProcessingStatus,
)
from src.rag_video_anh.retrieval.indexing_service import VideoRetrievalIndexingService
from src.rag_video_anh.retrieval.retrieval_service import VideoRetrievalService
from src.rag_video_anh.retrieval.retrieval_units import ImageRetrievalUnitBuilder
from src.rag_video_anh.retrieval.retriever import (
    MEDIA_KIND_IMAGE,
    MEDIA_KIND_VIDEO_FRAME,
    MediaClipHit,
    VideoRetriever,
)
from src.rag_video_anh.schemas import (
    CaptionResult,
    CaptionResultSet,
    DetectionResultSet,
    MediaInput,
    OCRResult,
    OCRResultSet,
    PipelineStatus,
    StageStatus,
)

# --------------------------------------------------------------------------
# Validate + route
# --------------------------------------------------------------------------


def test_validator_accepts_image_media() -> None:
    result = MediaValidatorService().validate(
        MediaInput(media_id="img-1", media_type="image", media_path="/tmp/anh.jpg")
    )

    assert result.is_valid is True, result.errors


def test_image_route_skips_keyframe_extraction_and_asr() -> None:
    validation = MediaValidatorService().validate(
        MediaInput(media_id="img-1", media_type="image", media_path="/tmp/anh.jpg")
    )

    route = MediaRouterService().select_route(validation)

    assert route.route_type == "image"
    # Ảnh tự nó đã là khung hình, và không có tiếng.
    assert route.requires_keyframes is False
    assert route.requires_asr is False
    # Nhưng vẫn phải đọc chữ và sinh caption.
    assert route.requires_ocr is True
    assert route.requires_caption is True


def test_video_route_is_unchanged() -> None:
    validation = MediaValidatorService().validate(
        MediaInput(media_id="vid-1", media_type="video", media_path="/tmp/clip.mp4")
    )

    route = MediaRouterService().select_route(validation)

    assert route.route_type == "video"
    assert route.requires_keyframes is True


def test_validator_rejects_video_extension_for_image_media() -> None:
    """Ảnh và video có tập đuôi riêng, không được lẫn."""
    result = MediaValidatorService().validate(
        MediaInput(media_id="img-1", media_type="image", media_path="/tmp/anh.mp4")
    )

    assert result.is_valid is False
    assert any("extension" in error for error in result.errors)


def test_validator_still_rejects_other_media_types() -> None:
    result = MediaValidatorService().validate(
        MediaInput(media_id="a-1", media_type="audio", media_path="/tmp/a.mp3")
    )

    assert result.is_valid is False
    assert any("unsupported media_type" in error for error in result.errors)


# --------------------------------------------------------------------------
# Khung đơn
# --------------------------------------------------------------------------


class ExplodingKeyframeExtractor:
    def extract(self, request):  # pragma: no cover - chỉ để chứng minh là không bị gọi
        raise AssertionError("ảnh không được đi qua bước tách keyframe")


def test_pipeline_wraps_image_into_single_frame_without_extractor(tmp_path: Path) -> None:
    image_path = tmp_path / "anh.jpg"
    image_path.write_bytes(b"fake-jpeg")

    frames = PipelineService(keyframe_extractor=ExplodingKeyframeExtractor())._single_frame_set(
        MediaInput(media_id="img-1", media_type="image", media_path=str(image_path))
    )

    assert frames.status == StageStatus.DONE
    assert len(frames.frames) == 1
    frame = frames.frames[0]
    assert frame.image_path == str(image_path)
    assert frame.frame_index == 0
    assert frame.selection_reason == "static_image"


def test_single_frame_requires_a_readable_path() -> None:
    with pytest.raises(ValueError, match="no media_path"):
        PipelineService()._single_frame_set(MediaInput(media_id="img-1", media_type="image"))


class ExplodingTranscriptMapper:
    def map(self, request):  # pragma: no cover - chỉ để chứng minh là không bị gọi
        raise AssertionError("ảnh không có lời thoại thì không được gióng transcript")


class FakeVisionService:
    """Trả OCR + caption đã xong, không gọi ra ngoài mạng."""

    def analyze(self, *, media_id, keyframes, ocr_policy=None, caption_policy=None):
        frame_id = keyframes.frames[0].frame_id
        return (
            OCRResultSet(
                media_id=media_id,
                status=StageStatus.DONE,
                results=[OCRResult(frame_id=frame_id, full_text="HIT Club", status=StageStatus.DONE)],
            ),
            CaptionResultSet(
                media_id=media_id,
                status=StageStatus.DONE,
                results=[
                    CaptionResult(
                        frame_id=frame_id,
                        caption_text="Nhóm sinh viên chụp ảnh chung ngoài trời",
                        status=StageStatus.DONE,
                    )
                ],
            ),
        )


class FakeDetectionService:
    def detect(self, request):
        return DetectionResultSet(media_id=request.media_id, status=StageStatus.DONE, results=[])


def test_image_run_is_success_not_partial_when_every_enabled_stage_finishes(tmp_path: Path) -> None:
    """Ảnh không bao giờ có transcript.

    Nếu khâu gióng transcript báo NOT_FOUND thay vì 'bỏ qua theo route' thì mọi
    ảnh đều bị chấm partial_success, và lỗi thật lẫn vào đó không nhận ra được.
    """
    image_path = tmp_path / "anh.jpg"
    image_path.write_bytes(b"fake-jpeg")

    result = PipelineService(
        vision_service=FakeVisionService(),
        detection_service=FakeDetectionService(),
        transcript_mapper=ExplodingTranscriptMapper(),
    ).process(MediaInput(media_id="img-1", media_type="image", media_path=str(image_path)))

    assert result.route is not None and result.route.requires_asr is False
    assert result.aligned_context is not None
    assert result.aligned_context.status == StageStatus.SKIPPED
    assert "disabled by route" in (result.aligned_context.reason or "")
    assert result.status == PipelineStatus.SUCCESS


# --------------------------------------------------------------------------
# Dựng unit từ DB
# --------------------------------------------------------------------------


class FakeStorage:
    def __init__(self, existing: set[tuple[str, str]]) -> None:
        self.existing = existing

    def object_exists(self, bucket_name: str, object_key: str) -> bool:
        return (bucket_name, object_key) in self.existing


class FakeMediaRepository:
    def __init__(self, media: MediaRecord) -> None:
        self.media = media

    def get_media(self, media_id):
        return self.media if str(media_id) == str(self.media.media_id) else None


class FakeResultRepository:
    def __init__(self, ocr=None, caption=None, objects=None) -> None:
        self.ocr = ocr
        self.caption = caption
        self.objects = objects

    def get_ocr_result(self, media_id):
        return self.ocr

    def get_caption_result(self, media_id):
        return self.caption

    def get_object_result(self, media_id):
        return self.objects


class FakeUnitOfWork:
    def __init__(self, media, results) -> None:
        self.media = media
        self.results = results

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return None


def _image_builder(media: MediaRecord, *, ocr=None, caption=None, existing=None) -> ImageRetrievalUnitBuilder:
    uow = FakeUnitOfWork(FakeMediaRepository(media), FakeResultRepository(ocr=ocr, caption=caption))
    return ImageRetrievalUnitBuilder(
        uow_factory=lambda: uow,
        storage=FakeStorage(existing if existing is not None else {(media.bucket_name, media.object_key)}),
    )


def _image_media(**overrides) -> MediaRecord:
    defaults = dict(
        media_id=uuid4(),
        post_id=uuid4(),
        media_type=MediaType.IMAGE.value,
        bucket_name="hit-mira-media",
        object_key="events/post-1/media/photo_01.jpg",
    )
    defaults.update(overrides)
    return MediaRecord(**defaults)


def test_image_unit_carries_caption_and_ocr_but_no_timestamp() -> None:
    media = _image_media()
    unit = _image_builder(
        media,
        ocr=OcrResultRecord(media_id=media.media_id, ocr_status=ProcessingStatus.DONE.value, ocr_text="Bản Lác Mai Châu"),
        caption=CaptionResultRecord(
            media_id=media.media_id,
            caption_status=ProcessingStatus.DONE.value,
            caption_text="Một người đọc bản đồ bên cạnh ba lô đỏ.",
        ),
    ).build(media.media_id)

    assert unit is not None
    assert unit.caption == "Một người đọc bản đồ bên cạnh ba lô đỏ."
    assert unit.ocr_text == "Bản Lác Mai Châu"
    assert unit.post_id == str(media.post_id)
    assert unit.unit_id == f"image:{media.media_id}"
    # Ảnh không nằm trên trục thời gian nào.
    assert not hasattr(unit, "timestamp_sec")
    assert not hasattr(unit, "video_id")


def test_image_unit_is_skipped_when_object_is_missing() -> None:
    media = _image_media()

    assert _image_builder(media, existing=set()).build(media.media_id) is None


def test_image_builder_rejects_non_image_media() -> None:
    media = _image_media(media_type=MediaType.VIDEO.value)

    with pytest.raises(ValueError, match="expected media_type='image'"):
        _image_builder(media).build(media.media_id)


def test_image_unit_drops_error_like_caption() -> None:
    """Caption lỗi phải bị loại y như nhánh video, không được index rác."""
    media = _image_media()
    unit = _image_builder(
        media,
        caption=CaptionResultRecord(
            media_id=media.media_id,
            caption_status=ProcessingStatus.DONE.value,
            caption_text="Lỗi: timeout khi phân tích ảnh",
        ),
    ).build(media.media_id)

    assert unit is not None
    assert unit.caption == ""


# --------------------------------------------------------------------------
# Index
# --------------------------------------------------------------------------


class FakeImageBuilder:
    def __init__(self, units) -> None:
        self.units = units

    def build(self, media_id):
        return self.units.get(str(media_id))


class FakeEmbedder:
    def __init__(self) -> None:
        self.calls: list[list] = []

    def embed_images(self, paths):
        self.calls.append(list(paths))
        return [[0.1, 0.2]] * len(paths)


class FakeVectorStore:
    def __init__(self) -> None:
        self.upserts: list[dict] = []
        self.payload_updates: list[dict] = []

    def upsert_points(self, *, collection_name, point_ids, vectors, payloads):
        self.upserts.append(
            {"collection_name": collection_name, "point_ids": point_ids, "vectors": vectors, "payloads": payloads}
        )
        return {"upserted": len(point_ids)}

    def set_payloads(self, *, collection_name, point_ids, payloads):
        self.payload_updates.append(
            {"collection_name": collection_name, "point_ids": point_ids, "payloads": payloads}
        )
        return {"updated": len(point_ids)}


class FakeDownloadStorage:
    """Giả MinIO: ghi ra file thật để bước nhúng có đường dẫn hợp lệ."""

    def download_object(self, bucket_name, object_key, local_path) -> None:
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        Path(local_path).write_bytes(b"fake-image-bytes")


def test_index_images_writes_into_media_clip_with_image_kind() -> None:
    media = _image_media()
    unit = _image_builder(
        media,
        caption=CaptionResultRecord(
            media_id=media.media_id, caption_status=ProcessingStatus.DONE.value, caption_text="Sinh hoạt CLB"
        ),
    ).build(media.media_id)
    embedder = FakeEmbedder()
    store = FakeVectorStore()

    summary = VideoRetrievalIndexingService(
        builder=object(),
        image_builder=FakeImageBuilder({str(media.media_id): unit}),
        storage=FakeDownloadStorage(),
        image_embedder=embedder,
        vector_store=store,
    ).index_images([str(media.media_id)])

    assert summary.media_clip_indexed == 1
    assert len(store.upserts) == 1
    upsert = store.upserts[0]
    # Ảnh và keyframe dùng chung collection để một truy vấn xếp hạng được cả hai.
    assert upsert["collection_name"] == "media_clip"
    payload = upsert["payloads"][0]
    assert payload["media_kind"] == MEDIA_KIND_IMAGE
    assert payload["caption"] == "Sinh hoạt CLB"
    assert payload["post_id"] == str(media.post_id)
    assert "timestamp_sec" not in payload
    assert "video_id" not in payload


class ExplodingEmbedder:
    def embed_images(self, paths):  # pragma: no cover - chỉ để chứng minh là không bị gọi
        raise AssertionError("cập nhật payload thì không được nhúng lại")


def test_refresh_image_payloads_writes_caption_without_embedding_again() -> None:
    """Caption có sau lúc index, mà vector thì không đổi.

    Nhúng lại 1.628 ảnh tốn cả tiếng hạn mức Jina một cách vô ích, nên phải ghi
    được phần chữ mà không đụng tới vector.
    """
    media = _image_media()
    unit = _image_builder(
        media,
        caption=CaptionResultRecord(
            media_id=media.media_id,
            caption_status=ProcessingStatus.DONE.value,
            caption_text="Lễ trao giải trên sân khấu",
        ),
    ).build(media.media_id)
    store = FakeVectorStore()

    summary = VideoRetrievalIndexingService(
        builder=object(),
        image_builder=FakeImageBuilder({str(media.media_id): unit}),
        storage=FakeDownloadStorage(),
        image_embedder=ExplodingEmbedder(),
        vector_store=store,
    ).refresh_image_payloads([str(media.media_id)])

    assert summary.media_clip_indexed == 1
    assert store.upserts == []
    assert len(store.payload_updates) == 1
    update = store.payload_updates[0]
    assert update["collection_name"] == "media_clip"
    assert update["point_ids"] == [str(media.media_id)]
    assert update["payloads"][0]["caption"] == "Lễ trao giải trên sân khấu"
    assert update["payloads"][0]["media_kind"] == MEDIA_KIND_IMAGE


def test_index_images_skips_units_that_cannot_be_built() -> None:
    store = FakeVectorStore()

    summary = VideoRetrievalIndexingService(
        builder=object(),
        image_builder=FakeImageBuilder({}),
        storage=FakeDownloadStorage(),
        image_embedder=FakeEmbedder(),
        vector_store=store,
    ).index_images(["khong-ton-tai"])

    assert summary.media_clip_indexed == 0
    assert store.upserts == []
    assert summary.skipped_by_reason.get("missing_image_object") == 1


# --------------------------------------------------------------------------
# Truy hồi
# --------------------------------------------------------------------------


def test_hit_reads_media_kind_from_payload() -> None:
    hit = VideoRetriever._as_clip_hit(
        {"score": 0.5, "payload": {"media_kind": "image", "post_id": "p-1", "caption": "ảnh"}}
    )

    assert hit.media_kind == MEDIA_KIND_IMAGE
    assert hit.is_image is True
    assert hit.as_dict()["media_kind"] == MEDIA_KIND_IMAGE


def test_hit_infers_kind_for_points_indexed_before_media_kind_existed() -> None:
    """Điểm cũ trong Qdrant không có khoá media_kind vẫn phải đọc đúng."""
    old_video_point = VideoRetriever._as_clip_hit(
        {"score": 0.5, "payload": {"video_id": "v-1", "timestamp_sec": 12.0}}
    )

    assert old_video_point.media_kind == MEDIA_KIND_VIDEO_FRAME
    assert old_video_point.is_image is False


def _hit(**overrides) -> MediaClipHit:
    defaults = dict(
        score=0.9,
        video_id=None,
        unit_id="image:1",
        timestamp_sec=None,
        caption="Ảnh trao giải",
        ocr_text="",
        bucket_name="hit-mira-media",
        frame_object_key="events/post-1/media/photo_01.jpg",
        media_kind=MEDIA_KIND_IMAGE,
        payload={"post_id": "post-1"},
    )
    defaults.update(overrides)
    return MediaClipHit(**defaults)


def test_context_cites_image_by_post_not_by_timestamp() -> None:
    context = VideoRetrievalService._format_context([_hit()], [])

    assert "[1] ảnh trong bài post-1" in context
    assert "Ảnh trao giải" in context
    # Không được bịa mốc thời gian cho ảnh.
    assert "tại" not in context
    assert "video" not in context


def test_context_still_cites_video_frames_with_timestamp() -> None:
    frame = _hit(
        media_kind=MEDIA_KIND_VIDEO_FRAME,
        video_id="video-1",
        timestamp_sec=75.0,
        caption="Cảnh trao bằng khen",
        payload={"post_id": "post-1"},
    )

    context = VideoRetrievalService._format_context([frame], [])

    assert "[1] video video-1 tại 01:15" in context
