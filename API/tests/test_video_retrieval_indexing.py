from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from src.rag_video_anh.retrieval.indexing_service import VideoRetrievalIndexingService
from src.rag_video_anh.retrieval.retrieval_units import (
    MediaClipUnit,
    RetrievalUnitBuildResult,
    RetrievalUnitBuildSummary,
    TranscriptContext,
    TranscriptContextSegment,
    VideoTranscriptUnit,
)

# US-405.1: link bài gốc phải đi hết đường từ unit tới payload Qdrant.
POST_URL = "https://www.facebook.com/1679787570636388/posts/1001928175089001"


class FakeBuilder:
    def __init__(self, result: RetrievalUnitBuildResult) -> None:
        self.result = result
        self.called_with = None

    def build(self, video_media_id):
        self.called_with = video_media_id
        return self.result


class FakeStorage:
    def __init__(self, available: set[tuple[str, str]]) -> None:
        self.available = available
        self.downloaded: list[tuple[str, str]] = []

    def download_object(self, bucket_name: str, object_key: str, local_path: Path) -> None:
        self.downloaded.append((bucket_name, object_key))
        if (bucket_name, object_key) not in self.available:
            raise FileNotFoundError(object_key)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_bytes(b"fake image")


class FakeImageEmbedder:
    def __init__(self) -> None:
        self.paths: list[Path] = []

    def embed_images(self, image_paths):
        self.paths = [Path(path) for path in image_paths]
        return [[1.0, 0.0, 0.5] for _ in image_paths]


class FakeTextEmbedder:
    def __init__(self) -> None:
        self.texts: list[str] = []

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.texts = texts
        return [[0.2, 0.3] for _ in texts]


class FakeVectorStore:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def upsert_points(self, *, collection_name, point_ids, vectors, payloads):
        self.calls.append(
            {
                "collection_name": collection_name,
                "point_ids": point_ids,
                "vectors": vectors,
                "payloads": payloads,
            }
        )
        return {"collection_name": collection_name, "upserted": len(point_ids)}


def test_index_video_uses_builder_embeds_and_upserts_clean_units() -> None:
    video_media_id = str(uuid4())
    video_id = str(uuid4())
    post_id = str(uuid4())
    good_frame_media_id = str(uuid4())
    missing_frame_media_id = str(uuid4())

    transcript_context = TranscriptContext(
        text="mở đầu hoạt động",
        source_segment_ids=["seg-1", "seg-2"],
        segments=[
            TranscriptContextSegment(segment_id="seg-1", start_sec=1.0, end_sec=2.0, text="mở đầu"),
            TranscriptContextSegment(segment_id="seg-2", start_sec=2.0, end_sec=3.5, text="hoạt động"),
        ],
    )
    media_units = [
        MediaClipUnit(
            unit_id=f"media_clip:{good_frame_media_id}",
            video_id=video_id,
            video_media_id=video_media_id,
            post_id=post_id,
            frame_media_id=good_frame_media_id,
            frame_index=5,
            timestamp_sec=2.5,
            bucket_name="mira-data",
            frame_object_key="frames/good.jpg",
            caption="Sinh viên thuyết trình.",
            ocr_text="HIT MIRA",
            vision_metadata={"scene": "hội trường"},
            detected_objects=[{"label": "person", "confidence": 0.9}],
            object_counts={"person": 1},
            transcript_context=transcript_context,
            source_url=POST_URL,
        ),
        MediaClipUnit(
            unit_id=f"media_clip:{missing_frame_media_id}",
            video_id=video_id,
            video_media_id=video_media_id,
            post_id=post_id,
            frame_media_id=missing_frame_media_id,
            frame_index=6,
            timestamp_sec=4.0,
            bucket_name="mira-data",
            frame_object_key="frames/missing.jpg",
            caption="",
            ocr_text="",
            vision_metadata={},
            detected_objects=[],
            object_counts={},
            transcript_context=TranscriptContext(text="", source_segment_ids=[], segments=[]),
        ),
    ]
    transcript_units = [
        VideoTranscriptUnit(
            unit_id=f"video_transcript:{video_id}:seg-1",
            video_id=video_id,
            post_id=post_id,
            start_sec=1.0,
            end_sec=2.0,
            text="mở đầu",
            language="vi",
            source_segment_ids=["seg-1"],
            source_url=POST_URL,
        ),
        VideoTranscriptUnit(
            unit_id=f"video_transcript:{video_id}:seg-2",
            video_id=video_id,
            post_id=post_id,
            start_sec=2.0,
            end_sec=3.5,
            text="hoạt động",
            language="vi",
            source_segment_ids=["seg-2"],
            source_url=POST_URL,
        ),
    ]
    build_result = RetrievalUnitBuildResult(
        media_clip_units=media_units,
        video_transcript_units=transcript_units,
        summary=RetrievalUnitBuildSummary(video_id=video_id, video_media_id=video_media_id, post_id=post_id),
    )
    builder = FakeBuilder(build_result)
    storage = FakeStorage({("mira-data", "frames/good.jpg")})
    image_embedder = FakeImageEmbedder()
    text_embedder = FakeTextEmbedder()
    vector_store = FakeVectorStore()

    indexer = VideoRetrievalIndexingService(
        builder=builder,
        storage=storage,
        image_embedder=image_embedder,
        text_embedder=text_embedder,
        vector_store=vector_store,
    )
    summary = indexer.index_video(video_media_id).to_dict()

    assert builder.called_with == video_media_id
    assert storage.downloaded == [("mira-data", "frames/good.jpg"), ("mira-data", "frames/missing.jpg")]
    assert len(image_embedder.paths) == 1
    assert text_embedder.texts == ["mở đầu", "hoạt động"]

    media_call = vector_store.calls[0]
    transcript_call = vector_store.calls[1]
    assert media_call["collection_name"] == "media_clip"
    assert media_call["point_ids"] == [good_frame_media_id]
    assert media_call["vectors"] == [[1.0, 0.0, 0.5]]
    media_payload = media_call["payloads"][0]
    assert media_payload["unit_id"] == f"media_clip:{good_frame_media_id}"
    assert media_payload["video_id"] == video_id
    assert media_payload["post_id"] == post_id
    assert media_payload["frame_media_id"] == good_frame_media_id
    assert media_payload["frame_index"] == 5
    assert media_payload["timestamp_sec"] == 2.5
    assert media_payload["bucket_name"] == "mira-data"
    assert media_payload["frame_object_key"] == "frames/good.jpg"
    assert media_payload["caption"] == "Sinh viên thuyết trình."
    assert media_payload["ocr_text"] == "HIT MIRA"
    assert media_payload["vision_metadata"] == {"scene": "hội trường"}
    assert media_payload["detected_objects"] == [{"label": "person", "confidence": 0.9}]
    assert media_payload["object_counts"] == {"person": 1}
    assert media_payload["transcript_context"]["text"] == "mở đầu hoạt động"
    assert media_payload["transcript_context_start_sec"] == 1.0
    assert media_payload["transcript_context_end_sec"] == 3.5

    assert transcript_call["collection_name"] == "video_transcript"
    assert transcript_call["point_ids"] == [f"video_transcript:{video_id}:seg-1", f"video_transcript:{video_id}:seg-2"]
    assert transcript_call["vectors"] == [[0.2, 0.3], [0.2, 0.3]]
    assert media_payload["source_url"] == POST_URL
    assert transcript_call["payloads"][0] == {
        "unit_id": f"video_transcript:{video_id}:seg-1",
        "video_id": video_id,
        "post_id": post_id,
        "source_url": POST_URL,
        # None ở đây vì `MediaClipUnit` trong test này không khai hai trường đó,
        # nhưng khoá vẫn phải có mặt: ba nguồn dùng chung một bộ khoá thì bộ lọc
        # theo năm mới không âm thầm bỏ sót transcript.
        "post_created_at": None,
        "event_key": None,
        "start_sec": 1.0,
        "end_sec": 2.0,
        "text": "mở đầu",
        "language": "vi",
        "source_segment_ids": ["seg-1"],
    }

    assert summary["video_id"] == video_id
    assert summary["media_clip_units_received"] == 2
    assert summary["video_transcript_units_received"] == 2
    assert summary["media_clip_indexed"] == 1
    assert summary["video_transcript_indexed"] == 2
    assert summary["skipped_by_reason"] == {"frame_download_failed": 1}
    assert summary["collections"] == {"media_clip": "media_clip", "video_transcript": "video_transcript"}


def test_index_video_skips_empty_transcript_units() -> None:
    video_media_id = str(uuid4())
    video_id = str(uuid4())
    post_id = str(uuid4())
    build_result = RetrievalUnitBuildResult(
        media_clip_units=[],
        video_transcript_units=[
            VideoTranscriptUnit(
                unit_id=f"video_transcript:{video_id}:empty",
                video_id=video_id,
                post_id=post_id,
                start_sec=0.0,
                end_sec=1.0,
                text="   ",
                language="vi",
                source_segment_ids=["empty"],
            )
        ],
        summary=RetrievalUnitBuildSummary(video_id=video_id, video_media_id=video_media_id, post_id=post_id),
    )
    vector_store = FakeVectorStore()
    indexer = VideoRetrievalIndexingService(
        builder=FakeBuilder(build_result),
        storage=FakeStorage(set()),
        image_embedder=FakeImageEmbedder(),
        text_embedder=FakeTextEmbedder(),
        vector_store=vector_store,
    )

    summary = indexer.index_video(video_media_id).to_dict()

    assert vector_store.calls == []
    assert summary["video_transcript_indexed"] == 0
    assert summary["skipped_by_reason"] == {"empty_transcript": 1}


def test_transcript_only_khong_dung_toi_keyframe() -> None:
    """Vá lời thoại thì không được nhúng lại keyframe.

    Hai nhánh vốn độc lập. Khi chỉ transcript đổi (ví dụ vá lại mốc giây 0 bị
    ghi nhầm thành NULL), nhúng lại hàng trăm keyframe chỉ để ra đúng vector cũ
    là đốt công và đốt tiền GPU. Đợt vá thật: 56 video, 0 keyframe bị đụng tới.
    """
    video_media_id = str(uuid4())
    video_id = str(uuid4())
    post_id = str(uuid4())
    build_result = RetrievalUnitBuildResult(
        media_clip_units=[
            MediaClipUnit(
                unit_id=f"media_clip:{uuid4()}",
                video_id=video_id,
                video_media_id=video_media_id,
                post_id=post_id,
                frame_media_id=str(uuid4()),
                frame_index=1,
                timestamp_sec=0.0,
                bucket_name="mira-data",
                frame_object_key="frames/good.jpg",
                caption="",
                ocr_text="",
                vision_metadata={},
                detected_objects=[],
                object_counts={},
                transcript_context=TranscriptContext(text="", source_segment_ids=[], segments=[]),
            )
        ],
        video_transcript_units=[
            VideoTranscriptUnit(
                unit_id=f"video_transcript:{video_id}:seg-0",
                video_id=video_id,
                post_id=post_id,
                start_sec=0.0,
                end_sec=60.0,
                text="chào mừng đến với câu lạc bộ",
                language="vi",
                source_segment_ids=["seg-0"],
            )
        ],
        summary=RetrievalUnitBuildSummary(video_id=video_id, video_media_id=video_media_id, post_id=post_id),
    )
    image_embedder = FakeImageEmbedder()
    vector_store = FakeVectorStore()
    indexer = VideoRetrievalIndexingService(
        builder=FakeBuilder(build_result),
        storage=FakeStorage({("mira-data", "frames/good.jpg")}),
        image_embedder=image_embedder,
        text_embedder=FakeTextEmbedder(),
        vector_store=vector_store,
    )

    summary = indexer.index_video(video_media_id, transcript_only=True).to_dict()

    assert summary["video_transcript_indexed"] == 1
    assert summary["media_clip_indexed"] == 0
    assert image_embedder.paths == []
    assert [call["collection_name"] for call in vector_store.calls] == ["video_transcript"]


def test_giay_0_la_moc_hop_le_khong_phai_thieu_moc() -> None:
    """`start_sec=0.0` phải đi hết đường vào payload Qdrant.

    Đây là điều kiện để đoạn mở đầu mỗi video tìm kiếm được — chỗ nói tên câu
    lạc bộ và tên sự kiện, tức phần nhận dạng rõ nhất của cả video.
    """
    video_media_id = str(uuid4())
    video_id = str(uuid4())
    post_id = str(uuid4())
    build_result = RetrievalUnitBuildResult(
        media_clip_units=[],
        video_transcript_units=[
            VideoTranscriptUnit(
                unit_id=f"video_transcript:{video_id}:seg-0",
                video_id=video_id,
                post_id=post_id,
                start_sec=0.0,
                end_sec=60.0,
                text="chào mừng bạn đến với câu lạc bộ tin học HIT",
                language="vi",
                source_segment_ids=["seg-0"],
            )
        ],
        summary=RetrievalUnitBuildSummary(video_id=video_id, video_media_id=video_media_id, post_id=post_id),
    )
    vector_store = FakeVectorStore()
    indexer = VideoRetrievalIndexingService(
        builder=FakeBuilder(build_result),
        storage=FakeStorage(set()),
        image_embedder=FakeImageEmbedder(),
        text_embedder=FakeTextEmbedder(),
        vector_store=vector_store,
    )

    summary = indexer.index_video(video_media_id).to_dict()

    assert summary["video_transcript_indexed"] == 1
    assert summary["skipped_by_reason"] == {}
    payload = vector_store.calls[0]["payloads"][0]
    assert payload["start_sec"] == 0.0
