"""Test pipeline OFFLINE — TC-201/202/205 + chuỗi frames→embed→index.

frames dùng ffmpeg thật (skip nếu thiếu ffmpeg); embed/index dùng fake để khỏi cần
torch/Qdrant. Đối xứng với test_rag_chain.py cho nhánh online.
"""
from __future__ import annotations

import shutil
import subprocess

import pytest

from pipeline.asr import chunk_transcript, extract_transcript
from pipeline.caption import generate_caption
from pipeline.embed import embed_images
from pipeline.frames import extract_keyframes
from pipeline.index import build_index
from pipeline.run import process_video
from shared.providers.asr import TranscriptSegment

HAS_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None
needs_ffmpeg = pytest.mark.skipif(not HAS_FFMPEG, reason="cần ffmpeg/ffprobe")


@pytest.fixture
def sample_video(tmp_path):
    """Sinh video test 3s (testsrc, KHÔNG audio) bằng ffmpeg."""
    path = tmp_path / "sample.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc=duration=3:size=128x128:rate=10",
         "-pix_fmt", "yuv420p", str(path)],
        capture_output=True, check=True,
    )
    return str(path)


@pytest.fixture
def audio_video(tmp_path):
    """Video 3s CÓ audio (sine) để test tách audio + ASR."""
    path = tmp_path / "with_audio.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc=duration=3:size=128x128:rate=10",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=3",
         "-pix_fmt", "yuv420p", "-shortest", str(path)],
        capture_output=True, check=True,
    )
    return str(path)


class FakeASR:
    """Trả segment cố định bất kể audio (đóng vai PhoWhisper khi test)."""

    def transcribe(self, audio_path):
        return [
            TranscriptSegment(0.0, 2.0, "chào mừng tân sinh viên"),
            TranscriptSegment(2.0, 4.0, "khai giảng câu lạc bộ"),
        ]


class FakeCaptioner:
    def caption(self, image_path):
        return "sinh viên chụp ảnh tại lễ khai giảng"


class BoomCaptioner:
    def caption(self, image_path):
        raise RuntimeError("hết quota vision")


class FakeImageEmbedder:
    dim = 4

    def embed(self, image_paths):
        return [[0.1, 0.2, 0.3, 0.4] for _ in image_paths]

    def embed_query(self, text):
        return [0.1, 0.2, 0.3, 0.4]


class FakeStore:
    def __init__(self):
        self.ensured = False
        self.points = None

    def ensure_collection(self):
        self.ensured = True

    def upsert(self, ids, vectors, payloads):
        self.points = list(zip(ids, vectors, payloads, strict=True))


# ---- T-10 ----
@needs_ffmpeg
def test_frames_have_timestamp(sample_video, tmp_path):
    frames = extract_keyframes(sample_video, media_asset_id=1,
                               step_sec=1.0, out_dir=str(tmp_path / "f"))
    assert len(frames) >= 2
    ts = [f.timestamp_sec for f in frames]
    assert ts == sorted(ts) and len(set(ts)) == len(ts)   # tăng dần, không trùng
    assert all(t < 3.0 for t in ts)                       # trong thời lượng
    import os
    assert all(os.path.getsize(f.frame_path) > 0 for f in frames)  # file thật


def test_corrupt_video_returns_empty(tmp_path):
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"not a video")
    assert extract_keyframes(str(bad), media_asset_id=9) == []   # không raise


# ---- T-13 ----
def test_embed_same_dim(tmp_path):
    paths = []
    for i in range(3):
        p = tmp_path / f"{i}.jpg"
        p.write_bytes(b"x")
        paths.append(str(p))
    vecs = embed_images(paths, embedder=FakeImageEmbedder())
    assert len(vecs) == 3
    assert {len(v) for v in vecs} == {4}


def test_embed_skips_missing_file(tmp_path):
    p = tmp_path / "ok.jpg"
    p.write_bytes(b"x")
    vecs = embed_images([str(p), str(tmp_path / "missing.jpg")], embedder=FakeImageEmbedder())
    assert len(vecs) == 1                                  # bỏ file thiếu, không vỡ


# ---- T-14 ----
def test_index_upsert():
    store = FakeStore()
    build_index("media_clip", 4, ["a", "b"], [[1, 2, 3, 4], [5, 6, 7, 8]],
                [{"video_id": 1}, {"video_id": 1}], store=store)
    assert store.ensured
    assert len(store.points) == 2


def test_index_rejects_wrong_dim():
    store = FakeStore()
    with pytest.raises(ValueError, match="sai chiều"):
        build_index("media_clip", 4, ["a"], [[1, 2, 3]], [{"video_id": 1}], store=store)
    assert store.points is None                            # từ chối, không upsert


# ---- T-12 caption ----
def test_caption_generated(tmp_path):
    p = tmp_path / "frame.jpg"
    p.write_bytes(b"x")
    assert generate_caption(str(p), captioner=FakeCaptioner()) == "sinh viên chụp ảnh tại lễ khai giảng"


def test_caption_error_returns_empty(tmp_path):
    p = tmp_path / "frame.jpg"
    p.write_bytes(b"x")
    assert generate_caption(str(p), captioner=BoomCaptioner()) == ""   # gắn cờ review, không raise


# ---- T-11 transcript ----
def test_chunk_transcript_windows_and_timestamps():
    segs = [TranscriptSegment(float(i), float(i + 1), "câu " * 40) for i in range(6)]
    chunks = chunk_transcript(segs, media_asset_id=3, max_chars=300, overlap_chars=60)
    assert len(chunks) >= 2                                # cắt thành nhiều cửa sổ
    assert all(c["video_id"] == 3 for c in chunks)
    assert all(c["end_sec"] >= c["start_sec"] for c in chunks)   # giữ timestamp hợp lệ
    assert chunks[0]["start_sec"] == 0.0


def test_chunk_transcript_short_returns_one():
    segs = [TranscriptSegment(0.0, 1.0, "ngắn")]
    assert len(chunk_transcript(segs, media_asset_id=1)) == 1


@needs_ffmpeg
def test_transcript_segments(audio_video, tmp_path):
    segs = extract_transcript(audio_video, media_asset_id=1, asr=FakeASR(),
                              out_dir=str(tmp_path / "a"))
    assert len(segs) == 2
    assert segs[0].start_sec <= segs[1].start_sec


def test_transcript_empty_when_no_audio(sample_video, tmp_path):
    # testsrc không có audio → _extract_audio None → [] (không raise, không gọi ASR)
    if not HAS_FFMPEG:
        pytest.skip("cần ffmpeg")
    segs = extract_transcript(sample_video, media_asset_id=1, asr=FakeASR(),
                              out_dir=str(tmp_path / "a"))
    assert segs == []


# ---- chuỗi offline end-to-end ----
@needs_ffmpeg
def test_offline_chain_indexes_media(sample_video, tmp_path):
    store = FakeStore()
    summary = process_video(sample_video, media_asset_id=42, collection="media_clip",
                            out_dir=str(tmp_path / "f"),
                            embedder=FakeImageEmbedder(), store=store)
    assert summary["n_frames"] >= 2
    assert summary["n_indexed"] == summary["n_frames"]
    # payload đủ trường cho retrieve_media/synthesize_answer
    _id, _vec, payload = store.points[0]
    assert payload["video_id"] == 42 and "timestamp" in payload


@needs_ffmpeg
def test_offline_chain_captions_media(sample_video, tmp_path):
    store = FakeStore()
    summary = process_video(sample_video, media_asset_id=1, out_dir=str(tmp_path / "f"),
                            embedder=FakeImageEmbedder(), store=store,
                            captioner=FakeCaptioner())
    assert summary["n_captioned"] == summary["n_indexed"] >= 2
    _id, _vec, payload = store.points[0]
    assert payload["caption"] == "sinh viên chụp ảnh tại lễ khai giảng"


@needs_ffmpeg
def test_offline_chain_indexes_transcript(audio_video, tmp_path):
    media_store, transcript_store = FakeStore(), FakeStore()
    summary = process_video(
        audio_video, media_asset_id=7, out_dir=str(tmp_path / "f"),
        embedder=FakeImageEmbedder(), store=media_store,
        asr=FakeASR(), text_embedder=FakeTextEmbedder(), transcript_store=transcript_store,
    )
    assert summary["n_transcript"] >= 1
    _id, _vec, payload = transcript_store.points[0]
    assert payload["video_id"] == 7 and "text" in payload and "start_sec" in payload


class FakeTextEmbedder:
    dim = 4

    def embed(self, texts):
        return [[0.1, 0.2, 0.3, 0.4] for _ in texts]
