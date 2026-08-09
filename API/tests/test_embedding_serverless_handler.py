"""Contract của worker RunPod Serverless cho jina-clip-v2.

Model thật được thay bằng fake để test không cần GPU, mạng hay tải trọng số.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

EMBEDDING_SERVER = Path(__file__).resolve().parents[2] / "embedding_server"
sys.path.insert(0, str(EMBEDDING_SERVER))

import encoder as encoder_module  # noqa: E402
import runpod_handler  # noqa: E402

DIM = 1024


class _EncoderGia:
    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str]]] = []

    @staticmethod
    def _vectors(count: int) -> list[list[float]]:
        return [[float(index)] + [0.0] * (DIM - 1) for index in range(count)]

    def encode_texts(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(("text", texts))
        return self._vectors(len(texts))

    def encode_images(self, images: list[str]) -> list[list[float]]:
        self.calls.append(("image", images))
        return self._vectors(len(images))


@pytest.fixture
def encoder_gia(monkeypatch):
    fake = _EncoderGia()
    monkeypatch.setattr(runpod_handler, "encoder", fake)
    monkeypatch.setattr(runpod_handler, "MAX_BATCH_SIZE", 64)
    return fake


def test_text_job_giu_nguyen_jina_response_contract(encoder_gia):
    output = runpod_handler.handler(
        {
            "id": "job-1",
            "input": {
                "model": "jina-clip-v2",
                "input": [{"text": "sự kiện mùa hè"}, {"text": "HIT Open Day"}],
                "task": "retrieval.query",
                "dimensions": DIM,
                "normalized": True,
            },
        }
    )

    assert output["object"] == "list"
    assert output["usage"]["items"] == 2
    assert [row["index"] for row in output["data"]] == [0, 1]
    assert all(len(row["embedding"]) == DIM for row in output["data"])
    assert encoder_gia.calls == [("text", ["sự kiện mùa hè", "HIT Open Day"])]


def test_image_job_di_vao_image_encoder(encoder_gia):
    output = runpod_handler.handler(
        {"input": {"input": [{"image": "base64-a"}, {"image": "base64-b"}]}}
    )

    assert output["usage"]["items"] == 2
    assert encoder_gia.calls == [("image", ["base64-a", "base64-b"])]


@pytest.mark.parametrize(
    "job, message",
    [
        ({}, "job.input phải là một object"),
        ({"input": []}, "job.input phải là một object"),
        ({"input": {}}, "job.input sai schema"),
        ({"input": {"input": [{}]}}, "phải có đúng một trường"),
        ({"input": {"input": [{"text": "", "image": "b"}]}}, "phải có đúng một trường"),
        ({"input": {"input": [{"text": "  "}]}}, "text rỗng"),
        (
            {"input": {"input": [{"text": "a"}, {"image": "b"}]}},
            "chỉ được chứa toàn text hoặc toàn image",
        ),
        ({"input": {"input": [{"text": "a"}], "dimensions": 512}}, "1024 chiều"),
        (
            {"input": {"input": [{"text": "a"}], "task": "retrieval.passage"}},
            "retrieval.query",
        ),
        ({"input": {"input": [{"text": "a"}], "normalized": False}}, "normalized"),
    ],
)
def test_job_sai_phai_fail_ro_rang(job, message, encoder_gia):
    with pytest.raises(runpod_handler.RunPodEmbeddingInputError, match=message):
        runpod_handler.handler(job)
    assert encoder_gia.calls == []


def test_khong_nhan_batch_lon_hon_gioi_han(encoder_gia, monkeypatch):
    monkeypatch.setattr(runpod_handler, "MAX_BATCH_SIZE", 2)
    job = {"input": {"input": [{"text": "a"}, {"text": "b"}, {"text": "c"}]}}

    with pytest.raises(runpod_handler.RunPodEmbeddingInputError, match="tối đa 2"):
        runpod_handler.handler(job)


def test_model_tra_thieu_vector_phai_fail_job(monkeypatch):
    fake = _EncoderGia()
    monkeypatch.setattr(fake, "encode_texts", lambda _: [])
    monkeypatch.setattr(runpod_handler, "encoder", fake)

    with pytest.raises(RuntimeError, match="Model trả 0 vector cho 1 input"):
        runpod_handler.handler({"input": {"input": [{"text": "a"}]}})


def test_loi_schema_khong_lap_lai_base64_vao_message(encoder_gia):
    marker = "base64-khong-duoc-lo-ra-log"
    with pytest.raises(runpod_handler.RunPodEmbeddingInputError) as caught:
        runpod_handler.handler({"input": {"input": marker}})
    assert marker not in str(caught.value)


def test_resolve_runpod_cached_model_uu_tien_refs_main(tmp_path, monkeypatch):
    cache_root = tmp_path / "hub"
    model_root = cache_root / "models--jinaai--jina-clip-v2"
    wanted = model_root / "snapshots" / "revision-b"
    other = model_root / "snapshots" / "revision-a"
    wanted.mkdir(parents=True)
    other.mkdir(parents=True)
    (model_root / "refs").mkdir()
    (model_root / "refs" / "main").write_text("revision-b\n", encoding="utf-8")
    monkeypatch.setattr(encoder_module, "RUNPOD_HF_CACHE_ROOT", cache_root)
    monkeypatch.setenv("EMBED_REQUIRE_CACHED_MODEL", "1")

    source, local_only = encoder_module._resolve_model_source()

    assert source == str(wanted)
    assert local_only is True


def test_thieu_cached_model_khong_am_tham_tai_huggingface(tmp_path, monkeypatch):
    monkeypatch.setattr(encoder_module, "RUNPOD_HF_CACHE_ROOT", tmp_path)
    monkeypatch.setenv("EMBED_REQUIRE_CACHED_MODEL", "1")

    with pytest.raises(encoder_module.EncoderError, match="Cached model"):
        encoder_module._resolve_model_source()
