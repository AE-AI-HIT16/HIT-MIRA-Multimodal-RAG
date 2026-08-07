"""Chọn đường nhúng — TC-902 (giao thức Jina trực tiếp hay hàng đợi RunPod).

Không lời gọi mạng nào: test chỉ nói về việc chọn và về cấu hình client dựng ra.
"""

from __future__ import annotations

import pytest

from src.configuration import AppConfig
from src.rag_video_anh.embedding.embedding_service import ImageEmbeddingConfigurationError
from src.rag_video_anh.embedding.provider import (
    ONLINE_JOB_TIMEOUT_SECONDS,
    PROVIDER_JINA,
    PROVIDER_RUNPOD,
    build_media_embedder,
    resolve_media_embedding_provider,
)
from src.rag_video_anh.embedding.runpod_transport import RunPodEmbeddingTransport

ENDPOINT = "z1zyigyd4vl2fy"
POD_URL = "http://10.0.0.5:8100/v1/embeddings"


def dat_runpod(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JINA_RUNPOD_ENDPOINT_ID", ENDPOINT)
    monkeypatch.setenv("RUNPOD_API_KEY", "rpa_test")


# ---------------------------------------------------------------- tự dò


def test_khong_khai_gi_thi_giu_nhanh_truc_tiep() -> None:
    """Người vận hành phải nhận đúng thông báo cũ, không phải lỗi về RunPod."""
    assert resolve_media_embedding_provider() == PROVIDER_JINA

    service = build_media_embedder()
    with pytest.raises(ImageEmbeddingConfigurationError, match="JINA_API_KEY"):
        service.embed_texts(["thử"])


def test_chi_khai_runpod_thi_di_hang_doi(monkeypatch: pytest.MonkeyPatch) -> None:
    dat_runpod(monkeypatch)
    assert resolve_media_embedding_provider() == PROVIDER_RUNPOD

    service = build_media_embedder()
    assert service.base_url.endswith(f"/{ENDPOINT}/run")
    assert isinstance(service._http_post, RunPodEmbeddingTransport)


def test_chi_khai_endpoint_truc_tiep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MEDIA_IMAGE_EMBEDDING_BASE_URL", POD_URL)
    assert resolve_media_embedding_provider() == PROVIDER_JINA

    service = build_media_embedder()
    assert service.base_url == POD_URL
    assert service._http_post is None


def test_khai_ca_hai_thi_truc_tiep_thang(monkeypatch: pytest.MonkeyPatch) -> None:
    """Không cold start, và giống hệt hành vi trước khi có adapter."""
    dat_runpod(monkeypatch)
    monkeypatch.setenv("MEDIA_IMAGE_EMBEDDING_BASE_URL", POD_URL)

    assert resolve_media_embedding_provider() == PROVIDER_JINA
    assert build_media_embedder().base_url == POD_URL


def test_placeholder_chua_thay_khong_tinh_la_da_cau_hinh(monkeypatch: pytest.MonkeyPatch) -> None:
    """`${...}` sót trong YAML từng bị coi là 'đã có endpoint trực tiếp'."""
    dat_runpod(monkeypatch)
    monkeypatch.setenv("MEDIA_IMAGE_EMBEDDING_BASE_URL", "${MEDIA_IMAGE_EMBEDDING_BASE_URL}")

    assert resolve_media_embedding_provider() == PROVIDER_RUNPOD


# ---------------------------------------------------------------- ép tường minh


def test_ep_runpod_du_da_co_endpoint_truc_tiep(monkeypatch: pytest.MonkeyPatch) -> None:
    """Đẩy mẻ index sang GPU trong khi đường online vẫn dùng Pod."""
    dat_runpod(monkeypatch)
    monkeypatch.setenv("MEDIA_IMAGE_EMBEDDING_BASE_URL", POD_URL)
    monkeypatch.setenv("MEDIA_EMBEDDING_PROVIDER", "runpod")

    assert resolve_media_embedding_provider() == PROVIDER_RUNPOD
    assert build_media_embedder().base_url.endswith(f"/{ENDPOINT}/run")


def test_ep_jina_du_chi_khai_runpod(monkeypatch: pytest.MonkeyPatch) -> None:
    dat_runpod(monkeypatch)
    monkeypatch.setenv("MEDIA_EMBEDDING_PROVIDER", "JINA")

    assert resolve_media_embedding_provider() == PROVIDER_JINA


def test_gia_tri_la_bi_chan_ngay(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MEDIA_EMBEDDING_PROVIDER", "openai")
    with pytest.raises(ImageEmbeddingConfigurationError, match="không hợp lệ"):
        resolve_media_embedding_provider()


# ---------------------------------------------------------------- online vs offline


def test_truy_van_online_qua_hang_doi_bi_rut_han_cho(monkeypatch: pytest.MonkeyPatch) -> None:
    """Người đang chờ khung chat không nên bị treo 15 phút của batch."""
    dat_runpod(monkeypatch)
    online = build_media_embedder(for_online_queries=True)
    offline = build_media_embedder()

    assert online._http_post.job_timeout == ONLINE_JOB_TIMEOUT_SECONDS
    assert offline._http_post.job_timeout > ONLINE_JOB_TIMEOUT_SECONDS


def test_truy_van_online_qua_hang_doi_phai_keu_to(monkeypatch: pytest.MonkeyPatch, caplog) -> None:
    """Vượt ranh giới online/offline thì phải để lại dấu vết, không im lặng."""
    dat_runpod(monkeypatch)
    with caplog.at_level("WARNING"):
        build_media_embedder(for_online_queries=True)
    assert "cold start" in caplog.text


def test_online_qua_endpoint_truc_tiep_thi_khong_canh_bao(
    monkeypatch: pytest.MonkeyPatch, caplog
) -> None:
    monkeypatch.setenv("MEDIA_IMAGE_EMBEDDING_BASE_URL", POD_URL)
    with caplog.at_level("WARNING"):
        build_media_embedder(for_online_queries=True)
    assert "cold start" not in caplog.text


# ---------------------------------------------------------------- điểm dựng thật


def test_ba_diem_dung_deu_di_qua_factory(monkeypatch: pytest.MonkeyPatch) -> None:
    """Đường index và đường truy vấn cùng đọc một cấu hình, không lệch nhau."""
    dat_runpod(monkeypatch)
    from src.rag_video_anh.retrieval.indexing_service import VideoRetrievalIndexingService
    from src.rag_video_anh.retrieval.retriever import VideoRetriever

    class FakeVectorStore:
        pass

    indexer = VideoRetrievalIndexingService(
        builder=object(), image_builder=object(), vector_store=FakeVectorStore()
    )
    retriever = VideoRetriever(vector_store=FakeVectorStore(), config=AppConfig())

    assert isinstance(indexer.image_embedder._http_post, RunPodEmbeddingTransport)
    assert isinstance(retriever.embedding_service._http_post, RunPodEmbeddingTransport)
    # Transcript vẫn dùng chung client với ảnh — đó là thứ giữ một vector truy vấn.
    assert indexer.text_embedder is indexer.image_embedder
