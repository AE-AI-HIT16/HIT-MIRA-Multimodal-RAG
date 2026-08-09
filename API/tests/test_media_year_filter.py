"""Lọc kết quả truy hồi theo năm của bài đăng.

`post_created_at` đã nằm trong payload Qdrant (xem `test_post_created_at_payload`),
nhưng ghi được mà không lọc được thì câu hỏi "ảnh Open Day 2024" vẫn trả về cả
sáu năm. Ba thứ ở đây dễ sai mà lại không gây lỗi — kết quả vẫn trông hợp lý:

1. **Múi giờ.** Dữ liệu lưu UTC, người hỏi nghĩ theo lịch Việt Nam. Cắt mốc
   theo UTC làm bài đăng tối 31/12 giờ VN rơi sang năm sau.
2. **Ghép điều kiện.** Video + năm phải là GIAO, còn nhiều năm phải là HỢP.
   Ghép nhầm thành một `should` phẳng thì lọc năm 2024 kéo về cả năm khác.
3. **Rỗng vì lọc.** Không nói ra thì người dùng đọc "0 kết quả" thành "CLB
   không có ảnh này" — đúng kiểu hiểu sai mà `notes` sinh ra để chặn.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from src.rag_video_anh.retrieval.retrieval_service import VideoRetrievalService
from src.rag_video_anh.retrieval.retriever import VideoRetriever
from src.rag_video_anh.vector_store.vector_store import (
    POST_CREATED_AT_KEY,
    QdrantVideoVectorStore,
)

MEDIA_CLIP = "media_clip"
VIDEO_TRANSCRIPT = "video_transcript"


class FakeQdrantClient:
    def __init__(self) -> None:
        self.indexes: list[dict] = []

    def collection_exists(self, collection_name):  # noqa: ARG002
        return True

    def create_payload_index(self, **kwargs):
        self.indexes.append(kwargs)


def build_store() -> QdrantVideoVectorStore:
    return QdrantVideoVectorStore(url="http://localhost:6333", client=FakeQdrantClient())


class FakeEmbedder:
    def embed_texts(self, texts):
        return [[0.1, 0.2, 0.3] for _ in texts]


class FakeVectorStore:
    """Ghi lại filter đã dựng, trả về đúng những gì test dựng sẵn."""

    def __init__(self, results: dict | None = None) -> None:
        self.results = results or {}
        self.calls: list[dict] = []

    def search_points(self, *, collection_name, vector, limit, query_filter=None):  # noqa: ARG002
        self.calls.append({"collection_name": collection_name, "query_filter": query_filter})
        return list(self.results.get(collection_name, []))

    def search_filter(self, video_ids=None, years=None, events=None):
        return {"video_ids": video_ids, "years": years, "events": events}


def build_service(store: FakeVectorStore) -> VideoRetrievalService:
    config = SimpleNamespace(
        retrieval=SimpleNamespace(top_k=5),
        qdrant=SimpleNamespace(url="http://localhost:6333", api_key=None, distance="COSINE"),
    )
    return VideoRetrievalService(
        retriever=VideoRetriever(
            embedding_service=FakeEmbedder(), vector_store=store, config=config
        )
    )


# ── Chuẩn hoá đầu vào ────────────────────────────────────────────────────────


def test_normalize_years_khu_trung_lap_va_giu_thu_tu():
    assert QdrantVideoVectorStore.normalize_years([2025, 2024, 2025]) == [2025, 2024]
    assert QdrantVideoVectorStore.normalize_years(None) == []
    assert QdrantVideoVectorStore.normalize_years([]) == []


def test_normalize_years_nhan_chuoi_so():
    """Query string của GET và form multipart đều gửi chữ, không gửi int."""
    assert QdrantVideoVectorStore.normalize_years([" 2024 "]) == [2024]


@pytest.mark.parametrize("gia_tri", ["hai nghìn", "20o4", None, 202, 20244, True])
def test_normalize_years_tu_choi_gia_tri_rac(gia_tri):
    """`True` nằm trong danh sách này có lý do: bool là int trong Python, không
    chặn riêng thì `years=[True]` lọt thành năm 1 và trả về 0 kết quả lặng lẽ."""
    with pytest.raises(ValueError):
        QdrantVideoVectorStore.normalize_years([gia_tri])


# ── Dựng filter Qdrant ───────────────────────────────────────────────────────


def test_year_filter_cat_moc_theo_gio_viet_nam():
    store = build_store()

    loc = store.search_filter(years=[2024])

    assert len(loc.should) == 1
    khoang = loc.should[0].range
    assert loc.should[0].key == POST_CREATED_AT_KEY
    vn = timezone(timedelta(hours=7))
    assert khoang.gte == datetime(2024, 1, 1, tzinfo=vn)
    # Nửa mở: `lt` mốc đầu năm sau, không phải `lte` một mốc "cuối ngày" tự chế.
    assert khoang.lt == datetime(2025, 1, 1, tzinfo=vn)
    assert khoang.lte is None


def test_nhieu_nam_la_hop_khong_phai_giao():
    store = build_store()

    loc = store.search_filter(years=[2024, 2025])

    assert len(loc.should) == 2, "hai năm ghép bằng must thì không bài nào khớp cả hai"
    assert not loc.must


def test_video_va_nam_ghep_bang_giao():
    store = build_store()

    loc = store.search_filter(video_ids=["video-1"], years=[2024])

    # must giữa hai nhóm, should bên trong mỗi nhóm.
    assert len(loc.must) == 2
    assert not loc.should
    khoa = {dieu_kien.should[0].key for dieu_kien in loc.must}
    assert khoa == {"video_id", POST_CREATED_AT_KEY}


def test_khong_loc_gi_thi_khong_dung_filter():
    store = build_store()

    assert store.search_filter() is None
    assert store.search_filter(video_ids=[], years=[]) is None


def test_collection_moi_duoc_danh_index_datetime():
    """video_transcript sinh sau — không được bắt người vận hành nhớ chạy tay
    như `media_clip` đã phải làm."""
    client = FakeQdrantClient()
    client.collection_exists = lambda collection_name: False  # noqa: ARG005
    store = QdrantVideoVectorStore(url="http://localhost:6333", client=client)
    store.client.create_collection = lambda **kwargs: None

    store.ensure_collection(collection_name=VIDEO_TRANSCRIPT, vector_size=1024)

    assert client.indexes[0]["field_name"] == POST_CREATED_AT_KEY


# ── Đường đi qua service ─────────────────────────────────────────────────────


def test_service_ap_loc_nam_cho_ca_hai_nhanh():
    """Chỉ lọc nhánh ảnh sẽ làm lời thoại của năm khác lọt vào cùng câu trả lời."""
    store = FakeVectorStore()
    build_service(store).retrieve("open day", years=[2024])

    collections = {goi["collection_name"] for goi in store.calls}
    assert collections == {MEDIA_CLIP, VIDEO_TRANSCRIPT}
    assert all(goi["query_filter"]["years"] == [2024] for goi in store.calls)


def test_service_tra_lai_nam_da_chuan_hoa():
    store = FakeVectorStore()

    ket_qua = build_service(store).retrieve("open day", years=[" 2024 ", 2024])

    assert ket_qua["years"] == [2024]


def test_service_noi_ro_khi_rong_vi_loc_nam():
    store = FakeVectorStore()

    ket_qua = build_service(store).retrieve("open day", years=[2019])

    assert ket_qua["found"] is False
    assert any("2019" in note for note in ket_qua["notes"])


def test_service_khong_them_note_khi_van_co_ket_qua():
    store = FakeVectorStore(
        {MEDIA_CLIP: [{"score": 0.5, "payload": {"unit_id": "image:1", "caption": "c"}}]}
    )

    ket_qua = build_service(store).retrieve("open day", years=[2024])

    assert ket_qua["found"] is True
    assert ket_qua["notes"] == []


def test_service_tu_choi_nam_rac_truoc_khi_goi_qdrant():
    """Ném ValueError để router map thành 422; quan trọng là KHÔNG tìm bừa."""
    store = FakeVectorStore()

    with pytest.raises(ValueError):
        build_service(store).retrieve("open day", years=["hai nghìn"])

    assert store.calls == [], "sai đầu vào mà vẫn gọi Qdrant là đã tốn một lượt nhúng"
