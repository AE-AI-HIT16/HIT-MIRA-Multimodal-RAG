"""Lọc kết quả truy hồi theo chuỗi sự kiện của bài đăng.

Song song với bộ lọc năm (`test_media_year_filter`), nhưng có một chỗ khác biệt
đủ để đáng một file riêng: **khoá lọc không phải thứ người dùng gõ**. Payload
chứa slug (`hit-contest-series`) do `scripts/extract_post_events.py` sinh ra,
còn người hỏi gõ "HIT Contest Series". Hai bên phải quy về cùng một chuỗi bằng
CÙNG MỘT hàm — lệch nhau thì bộ lọc trả rỗng cho một sự kiện có thật, và không
có lỗi nào được ném ra.

Khác với năm, tên sự kiện sai KHÔNG bị từ chối: năm có khoảng hợp lệ để kiểm
tra, còn tên thì không — "sự kiện abc" chỉ đơn giản là không khớp gì cả.
"""

from __future__ import annotations

from types import SimpleNamespace

from src.rag_video_anh.retrieval.retrieval_service import VideoRetrievalService
from src.rag_video_anh.retrieval.retriever import VideoRetriever
from src.rag_video_anh.vector_store.vector_store import (
    EVENT_KEY,
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
        retriever=VideoRetriever(embedding_service=FakeEmbedder(), vector_store=store, config=config)
    )


# ── Quy tên người gõ về slug trong payload ───────────────────────────────────


def test_ten_day_du_va_slug_ra_cung_mot_khoa():
    """Bắt người gọi tự slug hoá là giao cho họ một luật ngầm mà sai thì im lặng."""
    assert QdrantVideoVectorStore.normalize_event_keys(["HIT Contest Series"]) == ["hit-contest-series"]
    assert QdrantVideoVectorStore.normalize_event_keys(["hit-contest-series"]) == ["hit-contest-series"]


def test_giu_duoc_chu_d_gach():
    """NFKD không tách `đ` — bỏ dấu kiểu thô sẽ biến "đại hội" thành "ai hoi"."""
    assert QdrantVideoVectorStore.normalize_event_keys(["Đại hội Đoàn"]) == ["dai-hoi-doan"]


def test_khu_trung_lap_va_giu_thu_tu():
    ten = ["HIT Open Day", "hit open day", "HIT Gala"]

    assert QdrantVideoVectorStore.normalize_event_keys(ten) == ["hit-open-day", "hit-gala"]


def test_ten_vo_nghia_bi_bo_qua_chu_khong_nem_loi():
    """Khác với năm: không có khoảng hợp lệ nào để kiểm tra một cái tên."""
    assert QdrantVideoVectorStore.normalize_event_keys(["🔥", "   ", None]) == []
    assert QdrantVideoVectorStore.normalize_event_keys(None) == []


# ── Dựng filter Qdrant ───────────────────────────────────────────────────────


def test_loc_mot_su_kien_dung_khoa_payload():
    loc = build_store().search_filter(events=["HIT Contest Series"])

    assert len(loc.should) == 1
    assert loc.should[0].key == EVENT_KEY
    assert loc.should[0].match.value == "hit-contest-series"


def test_nhieu_su_kien_la_hop_khong_phai_giao():
    loc = build_store().search_filter(events=["HIT Open Day", "HIT Gala"])

    assert len(loc.should) == 2, "hai sự kiện ghép bằng must thì không bài nào khớp cả hai"
    assert not loc.must


def test_su_kien_va_nam_ghep_bang_giao():
    """"Open Day năm 2024" là GIAO. Ghép phẳng vào một `should` vẫn ra kết quả
    trông hợp lý — gồm cả Open Day 2021 và cả ảnh 2024 của sự kiện khác."""
    loc = build_store().search_filter(events=["HIT Open Day"], years=[2024])

    assert len(loc.must) == 2
    assert not loc.should
    assert {dieu_kien.should[0].key for dieu_kien in loc.must} == {EVENT_KEY, POST_CREATED_AT_KEY}


def test_ba_tieu_chi_van_ghep_bang_giao():
    loc = build_store().search_filter(video_ids=["video-1"], years=[2024], events=["HIT Open Day"])

    assert len(loc.must) == 3
    assert {dieu_kien.should[0].key for dieu_kien in loc.must} == {
        "video_id",
        POST_CREATED_AT_KEY,
        EVENT_KEY,
    }


def test_ten_vo_nghia_khong_tao_ra_filter_rong():
    """Filter rỗng khác hẳn không filter: một cái khớp không gì, cái kia khớp tất."""
    assert build_store().search_filter(events=["🔥"]) is None


def test_collection_moi_duoc_danh_index_ca_hai_khoa_loc():
    client = FakeQdrantClient()
    client.collection_exists = lambda collection_name: False  # noqa: ARG005
    store = QdrantVideoVectorStore(url="http://localhost:6333", client=client)
    store.client.create_collection = lambda **kwargs: None

    store.ensure_collection(collection_name=VIDEO_TRANSCRIPT, vector_size=1024)

    assert {chi_muc["field_name"] for chi_muc in client.indexes} == {POST_CREATED_AT_KEY, EVENT_KEY}


# ── Đường đi qua service ─────────────────────────────────────────────────────


def test_service_ap_loc_su_kien_cho_ca_hai_nhanh():
    store = FakeVectorStore()
    build_service(store).retrieve("khai mạc", events=["HIT Open Day"])

    assert {goi["collection_name"] for goi in store.calls} == {MEDIA_CLIP, VIDEO_TRANSCRIPT}
    assert all(goi["query_filter"]["events"] == ["hit-open-day"] for goi in store.calls)


def test_service_tra_lai_su_kien_da_chuan_hoa():
    ket_qua = build_service(FakeVectorStore()).retrieve("khai mạc", events=["HIT Open Day"])

    assert ket_qua["events"] == ["hit-open-day"]


def test_service_noi_ro_khi_rong_vi_loc_su_kien():
    """Không nói ra thì "0 kết quả" bị đọc thành "CLB chưa từng tổ chức việc này"."""
    ket_qua = build_service(FakeVectorStore()).retrieve("khai mạc", events=["Sự kiện chưa có"])

    assert ket_qua["found"] is False
    assert any("su-kien-chua-co" in note for note in ket_qua["notes"])


def test_note_cua_nam_va_su_kien_khong_de_len_nhau():
    """Lọc cả hai mà rỗng thì phải nói rõ cả hai, người dùng mới biết bỏ cái nào."""
    ket_qua = build_service(FakeVectorStore()).retrieve("khai mạc", years=[2019], events=["HIT Open Day"])

    assert len(ket_qua["notes"]) == 2
    assert any("2019" in note for note in ket_qua["notes"])
    assert any("hit-open-day" in note for note in ket_qua["notes"])


def test_khong_them_note_khi_van_co_ket_qua():
    store = FakeVectorStore(
        {MEDIA_CLIP: [{"score": 0.5, "payload": {"unit_id": "image:1", "caption": "c"}}]}
    )

    ket_qua = build_service(store).retrieve("khai mạc", events=["HIT Open Day"])

    assert ket_qua["found"] is True
    assert ket_qua["notes"] == []
