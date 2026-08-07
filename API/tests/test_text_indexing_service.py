"""Index văn bản nhánh media — TC-906.

Toàn bộ chạy với fake: không gọi nhà cung cấp nhúng, không chạm Qdrant, không
chạm MinIO. Ba tính chất phải giữ: không tải ảnh, một văn bản một lời gọi, và
vector–payload luôn khớp 1:1.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.rag_video_anh.retrieval import text_units as tu
from src.rag_video_anh.retrieval.text_indexing_service import (
    DEFAULT_MEDIA_TEXT_COLLECTION,
    MAX_TOKENS_PER_REQUEST,
    NGUON_HINH_ANH,
    IndexScope,
    MediaTextIndexingService,
)

CHIEU = 1536


class FakeEmbedder:
    model = "text-embedding-3-small"

    def __init__(self) -> None:
        self.lo_da_goi: list[list[str]] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.lo_da_goi.append(list(texts))
        return [[float(len(t))] * CHIEU for t in texts]


class FakeVectorStore:
    """Giữ point trong RAM để test được cả lượt chạy lại."""

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.deleted: list[str] = []
        self.points: dict[str, dict] = {}

    def upsert_points(self, *, collection_name, point_ids, vectors, payloads):
        self.calls.append({
            "collection_name": collection_name,
            "point_ids": point_ids,
            "vectors": vectors,
            "payloads": payloads,
        })
        for pid, payload in zip(point_ids, payloads):
            self.points[pid] = dict(payload)
        return {"upserted": len(point_ids)}

    def get_payloads(self, *, collection_name, point_ids):
        return {pid: self.points[pid] for pid in point_ids if pid in self.points}

    def delete_points(self, *, collection_name, point_ids):
        for pid in point_ids:
            self.points.pop(pid, None)
            self.deleted.append(pid)
        return {"deleted": len(point_ids)}

    def list_unit_ids_by_field(self, *, collection_name, field_name, value, source_types=None):
        return [
            pid
            for pid, p in self.points.items()
            if p.get(field_name) == value and (not source_types or p.get("source_type") in source_types)
        ]


def anh(media_id="img-1", caption="", ocr=""):
    return SimpleNamespace(
        image_media_id=media_id, post_id="post-1", bucket_name="b", object_key=f"{media_id}.jpg",
        caption=caption, ocr_text=ocr, source_url=None,
    )


def keyframe(frame_id="fr-1", caption="", ocr=""):
    return SimpleNamespace(
        video_id="vid-1", video_media_id="vm-1", frame_media_id=frame_id, frame_index=0,
        timestamp_sec=1.0, post_id="post-1", bucket_name="b", frame_object_key=f"{frame_id}.jpg",
        caption=caption, ocr_text=ocr, source_url=None,
    )


def doan_transcript(text):
    return SimpleNamespace(
        video_id="vid-1", post_id="post-1", start_sec=0.0, end_sec=60.0,
        text=text, language="vi", source_segment_ids=["s1"], source_url=None,
    )


def dung_service(**kwargs) -> tuple[MediaTextIndexingService, FakeEmbedder, FakeVectorStore]:
    emb, store = FakeEmbedder(), FakeVectorStore()
    svc = MediaTextIndexingService(
        image_builder=SimpleNamespace(build=lambda _: None),
        video_builder=SimpleNamespace(build=lambda _: None),
        text_embedder=emb,
        vector_store=store,
        **kwargs,
    )
    return svc, emb, store


# ---------------------------------------------------------------- không chạm ảnh


def test_khong_doi_anh_phai_ton_tai_trong_minio() -> None:
    """Caption và OCR đã nằm trong PostgreSQL; object bị xoá không xoá được chữ đã đọc."""
    svc = MediaTextIndexingService(text_embedder=FakeEmbedder(), vector_store=FakeVectorStore())
    assert svc.image_builder.check_image_exists is False
    assert svc.video_builder.check_frame_exists is False


def test_index_ghi_vao_collection_1536_rieng() -> None:
    """Không được ghi lẫn vào media_clip 1.024 chiều."""
    svc, _, store = dung_service()
    svc.index_units(tu.units_tu_anh(anh(caption="Một buổi sinh hoạt")))
    assert store.calls[0]["collection_name"] == DEFAULT_MEDIA_TEXT_COLLECTION
    assert DEFAULT_MEDIA_TEXT_COLLECTION != "media_clip"


# ---------------------------------------------------------------- gộp theo nội dung


def test_caption_trung_nhau_chi_goi_api_mot_lan() -> None:
    """Cùng caption trên nhiều frame: một vector, nhưng vẫn đủ point và metadata riêng."""
    units = tu.units_tu_keyframe(keyframe("fr-1", caption="Ảnh tập thể")) + \
            tu.units_tu_keyframe(keyframe("fr-2", caption="Ảnh tập thể"))
    svc, emb, store = dung_service()
    summary = svc.index_units(units)

    assert summary.units_built == 2
    assert summary.unique_texts == 1
    assert emb.lo_da_goi == [["Ảnh tập thể"]]
    assert summary.points_upserted == 2
    assert store.calls[0]["point_ids"] == ["caption:frame:fr-1", "caption:frame:fr-2"]


def test_vector_va_payload_khop_1_1() -> None:
    units = tu.units_tu_anh(anh(caption="mô tả", ocr="HIT"))
    svc, _, store = dung_service()
    svc.index_units(units)

    goi = store.calls[0]
    assert len(goi["point_ids"]) == len(goi["vectors"]) == len(goi["payloads"]) == 2
    for payload, unit in zip(goi["payloads"], units):
        assert payload["unit_id"] == unit.unit_id
        assert payload["source_type"] == unit.source_type
        assert payload["text"] == unit.text


def test_payload_ghi_model_va_so_chieu() -> None:
    """Đọc một point là biết ngay nó thuộc không gian vector nào."""
    svc, _, store = dung_service()
    svc.index_units(tu.units_tu_anh(anh(caption="x")))
    payload = store.calls[0]["payloads"][0]

    assert payload["embedding_model"] == "text-embedding-3-small"
    assert payload["embedding_dimensions"] == CHIEU
    assert payload["embedding_format_version"] == tu.EMBEDDING_FORMAT_VERSION


# ---------------------------------------------------------------- chia lô


def test_chia_lo_theo_so_luong() -> None:
    svc, emb, _ = dung_service(batch_size=2)
    units = [u for i in range(5) for u in tu.units_tu_anh(anh(f"img-{i}", caption=f"caption {i}"))]
    svc.index_units(units)

    assert [len(lo) for lo in emb.lo_da_goi] == [2, 2, 1]


def test_chia_lo_theo_tong_token(monkeypatch: pytest.MonkeyPatch) -> None:
    """Đủ số lượng nhưng vượt hạn mức token thì cả lô ăn 400."""
    svc, emb, _ = dung_service(batch_size=1000)
    monkeypatch.setattr(
        "src.rag_video_anh.retrieval.text_indexing_service.dem_token",
        lambda _: MAX_TOKENS_PER_REQUEST // 2,
    )
    units = [u for i in range(3) for u in tu.units_tu_anh(anh(f"img-{i}", caption=f"caption {i}"))]
    svc.index_units(units)

    assert len(emb.lo_da_goi) == 2
    assert [len(lo) for lo in emb.lo_da_goi] == [2, 1]


def test_nha_cung_cap_tra_thieu_vector_thi_no_ngay() -> None:
    """Lệch 1:1 mà đi tiếp là gán vector của văn bản này cho point của văn bản khác."""
    class ThieuVector(FakeEmbedder):
        def embed_documents(self, texts):
            return super().embed_documents(texts)[:-1]

    svc = MediaTextIndexingService(
        image_builder=SimpleNamespace(), video_builder=SimpleNamespace(),
        text_embedder=ThieuVector(), vector_store=FakeVectorStore(),
    )
    units = [u for i in range(3) for u in tu.units_tu_anh(anh(f"img-{i}", caption=f"caption {i}"))]
    with pytest.raises(RuntimeError, match="1 unit ↔ 1 vector"):
        svc.index_units(units)


# ---------------------------------------------------------------- dựng unit


def test_dung_unit_tu_video_gom_ca_keyframe_va_transcript() -> None:
    ket_qua = SimpleNamespace(
        media_clip_units=[keyframe("fr-1", caption="cảnh mở đầu", ocr="HIT")],
        video_transcript_units=[doan_transcript(" ".join(["nội dung buổi nói chuyện"] * 15))],
    )
    svc, _, _ = dung_service()
    svc.video_builder = SimpleNamespace(build=lambda _: ket_qua)

    group = svc.build_video_group("vid-1")
    assert {u.source_type for u in group.units} == {tu.SOURCE_CAPTION, tu.SOURCE_OCR, tu.SOURCE_TRANSCRIPT}
    # Scope đặt ở mức video, tách theo source_type: dọn transcript không chạm
    # caption/OCR của frame, và ngược lại.
    assert {(s.field_name, tuple(sorted(s.source_types))) for s in group.scopes} == {
        ("video_id", ("caption", "ocr")),
        ("video_id", ("transcript",)),
    }


def test_anh_khong_dung_duoc_thi_khong_sinh_unit() -> None:
    svc, _, store = dung_service()
    group = svc.build_image_group("thieu")
    assert group.units == []
    # Nhưng vẫn giữ danh tính cha, nếu không thì point cũ của ảnh này nằm lại mãi.
    assert group.scopes == (IndexScope("image_media_id", "thieu", NGUON_HINH_ANH),)
    assert svc.index_image("thieu").points_upserted == 0
    assert store.calls == []


# ---------------------------------------------------------------- gộp lô nhiều media


def test_index_many_gop_qua_nhieu_anh_thanh_mot_lo() -> None:
    """Gọi index_units từng ảnh thì batch_size vô nghĩa: 1.628 ảnh ~ 1.628 request."""
    svc, emb, _ = dung_service(batch_size=128, skip_unchanged=False, delete_stale=False)
    nhom = [tu.units_tu_anh(anh(f"img-{i}", caption=f"caption {i}")) for i in range(10)]
    tong = svc.index_many(nhom)

    assert len(emb.lo_da_goi) == 1          # một lời gọi cho cả 10 ảnh
    assert tong.units_built == 10
    assert tong.points_upserted == 10


def test_index_many_bat_duoc_caption_trung_giua_cac_anh() -> None:
    """Trùng nhau GIỮA các ảnh chỉ gộp được khi gom chung một lô."""
    svc, emb, _ = dung_service(batch_size=128, skip_unchanged=False, delete_stale=False)
    nhom = [tu.units_tu_anh(anh(f"img-{i}", caption="Ảnh tập thể CLB")) for i in range(6)]
    tong = svc.index_many(nhom)

    assert emb.lo_da_goi == [["Ảnh tập thể CLB"]]
    assert tong.points_upserted == 6


def test_index_many_xa_lo_khi_du_van_ban_khac_nhau() -> None:
    svc, emb, _ = dung_service(batch_size=3, skip_unchanged=False, delete_stale=False)
    svc.index_many([tu.units_tu_anh(anh(f"img-{i}", caption=f"caption {i}")) for i in range(7)])
    assert [len(lo) for lo in emb.lo_da_goi] == [3, 3, 1]


# ---------------------------------------------------------------- chạy lại


def test_chay_lai_khong_nhung_lai_thu_khong_doi() -> None:
    svc, emb, store = dung_service()
    units = tu.units_tu_anh(anh(caption="Một buổi sinh hoạt"))
    svc.index_units(units)
    assert len(emb.lo_da_goi) == 1

    lai = svc.index_units(units)
    assert len(emb.lo_da_goi) == 1          # không gọi thêm lần nào
    assert lai.points_unchanged == 1
    assert lai.points_upserted == 0


def test_doi_caption_thi_nhung_lai() -> None:
    svc, emb, _ = dung_service()
    svc.index_units(tu.units_tu_anh(anh(caption="bản cũ")))
    lai = svc.index_units(tu.units_tu_anh(anh(caption="bản mới")))

    assert len(emb.lo_da_goi) == 2
    assert lai.points_upserted == 1
    assert lai.points_unchanged == 0


def test_doi_model_thi_nhung_lai_du_van_ban_khong_doi() -> None:
    """Vector của model khác không so được với vector đang nằm trong collection."""
    svc, emb, _ = dung_service()
    units = tu.units_tu_anh(anh(caption="không đổi"))
    svc.index_units(units)

    svc.text_embedder.model = "text-embedding-3-large"
    lai = svc.index_units(units)
    assert lai.points_upserted == 1
    assert len(emb.lo_da_goi) == 2


def test_ocr_cat_lai_it_manh_hon_thi_xoa_point_mo_coi() -> None:
    """'ocr:image:x#5' của lượt trước sẽ nằm lại mãi, mang nội dung đã bị thay thế."""
    dai = "\n".join(f"dòng {i} với khá nhiều chữ tiếng Việt để tốn token" for i in range(200))
    svc, _, store = dung_service()
    svc.index_units(tu.units_tu_anh(anh(ocr=dai)))
    so_point_cu = len(store.points)
    assert so_point_cu > 1

    # Lượt cũ sinh 'ocr:image:img-1#0..#N', lượt mới chỉ còn một mảnh nên id
    # không mang hậu tố — toàn bộ point cũ đều mồ côi.
    lai = svc.index_units(tu.units_tu_anh(anh(ocr="HIT")))
    assert lai.points_deleted == so_point_cu
    assert set(store.points) == {"ocr:image:img-1"}


def test_khong_xoa_point_cua_anh_khac() -> None:
    svc, _, store = dung_service()
    svc.index_units(tu.units_tu_anh(anh("img-1", caption="một")))
    svc.index_units(tu.units_tu_anh(anh("img-2", caption="hai")))
    assert set(store.points) == {"caption:image:img-1", "caption:image:img-2"}
    assert store.deleted == []


# ---------------------------------------------------------------- chốt số chiều


def test_so_chieu_suy_tu_ten_collection() -> None:
    svc, _, _ = dung_service()
    assert svc.expected_dimensions == 1536


def test_so_chieu_lech_thi_no_truoc_khi_ghi() -> None:
    """Ghi 512 chiều vào collection tên _1536_ là trộn hai không gian vector."""
    from src.rag_video_anh.retrieval.text_indexing_service import DimensionMismatchError

    class Emb512(FakeEmbedder):
        def embed_documents(self, texts):
            self.lo_da_goi.append(list(texts))
            return [[0.1] * 512 for _ in texts]

    store = FakeVectorStore()
    svc = MediaTextIndexingService(
        image_builder=SimpleNamespace(), video_builder=SimpleNamespace(),
        text_embedder=Emb512(), vector_store=store,
    )
    with pytest.raises(DimensionMismatchError, match="1536"):
        svc.index_units(tu.units_tu_anh(anh(caption="x")))
    assert store.calls == []          # không ghi gì cả


def test_ten_collection_khong_co_so_chieu_thi_khong_chan() -> None:
    svc, _, store = dung_service(collection_name="media_text_v1")
    assert svc.expected_dimensions is None
    svc.index_units(tu.units_tu_anh(anh(caption="x")))
    assert len(store.calls) == 1


# ---------------------------------------------------------------- hỏng giữa chừng


def test_embedding_loi_thi_khong_xoa_point_cu() -> None:
    """Xoá trước rồi nhúng lỗi là kho rỗng đi một khoảng mà không ai chủ ý."""
    dai = "\n".join(f"dòng {i} có khá nhiều chữ tiếng Việt để tốn token" for i in range(200))
    svc, _, store = dung_service()
    svc.index_units(tu.units_tu_anh(anh(ocr=dai)))
    truoc = dict(store.points)
    assert len(truoc) > 1

    class No(FakeEmbedder):
        def embed_documents(self, texts):
            raise RuntimeError("nhà cung cấp sập")

    svc.text_embedder = No()
    with pytest.raises(RuntimeError, match="nhà cung cấp sập"):
        svc.index_units(tu.units_tu_anh(anh(ocr="HIT")))

    assert store.points == truoc          # còn nguyên
    assert store.deleted == []


def test_vector_sai_chieu_thi_khong_upsert_va_khong_delete() -> None:
    from src.rag_video_anh.retrieval.text_indexing_service import DimensionMismatchError

    dai = "\n".join(f"dòng {i} có khá nhiều chữ tiếng Việt để tốn token" for i in range(200))
    svc, _, store = dung_service()
    svc.index_units(tu.units_tu_anh(anh(ocr=dai)))
    truoc, so_lan_ghi = dict(store.points), len(store.calls)

    class Emb512(FakeEmbedder):
        def embed_documents(self, texts):
            self.lo_da_goi.append(list(texts))
            return [[0.1] * 512 for _ in texts]

    svc.text_embedder = Emb512()
    with pytest.raises(DimensionMismatchError):
        svc.index_units(tu.units_tu_anh(anh(ocr="HIT")))

    assert len(store.calls) == so_lan_ghi
    assert store.points == truoc
    assert store.deleted == []


# ---------------------------------------------------------------- media thành rỗng


def test_anh_mat_het_caption_thi_xoa_point_cu() -> None:
    """`units=[]` không nói được nó thuộc ảnh nào — nên scope phải đi kèm."""
    svc, _, store = dung_service()
    svc.image_builder = SimpleNamespace(build=lambda _: anh("img-1", caption="bản cũ"))
    svc.index_image("img-1")
    assert set(store.points) == {"caption:image:img-1"}

    svc.image_builder = SimpleNamespace(build=lambda _: anh("img-1", caption="", ocr=""))
    tong = svc.index_image("img-1")

    assert tong.points_deleted == 1
    assert store.points == {}


def test_video_mat_transcript_thi_giu_caption_ocr_cua_frame() -> None:
    """Lọc mỗi video_id sẽ quét luôn caption/OCR của mọi frame trong video đó."""
    co_transcript = SimpleNamespace(
        media_clip_units=[keyframe("fr-1", caption="cảnh mở đầu", ocr="HIT")],
        video_transcript_units=[doan_transcript(" ".join(["nội dung buổi nói chuyện"] * 15))],
    )
    mat_transcript = SimpleNamespace(
        media_clip_units=[keyframe("fr-1", caption="cảnh mở đầu", ocr="HIT")],
        video_transcript_units=[],
    )
    svc, _, store = dung_service()
    svc.video_builder = SimpleNamespace(build=lambda _: co_transcript)
    svc.index_video("vm-1")
    assert {p["source_type"] for p in store.points.values()} == {"caption", "ocr", "transcript"}

    svc.video_builder = SimpleNamespace(build=lambda _: mat_transcript)
    tong = svc.index_video("vm-1")

    assert tong.points_deleted == 1
    assert {p["source_type"] for p in store.points.values()} == {"caption", "ocr"}
    assert set(store.points) == {"caption:frame:fr-1", "ocr:frame:fr-1"}


def test_frame_mat_caption_thi_xoa_dung_point_do() -> None:
    svc, _, store = dung_service()
    svc.video_builder = SimpleNamespace(build=lambda _: SimpleNamespace(
        media_clip_units=[keyframe("fr-1", caption="có caption", ocr="HIT")],
        video_transcript_units=[],
    ))
    svc.index_video("vm-1")
    svc.video_builder = SimpleNamespace(build=lambda _: SimpleNamespace(
        media_clip_units=[keyframe("fr-1", caption="", ocr="HIT")],
        video_transcript_units=[],
    ))
    tong = svc.index_video("vm-1")

    assert tong.points_deleted == 1
    assert set(store.points) == {"ocr:frame:fr-1"}


# ---------------------------------------------------------------- cấu hình số chiều


def test_ten_collection_1536_di_voi_explicit_512_la_loi_cau_hinh() -> None:
    """Cho explicit ghi đè tên là vô hiệu hoá cam kết ghi trong chính tên collection."""
    from src.rag_video_anh.retrieval.text_indexing_service import CollectionConfigError

    with pytest.raises(CollectionConfigError, match="1536"):
        MediaTextIndexingService(
            image_builder=SimpleNamespace(), video_builder=SimpleNamespace(),
            text_embedder=FakeEmbedder(), vector_store=FakeVectorStore(),
            collection_name="media_text_te3s_1536_v1", expected_dimensions=512,
        )


def test_explicit_trung_voi_ten_thi_chap_nhan() -> None:
    svc, _, _ = dung_service(expected_dimensions=1536)
    assert svc.expected_dimensions == 1536


def test_ten_khong_co_so_chieu_thi_tat_skip_unchanged() -> None:
    """Không kiểm được không gian vector thì không được tin 'không đổi'."""
    svc, emb, _ = dung_service(collection_name="media_text_v1")
    assert svc.expected_dimensions is None
    assert svc.skip_unchanged is False

    units = tu.units_tu_anh(anh(caption="x"))
    svc.index_units(units)
    svc.index_units(units)
    assert len(emb.lo_da_goi) == 2      # nhúng lại chứ không bỏ qua


# ---------------------------------------------------------------- danh tính video


def ket_qua_video(keyframes=(), transcripts=(), video_id="vid-1"):
    """Bắt chước RetrievalUnitBuildResult, có `summary.video_id` như builder thật."""
    return SimpleNamespace(
        media_clip_units=list(keyframes),
        video_transcript_units=list(transcripts),
        summary=SimpleNamespace(video_id=video_id),
    )


def test_video_rong_hoan_toan_van_don_duoc_transcript_cu() -> None:
    """summary.video_id là nguồn duy nhất còn đúng khi cả hai danh sách đều rỗng."""
    svc, _, store = dung_service()
    svc.video_builder = SimpleNamespace(build=lambda _: ket_qua_video(
        transcripts=[doan_transcript(" ".join(["nội dung buổi nói chuyện"] * 15))]))
    svc.index_video("vm-1")
    assert any(p["source_type"] == "transcript" for p in store.points.values())

    # Transcript bị xoá VÀ không còn keyframe hợp lệ nào.
    svc.video_builder = SimpleNamespace(build=lambda _: ket_qua_video())
    tong = svc.index_video("vm-1")

    assert tong.points_deleted == 1
    assert store.points == {}


def test_frame_bi_builder_loai_thi_point_cu_van_bi_don() -> None:
    """Frame hỏng timestamp/object_key không sinh MediaClipUnit, nên không sinh scope frame."""
    svc, _, store = dung_service()
    svc.video_builder = SimpleNamespace(build=lambda _: ket_qua_video(
        keyframes=[keyframe("fr-1", caption="cảnh một"), keyframe("fr-2", caption="cảnh hai")],
        transcripts=[doan_transcript(" ".join(["lời thoại của video"] * 15))],
    ))
    svc.index_video("vm-1")
    assert set(store.points) == {"caption:frame:fr-1", "caption:frame:fr-2", *(
        k for k in store.points if k.startswith("transcript:"))}

    # fr-2 bị loại hẳn khỏi kết quả builder.
    svc.video_builder = SimpleNamespace(build=lambda _: ket_qua_video(
        keyframes=[keyframe("fr-1", caption="cảnh một")],
        transcripts=[doan_transcript(" ".join(["lời thoại của video"] * 15))],
    ))
    tong = svc.index_video("vm-1")

    assert tong.points_deleted == 1
    assert "caption:frame:fr-2" not in store.points
    assert "caption:frame:fr-1" in store.points
    assert any(k.startswith("transcript:") for k in store.points)   # transcript không bị chạm


def test_khong_suy_duoc_video_id_thi_khong_don_bua() -> None:
    """Thà bỏ sót còn hơn xoá nhầm khi không biết point thuộc về ai."""
    svc, _, store = dung_service()
    svc.video_builder = SimpleNamespace(build=lambda _: SimpleNamespace(
        media_clip_units=[], video_transcript_units=[], summary=SimpleNamespace(video_id=None)))
    group = svc.build_video_group("vm-1")

    assert group.scopes == ()
    assert svc.index_video("vm-1").points_deleted == 0
