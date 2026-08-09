"""Nạp media từ màn admin — T-62 / P4-3 ①.

Chạy offline: MinIO và Unit of Work đều là fake, tiêm qua `dependency_overrides`
đúng như các router khác trong repo.
"""

from __future__ import annotations

import uuid
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from src.routers import ingest
from src.routers.auth import yeu_cau_admin
from src.routers.ingest import facebook_post_id_tu_link, lay_storage, lay_uow_factory
from src.server import app

JPEG = b"\xff\xd8\xff\xe0-anh"
MP4 = b"\x00\x00\x00\x18ftypmp42-video"


class FakeStorage:
    bucket_name = "mira-data"

    def __init__(self, loi_khi: set[str] | None = None) -> None:
        self.uploaded: list[str] = []
        self.loi_khi = loi_khi or set()
        self.ensured = False

    def ensure_bucket(self) -> None:
        self.ensured = True

    def upload_file(self, local_path, object_key, content_type=None) -> str:
        if object_key in self.loi_khi:
            raise RuntimeError("MinIO từ chối")
        self.uploaded.append(object_key)
        return object_key


class FakePosts:
    def __init__(self) -> None:
        self.upserted: list = []

    def upsert_by_facebook_id(self, post):
        self.upserted.append(post)
        return SimpleNamespace(post_id=uuid.uuid4())


class FakeMedia:
    def __init__(self) -> None:
        self.created: list = []

    def create_media(self, media):
        self.created.append(media)
        return SimpleNamespace(media_id=uuid.uuid4())


class FakeSession:
    """`scalar` trả lần lượt các giá trị đặt sẵn — đủ để lái nhánh trùng/không trùng."""

    def __init__(self, ton_tai: list | None = None) -> None:
        self.ton_tai = list(ton_tai or [])

    def scalar(self, _stmt):
        return self.ton_tai.pop(0) if self.ton_tai else None


class FakeUow:
    def __init__(self, session: FakeSession) -> None:
        self.session = session
        self.posts = FakePosts()
        self.media = FakeMedia()

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False


@pytest.fixture
def client():
    app.dependency_overrides[yeu_cau_admin] = lambda: {"email": "admin@hit", "role": "admin"}
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def dat_fake(storage: FakeStorage, uow: FakeUow) -> None:
    app.dependency_overrides[lay_storage] = lambda: storage
    app.dependency_overrides[lay_uow_factory] = lambda: (lambda: uow)


def nap(client, files, **truong):
    du_lieu = {
        "source_url": "https://facebook.com/hit/posts/1028940839054401",
        "posted_at": "2026-08-05T14:30",
    }
    du_lieu.update(truong)
    return client.post("/api/ingest/upload", files=files, data=du_lieu)


def test_route_da_dang_ky() -> None:
    assert "/api/ingest/upload" in {route.path for route in app.routes}


def test_nap_anh_va_video_ghi_dung_object_key_theo_quy_uoc_crawl(client) -> None:
    """Sai quy ước key là cả dây script index sau đó không tìm thấy file."""
    storage, uow = FakeStorage(), FakeUow(FakeSession())
    dat_fake(storage, uow)

    body = nap(
        client,
        [
            ("files", ("anh_01.jpg", JPEG, "image/jpeg")),
            ("files", ("clip.mp4", MP4, "video/mp4")),
        ],
    ).json()

    assert storage.uploaded == [
        "events/1028940839054401/media/anh_01.jpg",
        "events/1028940839054401/media/clip.mp4",
    ]
    assert len(body["created_ids"]) == 2
    assert [m.media_type for m in uow.media.created] == ["image", "video"]
    assert storage.ensured is True


def test_media_id_tra_ve_la_uuid_dang_chuoi(client) -> None:
    """Kiểu cũ trên web khai `number[]` — id thật là UUID, khai sai thì không map được dòng nào."""
    dat_fake(FakeStorage(), FakeUow(FakeSession()))

    body = nap(client, [("files", ("a.jpg", JPEG, "image/jpeg"))]).json()

    uuid.UUID(body["created_ids"][0])


def test_dinh_dang_khong_ho_tro_bi_bo_qua_ma_khong_giet_ca_lo(client) -> None:
    """P4-3 ①: một file trượt thì phần còn lại vẫn phải vào kho, kèm lý do."""
    storage, uow = FakeStorage(), FakeUow(FakeSession())
    dat_fake(storage, uow)

    body = nap(
        client,
        [
            ("files", ("tailieu.pdf", b"%PDF-", "application/pdf")),
            ("files", ("anh.jpg", JPEG, "image/jpeg")),
        ],
    ).json()

    assert len(body["created_ids"]) == 1
    assert body["skipped"] == [{"file": "tailieu.pdf", "reason": "định dạng không hỗ trợ"}]


def test_file_da_co_trong_kho_bi_bo_qua_kem_ly_do(client) -> None:
    storage = FakeStorage()
    # Dòng media đã tồn tại cho file đầu, chưa có cho file thứ hai.
    dat_fake(storage, FakeUow(FakeSession(ton_tai=[object(), None])))

    body = nap(
        client,
        [
            ("files", ("cu.jpg", JPEG, "image/jpeg")),
            ("files", ("moi.jpg", JPEG, "image/jpeg")),
        ],
    ).json()

    assert storage.uploaded == ["events/1028940839054401/media/moi.jpg"]
    assert len(body["created_ids"]) == 1
    assert "đã có trong kho" in body["skipped"][0]["reason"]


def test_trung_ten_trong_cung_luot_nap_bi_bo_qua(client) -> None:
    """Hai file cùng tên sẽ ra cùng một object key — cái sau ghi đè cái trước."""
    storage = FakeStorage()
    dat_fake(storage, FakeUow(FakeSession()))

    body = nap(
        client,
        [
            ("files", ("anh.jpg", JPEG, "image/jpeg")),
            ("files", ("anh.jpg", b"anh-khac", "image/jpeg")),
        ],
    ).json()

    assert len(storage.uploaded) == 1
    assert body["skipped"][0]["reason"] == "trùng tên trong cùng lượt nạp"


def test_loi_minio_thanh_dong_skipped_chu_khong_phai_500(client) -> None:
    storage = FakeStorage(loi_khi={"events/1028940839054401/media/anh.jpg"})
    dat_fake(storage, FakeUow(FakeSession()))

    phan_hoi = nap(client, [("files", ("anh.jpg", JPEG, "image/jpeg"))])

    assert phan_hoi.status_code == 201
    assert "lỗi MinIO" in phan_hoi.json()["skipped"][0]["reason"]


def test_file_qua_lon_bi_chan(client, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ingest, "MAX_MEDIA_BYTES", 16)
    storage = FakeStorage()
    dat_fake(storage, FakeUow(FakeSession()))

    body = nap(client, [("files", ("to.mp4", b"0" * 100, "video/mp4"))]).json()

    assert body["created_ids"] == []
    assert "vượt" in body["skipped"][0]["reason"]
    assert storage.uploaded == []


def test_ngay_dang_khong_doc_duoc_tra_422(client) -> None:
    dat_fake(FakeStorage(), FakeUow(FakeSession()))

    phan_hoi = nap(client, [("files", ("a.jpg", JPEG, "image/jpeg"))], posted_at="hôm qua")

    assert phan_hoi.status_code == 422


def test_ten_su_kien_duoc_giu_lai_trong_noi_dung_bai(client) -> None:
    """Không có cột sự kiện trong DB — nhưng cũng không được vứt đi lặng lẽ."""
    uow = FakeUow(FakeSession())
    dat_fake(FakeStorage(), uow)

    nap(
        client,
        [("files", ("a.jpg", JPEG, "image/jpeg"))],
        event_name="Kết nạp 2024",
        caption_original="Ảnh tập thể",
    )

    noi_dung = uow.posts.upserted[0].content
    assert "Kết nạp 2024" in noi_dung and "Ảnh tập thể" in noi_dung


def test_ten_file_khong_thoat_ra_khoi_thu_muc_bai(client) -> None:
    """`../../` trong tên file đi thẳng vào object key nếu không cắt."""
    storage = FakeStorage()
    dat_fake(storage, FakeUow(FakeSession()))

    nap(client, [("files", ("../../evil.jpg", JPEG, "image/jpeg"))])

    assert storage.uploaded == ["events/1028940839054401/media/evil.jpg"]


def test_can_quyen_admin() -> None:
    app.dependency_overrides.clear()
    with TestClient(app) as khach:
        phan_hoi = khach.post(
            "/api/ingest/upload",
            files=[("files", ("a.jpg", JPEG, "image/jpeg"))],
            data={"source_url": "https://x", "posted_at": "2026-08-05T14:30"},
        )
    assert phan_hoi.status_code in (401, 403)


# ── Suy facebook_post_id ──────────────────────────────────────────────────────


def test_link_facebook_that_dung_lai_id_bai_de_gop_voi_du_lieu_da_crawl() -> None:
    assert (
        facebook_post_id_tu_link("https://facebook.com/hit/posts/1028940839054401?x=1")
        == "1028940839054401"
    )


def test_link_khong_phai_facebook_dung_chinh_link_lam_khoa() -> None:
    assert facebook_post_id_tu_link("https://hit.edu.vn/su-kien") == "https://hit.edu.vn/su-kien"


def test_link_qua_dai_duoc_bam_cho_vua_cot_255_ky_tu() -> None:
    khoa = facebook_post_id_tu_link("https://x.vn/" + "a" * 400)

    assert khoa.startswith("url-") and len(khoa) <= 255
