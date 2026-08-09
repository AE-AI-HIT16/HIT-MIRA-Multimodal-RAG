"""Quy ước object key và cách xử lý post.json của hai script đăng ký media.

Bản trước hardcode `events/` và có fallback `Path(key).parent.name`. Bộ dữ
liệu Google Drive nằm ở `raw/google-drive/data/<post_id>/media/<file>`, nên
fallback đó suy ra `event_id="media"` cho **mọi** object, rồi không tìm thấy
`events/media/post.json` nên trả `{}` và tạo một post giữ chỗ. Toàn bộ 1.628
ảnh và 59 video sẽ dồn vào một post giả, không content, không permalink —
tức là mất sạch provenance cho trích dẫn — mà script vẫn in ra "thành công".

Test ở đây khoá lại ba thứ: quy ước key là tham số chứ không phải hằng số,
key sai phải nổ chứ không được đoán, và thiếu post.json thì bỏ qua media chứ
không tạo post rỗng.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
for duong_dan in (str(PROJECT_ROOT), str(SCRIPTS_ROOT)):
    if duong_dan not in sys.path:
        sys.path.insert(0, duong_dan)

# Import đúng như hai script vẫn import (`from minio_registration import ...`),
# để chỉ có một bản ObjectKeyError trong tiến trình test.
import minio_registration as dang_ky  # noqa: E402
import register_minio_images  # noqa: E402
import register_minio_videos  # noqa: E402

from src.rag_video_anh.repository import MediaType, RepositoryUnitOfWork  # noqa: E402
from src.rag_video_anh.repository.models import Base, DatasetModel, MediaModel, PostModel  # noqa: E402

CENSUS_FILE = Path(__file__).parent / "fixtures" / "minio_dataset_census.txt"
GOOGLE_DRIVE_PREFIX = "raw/google-drive/data"
POST_ID = "213347192060163_1001928175089001"


# --------------------------------------------------------------------------
# Giả lập MinIO: chỉ cần list_objects/get_object nên không dựng server thật.
# --------------------------------------------------------------------------
class FakeResponse:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return self._payload

    def close(self) -> None:
        pass

    def release_conn(self) -> None:
        pass


class FakeStorage:
    """Storage đủ dùng cho hai script: một dict key -> bytes."""

    def __init__(self, objects: dict[str, bytes], bucket_name: str = "hit-mira-media") -> None:
        self.objects = dict(objects)
        self.bucket_name = bucket_name
        self.client = self
        self.doc_da_lay: list[str] = []

    def list_objects(self, bucket_name: str, prefix: str = "", recursive: bool = False):
        for key in sorted(self.objects):
            if key.startswith(prefix):
                yield SimpleNamespace(object_name=key)

    def get_object(self, bucket_name: str, object_key: str) -> FakeResponse:
        self.doc_da_lay.append(object_key)
        if object_key not in self.objects:
            raise FileNotFoundError(object_key)
        return FakeResponse(self.objects[object_key])


def post_json(post_id: str = POST_ID) -> bytes:
    return json.dumps(
        {
            "id": post_id,
            "created_time": "2024-04-11T12:59:01+0000",
            "message": "DU LỊCH 2024 – BẢN LÁC - MAI CHÂU CÓ GÌ?",
            "story": None,
            "permalink_url": f"https://www.facebook.com/1679787570636388/posts/{post_id}",
        },
        ensure_ascii=False,
    ).encode("utf-8")


def storage_google_drive(**them: bytes) -> FakeStorage:
    objects = {
        f"{GOOGLE_DRIVE_PREFIX}/{POST_ID}/post.json": post_json(),
        f"{GOOGLE_DRIVE_PREFIX}/{POST_ID}/media/photo_01.jpg": b"anh-1",
        f"{GOOGLE_DRIVE_PREFIX}/{POST_ID}/media/video_01.mp4": b"video-1",
    }
    objects.update(them)
    return FakeStorage(objects)


@pytest.fixture()
def uow_factory():
    """Unit of work trên sqlite trong RAM; models dùng GUID nên chạy được."""
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    yield lambda: RepositoryUnitOfWork(session=session), session
    session.close()


# --------------------------------------------------------------------------
# Suy post_id từ object key
# --------------------------------------------------------------------------
def test_layout_events_van_chay_voi_mac_dinh() -> None:
    """Bộ crawl cũ không được vỡ khi đổi sang prefix tham số."""
    assert dang_ky.event_id_from_object_key(f"events/{POST_ID}/media/photo_01.jpg") == POST_ID


def test_layout_google_drive() -> None:
    key = f"{GOOGLE_DRIVE_PREFIX}/{POST_ID}/media/photo_01.jpg"
    assert dang_ky.event_id_from_object_key(key, GOOGLE_DRIVE_PREFIX) == POST_ID


def test_prefix_co_hay_khong_co_dau_gach_cuoi_deu_nhu_nhau() -> None:
    key = f"{GOOGLE_DRIVE_PREFIX}/{POST_ID}/media/photo_01.jpg"
    assert (
        dang_ky.event_id_from_object_key(key, GOOGLE_DRIVE_PREFIX)
        == dang_ky.event_id_from_object_key(key, f"{GOOGLE_DRIVE_PREFIX}/")
        == dang_ky.event_id_from_object_key(key, f"/{GOOGLE_DRIVE_PREFIX}/")
        == POST_ID
    )


def test_key_khong_nam_duoi_prefix_thi_no() -> None:
    """Đây chính là lỗi cũ: key raw/... với prefix events/ phải nổ, không ra 'media'."""
    key = f"{GOOGLE_DRIVE_PREFIX}/{POST_ID}/media/photo_01.jpg"
    with pytest.raises(dang_ky.ObjectKeyError) as loi:
        dang_ky.event_id_from_object_key(key, "events")
    assert "không nằm dưới prefix" in str(loi.value)


def test_key_thieu_segment_media_thi_no() -> None:
    with pytest.raises(dang_ky.ObjectKeyError):
        dang_ky.event_id_from_object_key(f"events/{POST_ID}/photo_01.jpg", "events")


def test_key_khong_du_segment_thi_no() -> None:
    with pytest.raises(dang_ky.ObjectKeyError):
        dang_ky.event_id_from_object_key("events/photo_01.jpg", "events")
    with pytest.raises(dang_ky.ObjectKeyError):
        dang_ky.event_id_from_object_key("", "events")


def test_khong_con_fallback_doan_bua() -> None:
    """Fallback cũ trả về tên thư mục cha; giờ phải là lỗi."""
    with pytest.raises(dang_ky.ObjectKeyError):
        dang_ky.event_id_from_object_key("linh-tinh/media/photo_01.jpg", GOOGLE_DRIVE_PREFIX)


def test_mot_key_sai_lam_hong_ca_lo() -> None:
    """Prefix sai làm mọi key sai — phải dừng cả lượt chứ không lẳng lặng bỏ qua."""
    keys = [
        f"{GOOGLE_DRIVE_PREFIX}/{POST_ID}/media/photo_01.jpg",
        f"{GOOGLE_DRIVE_PREFIX}/khong-co-media.jpg",
    ]
    with pytest.raises(dang_ky.ObjectKeyError) as loi:
        dang_ky.map_keys_to_event_ids(keys, GOOGLE_DRIVE_PREFIX)
    assert "1/2" in str(loi.value)
    assert "--prefix" in str(loi.value)


def test_prefix_lech_han_cho_listing_rong_van_phai_no() -> None:
    """`--prefix events` trên bộ Google Drive: MinIO trả 0 object, không còn key nào để soi.

    Khe hở thật: lọc prefix xảy ra ở tầng `list_objects`, nên prefix lệch hẳn
    không sinh ra key sai nào — nó sinh ra danh sách rỗng, và "tìm thấy 0 ảnh"
    trông y hệt một lượt chạy sạch.
    """
    with pytest.raises(dang_ky.EmptyPrefixError) as loi:
        dang_ky.map_keys_to_event_ids([], "events")
    assert "--prefix" in str(loi.value)
    assert "--allow-empty" in str(loi.value)


def test_empty_prefix_error_van_la_object_key_error() -> None:
    """Hai script bắt chung `ObjectKeyError`; guard mới phải rơi vào cùng nhánh đó."""
    assert issubclass(dang_ky.EmptyPrefixError, dang_ky.ObjectKeyError)


def test_co_the_chap_nhan_prefix_rong_khi_tu_bat_co() -> None:
    assert dang_ky.map_keys_to_event_ids([], "events", allow_empty=True) == []


def test_dang_ky_anh_voi_prefix_lech_han_thi_dung_va_khong_ghi_gi(uow_factory) -> None:
    factory, session = uow_factory
    with pytest.raises(dang_ky.EmptyPrefixError):
        register_minio_images.register_images(
            prefix="events",
            apply=True,
            storage=storage_google_drive(),
            uow_factory=factory,
        )
    assert session.scalars(select(PostModel)).all() == []
    assert session.scalars(select(MediaModel)).all() == []


def test_dang_ky_video_voi_prefix_lech_han_thi_dung(uow_factory) -> None:
    factory, _ = uow_factory
    with pytest.raises(dang_ky.EmptyPrefixError):
        register_minio_videos.register_videos(
            prefix="events",
            apply=True,
            storage=storage_google_drive(),
            uow_factory=factory,
        )


def test_liet_ke_khong_lan_sang_prefix_anh_em() -> None:
    """`raw/google-drive/data` không được nuốt luôn `raw/google-drive/database`."""
    storage = storage_google_drive(
        **{"raw/google-drive/database/dump.jpg": b"khong-phai-bo-nay"},
    )
    keys = dang_ky.list_objects_by_extension(storage, GOOGLE_DRIVE_PREFIX, dang_ky.IMAGE_EXTENSIONS)
    assert keys == [f"{GOOGLE_DRIVE_PREFIX}/{POST_ID}/media/photo_01.jpg"]


def test_duong_dan_post_json_theo_prefix() -> None:
    assert dang_ky.post_metadata_key(POST_ID, GOOGLE_DRIVE_PREFIX) == f"{GOOGLE_DRIVE_PREFIX}/{POST_ID}/post.json"
    assert dang_ky.post_metadata_key(POST_ID, "events/") == f"events/{POST_ID}/post.json"


# --------------------------------------------------------------------------
# post.json thiếu / hỏng
# --------------------------------------------------------------------------
def test_post_json_hong_tra_ve_none() -> None:
    storage = storage_google_drive(**{f"{GOOGLE_DRIVE_PREFIX}/{POST_ID}/post.json": b"{khong-phai-json"})
    assert dang_ky.load_post_metadata(storage, POST_ID, GOOGLE_DRIVE_PREFIX) is None


def test_post_json_khong_phai_object_tra_ve_none() -> None:
    storage = storage_google_drive(**{f"{GOOGLE_DRIVE_PREFIX}/{POST_ID}/post.json": b"[1, 2, 3]"})
    assert dang_ky.load_post_metadata(storage, POST_ID, GOOGLE_DRIVE_PREFIX) is None


def test_post_json_thieu_tra_ve_none() -> None:
    storage = FakeStorage({f"{GOOGLE_DRIVE_PREFIX}/{POST_ID}/media/photo_01.jpg": b"anh"})
    assert dang_ky.load_post_metadata(storage, POST_ID, GOOGLE_DRIVE_PREFIX) is None


def test_chay_thu_phat_hien_thieu_post_json(uow_factory) -> None:
    """Không được đợi tới --apply mới biết bài không có metadata."""
    factory, _ = uow_factory
    storage = FakeStorage({f"{GOOGLE_DRIVE_PREFIX}/{POST_ID}/media/photo_01.jpg": b"anh"})
    summary = register_minio_images.register_images(
        prefix=GOOGLE_DRIVE_PREFIX,
        apply=False,
        storage=storage,
        uow_factory=factory,
    )
    assert summary["missing_metadata"] == [POST_ID]
    assert summary["skipped_missing_metadata"] == 1
    assert summary["created"] == 0


def test_thieu_metadata_thi_bo_qua_chu_khong_tao_post_gia(uow_factory) -> None:
    factory, session = uow_factory
    storage = FakeStorage({f"{GOOGLE_DRIVE_PREFIX}/{POST_ID}/media/photo_01.jpg": b"anh"})
    summary = register_minio_images.register_images(
        prefix=GOOGLE_DRIVE_PREFIX,
        apply=True,
        storage=storage,
        uow_factory=factory,
    )
    assert summary["created"] == 0
    assert summary["skipped_missing_metadata"] == 1
    assert session.scalars(select(PostModel)).all() == []
    assert session.scalars(select(MediaModel)).all() == []


def test_co_the_bat_buoc_dang_ky_khi_thieu_metadata(uow_factory) -> None:
    """Mất provenance là quyết định của người chạy, phải tự bật cờ mới có."""
    factory, session = uow_factory
    storage = FakeStorage({f"{GOOGLE_DRIVE_PREFIX}/{POST_ID}/media/photo_01.jpg": b"anh"})
    summary = register_minio_images.register_images(
        prefix=GOOGLE_DRIVE_PREFIX,
        apply=True,
        allow_missing_metadata=True,
        storage=storage,
        uow_factory=factory,
    )
    assert summary["created"] == 1
    posts = session.scalars(select(PostModel)).all()
    assert [p.facebook_post_id for p in posts] == [POST_ID]
    assert posts[0].content is None


# --------------------------------------------------------------------------
# Đăng ký thật trên layout raw/
# --------------------------------------------------------------------------
def test_dang_ky_anh_gan_dung_post_that(uow_factory) -> None:
    factory, session = uow_factory
    storage = storage_google_drive()
    summary = register_minio_images.register_images(
        prefix=GOOGLE_DRIVE_PREFIX,
        apply=True,
        storage=storage,
        uow_factory=factory,
    )
    assert summary["created"] == 1
    assert summary["posts"] == 1
    assert summary["missing_metadata"] == []

    posts = session.scalars(select(PostModel)).all()
    assert len(posts) == 1
    assert posts[0].facebook_post_id == POST_ID
    assert posts[0].facebook_post_id != "media"
    assert "MAI CHÂU" in posts[0].content
    assert posts[0].post_url.endswith(POST_ID)
    assert posts[0].created_time.year == 2024
    datasets = session.scalars(select(DatasetModel)).all()
    assert len(datasets) == 1
    assert datasets[0].bucket_name == "hit-mira-media"
    assert datasets[0].object_prefix == GOOGLE_DRIVE_PREFIX
    assert posts[0].dataset_id == datasets[0].dataset_id

    media = session.scalars(select(MediaModel)).all()
    assert len(media) == 1
    assert media[0].media_type == MediaType.IMAGE.value
    assert media[0].object_key == f"{GOOGLE_DRIVE_PREFIX}/{POST_ID}/media/photo_01.jpg"
    assert media[0].post_id == posts[0].post_id


def test_dang_ky_video_gan_dung_post_that(uow_factory) -> None:
    factory, session = uow_factory
    storage = storage_google_drive()
    summary = register_minio_videos.register_videos(
        prefix=GOOGLE_DRIVE_PREFIX,
        apply=True,
        storage=storage,
        uow_factory=factory,
    )
    assert summary["created"] == 1
    media = session.scalars(select(MediaModel)).all()
    assert [m.media_type for m in media] == [MediaType.VIDEO.value]
    assert media[0].object_key.endswith("video_01.mp4")


def test_chay_thu_khong_ghi_gi(uow_factory) -> None:
    factory, session = uow_factory
    summary = register_minio_images.register_images(
        prefix=GOOGLE_DRIVE_PREFIX,
        apply=False,
        storage=storage_google_drive(),
        uow_factory=factory,
    )
    assert summary["created"] == 1
    assert summary["applied"] is False
    assert session.scalars(select(DatasetModel)).all() == []
    assert session.scalars(select(PostModel)).all() == []
    assert session.scalars(select(MediaModel)).all() == []


def test_prefix_sai_thi_khong_ghi_dong_nao(uow_factory) -> None:
    """Chạy nhầm `--prefix events` trên bộ Google Drive phải dừng, không ghi gì."""
    factory, session = uow_factory
    with pytest.raises(dang_ky.ObjectKeyError):
        register_minio_images.register_images(
            prefix="raw",
            apply=True,
            storage=storage_google_drive(),
            uow_factory=factory,
        )
    assert session.scalars(select(PostModel)).all() == []
    assert session.scalars(select(MediaModel)).all() == []


def test_chay_lai_bo_qua_media_da_co(uow_factory) -> None:
    factory, session = uow_factory
    storage = storage_google_drive()
    dau = register_minio_images.register_images(
        prefix=GOOGLE_DRIVE_PREFIX, apply=True, storage=storage, uow_factory=factory
    )
    lai = register_minio_images.register_images(
        prefix=GOOGLE_DRIVE_PREFIX, apply=True, storage=storage, uow_factory=factory
    )
    assert dau["created"] == 1
    assert lai["created"] == 0
    assert lai["skipped"] == 1
    assert len(session.scalars(select(MediaModel)).all()) == 1


def test_chay_lai_van_kiem_post_json_cua_anh_da_dang_ky(uow_factory) -> None:
    """Media đã có trong DB không được phép bỏ qua khâu kiểm metadata.

    Khe hở thật: `continue` khi `exists is not None` đặt trước
    `has_metadata()` làm lượt chạy lại im lặng với đúng những bài đã đăng ký —
    tức là mất khả năng phát hiện post.json biến mất sau đó.
    """
    factory, _ = uow_factory
    storage = storage_google_drive()
    register_minio_images.register_images(
        prefix=GOOGLE_DRIVE_PREFIX, apply=True, storage=storage, uow_factory=factory
    )

    del storage.objects[f"{GOOGLE_DRIVE_PREFIX}/{POST_ID}/post.json"]
    lai = register_minio_images.register_images(
        prefix=GOOGLE_DRIVE_PREFIX, apply=True, storage=storage, uow_factory=factory
    )

    assert lai["missing_metadata"] == [POST_ID]
    # Ảnh đã nằm trong DB nên đây là "bỏ qua vì đã có", không phải bị chặn.
    assert lai["skipped"] == 1
    assert lai["skipped_missing_metadata"] == 0
    assert lai["created"] == 0


def test_chay_lai_van_kiem_post_json_cua_video_da_dang_ky(uow_factory) -> None:
    factory, _ = uow_factory
    storage = storage_google_drive()
    register_minio_videos.register_videos(
        prefix=GOOGLE_DRIVE_PREFIX, apply=True, storage=storage, uow_factory=factory
    )

    del storage.objects[f"{GOOGLE_DRIVE_PREFIX}/{POST_ID}/post.json"]
    lai = register_minio_videos.register_videos(
        prefix=GOOGLE_DRIVE_PREFIX, apply=True, storage=storage, uow_factory=factory
    )

    assert lai["missing_metadata"] == [POST_ID]
    assert lai["skipped"] == 1
    assert lai["skipped_missing_metadata"] == 0


def test_anh_chua_dang_ky_thieu_metadata_van_tinh_la_bi_chan(uow_factory) -> None:
    """Phân biệt hai kiểu 'bỏ qua': đã có trong DB, và không đăng ký được."""
    factory, _ = uow_factory
    storage = FakeStorage({f"{GOOGLE_DRIVE_PREFIX}/{POST_ID}/media/photo_01.jpg": b"anh"})
    summary = register_minio_images.register_images(
        prefix=GOOGLE_DRIVE_PREFIX, apply=True, storage=storage, uow_factory=factory
    )
    assert summary["skipped_missing_metadata"] == 1
    assert summary["skipped"] == 0


# --------------------------------------------------------------------------
# Mốc hồi quy trên corpus thật
# --------------------------------------------------------------------------
def doc_census() -> list[tuple[str, int, int]]:
    rows = []
    for dong in CENSUS_FILE.read_text(encoding="utf-8").splitlines():
        if not dong.strip() or dong.startswith("#"):
            continue
        post_id, so_anh, so_video = dong.split()
        rows.append((post_id, int(so_anh), int(so_video)))
    return rows


def test_corpus_that_ra_452_bai_co_anh_va_45_bai_co_video() -> None:
    """Trên bộ dữ liệu đang nằm trên MinIO, kết quả phải là hàng trăm bài — không phải một bài 'media'.

    Census trích ngày 2026-08-05 từ bucket hit-mira-media, prefix
    raw/google-drive/data (full scan). Xem docs/data-audit.md.
    """
    census = doc_census()
    keys = []
    for post_id, so_anh, so_video in census:
        keys += [f"{GOOGLE_DRIVE_PREFIX}/{post_id}/media/photo_{i:02d}.jpg" for i in range(1, so_anh + 1)]
        keys += [f"{GOOGLE_DRIVE_PREFIX}/{post_id}/media/video_{i:02d}.mp4" for i in range(1, so_video + 1)]

    pairs = dang_ky.map_keys_to_event_ids(keys, GOOGLE_DRIVE_PREFIX)
    bai_co_anh = {pid for key, pid in pairs if key.endswith(".jpg")}
    bai_co_video = {pid for key, pid in pairs if key.endswith(".mp4")}

    assert len(keys) == 1628 + 59
    assert len(bai_co_anh) == 452
    assert len(bai_co_video) == 45
    assert "media" not in bai_co_anh | bai_co_video
    assert len({pid for _, pid in pairs}) == 496
