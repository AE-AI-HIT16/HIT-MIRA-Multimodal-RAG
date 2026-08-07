"""Cụm bảng sự kiện: gán bài vào sự kiện và đưa `event_key` ra tới payload.

Bốn bảng sự kiện đã có trong schema từ lâu nhưng rỗng, và `_post_index_metadata`
để sẵn một nhánh `getattr(uow, "events", None)` trả None. Bộ test này ghim cả hai
đầu: repository đọc/ghi đúng, và cái seam kia thực sự nối được.

Ràng buộc quan trọng nhất được ghim ở đây là **im lặng khi mơ hồ**: một bài gắn
hai sự kiện khác nhau mà không nói cái nào là chính thì `event_key` phải là None.
Chọn bừa một cái sẽ khiến bộ lọc trả về ảnh của sự kiện khác — sai kiểu trông
vẫn hợp lý, đúng loại lỗi tệ nhất của dự án này.
"""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine, event, select
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from src.rag_video_anh.repository import DatasetCreate, RepositoryUnitOfWork
from src.rag_video_anh.repository.events import EventRepository, normalize_alias, slugify
from src.rag_video_anh.repository.models import Base, EventAliasModel, PostModel
from src.rag_video_anh.retrieval.retrieval_units import _post_index_metadata


@pytest.fixture()
def session() -> Session:
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)

    @event.listens_for(engine, "connect")
    def _foreign_keys(dbapi_connection, _connection_record) -> None:
        dbapi_connection.execute("PRAGMA foreign_keys=ON")

    Base.metadata.create_all(engine)
    db_session = Session(engine)
    yield db_session
    db_session.close()
    engine.dispose()


@pytest.fixture()
def repo(session: Session) -> EventRepository:
    return EventRepository(session)


def make_post(session: Session, facebook_id: str = "post-1") -> PostModel:
    with RepositoryUnitOfWork(session=session) as uow:
        dataset = uow.datasets.upsert_by_location(
            DatasetCreate(
                name="raw/google-drive/data",
                bucket_name="hit-mira-media",
                object_prefix="raw/google-drive/data",
            )
        )
    row = PostModel(facebook_post_id=facebook_id, dataset_id=dataset.dataset_id)
    session.add(row)
    session.flush()
    return row


def make_occurrence(repo: EventRepository, slug: str, ten: str, label: str, nam: int | None = None):
    series = repo.upsert_series(slug, ten)
    return series, repo.upsert_occurrence(series.series_id, label, event_year=nam)


# ── Chuẩn hoá tên ────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("goc", "mong_doi"),
    [
        ("HIT Open Day", "hit open day"),
        ("hit  open   day", "hit open day"),
        ("  HIT OPEN DAY  ", "hit open day"),
    ],
)
def test_cac_cach_viet_ve_cung_mot_dang_chuan(goc, mong_doi):
    assert normalize_alias(goc) == mong_doi


def test_chu_d_gach_khong_bi_xoa():
    """NFKD không tách `đ`, nên bỏ dấu kiểu thô sẽ xoá hẳn nó: "đại hội" → "ai hoi".
    Đúng cái bẫy đã làm chết bộ lọc boilerplate của ASR."""
    assert normalize_alias("Đại hội Đoàn") == "dai hoi doan"
    assert slugify("Đại hội Đoàn") == "dai-hoi-doan"


def test_slug_chi_con_ky_tu_an_toan():
    assert slugify("HIT Open Day 2024!") == "hit-open-day-2024"
    assert slugify("--- ***") == ""


# ── event_key: chỉ trả lời khi chắc chắn ─────────────────────────────────────


def test_bai_khong_gan_su_kien_thi_khong_co_khoa(session, repo):
    bai = make_post(session)
    assert repo.primary_series_slug(bai.post_id) is None


def test_lay_slug_cua_su_kien_chinh(session, repo):
    bai = make_post(session)
    _, ky = make_occurrence(repo, "hit-open-day", "HIT Open Day", "2024", 2024)
    repo.link_post(bai.post_id, ky.occurrence_id, is_primary=True, assigned_by="llm", confidence=0.9)

    assert repo.primary_series_slug(bai.post_id) == "hit-open-day"


def test_hai_su_kien_khac_nhau_khong_co_cai_chinh_thi_tra_none(session, repo):
    """Mơ hồ thì im lặng. Đây là ràng buộc quan trọng nhất của cả file."""
    bai = make_post(session)
    _, a = make_occurrence(repo, "hit-open-day", "HIT Open Day", "2024")
    _, b = make_occurrence(repo, "hit-gala", "HIT Gala", "2024")
    repo.link_post(bai.post_id, a.occurrence_id)
    repo.link_post(bai.post_id, b.occurrence_id)

    assert repo.primary_series_slug(bai.post_id) is None


def test_hai_ky_cua_cung_mot_chuoi_thi_khong_mo_ho(session, repo):
    """Bài tổng kết nhắc cả hai kỳ vẫn thuộc đúng một chuỗi — không có gì để đoán."""
    bai = make_post(session)
    series, ky_2023 = make_occurrence(repo, "hit-open-day", "HIT Open Day", "2023")
    ky_2024 = repo.upsert_occurrence(series.series_id, "2024")
    repo.link_post(bai.post_id, ky_2023.occurrence_id)
    repo.link_post(bai.post_id, ky_2024.occurrence_id)

    assert repo.primary_series_slug(bai.post_id) == "hit-open-day"


# ── Chạy lại không được đẻ thêm hàng ─────────────────────────────────────────


def test_upsert_series_khong_tao_ban_sao(repo):
    dau = repo.upsert_series("hit-open-day", "HIT Open Day")
    sau = repo.upsert_series("hit-open-day", "HIT Open Day 2024")

    assert dau.series_id == sau.series_id
    assert sau.canonical_name == "HIT Open Day 2024"
    assert repo.counts()["event_series"] == 1


def test_ky_dinh_danh_bang_label_chu_khong_phai_nam(repo):
    """Một chuỗi có thể tổ chức hai kỳ trong cùng một năm; lấy năm làm khoá là gộp nhầm."""
    series = repo.upsert_series("hit-seminar", "HIT Seminar")
    dot_1 = repo.upsert_occurrence(series.series_id, "dot-1", event_year=2024)
    dot_2 = repo.upsert_occurrence(series.series_id, "dot-2", event_year=2024)

    assert dot_1.occurrence_id != dot_2.occurrence_id
    assert repo.counts()["event_occurrences"] == 2


def test_upsert_ky_cu_thi_cap_nhat_chu_khong_them(repo):
    series = repo.upsert_series("hit-open-day", "HIT Open Day")
    repo.upsert_occurrence(series.series_id, "2024")
    lai = repo.upsert_occurrence(series.series_id, "2024", event_year=2024, display_name="Open Day 2024")

    assert lai.event_year == 2024
    assert lai.display_name == "Open Day 2024"
    assert repo.counts()["event_occurrences"] == 1


# ── Alias: cơ chế chống việc trích xuất đẻ ba series cho một sự kiện ─────────


def test_alias_tra_ve_series_du_duoc_gan_vao_ky(repo):
    series, ky = make_occurrence(repo, "hit-open-day", "HIT Open Day", "2024")
    repo.add_alias("Open Day 2024", occurrence_id=ky.occurrence_id)

    assert repo.series_by_alias("open  day 2024").series_id == series.series_id


def test_alias_da_tro_cho_khac_thi_khong_bi_cuop(repo):
    a = repo.upsert_series("hit-open-day", "HIT Open Day")
    b = repo.upsert_series("hit-gala", "HIT Gala")
    assert repo.add_alias("Ngày hội", series_id=a.series_id) is True

    assert repo.add_alias("ngay hoi", series_id=b.series_id) is False
    assert repo.series_by_alias("Ngày hội").series_id == a.series_id


def test_lich_chuan_duoc_quyen_cuop_alias(repo):
    """Alias do máy đặt sai phải sửa được, nếu không một lần trích sai sẽ đóng
    đinh vĩnh viễn — đúng cái đã xảy ra với "Tuyển CTV"."""
    a = repo.upsert_series("tuyen-thanh-vien-hit", "Tuyển thành viên HIT")
    b = repo.upsert_series("tuyen-ctv", "Tuyển CTV")
    repo.add_alias("Tuyển CTV", series_id=a.series_id)

    assert repo.add_alias("Tuyển CTV", series_id=b.series_id, ghi_de=True) is True
    assert repo.series_by_alias("tuyen ctv").slug == "tuyen-ctv"
    assert repo.counts()["event_aliases"] == 1, "ghi đè chứ không tạo hàng thứ hai"


def test_gan_lai_dung_alias_cu_van_bao_thanh_cong(repo):
    a = repo.upsert_series("hit-open-day", "HIT Open Day")
    repo.add_alias("Ngày hội", series_id=a.series_id)

    assert repo.add_alias("Ngày hội", series_id=a.series_id) is True
    assert repo.counts()["event_aliases"] == 1


@pytest.mark.parametrize("dich", [{}, {"series_id": "a", "occurrence_id": "b"}])
def test_alias_phai_tro_vao_dung_mot_dich(repo, dich):
    with pytest.raises(ValueError):
        repo.add_alias("Ngày hội", **dich)


def test_alias_rong_khong_tao_hang(repo):
    a = repo.upsert_series("hit-open-day", "HIT Open Day")

    assert repo.add_alias("   ", series_id=a.series_id) is False
    assert repo.counts()["event_aliases"] == 0


def test_alias_giu_nguyen_cach_viet_goc(repo, session):
    a = repo.upsert_series("hit-open-day", "HIT Open Day")
    repo.add_alias("Ngày hội HIT", series_id=a.series_id)

    row = session.scalar(select(EventAliasModel))
    assert (row.alias, row.normalized_alias) == ("Ngày hội HIT", "ngay hoi hit")


# ── Liên kết bài–kỳ ──────────────────────────────────────────────────────────


def test_doi_su_kien_chinh_go_co_cu_truoc(session, repo):
    """Có unique index một-phần trên `(post_id) WHERE is_primary`. Ghi cờ mới
    trước khi gỡ cờ cũ là chết ở tầng DB, nên thứ tự trong `link_post` là bắt buộc."""
    bai = make_post(session)
    _, cu = make_occurrence(repo, "hit-gala", "HIT Gala", "2023")
    _, moi = make_occurrence(repo, "hit-open-day", "HIT Open Day", "2024")
    repo.link_post(bai.post_id, cu.occurrence_id, is_primary=True)

    repo.link_post(bai.post_id, moi.occurrence_id, is_primary=True)

    assert repo.primary_series_slug(bai.post_id) == "hit-open-day"
    assert repo.counts()["post_event_occurrences"] == 2


def test_gan_lai_cung_ky_thi_cap_nhat_bang_chung(session, repo):
    bai = make_post(session)
    _, ky = make_occurrence(repo, "hit-open-day", "HIT Open Day", "2024")
    repo.link_post(bai.post_id, ky.occurrence_id, evidence={"trich": "cũ"})
    lai = repo.link_post(bai.post_id, ky.occurrence_id, evidence={"trich": "mới"}, confidence=0.8)

    assert lai.evidence == {"trich": "mới"}
    assert lai.confidence == 0.8
    assert repo.counts()["post_event_occurrences"] == 1


def test_go_lien_ket_de_chay_lai_sach(session, repo):
    """`--redo` mà chỉ ghi đè thì sự kiện gán sai ở lượt trước vẫn nằm nguyên."""
    bai = make_post(session)
    _, a = make_occurrence(repo, "hit-open-day", "HIT Open Day", "2024")
    _, b = make_occurrence(repo, "hit-gala", "HIT Gala", "2024")
    repo.link_post(bai.post_id, a.occurrence_id)
    repo.link_post(bai.post_id, b.occurrence_id)

    assert repo.unlink_post(bai.post_id) == 2
    assert repo.primary_series_slug(bai.post_id) is None


def test_xoa_chuoi_khong_con_bai_nao(session, repo):
    """Chuỗi rác của lượt trích cũ vẫn hiện ra như lựa chọn lọc hợp lệ, và chọn
    vào là rỗng — không có cách nào phân biệt với sự kiện chưa có ảnh.

    Chuỗi bị xoá ở đây CÓ alias và CÓ occurrence, vì phiên bản đầu chỉ xoá được
    chuỗi trơ: `session.delete()` set NULL `event_aliases.series_id` trước khi
    xoá, và cột đó có CHECK "trỏ vào đúng một đích" nên cả mẻ chết bằng
    CheckViolation — trên PostgreSQL thật, không phải trong test.
    """
    bai = make_post(session)
    _, con_dung = make_occurrence(repo, "hit-open-day", "HIT Open Day", "2024")
    cu, _ = make_occurrence(repo, "chuoi-cu", "Chuỗi cũ", "2023")
    repo.add_alias("Cách gọi cũ", series_id=cu.series_id)
    repo.link_post(bai.post_id, con_dung.occurrence_id, is_primary=True)

    assert repo.xoa_series_mo_coi() == ["chuoi-cu"]
    assert repo.counts() == {
        "event_series": 1,
        "event_occurrences": 1,
        "event_aliases": 0,
        "post_event_occurrences": 1,
    }
    assert repo.primary_series_slug(bai.post_id) == "hit-open-day"


def test_khong_xoa_nham_chuoi_dang_dung(session, repo):
    bai = make_post(session)
    _, ky = make_occurrence(repo, "hit-open-day", "HIT Open Day", "2024")
    repo.link_post(bai.post_id, ky.occurrence_id)

    assert repo.xoa_series_mo_coi() == []


def test_liet_ke_bai_da_gan_de_chay_tiep(session, repo):
    da_gan = make_post(session, "post-1")
    chua_gan = make_post(session, "post-2")
    _, ky = make_occurrence(repo, "hit-open-day", "HIT Open Day", "2024")
    repo.link_post(da_gan.post_id, ky.occurrence_id, is_primary=True)

    ids = repo.linked_post_ids()
    assert da_gan.post_id in ids
    assert chua_gan.post_id not in ids


# ── Seam: từ DB ra tới payload Qdrant ────────────────────────────────────────


def test_event_key_di_duoc_toi_metadata_index(session):
    """Chỗ này trước đây luôn trả None vì `uow.events` chưa tồn tại."""
    bai = make_post(session)
    with RepositoryUnitOfWork(session=session) as uow:
        assert uow.events is not None
        series = uow.events.upsert_series("hit-open-day", "HIT Open Day")
        ky = uow.events.upsert_occurrence(series.series_id, "2024", event_year=2024)
        uow.events.link_post(bai.post_id, ky.occurrence_id, is_primary=True, assigned_by="llm")

        meta = _post_index_metadata(uow, bai.post_id)

    assert meta.event_key == "hit-open-day"


def test_bai_chua_gan_su_kien_thi_metadata_khong_bia_khoa(session):
    bai = make_post(session)
    with RepositoryUnitOfWork(session=session) as uow:
        meta = _post_index_metadata(uow, bai.post_id)

    assert meta.event_key is None
