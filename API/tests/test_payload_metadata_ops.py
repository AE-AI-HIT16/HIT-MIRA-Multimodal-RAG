"""Ghi/xoá payload và tạo index — TC-908.

Ba tính chất phải giữ: cập nhật metadata cấp bài **không** chạm vector, tạo
index **không** nuốt lỗi, và chạy lại backfill phải thật sự idempotent — kể cả
khi nguồn chuẩn không còn giá trị nào để ghi.

Qdrant là fake; không kết nối gì cả.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.rag_video_anh.vector_store.vector_store import QdrantVideoVectorStore

REPO_ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "backfill_meta", REPO_ROOT / "scripts/backfill_media_text_post_metadata.py"
)
backfill = importlib.util.module_from_spec(_spec)
sys.modules["backfill_meta"] = backfill
_spec.loader.exec_module(backfill)


class FakeQdrant:
    """Đủ để quan sát: ghi gì, xoá gì, và có đụng vector không."""

    def __init__(self, ton_tai=("c",), schema: dict | None = None) -> None:
        self.ton_tai = list(ton_tai)
        self.schema = dict(schema or {})
        self.set_calls: list[dict] = []
        self.delete_calls: list[dict] = []
        self.index_calls: list[dict] = []
        self.upsert_calls: list[dict] = []
        self.no_khi_tao_index: Exception | None = None

    def get_collections(self):
        return SimpleNamespace(collections=[SimpleNamespace(name=n) for n in self.ton_tai])

    def collection_exists(self, *, collection_name):
        return collection_name in self.ton_tai

    def get_collection(self, collection_name):
        return SimpleNamespace(payload_schema={
            k: SimpleNamespace(data_type=SimpleNamespace(value=v)) for k, v in self.schema.items()
        })

    def set_payload(self, *, collection_name, payload, points):
        self.set_calls.append({"collection_name": collection_name, "payload": payload, "points": points})

    def delete_payload(self, *, collection_name, keys, points):
        self.delete_calls.append({"collection_name": collection_name, "keys": keys, "points": points})

    def create_payload_index(self, *, collection_name, field_name, field_schema):
        if self.no_khi_tao_index is not None:
            raise self.no_khi_tao_index
        self.index_calls.append({"field_name": field_name, "field_schema": field_schema})
        self.schema[field_name] = field_schema

    def upsert(self, **kwargs):        # pragma: no cover - chỉ để bắt lỗi nếu bị gọi
        self.upsert_calls.append(kwargs)


def dung_store(**kwargs) -> tuple[QdrantVideoVectorStore, FakeQdrant]:
    client = FakeQdrant(**kwargs)
    store = QdrantVideoVectorStore.__new__(QdrantVideoVectorStore)
    store.client = client
    store.models = _FakeModels()
    return store, client


class _FakeModels:
    Filter = staticmethod(lambda must: {"must": must})
    FieldCondition = staticmethod(lambda key, match: {"key": key, "match": match})
    MatchValue = staticmethod(lambda value: {"value": value})


# ---------------------------------------------------------------- set_payload_by_filter


def test_ghi_payload_theo_post_khong_dung_toi_vector() -> None:
    store, client = dung_store()
    store.set_payload_by_filter(
        collection_name="c", field_name="post_id", value="post-1",
        payload={"post_created_at": "2024-04-11T12:59:01Z"},
    )

    assert len(client.set_calls) == 1
    assert client.set_calls[0]["payload"] == {"post_created_at": "2024-04-11T12:59:01Z"}
    assert client.upsert_calls == []      # không nhúng lại, không ghi vector


def test_mot_request_cho_ca_bai_chu_khong_moi_point_mot_request() -> None:
    store, client = dung_store()
    store.set_payload_by_filter(
        collection_name="c", field_name="post_id", value="post-1", payload={"event_key": "team-building"}
    )
    assert client.set_calls[0]["points"] == {"must": [{"key": "post_id", "match": {"value": "post-1"}}]}


def test_payload_rong_thi_khong_goi_gi() -> None:
    store, client = dung_store()
    assert store.set_payload_by_filter(
        collection_name="c", field_name="post_id", value="p", payload={}
    )["updated"] == 0
    assert client.set_calls == []


def test_collection_chua_ton_tai_thi_bo_qua() -> None:
    store, client = dung_store(ton_tai=())
    store.set_payload_by_filter(collection_name="c", field_name="post_id", value="p", payload={"a": 1})
    assert client.set_calls == []


# ---------------------------------------------------------------- delete_payload_by_filter


def test_xoa_khoa_payload_khi_nguon_chuan_khong_con_gia_tri() -> None:
    """Bỏ qua im lặng thì Qdrant giữ mãi 'team-building' của lượt trước."""
    store, client = dung_store()
    store.delete_payload_by_filter(
        collection_name="c", field_name="post_id", value="post-1", keys=["event_key"]
    )

    assert client.delete_calls[0]["keys"] == ["event_key"]
    assert client.upsert_calls == []


def test_khong_co_khoa_nao_thi_khong_goi_xoa() -> None:
    store, client = dung_store()
    store.delete_payload_by_filter(collection_name="c", field_name="post_id", value="p", keys=[])
    assert client.delete_calls == []


# ---------------------------------------------------------------- ensure_payload_index


def test_tao_index_khi_chua_co() -> None:
    store, client = dung_store()
    assert store.ensure_payload_index(collection_name="c", field_name="event_key", schema="keyword") is True
    assert client.index_calls == [{"field_name": "event_key", "field_schema": "keyword"}]


def test_index_da_co_dung_kieu_thi_bo_qua() -> None:
    store, client = dung_store(schema={"event_key": "keyword"})
    assert store.ensure_payload_index(collection_name="c", field_name="event_key", schema="keyword") is False
    assert client.index_calls == []


def test_index_da_co_nhung_sai_kieu_thi_no() -> None:
    """Hai kiểu cùng tồn tại trên một trường là trạng thái không ai gỡ được về sau."""
    store, _ = dung_store(schema={"post_created_at": "keyword"})
    with pytest.raises(ValueError, match="đang là kiểu"):
        store.ensure_payload_index(collection_name="c", field_name="post_created_at", schema="datetime")


def test_loi_khi_tao_index_phai_noi_len_chu_khong_bi_nuot() -> None:
    """Nuốt mọi exception sẽ che luôn mất kết nối, thiếu quyền và sai schema."""
    store, client = dung_store()
    client.no_khi_tao_index = RuntimeError("mất kết nối Qdrant")

    with pytest.raises(RuntimeError, match="mất kết nối"):
        store.ensure_payload_index(collection_name="c", field_name="event_key", schema="keyword")


def test_collection_chua_ton_tai_thi_tao_index_la_loi() -> None:
    store, _ = dung_store(ton_tai=())
    with pytest.raises(ValueError, match="chưa tồn tại"):
        store.ensure_payload_index(collection_name="c", field_name="event_key", schema="keyword")


# ---------------------------------------------------------------- backfill chạy lại


def meta(post_created_at=None, event_key=None):
    return SimpleNamespace(post_created_at=post_created_at, event_key=event_key)


def test_tach_payload_ghi_gia_tri_co_va_xoa_gia_tri_none() -> None:
    ghi, xoa = backfill.tach_payload(meta("2024-04-11T12:59:01Z", "team-building"))
    assert ghi == {"post_created_at": "2024-04-11T12:59:01Z", "event_key": "team-building"}
    assert xoa == []


def test_truong_none_bi_xoa_chu_khong_bi_bo_qua() -> None:
    """Đây là điều làm cho lượt chạy lại thật sự idempotent."""
    ghi, xoa = backfill.tach_payload(meta("2024-04-11T12:59:01Z", None))
    assert ghi == {"post_created_at": "2024-04-11T12:59:01Z"}
    assert xoa == ["event_key"]


def test_khong_co_metadata_nao_thi_xoa_ca_hai() -> None:
    ghi, xoa = backfill.tach_payload(meta())
    assert ghi == {}
    assert sorted(xoa) == ["event_key", "post_created_at"]


def test_khong_bao_gio_ghi_unknown_hay_null() -> None:
    ghi, _ = backfill.tach_payload(meta(None, None))
    assert "unknown" not in str(ghi).lower()
    assert None not in ghi.values()
