#!/usr/bin/env python3
"""Bổ sung `post_created_at` và `event_key` vào point đã index — KHÔNG nhúng lại.

Không dùng `index_media_text_units.py` cho việc này: nó coi point là "không đổi"
khi `content_hash`, model, format và số chiều đều khớp, rồi bỏ qua — mà metadata
cấp bài không nằm trong `content_hash`, nên sẽ không bao giờ được ghi.

Script này chỉ gọi `set_payload`: vector giữ nguyên, không một lời gọi nào tới
nhà cung cấp nhúng, không tốn tiền. Một request Qdrant cho mỗi bài, không phải
cho mỗi point.

`event_key` lấy từ `scripts/extract_post_events.py` — bài nào chưa được gán sự
kiện thì khoá đó bị XOÁ khỏi payload chứ không ghi rỗng (xem `tach_payload`).
Chạy lại script này sau mỗi lần trích xuất sự kiện là đủ để payload theo kịp.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "API"))

from src.rag_video_anh.repository import RepositoryUnitOfWork  # noqa: E402
from src.rag_video_anh.retrieval.retrieval_units import _post_index_metadata  # noqa: E402
from src.rag_video_anh.retrieval.text_indexing_service import (  # noqa: E402
    DEFAULT_MEDIA_TEXT_COLLECTION,
)
from src.rag_video_anh.vector_store.vector_store import QdrantVideoVectorStore  # noqa: E402

CHI_MUC = {"event_key": "keyword", "post_created_at": "datetime"}
TRUONG = ("post_created_at", "event_key")


def tach_payload(meta: Any) -> tuple[dict[str, str], list[str]]:
    """Tách metadata thành (ghi vào, xoá đi).

    Trường `None` KHÔNG được bỏ qua im lặng: bài từng có `event_key` rồi bị gỡ
    liên kết sẽ giữ nguyên giá trị cũ trong Qdrant, và filter vẫn tìm ra nó theo
    một sự kiện nó không còn thuộc về. Phải xoá hẳn khoá đó thì chạy lại mới
    thật sự idempotent.
    """
    ghi, xoa = {}, []
    for ten in TRUONG:
        gia_tri = getattr(meta, ten, None)
        if gia_tri is None:
            xoa.append(ten)      # không ghi "unknown", không ghi null — xoá hẳn
        else:
            ghi[ten] = gia_tri
    return ghi, xoa


def liet_ke_post_id(store: QdrantVideoVectorStore, collection: str, limit: int = 1024) -> dict[str, int]:
    """Quét collection lấy `post_id`, không đọc vector.

    Chưa index lần nào là trạng thái hợp lệ (giống `search_points`), không phải
    lỗi — trả rỗng để người chạy thấy 0 bài chứ không thấy một traceback Qdrant.
    """
    if collection not in {c.name for c in store.client.get_collections().collections}:
        return {}
    dem: dict[str, int] = {}
    offset = None
    while True:
        diem, offset = store.client.scroll(
            collection_name=collection, limit=limit, offset=offset,
            with_payload=["post_id"], with_vectors=False,
        )
        for p in diem:
            pid = (getattr(p, "payload", None) or {}).get("post_id")
            if pid:
                dem[str(pid)] = dem.get(str(pid), 0) + 1
        if offset is None:
            return dem


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--collection", default=DEFAULT_MEDIA_TEXT_COLLECTION)
    p.add_argument("--limit", type=int, default=None, help="Chỉ xử lý N bài đầu tiên.")
    p.add_argument("--apply", action="store_true", help="Thực sự ghi payload và tạo index.")
    args = p.parse_args()

    store = QdrantVideoVectorStore()
    theo_post = liet_ke_post_id(store, args.collection)
    post_ids = sorted(theo_post)[: args.limit] if args.limit else sorted(theo_post)

    ke_hoach = {
        "collection": args.collection,
        "so_bai": len(post_ids),
        "so_point": sum(theo_post[pid] for pid in post_ids),
        "applied": args.apply,
    }
    print(json.dumps(ke_hoach, ensure_ascii=False, indent=2))
    if not post_ids:
        print("Collection chưa có point nào.")
        return 0

    co_thoi_gian = co_su_kien = da_ghi = da_xoa = 0
    with RepositoryUnitOfWork() as uow:
        for i, post_id in enumerate(post_ids, start=1):
            meta = _post_index_metadata(uow, post_id)
            payload, can_xoa = tach_payload(meta)
            co_thoi_gian += meta.post_created_at is not None
            co_su_kien += meta.event_key is not None
            if args.apply:
                # Một request cho cả bài, không phải một request cho mỗi point.
                if payload:
                    store.set_payload_by_filter(
                        collection_name=args.collection, field_name="post_id",
                        value=post_id, payload=payload,
                    )
                    da_ghi += 1
                if can_xoa:
                    store.delete_payload_by_filter(
                        collection_name=args.collection, field_name="post_id",
                        value=post_id, keys=can_xoa,
                    )
                    da_xoa += 1
            if i % 100 == 0 or i == len(post_ids):
                print(f"  ... {i}/{len(post_ids)} bài", flush=True)

    if args.apply:
        for ten, kieu in CHI_MUC.items():
            store.ensure_payload_index(collection_name=args.collection, field_name=ten, schema=kieu)

    print(json.dumps({
        "bai_co_post_created_at": co_thoi_gian,
        "bai_co_event_key": co_su_kien,
        "bai_da_ghi_payload": da_ghi,
        "bai_da_xoa_truong_rong": da_xoa,
        "chi_muc_payload": sorted(CHI_MUC) if args.apply else [],
    }, ensure_ascii=False, indent=2))
    if not args.apply:
        print("\nDry-run: chưa ghi payload và chưa tạo index. Thêm --apply để thực hiện.")
    if co_su_kien == 0:
        print(
            "\nLƯU Ý: không bài nào có event_key — filter theo sự kiện sẽ trả rỗng. "
            "Chạy `scripts/extract_post_events.py --apply` trước."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
