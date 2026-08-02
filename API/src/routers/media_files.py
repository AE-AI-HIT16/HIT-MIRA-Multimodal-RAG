"""Phục vụ file media cho trình duyệt — T-06 / US-105.1, và nền cho T-52.

Vì sao cần: kết quả truy xuất chỉ mang `bucket_name` + `frame_object_key`.
Trình duyệt không nói được giao thức MinIO, và MinIO thì không mở ra Internet,
nên không có tầng này thì mọi thẻ kết quả trên web đều là ô ảnh vỡ.

Cách phục vụ: **chuyển hướng 307 sang presigned URL của MinIO**, không proxy
từng byte qua FastAPI. Proxy thì mỗi tấm ảnh chiếm một worker suốt thời gian
tải, mà một lượt tìm kiếm trả về 5–10 ảnh cùng lúc — API sẽ nghẽn vì việc mà
MinIO làm tốt hơn hẳn. Đổi lại, presigned URL có hạn dùng, nên đặt ngắn
(`MEDIA_FILE_URL_TTL`, mặc định 1 giờ) vừa đủ cho một phiên xem.

**Chỉ đọc, và chỉ trong bucket đã cấu hình.** `bucket` là tham số của client
nên không được tin: nhận bừa thì thành đường đọc mọi bucket trong MinIO.
"""

from __future__ import annotations

import os
from functools import lru_cache

from fastapi import APIRouter, HTTPException, Query, status
from fastapi.responses import RedirectResponse

from src.rag_video_anh.pipeline.minio_storage import MinioStorage
from src.rag_video_anh.repository.database import get_session_manager

router = APIRouter(prefix="/media-files", tags=["media-files"])

TTL_MAC_DINH = 3600


@lru_cache(maxsize=1)
def get_storage() -> MinioStorage:
    """Một MinioStorage cho cả tiến trình — mỗi lần dựng là một pool kết nối."""
    return MinioStorage()


def _ttl() -> int:
    try:
        return max(60, int(os.getenv("MEDIA_FILE_URL_TTL") or TTL_MAC_DINH))
    except ValueError:
        return TTL_MAC_DINH


def _url_hoac_404(bucket: str | None, object_key: str) -> RedirectResponse:
    storage = get_storage()
    # Bucket do client gửi lên là dữ liệu không tin được. Chỉ chấp nhận đúng
    # bucket đã cấu hình; khác đi thì từ chối chứ không "chiều" client.
    if bucket and bucket != storage.bucket_name:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Chỉ phục vụ bucket '{storage.bucket_name}'",
        )
    if not object_key or object_key.startswith("/") or ".." in object_key:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="object_key không hợp lệ")
    if not storage.exists(object_key):
        # Có dòng DB mà mất object là chuyện đã xảy ra thật (keyframe mồ côi),
        # nên phải là 404 rõ ràng chứ không phải 500 hay redirect vào hư không.
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Không có object '{object_key}'")
    return RedirectResponse(
        url=storage.presigned_download_url(object_key, expires_seconds=_ttl()),
        status_code=status.HTTP_307_TEMPORARY_REDIRECT,
    )


@router.get("/by-key")
def lay_theo_object_key(
    object_key: str = Query(..., description="Khoá object trong MinIO, lấy từ kết quả truy xuất"),
    bucket: str | None = Query(None, description="Bỏ trống thì dùng bucket đã cấu hình"),
) -> RedirectResponse:
    """Đường dùng chính: kết quả truy xuất trả sẵn `bucket_name` + `frame_object_key`."""
    return _url_hoac_404(bucket, object_key)


@router.get("/video/{video_id}")
def lay_video_goc(video_id: str) -> RedirectResponse:
    """Video gốc theo `video_id` — khoá mà kết quả truy xuất trả về.

    Đủ để phát và **tua** trong trình duyệt: presigned URL của MinIO hỗ trợ HTTP
    Range, nên thẻ `<video src="...#t=125">` nhảy đúng giây 125. Nhờ vậy không
    cần dịch vụ cắt clip bằng ffmpeg mới xem được đúng khoảnh khắc — đổi lại là
    tải video đầy đủ thay vì một đoạn ngắn.
    """
    import sqlalchemy as sa

    with get_session_manager().session() as session:
        dong = session.execute(
            sa.text(
                "select m.bucket_name, m.object_key from videos v "
                "join media m on m.media_id = v.media_id where v.video_id = :vid"
            ),
            {"vid": video_id},
        ).first()
    if dong is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Không có video '{video_id}'")
    return _url_hoac_404(dong[0], dong[1])


@router.get("/{media_id}")
def lay_theo_media_id(media_id: str) -> RedirectResponse:
    """Tra `object_key` từ bảng `media` rồi chuyển hướng như trên.

    `media_id` là UUID, không phải số tự tăng — `web/lib/api.ts` bản đầu khai
    `mediaUrl(id: number)`, gọi kiểu đó sẽ không bao giờ khớp dòng nào.
    """
    import sqlalchemy as sa

    with get_session_manager().session() as session:
        dong = session.execute(
            sa.text("select bucket_name, object_key from media where media_id = :mid"),
            {"mid": media_id},
        ).first()
    if dong is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Không có media '{media_id}'")
    return _url_hoac_404(dong[0], dong[1])
