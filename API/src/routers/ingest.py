"""Nạp ảnh/video từ màn admin vào MinIO + PostgreSQL — T-62 (P4-3 ①).

Đi đúng đường mà bộ crawl đã đi: file nằm ở
`events/<facebook_post_id>/media/<tên file>` trong MinIO, metadata nằm ở
`posts` + `media`. Giữ nguyên quy ước đặt tên object là để
`scripts/minio_registration.py` và các script index sau đó vẫn đọc được — đặt
khác đi thì phải sửa cả một dây script đang chạy tốt.

**Không tạo dòng `videos`.** Dòng đó là kết quả của pipeline parse (duration,
fps, keyframe), không phải của việc tải file lên. Tạo sẵn một dòng rỗng sẽ làm
`scripts/index_all_videos.py` tưởng video đã parse xong rồi đi index một video
không có frame lẫn transcript.
"""

from __future__ import annotations

import hashlib
import re
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import sqlalchemy as sa
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from starlette.concurrency import run_in_threadpool

from src.log.logger import logger
from src.rag_video_anh.pipeline.minio_storage import MinioStorage
from src.rag_video_anh.repository import MediaCreate, MediaType, PostCreate, RepositoryUnitOfWork
from src.rag_video_anh.repository.models import MediaModel
from src.routers.auth import yeu_cau_admin

router = APIRouter(prefix="/ingest", tags=["ingest"])

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}
MAX_MEDIA_BYTES = 512 * 1024 * 1024
CHUNK_BYTES = 1024 * 1024

# Bài Facebook thật có id là một dãy số dài. Bắt được thì dùng đúng nó, để media
# nạp tay gộp vào cùng bài với dữ liệu đã crawl thay vì đẻ ra một bài song song.
_FACEBOOK_ID = re.compile(r"\d{10,}")


def facebook_post_id_tu_link(source_url: str) -> str:
    """Suy `facebook_post_id` từ link bài. Cột này UNIQUE nên phải tất định."""
    link = source_url.strip()
    so = _FACEBOOK_ID.findall(link)
    if so:
        return max(so, key=len)
    # Không phải link Facebook: dùng chính link làm khoá. Dài quá 255 ký tự thì
    # băm, vì cột chỉ chứa được ngần ấy.
    return link if len(link) <= 255 else f"url-{hashlib.sha1(link.encode()).hexdigest()}"


def loai_media(ten_file: str) -> str | None:
    duoi = Path(ten_file).suffix.lower()
    if duoi in IMAGE_EXTENSIONS:
        return MediaType.IMAGE.value
    if duoi in VIDEO_EXTENSIONS:
        return MediaType.VIDEO.value
    return None


def ten_file_an_toan(ten: str | None) -> str:
    """Chỉ giữ phần tên, bỏ mọi thành phần đường dẫn.

    Trình duyệt không gửi đường dẫn, nhưng client nào cũng có thể gửi
    `../../etc/passwd` — mà tên này đi thẳng vào object key.
    """
    return Path(ten or "").name


def _doc_ra_file_tam(upload: UploadFile, thu_muc: Path) -> tuple[Path, int]:
    """Ghi file tải lên ra đĩa theo từng khúc, chặn khi vượt hạn mức.

    Video có thể hàng trăm MB — đọc trọn vào RAM là cách chắc chắn nhất để một
    lượt nạp giết cả tiến trình API.
    """
    dich = thu_muc / ten_file_an_toan(upload.filename)
    tong = 0
    with dich.open("wb") as ra:
        while True:
            khuc = upload.file.read(CHUNK_BYTES)
            if not khuc:
                break
            tong += len(khuc)
            if tong > MAX_MEDIA_BYTES:
                raise ValueError(f"file vượt {MAX_MEDIA_BYTES // (1024 * 1024)}MB")
            ra.write(khuc)
    return dich, tong


def lay_storage() -> MinioStorage:
    """Tiêm được để test chạy offline; lỗi cấu hình thành 503 chứ không phải 500."""
    try:
        return MinioStorage()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Kho MinIO chưa được cấu hình đầy đủ: {exc}",
        ) from exc


def lay_uow_factory() -> Any:
    return RepositoryUnitOfWork


def _nap_mot_lot(
    files: list[tuple[str, Path]],
    *,
    storage: MinioStorage,
    uow_factory: Any,
    source_url: str,
    posted_at: datetime,
    caption_original: str | None,
    event_name: str | None,
) -> dict[str, Any]:
    """Đẩy file lên MinIO rồi ghi metadata. Chạy trong threadpool (toàn I/O đồng bộ)."""
    storage.ensure_bucket()
    facebook_post_id = facebook_post_id_tu_link(source_url)

    # `posts` không có cột sự kiện (BR-305 `/events` chưa làm), nên tên sự kiện
    # được gộp vào `content` chứ không bị vứt đi lặng lẽ.
    noi_dung = "\n".join(
        phan for phan in (f"Sự kiện: {event_name}" if event_name else None, caption_original) if phan
    )

    created_ids: list[str] = []
    skipped: list[dict[str, str]] = []

    with uow_factory() as uow:
        if uow.session is None or uow.posts is None or uow.media is None:
            raise RuntimeError("RepositoryUnitOfWork không mở được session/posts/media")

        post = uow.posts.upsert_by_facebook_id(
            PostCreate(
                facebook_post_id=facebook_post_id,
                content=noi_dung or None,
                post_url=source_url,
                created_time=posted_at,
            )
        )

        for ten_file, duong_dan in files:
            object_key = f"events/{facebook_post_id}/media/{ten_file}"
            da_co = uow.session.scalar(
                sa.select(MediaModel)
                .where(
                    MediaModel.bucket_name == storage.bucket_name,
                    MediaModel.object_key == object_key,
                )
                .limit(1)
            )
            if da_co is not None:
                skipped.append({"file": ten_file, "reason": "đã có trong kho (trùng object key)"})
                continue

            try:
                storage.upload_file(duong_dan, object_key)
            except Exception as exc:  # noqa: BLE001 — một file hỏng không được giết cả lô
                logger.warning(f"Không đẩy được '{object_key}' lên MinIO: {exc}")
                skipped.append({"file": ten_file, "reason": f"lỗi MinIO ({exc.__class__.__name__})"})
                continue

            ban_ghi = uow.media.create_media(
                MediaCreate(
                    post_id=post.post_id,
                    media_type=loai_media(ten_file) or MediaType.IMAGE.value,
                    bucket_name=storage.bucket_name,
                    object_key=object_key,
                )
            )
            created_ids.append(str(ban_ghi.media_id))

    return {"created_ids": created_ids, "skipped": skipped}


@router.post("/upload", status_code=status.HTTP_201_CREATED)
async def nap_media(
    # `_admin` PHẢI đứng trước mọi phụ thuộc khác: FastAPI giải phụ thuộc theo
    # thứ tự khai báo, nên đặt sau `lay_storage` thì một request chưa đăng nhập
    # đã kịp làm server đi mở kết nối MinIO trước khi bị từ chối.
    _admin: Any = Depends(yeu_cau_admin),
    files: list[UploadFile] = File(..., description="Ảnh/video, chọn nhiều được"),
    source_url: str = Form(...),
    posted_at: str = Form(...),
    caption_original: str | None = Form(default=None),
    event_name: str | None = Form(default=None),
    storage: MinioStorage = Depends(lay_storage),
    uow_factory: Any = Depends(lay_uow_factory),
) -> dict[str, Any]:
    """US-101.x: nạp ảnh/video kèm metadata, trả về cái đã tạo VÀ cái đã bỏ qua.

    Bỏ qua từng file chứ không hỏng cả lô: nạp 30 ảnh mà một cái sai định dạng
    thì 29 cái còn lại vẫn phải vào kho, và người nạp phải biết cái nào trượt
    vì lý do gì (P4-3 ①).
    """
    if not str(source_url).strip():
        raise HTTPException(status_code=422, detail="Thiếu link bài gốc.")
    try:
        thoi_diem = datetime.fromisoformat(str(posted_at).strip())
    except ValueError as exc:
        raise HTTPException(
            status_code=422, detail=f"Ngày đăng không đọc được: '{posted_at}'."
        ) from exc

    with tempfile.TemporaryDirectory(prefix="hit-mira-ingest-") as thu_muc_tam:
        thu_muc = Path(thu_muc_tam)
        hop_le: list[tuple[str, Path]] = []
        skipped: list[dict[str, str]] = []
        da_thay: set[str] = set()

        for upload in files:
            ten_file = ten_file_an_toan(upload.filename)
            if not ten_file:
                skipped.append({"file": str(upload.filename), "reason": "tên file không hợp lệ"})
                continue
            if loai_media(ten_file) is None:
                skipped.append({"file": ten_file, "reason": "định dạng không hỗ trợ"})
                continue
            if ten_file in da_thay:
                skipped.append({"file": ten_file, "reason": "trùng tên trong cùng lượt nạp"})
                continue
            try:
                duong_dan, _ = await run_in_threadpool(_doc_ra_file_tam, upload, thu_muc)
            except ValueError as exc:
                skipped.append({"file": ten_file, "reason": str(exc)})
                continue
            da_thay.add(ten_file)
            hop_le.append((ten_file, duong_dan))

        if not hop_le:
            return {"created_ids": [], "skipped": skipped}

        ket_qua = await run_in_threadpool(
            _nap_mot_lot,
            hop_le,
            storage=storage,
            uow_factory=uow_factory,
            source_url=source_url.strip(),
            posted_at=thoi_diem,
            caption_original=(caption_original or "").strip() or None,
            event_name=(event_name or "").strip() or None,
        )

    ket_qua["skipped"] = skipped + ket_qua["skipped"]
    logger.info(
        f"Nạp media: tạo {len(ket_qua['created_ids'])}, bỏ qua {len(ket_qua['skipped'])}"
    )
    return ket_qua
