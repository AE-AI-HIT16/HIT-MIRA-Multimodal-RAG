"""Object storage cho media gốc (hạ tầng — CHO SẴN). [P0-1 · BR-105]

Học phần nằm ở *dùng* store này trong ingest/media (lưu gì, đọc theo id ra sao),
không phải ở cú pháp minio/boto3. Hai backend sau cùng một interface:

  - FilesystemStorage : mặc định dev/test (zero-config, giống sqlite in-memory) — key = đường dẫn tương đối dưới `root`.
  - MinioStorage      : docker/prod (STORAGE_BACKEND=minio) — object trong 1 bucket.

Import nặng (minio) lazy trong __init__ để `import shared.providers.storage` không kéo SDK.
Factory theo STORAGE_BACKEND đặt ở `app/deps.py::get_storage` (không đọc config ở đây).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import BinaryIO


class StorageProvider(ABC):
    """Hợp đồng lưu/đọc media theo `key` (đường dẫn logic, vd 'media/2026/07/abc.jpg')."""

    @abstractmethod
    def put(self, key: str, data: bytes, *, content_type: str | None = None) -> str:
        """Ghi bytes vào `key`, trả lại chính `key` (để lưu vào media_assets.storage_key)."""
        raise NotImplementedError

    @abstractmethod
    def open(self, key: str) -> BinaryIO:
        """Mở stream đọc (cho StreamingResponse). Không tồn tại → FileNotFoundError."""
        raise NotImplementedError

    @abstractmethod
    def exists(self, key: str) -> bool:
        """True nếu object tồn tại (media/service.py dùng để phân biệt 404 vs file hỏng)."""
        raise NotImplementedError

    @abstractmethod
    def delete(self, key: str) -> None:
        """Xoá object; không tồn tại → no-op."""
        raise NotImplementedError


class FilesystemStorage(StorageProvider):
    """Lưu dưới thư mục `root` (mặc định ./data). Dùng cho dev/test — không cần service ngoài."""

    def __init__(self, root: str = "./data") -> None:
        from pathlib import Path

        self.root = Path(root)

    def _path(self, key: str):
        from pathlib import Path

        # Chặn path traversal: key phải nằm trong root.
        p = (self.root / key).resolve()
        if not p.is_relative_to(self.root.resolve()):
            raise ValueError(f"Key ngoài phạm vi storage: {key!r}")
        return Path(p)

    def put(self, key: str, data: bytes, *, content_type: str | None = None) -> str:
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        return key

    def open(self, key: str) -> BinaryIO:
        return self._path(key).open("rb")

    def exists(self, key: str) -> bool:
        return self._path(key).is_file()

    def delete(self, key: str) -> None:
        self._path(key).unlink(missing_ok=True)


class MinioStorage(StorageProvider):
    """MinIO/S3-compatible qua 1 bucket. Bật khi STORAGE_BACKEND=minio (docker/prod). [P0-1]"""

    def __init__(self, endpoint: str, access_key: str, secret_key: str,
                 bucket: str, *, secure: bool = False, client: object | None = None) -> None:
        self.bucket = bucket
        self._bucket_ready = False
        if client is not None:  # inject fake khi test
            self._client = client
            return
        from minio import Minio

        self._client = Minio(endpoint, access_key=access_key,
                             secret_key=secret_key, secure=secure)
        # KHÔNG gọi ensure_bucket() ở đây — lazy-init khi request đầu chạm
        # để api không crash nếu MinIO chưa sẵn sàng lúc boot. [P0-1 edge]

    def ensure_bucket(self) -> None:
        """Tạo bucket nếu chưa có. Gọi lazy lần đầu dùng, hoặc gọi tay khi re-init."""
        if not self._client.bucket_exists(self.bucket):
            self._client.make_bucket(self.bucket)
        self._bucket_ready = True

    def _lazy_ensure(self) -> None:
        """Đảm bảo bucket tồn tại (gọi 1 lần, sau đó skip)."""
        if not self._bucket_ready:
            self.ensure_bucket()

    def put(self, key: str, data: bytes, *, content_type: str | None = None) -> str:
        import io

        self._lazy_ensure()
        self._client.put_object(
            self.bucket, key, io.BytesIO(data), length=len(data),
            content_type=content_type or "application/octet-stream",
        )
        return key

    def open(self, key: str) -> BinaryIO:
        import io

        from minio.error import S3Error

        self._lazy_ensure()
        try:
            resp = self._client.get_object(self.bucket, key)
        except S3Error as exc:  # object không tồn tại → đồng nhất với filesystem
            raise FileNotFoundError(key) from exc
        try:
            return io.BytesIO(resp.read())
        finally:
            resp.close()
            resp.release_conn()

    def exists(self, key: str) -> bool:
        from minio.error import S3Error

        self._lazy_ensure()
        try:
            self._client.stat_object(self.bucket, key)
            return True
        except S3Error:
            return False

    def delete(self, key: str) -> None:
        self._lazy_ensure()
        self._client.remove_object(self.bucket, key)
