"""Tests cho StorageProvider — FilesystemStorage (zero-config, no MinIO). [P0-1]

pytest KHÔNG cần dựng MinIO: chỉ test FilesystemStorage + mock MinioStorage.
"""
from __future__ import annotations

import pytest

from shared.providers.storage import FilesystemStorage, MinioStorage, StorageProvider


# ────── FilesystemStorage (dev/test, zero-config) ──────

class TestFilesystemStorage:
    """TC liên quan T-06 / P0-1: đọc/ghi/xoá media bằng FilesystemStorage."""

    @pytest.fixture
    def store(self, tmp_path):
        return FilesystemStorage(str(tmp_path))

    def test_implements_interface(self, store):
        assert isinstance(store, StorageProvider)

    def test_put_and_open(self, store):
        payload = b"hello mira"
        key = store.put("media/test.bin", payload)
        assert key == "media/test.bin"
        with store.open(key) as f:
            assert f.read() == payload

    def test_put_creates_subdirs(self, store):
        store.put("a/b/c/deep.txt", b"deep")
        assert store.exists("a/b/c/deep.txt")

    def test_exists_true_false(self, store):
        assert not store.exists("nope.bin")
        store.put("yes.bin", b"data")
        assert store.exists("yes.bin")

    def test_open_missing_raises(self, store):
        with pytest.raises(FileNotFoundError):
            store.open("ghost.bin")

    def test_delete(self, store):
        store.put("del.bin", b"gone")
        assert store.exists("del.bin")
        store.delete("del.bin")
        assert not store.exists("del.bin")

    def test_delete_missing_noop(self, store):
        """Xoá key không tồn tại → không lỗi."""
        store.delete("not-here.bin")

    def test_path_traversal_blocked(self, store):
        """Key ../... phải bị chặn."""
        with pytest.raises(ValueError, match="ngoài phạm vi"):
            store.put("../../etc/passwd", b"hack")

    def test_content_type_ignored(self, store):
        """FilesystemStorage bỏ qua content_type (chỉ có ý nghĩa với MinIO)."""
        store.put("img.jpg", b"\xff\xd8", content_type="image/jpeg")
        assert store.exists("img.jpg")


# ────── MinioStorage (mock — không cần dựng MinIO) ──────

class _FakeHTTPResponse:
    """Minimal fake đủ cho S3Error.__init__ (minio 7.x)."""
    status = 404
    data = b""
    headers = {}
    def getheaders(self): return {}


class FakeMinioClient:
    """Giả lập minio.Minio đủ để test logic MinioStorage, không cần service thật."""

    def __init__(self):
        self._buckets: set[str] = set()
        self._objects: dict[str, bytes] = {}

    def bucket_exists(self, bucket: str) -> bool:
        return bucket in self._buckets

    def make_bucket(self, bucket: str) -> None:
        self._buckets.add(bucket)

    def put_object(self, bucket, key, data, length, content_type=""):
        self._objects[f"{bucket}/{key}"] = data.read()

    def _raise_no_key(self, key):
        from minio.error import S3Error
        raise S3Error(_FakeHTTPResponse(), code="NoSuchKey", message="not found",
                      resource=key, request_id="", host_id="")

    def get_object(self, bucket, key):
        import io
        full = f"{bucket}/{key}"
        if full not in self._objects:
            self._raise_no_key(key)
        class FakeResp:
            def __init__(self, raw):
                self._raw = raw
                self._stream = io.BytesIO(raw)
            def read(self):
                return self._raw
            def close(self):
                pass
            def release_conn(self):
                pass
        return FakeResp(self._objects[full])

    def stat_object(self, bucket, key):
        full = f"{bucket}/{key}"
        if full not in self._objects:
            self._raise_no_key(key)
        return True

    def remove_object(self, bucket, key):
        self._objects.pop(f"{bucket}/{key}", None)


class TestMinioStorage:
    """MinioStorage logic test bằng fake client (không cần MinIO service)."""

    @pytest.fixture
    def store(self):
        fake = FakeMinioClient()
        return MinioStorage("fake:9000", "a", "s", "test-bucket", client=fake)

    def test_implements_interface(self, store):
        assert isinstance(store, StorageProvider)

    def test_lazy_bucket_creation(self):
        """Bucket không được tạo trong __init__ (lazy-init). [P0-1 edge]"""
        fake = FakeMinioClient()
        s = MinioStorage("fake:9000", "a", "s", "lazy-bucket", client=fake)
        # __init__ chưa gọi ensure_bucket → bucket chưa tồn tại
        assert not fake.bucket_exists("lazy-bucket")
        # put đầu tiên trigger lazy ensure
        s.put("file.bin", b"data")
        assert fake.bucket_exists("lazy-bucket")

    def test_put_and_open(self, store):
        store.put("pic.jpg", b"\xff\xd8", content_type="image/jpeg")
        with store.open("pic.jpg") as f:
            assert f.read() == b"\xff\xd8"

    def test_exists(self, store):
        assert not store.exists("nope")
        store.put("yes.bin", b"y")
        assert store.exists("yes.bin")

    def test_open_missing(self, store):
        with pytest.raises(FileNotFoundError):
            store.open("ghost")

    def test_delete(self, store):
        store.put("del.bin", b"x")
        store.delete("del.bin")
        assert not store.exists("del.bin")
