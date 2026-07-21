"""MinIO storage helper for crawled media datasets."""

from __future__ import annotations

import mimetypes
from pathlib import Path

from minio import Minio
from minio.error import S3Error

from src.configuration import AppConfig, MinioConfig
from src.log.logger import logger


class MinioStorage:
    """Upload and download media assets using stable object keys."""

    def __init__(
        self,
        config: AppConfig | None = None,
        minio_config: MinioConfig | None = None,
        client: Minio | None = None,
    ) -> None:
        self.config = config or AppConfig()
        self.minio_config = minio_config or self.config.minio or MinioConfig()
        self.bucket_name = self.minio_config.bucket_name
        self.client = client or Minio(
            endpoint=self.minio_config.endpoint,
            access_key=self.minio_config.access_key,
            secret_key=self.minio_config.secret_key,
            secure=self.minio_config.secure,
        )
        self.ensure_bucket()

    def ensure_bucket(self) -> None:
        """Create the configured bucket if it does not exist."""
        if not self.client.bucket_exists(self.bucket_name):
            self.client.make_bucket(self.bucket_name)
            logger.info(f"Created MinIO bucket '{self.bucket_name}'")

    def upload_file(self, local_path: str | Path, object_key: str, content_type: str | None = None) -> str:
        """Upload one local file and return its object key."""
        source = Path(local_path)
        if not source.is_file():
            raise FileNotFoundError(f"Local file does not exist: {source}")

        normalized_key = self.normalize_object_key(object_key)
        detected_type = content_type or mimetypes.guess_type(source.name)[0] or "application/octet-stream"
        self.client.fput_object(
            bucket_name=self.bucket_name,
            object_name=normalized_key,
            file_path=str(source),
            content_type=detected_type,
        )
        logger.info(f"Uploaded '{source}' to MinIO object '{normalized_key}'")
        return normalized_key

    def download_file(self, object_key: str, local_path: str | Path) -> Path:
        """Download one object to a local file path."""
        target = Path(local_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        normalized_key = self.normalize_object_key(object_key)
        self.client.fget_object(self.bucket_name, normalized_key, str(target))
        logger.info(f"Downloaded MinIO object '{normalized_key}' to '{target}'")
        return target

    def delete_file(self, object_key: str) -> None:
        """Delete one object if the caller no longer needs it."""
        normalized_key = self.normalize_object_key(object_key)
        self.client.remove_object(self.bucket_name, normalized_key)
        logger.info(f"Deleted MinIO object '{normalized_key}'")

    def exists(self, object_key: str) -> bool:
        """Return whether an object key exists in the bucket."""
        normalized_key = self.normalize_object_key(object_key)
        try:
            self.client.stat_object(self.bucket_name, normalized_key)
            return True
        except S3Error as exc:
            if exc.code in {"NoSuchKey", "NoSuchObject", "NotFound"}:
                return False
            raise

    def upload_directory(self, local_dir: str | Path, prefix: str) -> list[str]:
        """Upload every file under a directory while preserving relative paths."""
        root = Path(local_dir)
        if not root.is_dir():
            raise NotADirectoryError(f"Local directory does not exist: {root}")

        uploaded: list[str] = []
        for file_path in sorted(path for path in root.rglob("*") if path.is_file()):
            relative_key = file_path.relative_to(root).as_posix()
            object_key = self.normalize_object_key(f"{prefix}/{relative_key}")
            uploaded.append(self.upload_file(file_path, object_key))
        return uploaded

    def upload_event_folder(self, event_dir: str | Path, base_prefix: str = "events") -> list[str]:
        """Upload one crawled event folder as events/<event_id>/..."""
        root = Path(event_dir)
        event_id = root.name
        return self.upload_directory(root, f"{base_prefix}/{event_id}")

    def upload_crawled_dataset(self, data_root: str | Path, base_prefix: str = "events") -> list[str]:
        """Upload a crawled dataset containing one or many event folders.

        If data_root itself contains post.json, it is treated as one event folder.
        Otherwise, each immediate child directory is treated as one event.
        """
        root = Path(data_root)
        if not root.is_dir():
            raise NotADirectoryError(f"Crawled data root does not exist: {root}")

        if (root / "post.json").is_file():
            return self.upload_event_folder(root, base_prefix=base_prefix)

        uploaded: list[str] = []
        event_dirs = sorted(path for path in root.iterdir() if path.is_dir())
        for event_dir in event_dirs:
            if not (event_dir / "post.json").is_file():
                logger.warning(f"Skipping '{event_dir}' because post.json was not found")
                continue
            uploaded.extend(self.upload_event_folder(event_dir, base_prefix=base_prefix))
        return uploaded

    @staticmethod
    def normalize_object_key(object_key: str) -> str:
        """Normalize object keys to S3-style forward slash paths."""
        normalized = object_key.replace("\\", "/").strip("/")
        if not normalized:
            raise ValueError("object_key cannot be empty")
        return normalized