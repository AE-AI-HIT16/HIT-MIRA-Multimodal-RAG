"""Phần dùng chung của hai script đăng ký media từ MinIO vào PostgreSQL.

Ảnh và video nằm cùng một bộ crawl nên cách đọc `post.json`, cách suy ra
post_id từ object key và cách dựng bản ghi post phải giống hệt nhau — tách ra
đây để hai script không trôi khác nhau.

QUY ƯỚC KEY (một quy ước duy nhất cho mọi bộ dữ liệu):

    <dataset_prefix>/<facebook_post_id>/post.json
    <dataset_prefix>/<facebook_post_id>/media/<filename>

`dataset_prefix` là tham số, không phải hằng số. Bộ crawl cũ nằm ở
`events`, bộ Google Drive hiện tại nằm ở `raw/google-drive/data`; cả hai
đọc được bằng cùng một hàm, chỉ khác giá trị truyền vào.

Bản trước hardcode `events/` ở hai chỗ và có một fallback im lặng
(`Path(key).parent.name`). Với layout `raw/google-drive/data/...` thì fallback
đó trả về `"media"` cho **mọi** object, rồi `load_post_metadata` không tìm thấy
`events/media/post.json` và trả về `{}` — kết quả là 1.628 ảnh và 59 video dồn
vào một post giả `facebook_post_id="media"`, không content, không permalink,
mà script vẫn báo chạy thành công. Vì vậy ở đây:

- không còn fallback: key sai cấu trúc là `ObjectKeyError`, không đoán bừa;
- key được soi hết **trước** khi mở kết nối DB, để lỡ truyền sai `--prefix`
  thì không ghi được dòng nào;
- thiếu `post.json` thì bỏ qua media đó chứ không tạo post rỗng, trừ khi
  người chạy tự bật `--allow-missing-metadata`.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from src.rag_video_anh.pipeline.minio_storage import MinioStorage
from src.rag_video_anh.repository import DatasetCreate, PostCreate

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}

# Bộ crawl đầu tiên nằm ở `events/`. Giữ làm mặc định để lệnh cũ trong docs và
# trong lịch sử shell không đổi nghĩa; bộ Google Drive phải truyền --prefix.
DEFAULT_DATASET_PREFIX = "events"

MEDIA_SEGMENT = "media"
POST_METADATA_FILENAME = "post.json"


class ObjectKeyError(ValueError):
    """Object key không khớp `<prefix>/<post_id>/media/<file>`."""


class EmptyPrefixError(ObjectKeyError):
    """Không có media nào dưới prefix — gần như luôn là sai `--prefix`.

    Kế thừa `ObjectKeyError` để hai script bắt chung một chỗ: cùng là "prefix
    không dùng được", chỉ khác cách lộ ra. Prefix sai mà *trùng một phần* (ví dụ
    `raw`) thì listing có key và mọi key hỏng khi soi; prefix sai mà *lệch hẳn*
    (ví dụ `events` trên bộ Google Drive) thì listing rỗng, không còn key nào để
    soi — không có guard này thì lượt chạy trôi qua êm ru và báo "tìm thấy 0".
    """


def normalize_prefix(prefix: str | None) -> str:
    """Đưa mọi cách viết prefix về một dạng: không dấu `/` đầu, không `/` cuối.

    `events`, `events/`, `/events/` là cùng một prefix; chuẩn hoá ở đây để lời
    gọi không phải nhớ viết kiểu nào.
    """
    return (prefix or "").strip().strip("/")


def event_id_from_object_key(object_key: str, prefix: str = DEFAULT_DATASET_PREFIX) -> str:
    """`<prefix>/<post_id>/media/photo_01.jpg` -> `<post_id>`.

    Sai cấu trúc thì ném `ObjectKeyError` kèm cả key lẫn prefix đang dùng —
    thông báo phải đủ để người chạy biết ngay là mình truyền nhầm `--prefix`.
    """
    base = normalize_prefix(prefix)
    key = (object_key or "").strip().strip("/")
    if not key:
        raise ObjectKeyError("Object key rỗng.")

    if base:
        if not key.startswith(f"{base}/"):
            raise ObjectKeyError(f"Object key '{object_key}' không nằm dưới prefix '{base}/'.")
        remainder = key[len(base) + 1 :]
    else:
        remainder = key

    parts = remainder.split("/")
    if len(parts) < 3 or not parts[0] or parts[1] != MEDIA_SEGMENT:
        raise ObjectKeyError(
            f"Object key '{object_key}' không đúng quy ước "
            f"'{base or '<gốc bucket>'}/<post_id>/{MEDIA_SEGMENT}/<file>'."
        )
    return parts[0]


def map_keys_to_event_ids(
    object_keys: list[str],
    prefix: str = DEFAULT_DATASET_PREFIX,
    *,
    allow_empty: bool = False,
) -> list[tuple[str, str]]:
    """Soi toàn bộ key một lượt; có key sai là hỏng cả lượt chạy, không ghi gì.

    Cố ý fail fast: nguyên nhân thực tế của một key sai gần như luôn là truyền
    nhầm `--prefix`, mà khi đó *mọi* key đều sai. Bỏ qua từng key một sẽ biến
    lỗi cấu hình thành một lượt chạy trống rỗng trông như thành công.

    Danh sách rỗng cũng là lỗi, vì `list_objects` đã lọc theo prefix từ trước:
    prefix lệch hẳn thì không còn key nào để soi, và "0 media" trông y hệt một
    lượt chạy sạch. Ai thật sự cần chạy trên prefix rỗng thì bật `allow_empty`.
    """
    base = normalize_prefix(prefix)
    if not object_keys and not allow_empty:
        raise EmptyPrefixError(
            f"Không có media nào dưới prefix '{base}/' — không đăng ký gì cả.\n"
            "Prefix lệch hẳn thì listing rỗng chứ không báo key sai, nên đây gần như luôn là "
            "sai --prefix (ví dụ: --prefix raw/google-drive/data).\n"
            "Nếu prefix đúng và bộ dữ liệu thật sự chưa có media, thêm --allow-empty."
        )

    pairs: list[tuple[str, str]] = []
    loi: list[str] = []
    for object_key in object_keys:
        try:
            pairs.append((object_key, event_id_from_object_key(object_key, prefix)))
        except ObjectKeyError as exc:
            loi.append(str(exc))
    if loi:
        vi_du = "\n  - ".join(loi[:5])
        raise ObjectKeyError(
            f"{len(loi)}/{len(object_keys)} object key không đúng quy ước, không đăng ký gì cả.\n"
            f"  - {vi_du}"
            + (f"\n  … và {len(loi) - 5} key nữa." if len(loi) > 5 else "")
            + "\nKiểm tra lại --prefix (ví dụ: --prefix raw/google-drive/data)."
        )
    return pairs


def list_objects_by_extension(storage: MinioStorage, prefix: str, extensions: set[str]) -> list[str]:
    """Liệt kê object dưới prefix, lọc theo đuôi file.

    Thêm `/` vào cuối trước khi liệt kê: prefix trần `raw/google-drive/data`
    còn khớp cả `raw/google-drive/database/...`, tức là quét lẫn sang bộ dữ
    liệu khác.
    """
    base = normalize_prefix(prefix)
    search_prefix = f"{base}/" if base else ""
    return sorted(
        obj.object_name
        for obj in storage.client.list_objects(storage.bucket_name, prefix=search_prefix, recursive=True)
        if Path(obj.object_name).suffix.lower() in extensions
    )


def post_metadata_key(event_id: str, prefix: str = DEFAULT_DATASET_PREFIX) -> str:
    base = normalize_prefix(prefix)
    return f"{base}/{event_id}/{POST_METADATA_FILENAME}" if base else f"{event_id}/{POST_METADATA_FILENAME}"


def load_post_metadata(
    storage: MinioStorage,
    event_id: str,
    prefix: str = DEFAULT_DATASET_PREFIX,
) -> dict | None:
    """Đọc `post.json` của bài. Trả `None` khi không đọc được hoặc file hỏng.

    Phân biệt `None` (không có metadata) với `{}` (có file nhưng rỗng) là điều
    kiện để tầng trên bỏ qua media thay vì tạo post giữ chỗ.
    """
    object_key = post_metadata_key(event_id, prefix)
    try:
        response = storage.client.get_object(storage.bucket_name, object_key)
        try:
            metadata = json.loads(response.read().decode("utf-8"))
        finally:
            response.close()
            response.release_conn()
    except Exception as exc:
        print(f"  ! không đọc được {object_key}: {exc.__class__.__name__}")
        return None
    if not isinstance(metadata, dict):
        print(f"  ! {object_key} không phải object JSON: {type(metadata).__name__}")
        return None
    return metadata


def parse_created_time(value) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).replace(tzinfo=None)
    except ValueError:
        return None


def build_dataset_create(storage: MinioStorage, prefix: str) -> DatasetCreate:
    """Dựng catalog entry ổn định cho vị trí đang import."""
    base = normalize_prefix(prefix)
    return DatasetCreate(
        name=base or "bucket-root",
        source_type="facebook-crawl",
        bucket_name=storage.bucket_name,
        object_prefix=base,
        metadata={"layout": "<dataset_prefix>/<facebook_post_id>/media/<file>"},
    )


def build_post_create(event_id: str, metadata: dict, *, dataset_id=None) -> PostCreate:
    """Dựng PostCreate từ post.json; không có metadata thì chỉ giữ facebook_post_id."""
    return PostCreate(
        facebook_post_id=event_id,
        content=metadata.get("message") or metadata.get("story"),
        author="facebook-crawler",
        post_url=metadata.get("permalink_url"),
        created_time=parse_created_time(metadata.get("created_time")),
        dataset_id=dataset_id,
    )


class PostMetadataResolver:
    """Upsert post theo event_id, đọc post.json đúng một lần cho mỗi bài.

    `metadata()` không cần tới `uow`, nên chạy thử cũng kiểm tra được post.json
    — chờ tới `--apply` mới phát hiện thiếu metadata thì đã muộn.
    """

    def __init__(
        self,
        storage: MinioStorage,
        uow=None,
        *,
        prefix: str = DEFAULT_DATASET_PREFIX,
        allow_missing_metadata: bool = False,
        dataset_id=None,
    ) -> None:
        self.storage = storage
        self.uow = uow
        self.prefix = normalize_prefix(prefix)
        self.allow_missing_metadata = allow_missing_metadata
        self.dataset_id = dataset_id
        self._metadata: dict[str, dict | None] = {}
        self._posts: dict[str, object] = {}

    def metadata(self, event_id: str) -> dict | None:
        if event_id not in self._metadata:
            self._metadata[event_id] = load_post_metadata(self.storage, event_id, self.prefix)
        return self._metadata[event_id]

    def has_metadata(self, event_id: str) -> bool:
        return self.metadata(event_id) is not None

    def upsert(self, event_id: str):
        if event_id in self._posts:
            return self._posts[event_id]
        if self.uow is None:
            raise RuntimeError("PostMetadataResolver.upsert cần unit of work")
        metadata = self.metadata(event_id)
        if metadata is None and not self.allow_missing_metadata:
            raise RuntimeError(
                f"Không có {post_metadata_key(event_id, self.prefix)}; "
                "không tạo post giữ chỗ (dùng --allow-missing-metadata nếu thực sự muốn)."
            )
        post = self.uow.posts.upsert_by_facebook_id(
            build_post_create(event_id, metadata or {}, dataset_id=self.dataset_id)
        )
        self._posts[event_id] = post
        return post
