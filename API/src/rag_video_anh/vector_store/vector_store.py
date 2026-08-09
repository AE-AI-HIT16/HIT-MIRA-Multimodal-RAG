"""Qdrant vector-store adapter for video retrieval collections."""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from src.configuration import AppConfig
from src.log.logger import logger

# Khoá payload mang mốc thời gian của BÀI ĐĂNG (không phải của file media).
POST_CREATED_AT_KEY = "post_created_at"

# Người hỏi "ảnh năm 2024" đang nghĩ theo lịch Việt Nam, còn `post_created_at`
# lưu theo UTC ('2024-07-06T12:59:02Z'). Cắt mốc năm theo UTC sẽ đẩy bài đăng
# từ 07:00 tối 31/12 giờ VN trở đi sang năm sau — hiếm, nhưng đúng vào dịp
# tổng kết cuối năm, và khi xảy ra thì ảnh "biến mất" khỏi năm người ta nhớ.
GIO_VIET_NAM = timezone(timedelta(hours=7))

# Năm ngoài khoảng này chắc chắn là gõ nhầm (2o24, 202, 20244). Chặn sớm để
# lỗi thành 422 có chỗ sửa, thay vì một truy vấn hợp lệ trả về 0 kết quả.
NAM_NHO_NHAT = 1970
NAM_LON_NHAT = 2100


class VideoVectorStoreConfigurationError(ValueError):
    """Raised when Qdrant configuration is incomplete or invalid."""


class VideoVectorStoreError(RuntimeError):
    """Raised when Qdrant rejects an operation or returns invalid metadata."""


class QdrantVideoVectorStore:
    """Qdrant store for media_clip and video_transcript retrieval points."""

    def __init__(
        self,
        url: str | None = None,
        api_key: str | None = None,
        distance: str | None = None,
        client: Any | None = None,
        config: AppConfig | None = None,
    ) -> None:
        app_config = config or AppConfig()
        qdrant_config = app_config.qdrant
        self.url = url if url is not None else qdrant_config.url
        self.api_key = api_key if api_key is not None else qdrant_config.api_key
        self.distance_name = (distance or getattr(qdrant_config, "distance", None) or "COSINE").upper()
        self.client = client or self._create_client()
        self.models = self._load_models()

    def upsert_points(
        self,
        *,
        collection_name: str,
        point_ids: list[str],
        vectors: list[list[float]],
        payloads: list[dict[str, Any]],
    ) -> dict[str, Any]:
        self._validate_upsert_inputs(point_ids, vectors, payloads)
        vector_size = len(vectors[0])
        self.ensure_collection(collection_name=collection_name, vector_size=vector_size)
        points = [
            self.models.PointStruct(
                id=self._point_id(point_id),
                vector=vector,
                payload=payload,
            )
            for point_id, vector, payload in zip(point_ids, vectors, payloads)
        ]
        self.client.upsert(collection_name=collection_name, points=points)
        logger.info(f"Upserted {len(points)} point(s) into Qdrant collection '{collection_name}'")
        return {"collection_name": collection_name, "upserted": len(points)}

    def set_payloads(
        self,
        *,
        collection_name: str,
        point_ids: list[str],
        payloads: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Ghi đè payload mà giữ nguyên vector.

        Caption và OCR có sau lúc index (khâu phân tích chạy riêng), nên phải
        cập nhật được phần chữ mà không nhúng lại — nhúng lại 1.628 ảnh tốn cả
        tiếng đồng hồ hạn mức Jina mà vector thì không hề đổi.
        """
        if len(point_ids) != len(payloads):
            raise ValueError("point_ids and payloads must have the same length")

        for point_id, payload in zip(point_ids, payloads):
            if not isinstance(payload, dict):
                raise TypeError(f"payload for point '{point_id}' must be a dict")
            self.client.set_payload(
                collection_name=collection_name,
                payload=payload,
                points=[self.point_id(point_id)],
            )
        logger.info(f"Updated payload of {len(point_ids)} point(s) in Qdrant collection '{collection_name}'")
        return {"collection_name": collection_name, "updated": len(point_ids)}

    def set_payload_by_filter(
        self, *, collection_name: str, field_name: str, value: Any, payload: dict[str, Any]
    ) -> dict[str, Any]:
        """Bổ sung payload cho MỌI point khớp một điều kiện, giữ nguyên vector.

        Dành cho việc thêm metadata cấp bài vào dữ liệu đã index: đọc từng point
        rồi ghi lại từng cái là hàng nghìn lượt gọi cho một giá trị giống hệt
        nhau, còn chạy lại indexer thì nó thấy `content_hash` không đổi và bỏ
        qua — metadata mới sẽ không bao giờ được ghi.
        """
        if not payload:
            return {"collection_name": collection_name, "updated": 0}
        if not self._collection_exists(collection_name):
            return {"collection_name": collection_name, "updated": 0}
        self.client.set_payload(
            collection_name=collection_name,
            payload=payload,
            points=self.models.Filter(
                must=[self.models.FieldCondition(key=field_name, match=self.models.MatchValue(value=value))]
            ),
        )
        logger.info(
            f"Set payload {sorted(payload)} for points with {field_name}={value} in '{collection_name}'"
        )
        return {"collection_name": collection_name, "updated": 1}

    def delete_payload_by_filter(
        self, *, collection_name: str, field_name: str, value: Any, keys: list[str]
    ) -> dict[str, Any]:
        """Xoá hẳn một số khoá payload khỏi mọi point khớp điều kiện.

        Cần cho việc chạy lại: khi nguồn chuẩn trả `None` (bài bị gỡ liên kết sự
        kiện chẳng hạn), chỉ *bỏ qua* khoá đó là để giá trị cũ nằm lại vĩnh viễn
        trong Qdrant, và filter vẫn tìm ra bài theo một sự kiện nó không còn thuộc về.
        """
        if not keys or not self._collection_exists(collection_name):
            return {"collection_name": collection_name, "deleted_keys": 0}
        self.client.delete_payload(
            collection_name=collection_name,
            keys=list(keys),
            points=self.models.Filter(
                must=[self.models.FieldCondition(key=field_name, match=self.models.MatchValue(value=value))]
            ),
        )
        logger.info(f"Deleted payload keys {sorted(keys)} for {field_name}={value} in '{collection_name}'")
        return {"collection_name": collection_name, "deleted_keys": len(keys)}

    def payload_index_schema(self, collection_name: str) -> dict[str, str]:
        """Kiểu index payload hiện có, khoá theo tên trường."""
        thong_tin = getattr(self.client.get_collection(collection_name), "payload_schema", None) or {}
        ra: dict[str, str] = {}
        for ten, mo_ta in thong_tin.items():
            kieu = getattr(mo_ta, "data_type", None) or mo_ta
            ra[str(ten)] = str(getattr(kieu, "value", kieu)).lower()
        return ra

    def ensure_payload_index(self, *, collection_name: str, field_name: str, schema: str) -> bool:
        """Tạo index payload nếu chưa có; **không nuốt lỗi**.

        Bắt mọi exception rồi coi như "đã tồn tại" sẽ che luôn chuỗi datetime
        sai định dạng, sai schema, mất kết nối, thiếu quyền và collection hỏng —
        rồi job vẫn báo đã tạo index trong khi không có index nào. Ở đây chỉ đúng
        một trường hợp được bỏ qua: **index đã có sẵn với đúng kiểu**.
        """
        if not self._collection_exists(collection_name):
            raise ValueError(
                f"Collection '{collection_name}' chưa tồn tại nên không tạo được index '{field_name}'."
            )
        dang_co = self.payload_index_schema(collection_name).get(field_name)
        if dang_co is not None:
            if dang_co == schema.lower():
                logger.info(f"Payload index '{field_name}' ({schema}) đã có trên '{collection_name}'")
                return False
            raise ValueError(
                f"Payload index '{field_name}' trên '{collection_name}' đang là kiểu "
                f"'{dang_co}' nhưng được yêu cầu '{schema}'. Xoá index cũ trước, "
                "đừng để hai kiểu cùng tồn tại trên một trường."
            )
        self.client.create_payload_index(
            collection_name=collection_name, field_name=field_name, field_schema=schema
        )
        logger.info(f"Created payload index '{field_name}' ({schema}) on '{collection_name}'")
        return True

    def get_payloads(self, *, collection_name: str, point_ids: list[str]) -> dict[str, dict[str, Any]]:
        """Đọc payload của những point đã có, khoá theo `unit_id` nghiệp vụ.

        Dùng để chạy lại mà không nhúng lại: point nào còn nguyên `content_hash`
        thì bỏ qua. Collection chưa tồn tại thì coi như chưa có gì.
        """
        if not point_ids:
            return {}
        if not self._collection_exists(collection_name):
            return {}
        diem = self.client.retrieve(
            collection_name=collection_name,
            ids=[self.point_id(p) for p in point_ids],
            with_payload=True,
            with_vectors=False,
        )
        return {
            str(p.payload.get("unit_id")): dict(p.payload)
            for p in diem
            if getattr(p, "payload", None) and p.payload.get("unit_id")
        }

    def delete_points(self, *, collection_name: str, point_ids: list[str]) -> dict[str, Any]:
        """Xoá point theo khoá nghiệp vụ.

        Cần khi caption/OCR đổi và số mảnh cắt giảm đi: `ocr:image:x#5` của lượt
        trước sẽ nằm lại vĩnh viễn, mang nội dung đã bị thay thế, và vẫn được
        trả về trong kết quả tìm kiếm.
        """
        if not point_ids or not self._collection_exists(collection_name):
            return {"collection_name": collection_name, "deleted": 0}
        self.client.delete(
            collection_name=collection_name,
            points_selector=self.models.PointIdsList(points=[self.point_id(p) for p in point_ids]),
        )
        logger.info(f"Deleted {len(point_ids)} stale point(s) from Qdrant collection '{collection_name}'")
        return {"collection_name": collection_name, "deleted": len(point_ids)}

    def list_unit_ids_by_field(
        self,
        *,
        collection_name: str,
        field_name: str,
        value: Any,
        source_types: list[str] | None = None,
        limit: int = 1024,
    ) -> list[str]:
        """Liệt kê `unit_id` của point thuộc cùng một cha, lọc thêm theo `source_type`.

        Lọc mỗi `video_id` là không đủ: transcript và caption/OCR của các frame
        cùng video đều mang `video_id` đó, nên dọn transcript sẽ quét luôn cả
        caption của frame.
        """
        if not self._collection_exists(collection_name):
            return []
        dieu_kien = [self.models.FieldCondition(key=field_name, match=self.models.MatchValue(value=value))]
        if source_types:
            dieu_kien.append(
                self.models.FieldCondition(
                    key="source_type", match=self.models.MatchAny(any=list(source_types))
                )
            )
        loc = self.models.Filter(must=dieu_kien)
        ra: list[str] = []
        offset = None
        while True:
            diem, offset = self.client.scroll(
                collection_name=collection_name,
                scroll_filter=loc,
                limit=limit,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            ra.extend(
                str(p.payload["unit_id"]) for p in diem if getattr(p, "payload", None) and p.payload.get("unit_id")
            )
            if offset is None:
                return ra

    def search_points(
        self,
        *,
        collection_name: str,
        vector: list[float],
        limit: int,
        query_filter: Any | None = None,
    ) -> list[dict[str, Any]]:
        """Tìm điểm gần nhất trong một collection.

        Collection chưa tồn tại (video chưa được index) là trạng thái hợp lệ ->
        trả về [] để tầng trên gọi luồng "không tìm thấy", không bịa kết quả.
        """
        self._validate_search_inputs(vector, limit)
        if not self._collection_exists(collection_name):
            logger.warning(
                f"Qdrant collection '{collection_name}' does not exist yet; returning no results"
            )
            return []

        raw_points = self._query_points(
            collection_name=collection_name,
            vector=vector,
            limit=limit,
            query_filter=query_filter,
        )
        results = [self._as_search_result(point) for point in raw_points]
        logger.info(f"Qdrant search on '{collection_name}' returned {len(results)} point(s)")
        return results

    def search_filter(
        self,
        video_ids: list[str] | None = None,
        years: list[int] | None = None,
    ) -> Any | None:
        """Build filter thu hẹp phạm vi tìm kiếm (None nếu không lọc gì).

        Hai tiêu chí ghép bằng `must`, còn nhiều giá trị của CÙNG một tiêu chí
        ghép bằng `should`: "video-1 và năm 2024" phải là giao, còn "2024, 2025"
        phải là hợp. Ghép phẳng hết vào một `should` thì lọc năm 2024 sẽ kéo về
        cả video khác năm — sai theo kiểu vẫn có kết quả nên khó thấy.
        """
        nhom: list[Any] = []
        for dieu_kien in (self._video_id_conditions(video_ids), self._year_conditions(years)):
            if dieu_kien:
                nhom.append(self.models.Filter(should=dieu_kien))
        if not nhom:
            return None
        # Một tiêu chí thì trả thẳng, khỏi bọc thêm một tầng Filter vô nghĩa.
        return nhom[0] if len(nhom) == 1 else self.models.Filter(must=nhom)

    def _video_id_conditions(self, video_ids: list[str] | None) -> list[Any]:
        clean_ids = [str(item).strip() for item in video_ids or [] if str(item or "").strip()]
        return [
            self.models.FieldCondition(key="video_id", match=self.models.MatchValue(value=video_id))
            for video_id in clean_ids
        ]

    def _year_conditions(self, years: list[int] | None) -> list[Any]:
        """Mỗi năm thành một khoảng nửa mở [01/01 năm đó, 01/01 năm sau).

        Nửa mở chứ không phải `lte=31/12`: giữa 31/12 23:59:59 và nửa đêm vẫn
        còn phần giây lẻ, và `lte` một mốc "cuối ngày" tự chọn sẽ đánh rơi đúng
        khoảng đó. Dùng datetime có timezone thay vì chuỗi để múi giờ là thứ
        được khai rõ, không phải thứ Qdrant tự đoán.
        """
        return [
            self.models.FieldCondition(
                key=POST_CREATED_AT_KEY,
                range=self.models.DatetimeRange(
                    gte=datetime(nam, 1, 1, tzinfo=GIO_VIET_NAM),
                    lt=datetime(nam + 1, 1, 1, tzinfo=GIO_VIET_NAM),
                ),
            )
            for nam in self.normalize_years(years)
        ]

    @staticmethod
    def normalize_years(years: list[int] | None) -> list[int]:
        """Lọc rác, khử trùng lặp, giữ nguyên thứ tự người dùng nhập."""
        ket_qua: list[int] = []
        for gia_tri in years or []:
            if isinstance(gia_tri, bool):
                # bool là int trong Python: True sẽ lọt thành năm 1.
                raise ValueError(f"năm không hợp lệ: {gia_tri!r}")
            try:
                nam = int(str(gia_tri).strip())
            except (TypeError, ValueError) as exc:
                raise ValueError(f"năm không hợp lệ: {gia_tri!r}") from exc
            if not NAM_NHO_NHAT <= nam <= NAM_LON_NHAT:
                raise ValueError(f"năm phải nằm trong {NAM_NHO_NHAT}..{NAM_LON_NHAT}, nhận được {nam}")
            if nam not in ket_qua:
                ket_qua.append(nam)
        return ket_qua

    def _query_points(
        self,
        *,
        collection_name: str,
        vector: list[float],
        limit: int,
        query_filter: Any | None,
    ) -> list[Any]:
        # qdrant-client >= 1.10 dùng query_points, bản cũ chỉ có search.
        if hasattr(self.client, "query_points"):
            response = self.client.query_points(
                collection_name=collection_name,
                query=vector,
                limit=limit,
                query_filter=query_filter,
                with_payload=True,
            )
            points = getattr(response, "points", response)
            return list(points or [])
        response = self.client.search(
            collection_name=collection_name,
            query_vector=vector,
            limit=limit,
            query_filter=query_filter,
            with_payload=True,
        )
        return list(response or [])

    @staticmethod
    def _as_search_result(point: Any) -> dict[str, Any]:
        payload = getattr(point, "payload", None)
        if not isinstance(payload, dict):
            payload = {}
        return {
            "id": str(getattr(point, "id", "") or ""),
            "score": float(getattr(point, "score", 0.0) or 0.0),
            "payload": dict(payload),
        }

    @staticmethod
    def _validate_search_inputs(vector: list[float], limit: int) -> None:
        if not isinstance(vector, list) or not vector:
            raise ValueError("vector must be a non-empty list")
        if not all(isinstance(value, (int, float)) for value in vector):
            raise ValueError("vector contains non-numeric values")
        if not isinstance(limit, int) or limit <= 0:
            raise ValueError("limit must be a positive integer")

    def ensure_collection(self, *, collection_name: str, vector_size: int) -> None:
        if not isinstance(collection_name, str) or not collection_name.strip():
            raise ValueError("collection_name must be a non-empty string")
        if not isinstance(vector_size, int) or vector_size <= 0:
            raise ValueError("vector_size must be a positive integer")

        if not self._collection_exists(collection_name):
            distance = getattr(self.models.Distance, self.distance_name, None)
            if distance is None:
                raise VideoVectorStoreConfigurationError(f"Unsupported Qdrant distance: {self.distance_name}")
            logger.info(f"Creating Qdrant collection '{collection_name}' with vector size {vector_size}")
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=self.models.VectorParams(size=vector_size, distance=distance),
            )
            self._ensure_post_created_at_index(collection_name)
            return

        existing_size = self._collection_vector_size(collection_name)
        if existing_size is not None and existing_size != vector_size:
            raise VideoVectorStoreConfigurationError(
                "Qdrant collection vector dimension mismatch: "
                f"collection '{collection_name}' has {existing_size}, requested {vector_size}."
            )

    def _ensure_post_created_at_index(self, collection_name: str) -> None:
        """Đánh index datetime cho khoá lọc năm, ngay khi tạo collection.

        Không có index thì Qdrant vẫn lọc đúng — chỉ là quét tuần tự. Tạo ở đây
        để collection sinh sau (video_transcript) không phải nhớ chạy tay như
        `media_clip` đã phải làm. Lỗi ở bước này không được chặn việc index dữ
        liệu: mất index chỉ làm truy vấn chậm, còn ném lên thì mất cả lô.
        """
        try:
            self.client.create_payload_index(
                collection_name=collection_name,
                field_name=POST_CREATED_AT_KEY,
                field_schema=self.models.PayloadSchemaType.DATETIME,
            )
        except Exception as exc:
            logger.warning(
                f"Không tạo được payload index '{POST_CREATED_AT_KEY}' cho "
                f"'{collection_name}': {exc.__class__.__name__}: {exc}"
            )

    def _create_client(self) -> Any:
        if self._is_missing(self.url):
            raise VideoVectorStoreConfigurationError("Missing Qdrant URL. Set QDRANT_URL.")
        try:
            from qdrant_client import QdrantClient
        except ImportError as exc:
            raise VideoVectorStoreConfigurationError("Missing dependency 'qdrant-client'.") from exc

        kwargs: dict[str, Any] = {"url": self.url}
        if not self._is_missing(self.api_key):
            kwargs["api_key"] = self.api_key
        return QdrantClient(**kwargs)

    @staticmethod
    def _load_models() -> Any:
        try:
            from qdrant_client import models
        except ImportError as exc:
            raise VideoVectorStoreConfigurationError("Missing dependency 'qdrant-client'.") from exc
        return models

    @staticmethod
    def _is_missing(value: Any) -> bool:
        if value is None:
            return True
        if not isinstance(value, str):
            return False
        stripped = value.strip()
        return (
            not stripped
            or (stripped.startswith("${") and stripped.endswith("}"))
            or (stripped.startswith("your_") and stripped.endswith("_here"))
        )

    def _collection_exists(self, collection_name: str) -> bool:
        if hasattr(self.client, "collection_exists"):
            return bool(self.client.collection_exists(collection_name=collection_name))
        try:
            self.client.get_collection(collection_name=collection_name)
            return True
        except Exception:
            return False

    def _collection_vector_size(self, collection_name: str) -> int | None:
        collection = self.client.get_collection(collection_name=collection_name)
        vectors = getattr(getattr(collection, "config", None), "params", None)
        vectors = getattr(vectors, "vectors", None)
        if vectors is None:
            return None
        if hasattr(vectors, "size"):
            return int(vectors.size)
        if isinstance(vectors, dict):
            sizes = [getattr(value, "size", None) for value in vectors.values()]
            sizes = [int(size) for size in sizes if size is not None]
            if len(set(sizes)) == 1:
                return sizes[0]
        return None

    @classmethod
    def _validate_upsert_inputs(
        cls,
        point_ids: list[str],
        vectors: list[list[float]],
        payloads: list[dict[str, Any]],
    ) -> None:
        if not point_ids:
            raise ValueError("point_ids must not be empty")
        if len(point_ids) != len(vectors) or len(point_ids) != len(payloads):
            raise ValueError(
                "point_ids/vectors/payloads length mismatch: "
                f"{len(point_ids)} != {len(vectors)} != {len(payloads)}"
            )
        dimension: int | None = None
        for index, embedding in enumerate(vectors):
            if not isinstance(embedding, list) or not embedding:
                raise ValueError(f"vectors[{index}] must be a non-empty list")
            if not all(isinstance(value, (int, float)) for value in embedding):
                raise ValueError(f"vectors[{index}] contains non-numeric values")
            if dimension is None:
                dimension = len(embedding)
            elif len(embedding) != dimension:
                raise ValueError("vector dimensions are not consistent")
        for index, point_id in enumerate(point_ids):
            if not str(point_id or "").strip():
                raise ValueError(f"point_ids[{index}] must be non-empty")
        for index, payload in enumerate(payloads):
            if not isinstance(payload, dict):
                raise TypeError(f"payloads[{index}] must be a dict")

    @staticmethod
    def point_id(value: str) -> str:
        """Suy point id từ khoá nghiệp vụ, luôn cho ra cùng một giá trị.

        Nhờ tính tất định này mà index lại là ghi đè chứ không nhân bản điểm,
        và công cụ bên ngoài hỏi được 'khoá này đã index chưa' mà không cần
        chạm vào phương thức riêng tư.
        """
        text = str(value).strip()
        try:
            return str(uuid.UUID(text))
        except ValueError:
            return str(uuid.uuid5(uuid.NAMESPACE_URL, text))

    @classmethod
    def _point_id(cls, value: str) -> str:
        return cls.point_id(value)
