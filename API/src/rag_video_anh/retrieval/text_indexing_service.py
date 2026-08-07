"""Index caption/OCR/transcript vào collection văn bản riêng.

Tách hẳn khỏi `indexing_service.py` (nhánh Jina, 1.024 chiều) thay vì sửa nó:
hai đường đi hai không gian vector khác nhau, và `media_clip` đang có 1.628 điểm
thật. Trộn chung một service thì một lần đổi cấu hình là ghi nhầm vector 1.536
chiều vào collection 1.024 chiều.

Khác nhánh cũ ở ba chỗ:

* **không tải ảnh về đĩa tạm** và không gọi `embed_images` — ảnh/keyframe chỉ
  còn là metadata cha;
* **không đòi ảnh phải tồn tại trong MinIO** mới cho tạo unit: caption và OCR đã
  nằm trong PostgreSQL, một object bị xoá không xoá được chữ đã đọc từ nó;
* **gộp theo `content_hash` trước khi gọi API**: caption trùng nhau giữa nhiều
  frame chỉ tốn một lời gọi, nhưng vẫn sinh đủ point với metadata riêng.

Đơn vị đầu vào là `IndexGroup`, không phải một danh sách unit trần. Danh sách
trần không mang danh tính cha, nên một ảnh **mất hết** caption/OCR sẽ trả về
`[]` và indexer không biết phải dọn point cũ của ảnh nào — point mồ côi nằm lại
vĩnh viễn và vẫn được trả về khi tìm kiếm.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from src.common_utils.text_embedding import TextEmbeddingService
from src.log.logger import logger
from src.rag_video_anh.retrieval.retrieval_units import (
    ImageRetrievalUnitBuilder,
    VideoRetrievalUnitBuilder,
)
from src.rag_video_anh.retrieval.text_units import (
    EMBEDDING_FORMAT_VERSION,
    SOURCE_CAPTION,
    SOURCE_OCR,
    SOURCE_TRANSCRIPT,
    TextRetrievalUnit,
    dem_token,
    units_tu_anh,
    units_tu_keyframe,
    units_tu_transcript,
)
from src.rag_video_anh.vector_store.vector_store import QdrantVideoVectorStore

# Model và số chiều nằm trong tên: đổi model là đổi collection, không bao giờ
# trộn hai không gian vector vào cùng một chỗ.
DEFAULT_MEDIA_TEXT_COLLECTION = "media_text_te3s_1536_v1"
# Nhà cung cấp giới hạn 300.000 token mỗi request; chừa biên an toàn.
MAX_TOKENS_PER_REQUEST = 250_000
DEFAULT_BATCH_SIZE = 128
NGUON_HINH_ANH = frozenset({SOURCE_CAPTION, SOURCE_OCR})
NGUON_LOI_THOAI = frozenset({SOURCE_TRANSCRIPT})


class DimensionMismatchError(RuntimeError):
    """Số chiều thật khác số chiều mà collection cam kết trong chính tên nó."""


class CollectionConfigError(ValueError):
    """Cấu hình collection tự mâu thuẫn; phát hiện ngay lúc dựng service."""


@dataclass(frozen=True)
class IndexScope:
    """Phạm vi dọn dẹp: point nào thuộc về cha này và thuộc những nguồn nào.

    `source_types` là phần bắt buộc. Lọc mỗi `video_id` sẽ quét luôn caption và
    OCR của mọi frame trong video đó, nên dọn transcript hoá ra xoá nhầm.
    """

    field_name: str
    value: Any
    source_types: frozenset[str]


@dataclass(frozen=True)
class IndexGroup:
    """Unit của một media, kèm danh tính cha để dọn được cả khi `units` rỗng."""

    scopes: tuple[IndexScope, ...]
    units: list[TextRetrievalUnit] = field(default_factory=list)


@dataclass
class TextIndexSummary:
    units_built: int = 0
    points_upserted: int = 0
    points_unchanged: int = 0
    points_deleted: int = 0
    unique_texts: int = 0
    embed_calls: int = 0
    by_source_type: dict[str, int] = field(default_factory=dict)
    skipped: list[dict[str, Any]] = field(default_factory=list)

    def cong(self, khac: TextIndexSummary) -> None:
        self.units_built += khac.units_built
        self.points_upserted += khac.points_upserted
        self.points_unchanged += khac.points_unchanged
        self.points_deleted += khac.points_deleted
        self.unique_texts += khac.unique_texts
        self.embed_calls += khac.embed_calls
        for k, v in khac.by_source_type.items():
            self.by_source_type[k] = self.by_source_type.get(k, 0) + v
        self.skipped.extend(khac.skipped)

    def to_dict(self) -> dict[str, Any]:
        return {
            "units_built": self.units_built,
            "points_upserted": self.points_upserted,
            "points_unchanged": self.points_unchanged,
            "points_deleted": self.points_deleted,
            "unique_texts": self.unique_texts,
            "embed_calls": self.embed_calls,
            "by_source_type": dict(sorted(self.by_source_type.items())),
            "skipped": self.skipped,
        }


class MediaTextIndexingService:
    def __init__(
        self,
        *,
        image_builder: Any | None = None,
        video_builder: Any | None = None,
        text_embedder: Any | None = None,
        vector_store: Any | None = None,
        collection_name: str = DEFAULT_MEDIA_TEXT_COLLECTION,
        batch_size: int = DEFAULT_BATCH_SIZE,
        expected_dimensions: int | None = None,
        skip_unchanged: bool = True,
        delete_stale: bool = True,
    ) -> None:
        # check_*_exists=False: unit văn bản không phụ thuộc vào việc object ảnh
        # còn nằm trong MinIO hay không.
        self.image_builder = image_builder or ImageRetrievalUnitBuilder(check_image_exists=False)
        self.video_builder = video_builder or VideoRetrievalUnitBuilder(check_frame_exists=False)
        self.text_embedder = text_embedder or TextEmbeddingService()
        self.vector_store = vector_store or QdrantVideoVectorStore()
        self.collection_name = collection_name
        self.batch_size = max(1, int(batch_size))
        self.delete_stale = delete_stale
        self.expected_dimensions = self._chot_so_chieu(expected_dimensions, collection_name)
        self.skip_unchanged = skip_unchanged and self._cho_phep_bo_qua()

    # ------------------------------------------------------------ số chiều

    @staticmethod
    def _chot_so_chieu(explicit: int | None, collection_name: str) -> int | None:
        """Số chiều trong TÊN collection là một cam kết, không phải gợi ý.

        Cho `expected_dimensions` ghi đè tên thì cấu hình
        `media_text_te3s_1536_v1` + `expected_dimensions=512` vẫn chạy trót lọt,
        và cam kết ghi trong tên bị vô hiệu hoá một cách âm thầm. Ở đây tên luôn
        được đọc, và mâu thuẫn thì nổ **ngay lúc dựng service** — trước khi tiêu
        một đồng nào cho lời gọi nhúng.
        """
        khop = re.search(r"_(\d{2,5})(?:_v\d+)?$", collection_name)
        tu_ten = int(khop.group(1)) if khop else None
        if explicit is None:
            return tu_ten
        if tu_ten is not None and int(explicit) != tu_ten:
            raise CollectionConfigError(
                f"Collection '{collection_name}' cam kết {tu_ten} chiều nhưng "
                f"expected_dimensions={explicit}. Sửa một trong hai, đừng ghi đè cam kết."
            )
        return int(explicit)

    def _cho_phep_bo_qua(self) -> bool:
        """Không biết số chiều thì không được tin 'không đổi'."""
        if self.expected_dimensions is not None:
            return True
        logger.warning(
            f"Collection '{self.collection_name}' không có số chiều trong tên, nên không "
            "kiểm tra được không gian vector. Đã TẮT skip_unchanged: bỏ qua theo một "
            "điều kiện không kiểm được thì có ngày giữ lại vector của model khác."
        )
        return False

    def _kiem_so_chieu(self, vectors: list[list[float]]) -> None:
        if self.expected_dimensions is None or not vectors:
            return
        thuc_te = {len(v) for v in vectors}
        if thuc_te != {self.expected_dimensions}:
            raise DimensionMismatchError(
                f"Collection '{self.collection_name}' đòi {self.expected_dimensions} chiều, "
                f"nhà cung cấp trả {sorted(thuc_te)}. Kiểm EMBEDDING_DIMENSIONS, hoặc dùng "
                "collection khác — không được trộn hai không gian vector vào một chỗ."
            )

    # ------------------------------------------------------------ dựng nhóm

    def build_image_group(self, image_media_id: str) -> IndexGroup:
        """Giữ `image_media_id` kể cả khi ảnh không còn caption/OCR nào."""
        scope = (IndexScope("image_media_id", str(image_media_id), NGUON_HINH_ANH),)
        unit = self.image_builder.build(image_media_id)
        return IndexGroup(scopes=scope, units=units_tu_anh(unit) if unit is not None else [])

    def build_video_group(self, video_media_id: str) -> IndexGroup:
        """Scope đặt ở mức **video**, không ở mức frame.

        Suy `video_id` từ chính unit thì một video mất sạch cả transcript lẫn
        keyframe sẽ ra `scopes=[]`, và point cũ của nó nằm lại vĩnh viễn. Còn
        scope theo từng `frame_media_id` thì không phủ được frame bị builder
        loại (frame_index/timestamp/object_key hỏng) — frame đó không sinh
        `MediaClipUnit` nên cũng không sinh scope.

        Payload của mọi point frame đều mang `video_id`, nên một scope mức video
        kèm `source_types={caption, ocr}` phủ hết cả hai lỗ hổng đó, mà vẫn
        không đụng transcript vì transcript nằm ở scope `source_types` khác.
        """
        ket_qua = self.video_builder.build(video_media_id)
        keyframes = list(getattr(ket_qua, "media_clip_units", None) or [])
        transcripts = list(getattr(ket_qua, "video_transcript_units", None) or [])

        units: list[TextRetrievalUnit] = []
        for keyframe in keyframes:
            units.extend(units_tu_keyframe(keyframe))
        units.extend(units_tu_transcript(transcripts))

        # Builder đặt `summary.video_id` trước mọi nhánh có thể loại frame, nên
        # đó là nguồn duy nhất còn đúng khi cả hai danh sách đều rỗng.
        video_id = getattr(getattr(ket_qua, "summary", None), "video_id", None)
        if video_id is None:
            # Chỉ để fake/test cũ không phải khai thêm summary.
            video_id = next(
                (x.video_id for x in (*transcripts, *keyframes) if getattr(x, "video_id", None)), None
            )
        if video_id is None:
            logger.warning(
                f"Video media '{video_media_id}' không suy được video_id; bỏ qua bước dọn point cũ."
            )
            return IndexGroup(scopes=(), units=units)

        return IndexGroup(
            scopes=(
                IndexScope("video_id", str(video_id), NGUON_HINH_ANH),
                IndexScope("video_id", str(video_id), NGUON_LOI_THOAI),
            ),
            units=units,
        )

    # ------------------------------------------------------------ index

    def index_image(self, image_media_id: str) -> TextIndexSummary:
        return self.index_groups([self.build_image_group(image_media_id)])

    def index_video(self, video_media_id: str) -> TextIndexSummary:
        return self.index_groups([self.build_video_group(video_media_id)])

    def index_units(self, units: list[TextRetrievalUnit]) -> TextIndexSummary:
        """Lối vào tiện dụng khi đã cầm sẵn unit; scope suy ra từ chính chúng.

        Không dùng được cho lượt chạy lại của media đã mất hết nội dung — lúc đó
        `units` rỗng nên không suy ra được cha nào. Đường sản xuất dùng
        `index_groups`.
        """
        return self.index_groups([IndexGroup(scopes=self._scope_tu_units(units), units=units)])

    def index_many(self, nhom: Iterable[list[TextRetrievalUnit]]) -> TextIndexSummary:
        return self.index_groups(
            IndexGroup(scopes=self._scope_tu_units(units), units=units) for units in nhom
        )

    def index_groups(self, groups: Iterable[IndexGroup]) -> TextIndexSummary:
        """Gom nhiều media rồi mới gọi API.

        Gọi từng ảnh một thì `batch_size` vô nghĩa và việc gộp theo nội dung
        không bao giờ bắt được caption trùng nhau **giữa các ảnh** — 1.628 ảnh
        thành gần 1.628 request.
        """
        tong = TextIndexSummary()
        cho: list[IndexGroup] = []
        so_van_ban: set[str] = set()
        for group in groups:
            cho.append(group)
            so_van_ban.update(u.content_hash for u in group.units)
            if len(so_van_ban) >= self.batch_size:
                tong.cong(self._xu_ly(cho))
                cho, so_van_ban = [], set()
        if cho:
            tong.cong(self._xu_ly(cho))
        return tong

    # ------------------------------------------------------------ một mẻ

    def _xu_ly(self, groups: list[IndexGroup]) -> TextIndexSummary:
        units = [u for g in groups for u in g.units]
        summary = TextIndexSummary(units_built=len(units))
        for u in units:
            summary.by_source_type[u.source_type] = summary.by_source_type.get(u.source_type, 0) + 1

        # Tìm trước, xoá sau. Xoá trước rồi nhúng lỗi là mất point cũ mà chưa có
        # point mới — kho rỗng đi một khoảng mà không ai chủ ý.
        mo_coi = self._tim_mo_coi(groups)

        can_lam = self._loc_don_vi_da_doi(units, summary)
        if can_lam:
            vector_theo_hash = self._embed_unique(can_lam, summary)
            ids, vectors, payloads = [], [], []
            for u in can_lam:
                vector = vector_theo_hash.get(u.content_hash)
                if vector is None:
                    summary.skipped.append({"unit_id": u.unit_id, "reason": "khong_co_vector"})
                    continue
                ids.append(u.unit_id)
                vectors.append(vector)
                payloads.append({
                    **u.to_dict(),
                    "embedding_model": self.text_embedder.model,
                    "embedding_dimensions": len(vector),
                })
            if ids:
                self._kiem_so_chieu(vectors)
                self.vector_store.upsert_points(
                    collection_name=self.collection_name,
                    point_ids=ids,
                    vectors=vectors,
                    payloads=payloads,
                )
                summary.points_upserted = len(ids)

        # Tới đây thì hoặc đã ghi xong, hoặc không có gì phải ghi. Cả hai đều là
        # lúc an toàn để dọn; lỗi ở trên đã ném ra ngoài và không chạm tới đây.
        if mo_coi:
            self.vector_store.delete_points(collection_name=self.collection_name, point_ids=mo_coi)
            summary.points_deleted = len(mo_coi)

        logger.info(f"Indexed media text units: {summary.to_dict()}")
        return summary

    # ------------------------------------------------------------ chạy lại

    def _loc_don_vi_da_doi(
        self, units: list[TextRetrievalUnit], summary: TextIndexSummary
    ) -> list[TextRetrievalUnit]:
        """Bỏ qua point có cùng `content_hash`, cùng model, cùng format và cùng số chiều."""
        if not units or not self.skip_unchanged:
            return units
        da_co = self.vector_store.get_payloads(
            collection_name=self.collection_name, point_ids=[u.unit_id for u in units]
        )
        if not da_co:
            return units

        can_lam = []
        for u in units:
            cu = da_co.get(u.unit_id)
            khong_doi = (
                cu is not None
                and cu.get("content_hash") == u.content_hash
                and cu.get("embedding_model") == self.text_embedder.model
                and cu.get("embedding_format_version") == EMBEDDING_FORMAT_VERSION
                and cu.get("embedding_dimensions") == self.expected_dimensions
            )
            if khong_doi:
                summary.points_unchanged += 1
            else:
                can_lam.append(u)
        return can_lam

    def _tim_mo_coi(self, groups: list[IndexGroup]) -> list[str]:
        """`unit_id` của lượt trước mà lượt này không còn sinh ra — chưa xoá vội."""
        if not self.delete_stale:
            return []
        con_giu = {u.unit_id for g in groups for u in g.units}
        mo_coi: set[str] = set()
        for scope in {s for g in groups for s in g.scopes}:
            cu = self.vector_store.list_unit_ids_by_field(
                collection_name=self.collection_name,
                field_name=scope.field_name,
                value=scope.value,
                source_types=sorted(scope.source_types),
            )
            mo_coi.update(set(cu) - con_giu)
        return sorted(mo_coi)

    @staticmethod
    def _scope_tu_units(units: list[TextRetrievalUnit]) -> tuple[IndexScope, ...]:
        ra: set[IndexScope] = set()
        for u in units:
            if u.source_type == SOURCE_TRANSCRIPT:
                if u.payload.get("video_id") is not None:
                    ra.add(IndexScope("video_id", u.payload["video_id"], NGUON_LOI_THOAI))
                continue
            for khoa in ("image_media_id", "frame_media_id"):
                if u.payload.get(khoa) is not None:
                    ra.add(IndexScope(khoa, u.payload[khoa], NGUON_HINH_ANH))
                    break
        return tuple(sorted(ra, key=lambda s: (s.field_name, str(s.value))))

    # ------------------------------------------------------------ nhúng

    def _embed_unique(self, units: list[TextRetrievalUnit], summary: TextIndexSummary) -> dict[str, list[float]]:
        """Một văn bản một lần gọi, dù nó xuất hiện ở bao nhiêu point."""
        theo_hash: dict[str, str] = {}
        for u in units:
            theo_hash.setdefault(u.content_hash, u.text)
        summary.unique_texts = len(theo_hash)

        hashes = list(theo_hash)
        ra: dict[str, list[float]] = {}
        for lo in self._chia_lo([theo_hash[h] for h in hashes]):
            vectors = self.text_embedder.embed_documents(lo["texts"])
            if len(vectors) != len(lo["texts"]):
                raise RuntimeError(
                    f"Nhà cung cấp trả {len(vectors)} vector cho {len(lo['texts'])} văn bản; "
                    "quan hệ 1 unit ↔ 1 vector bị phá."
                )
            summary.embed_calls += 1
            for offset, vector in enumerate(vectors):
                ra[hashes[lo["start"] + offset]] = vector
        return ra

    def _chia_lo(self, texts: list[str]) -> list[dict[str, Any]]:
        """Chia theo cả số lượng lẫn tổng token — vượt hạn mức token là 400 cả lô."""
        lo: list[dict[str, Any]] = []
        hien_tai: list[str] = []
        bat_dau = 0
        token = 0
        for i, text in enumerate(texts):
            n = dem_token(text)
            if hien_tai and (len(hien_tai) >= self.batch_size or token + n > MAX_TOKENS_PER_REQUEST):
                lo.append({"start": bat_dau, "texts": hien_tai})
                bat_dau, hien_tai, token = i, [], 0
            hien_tai.append(text)
            token += n
        if hien_tai:
            lo.append({"start": bat_dau, "texts": hien_tai})
        return lo


__all__ = [
    "DEFAULT_MEDIA_TEXT_COLLECTION",
    "EMBEDDING_FORMAT_VERSION",
    "CollectionConfigError",
    "DimensionMismatchError",
    "IndexGroup",
    "IndexScope",
    "MediaTextIndexingService",
    "TextIndexSummary",
]
