"""Truy hồi media qua ba nhánh văn bản: caption, OCR, transcript.

Một query vector, ba lần tìm có filter, rồi mới hợp nhất ở mức **media cha**.

Vì sao không tìm global top-k trên cả ba nguồn dù chúng cùng một không gian
vector: số point, độ dài và phân bố cosine của ba nguồn khác hẳn nhau. Transcript
có nhiều chunk nên dễ lấn hết slot của caption; OCR nhiễu thì chiếm chỗ của cả
hai. Ba nhánh riêng là cách rẻ nhất để mỗi nguồn có hạn ngạch của nó.

Hai thứ tự trong file này là **có chủ ý và không được đảo**:

1. **Ngưỡng chạy trước hợp nhất.** RRF chỉ nhìn *hạng*, không nhìn *điểm*: một
   câu hỏi hoàn toàn ngoài phạm vi vẫn có kết quả hạng 1, 2, 3 và vẫn nhận điểm
   RRF y hệt một câu trúng đích. Ngưỡng đặt sau RRF sẽ không bao giờ chặn được
   câu ngoài phạm vi — đúng thứ ràng buộc "không được bịa" cấm.
2. **Ngưỡng chạy trước gom nhóm**, ở mức từng point. Lọc sau khi gom thì một
   media có caption tốt sẽ kéo theo cả đoạn transcript 0,2 điểm vào danh sách
   bằng chứng, và tầng trả lời trích dẫn đúng đoạn vô nghĩa đó.

Điểm của một nguồn trong một media lấy `max`, không lấy tổng: lấy tổng thì video
dài hoặc ảnh nhiều mảnh OCR được xếp cao chỉ vì có nhiều point.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.log.logger import logger
from src.rag_video_anh.retrieval.text_indexing_service import DEFAULT_MEDIA_TEXT_COLLECTION
from src.rag_video_anh.retrieval.text_units import (
    SOURCE_CAPTION,
    SOURCE_OCR,
    SOURCE_TRANSCRIPT,
    chuan_hoa_caption,
)

# Transcript lấy nhiều hơn vì nhiều chunk có thể cùng thuộc một video; sau khi
# gom theo video thì số kết quả tụt xuống rất nhanh.
DEFAULT_CANDIDATES = {SOURCE_CAPTION: 20, SOURCE_OCR: 20, SOURCE_TRANSCRIPT: 30}
DEFAULT_TOP_K = 5
# Hằng số RRF quen dùng. 60 làm phẳng chênh lệch giữa các hạng đầu, nên một
# media chỉ hơn nửa hạng ở một nhánh không lật được media trúng ở hai nhánh.
RRF_K = 60
DEFAULT_WEIGHTS = {SOURCE_CAPTION: 1.0, SOURCE_OCR: 1.0, SOURCE_TRANSCRIPT: 1.0}
# Chưa có dữ liệu để hiệu chỉnh, nên mặc định để 0.0 = không lọc. Đặt bừa một
# ngưỡng "trông hợp lý" còn tệ hơn không lọc: nó im lặng cắt mất kết quả đúng
# và không ai biết vì sao. Chốt bằng development set có câu ngoài phạm vi.
DEFAULT_THRESHOLDS = {SOURCE_CAPTION: 0.0, SOURCE_OCR: 0.0, SOURCE_TRANSCRIPT: 0.0}
# Số moment lời thoại giữ lại cho mỗi video, ưu tiên moment không chồng thời gian.
MAX_TRANSCRIPT_MOMENTS = 3
# Không để một bài chiếm hết trang kết quả.
DEFAULT_MAX_PER_POST = 3


@dataclass(frozen=True)
class Evidence:
    source_type: str
    text: str
    score: float
    unit_id: str
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        giu = (
            "frame_media_id", "image_media_id", "timestamp_sec",
            "start_sec", "end_sec", "bucket_name", "object_key",
        )
        return {
            "source_type": self.source_type,
            "text": self.text,
            "score": round(self.score, 6),
            "unit_id": self.unit_id,
            **{k: self.payload[k] for k in giu if self.payload.get(k) is not None},
        }


@dataclass
class MediaResult:
    entity_type: str          # "image" | "video"
    entity_id: str
    post_id: str | None
    source_url: str | None
    score: float = 0.0
    scores_by_source: dict[str, float] = field(default_factory=dict)
    ranks_by_source: dict[str, int] = field(default_factory=dict)
    evidence: list[Evidence] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "entity_type": self.entity_type,
            "entity_id": self.entity_id,
            "post_id": self.post_id,
            "source_url": self.source_url,
            "score": round(self.score, 6),
            "scores_by_source": {k: round(v, 6) for k, v in sorted(self.scores_by_source.items())},
            "ranks_by_source": dict(sorted(self.ranks_by_source.items())),
            "evidence": [e.to_dict() for e in self.evidence],
        }


class MediaTextRetrievalService:
    """Service **riêng**, không sửa `VideoRetriever`.

    `VideoRetriever` đang trỏ `media_clip` (1.024 chiều, Jina) và
    `video_transcript`. Dùng chung một service thì một lần nhầm cấu hình là đem
    vector 1.536 chiều đi tìm trong collection 1.024 chiều.
    """

    def __init__(
        self,
        *,
        text_embedder: Any | None = None,
        vector_store: Any | None = None,
        collection_name: str = DEFAULT_MEDIA_TEXT_COLLECTION,
        top_k: int = DEFAULT_TOP_K,
        candidates: dict[str, int] | None = None,
        thresholds: dict[str, float] | None = None,
        weights: dict[str, float] | None = None,
        max_per_post: int = DEFAULT_MAX_PER_POST,
    ) -> None:
        if text_embedder is None or vector_store is None:
            from src.common_utils.text_embedding import TextEmbeddingService
            from src.rag_video_anh.vector_store.vector_store import QdrantVideoVectorStore

            text_embedder = text_embedder or TextEmbeddingService()
            vector_store = vector_store or QdrantVideoVectorStore()
        self.text_embedder = text_embedder
        self.vector_store = vector_store
        self.collection_name = collection_name
        self.top_k = max(1, int(top_k))
        self.candidates = {**DEFAULT_CANDIDATES, **(candidates or {})}
        self.thresholds = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
        self.weights = {**DEFAULT_WEIGHTS, **(weights or {})}
        self.max_per_post = max(1, int(max_per_post))

    # ------------------------------------------------------------ đường chính

    def retrieve(self, query: str) -> dict[str, Any]:
        cau_hoi = chuan_hoa_caption(query)
        if not cau_hoi:
            raise ValueError("query phải là chuỗi không rỗng")

        # Ba nguồn cùng model, cùng số chiều, cùng collection -> một vector dùng
        # lại cho cả ba nhánh. Gọi nhúng ba lần là trả tiền ba lần cho cùng một thứ.
        vector = self.text_embedder.embed_query(cau_hoi)

        theo_nguon: dict[str, list[dict[str, Any]]] = {}
        bi_loai: dict[str, int] = {}
        for nguon in (SOURCE_CAPTION, SOURCE_OCR, SOURCE_TRANSCRIPT):
            tho = self.vector_store.search_points(
                collection_name=self.collection_name,
                vector=vector,
                limit=self.candidates[nguon],
                query_filter=self._loc_theo_nguon(nguon),
            )
            nguong = float(self.thresholds.get(nguon, 0.0))
            giu = [p for p in tho if float(p.get("score", 0.0)) >= nguong]
            bi_loai[nguon] = len(tho) - len(giu)
            theo_nguon[nguon] = giu

        if not any(theo_nguon.values()):
            # Không nguồn nào vượt ngưỡng -> không tìm thấy. Tầng trên KHÔNG
            # được gọi LLM ở đây; không có bằng chứng thì không có câu trả lời.
            return self._khong_tim_thay(cau_hoi, bi_loai)

        nhom = self._gom_theo_media(theo_nguon)
        ket_qua = self._rrf(nhom)
        ket_qua = self._da_dang_hoa(ket_qua)[: self.top_k]
        return {
            "query": cau_hoi,
            "found": bool(ket_qua),
            "results": [r.to_dict() for r in ket_qua],
            "notes": self._ghi_chu(theo_nguon, bi_loai),
        }

    # ------------------------------------------------------------ tìm

    def _loc_theo_nguon(self, source_type: str) -> Any:
        models = getattr(self.vector_store, "models", None)
        if models is None:      # fake trong test không cần dựng filter thật
            return {"source_type": source_type}
        return models.Filter(
            must=[models.FieldCondition(key="source_type", match=models.MatchValue(value=source_type))]
        )

    # ------------------------------------------------------------ gom nhóm

    def _gom_theo_media(self, theo_nguon: dict[str, list[dict]]) -> dict[tuple[str, str], MediaResult]:
        """Gom point về media cha, mỗi nguồn lấy điểm cao nhất.

        Ảnh gom theo `image_media_id`. Video gom theo `video_id`, nhưng caption
        và OCR của keyframe được lấy `max` **theo từng frame trước**, để một
        frame có cả caption lẫn OCR không đội điểm video lên hai lần.
        """
        nhom: dict[tuple[str, str], MediaResult] = {}
        tot_nhat_theo_frame: dict[tuple[str, str, str], dict] = {}

        for nguon, diem in theo_nguon.items():
            for p in diem:
                pl = p.get("payload") or {}
                khoa = self._khoa_media(pl)
                if khoa is None:
                    continue
                kq = nhom.get(khoa)
                if kq is None:
                    kq = nhom[khoa] = MediaResult(
                        entity_type=khoa[0],
                        entity_id=khoa[1],
                        post_id=pl.get("post_id"),
                        source_url=pl.get("source_url"),
                    )
                score = float(p.get("score", 0.0))
                if score > kq.scores_by_source.get(nguon, float("-inf")):
                    kq.scores_by_source[nguon] = score

                frame_id = pl.get("frame_media_id")
                if nguon != SOURCE_TRANSCRIPT and frame_id:
                    # Giữ bằng chứng tốt nhất của từng (frame, nguồn).
                    k = (khoa[1], str(frame_id), nguon)
                    if score > float((tot_nhat_theo_frame.get(k) or {}).get("score", float("-inf"))):
                        tot_nhat_theo_frame[k] = p
                else:
                    kq.evidence.append(self._bang_chung(nguon, p))

        for (entity_id, _frame, nguon), p in tot_nhat_theo_frame.items():
            for khoa, kq in nhom.items():
                if khoa[1] == entity_id:
                    kq.evidence.append(self._bang_chung(nguon, p))
                    break

        for kq in nhom.values():
            kq.evidence = self._chon_bang_chung(kq.evidence)
        return nhom

    @staticmethod
    def _khoa_media(payload: dict[str, Any]) -> tuple[str, str] | None:
        """Ảnh tĩnh về `image_media_id`; mọi thứ thuộc video về `video_id`."""
        if payload.get("video_id"):
            return ("video", str(payload["video_id"]))
        if payload.get("image_media_id"):
            return ("image", str(payload["image_media_id"]))
        return None

    @staticmethod
    def _bang_chung(source_type: str, point: dict[str, Any]) -> Evidence:
        pl = point.get("payload") or {}
        return Evidence(
            source_type=source_type,
            text=str(pl.get("text") or ""),
            score=float(point.get("score", 0.0)),
            unit_id=str(pl.get("unit_id") or point.get("id") or ""),
            payload=pl,
        )

    def _chon_bang_chung(self, evidence: list[Evidence]) -> list[Evidence]:
        """Giữ bằng chứng đáng đọc, và chỉ giữ moment lời thoại KHÔNG chồng nhau.

        Ba đoạn transcript chồng thời gian là ba lần kể cùng một câu; chúng đẩy
        hai nguồn kia ra khỏi ngữ cảnh mà không thêm thông tin gì.
        """
        khong_loi = sorted(
            (e for e in evidence if e.source_type != SOURCE_TRANSCRIPT), key=lambda e: -e.score
        )
        moment: list[Evidence] = []
        for e in sorted(
            (e for e in evidence if e.source_type == SOURCE_TRANSCRIPT), key=lambda e: -e.score
        ):
            if len(moment) >= MAX_TRANSCRIPT_MOMENTS:
                break
            if not any(self._chong_thoi_gian(e, cu) for cu in moment):
                moment.append(e)
        return [*khong_loi, *sorted(moment, key=lambda e: e.payload.get("start_sec") or 0.0)]

    @staticmethod
    def _chong_thoi_gian(a: Evidence, b: Evidence) -> bool:
        a1, a2 = a.payload.get("start_sec"), a.payload.get("end_sec")
        b1, b2 = b.payload.get("start_sec"), b.payload.get("end_sec")
        if None in (a1, a2, b1, b2):
            return False
        return float(a1) < float(b2) and float(b1) < float(a2)

    # ------------------------------------------------------------ hợp nhất

    def _rrf(self, nhom: dict[tuple[str, str], MediaResult]) -> list[MediaResult]:
        """RRF chỉ để **xếp thứ tự** giữa ba nguồn không so điểm được với nhau.

        Không dùng trọng số kiểu `caption×0.5 + ocr×0.3 + transcript×0.2` ở
        baseline: những con số đó chưa có bằng chứng nào chống lưng.
        """
        for nguon in (SOURCE_CAPTION, SOURCE_OCR, SOURCE_TRANSCRIPT):
            co = [kq for kq in nhom.values() if nguon in kq.scores_by_source]
            for hang, kq in enumerate(sorted(co, key=lambda k: -k.scores_by_source[nguon]), start=1):
                kq.ranks_by_source[nguon] = hang

        for kq in nhom.values():
            kq.score = sum(
                float(self.weights.get(nguon, 1.0)) / (RRF_K + hang)
                for nguon, hang in kq.ranks_by_source.items()
            )
        return sorted(nhom.values(), key=lambda k: (-k.score, k.entity_id))

    def _da_dang_hoa(self, ket_qua: list[MediaResult]) -> list[MediaResult]:
        """Một bài đăng nhiều ảnh không được chiếm hết trang kết quả."""
        dem: dict[str, int] = {}
        ra: list[MediaResult] = []
        du: list[MediaResult] = []
        for kq in ket_qua:
            khoa = kq.post_id or kq.entity_id
            if dem.get(khoa, 0) < self.max_per_post:
                dem[khoa] = dem.get(khoa, 0) + 1
                ra.append(kq)
            else:
                du.append(kq)
        # Phần bị nén xuống cuối chứ không bị vứt: thiếu kết quả còn tệ hơn lệch bài.
        return [*ra, *du]

    # ------------------------------------------------------------ không tìm thấy

    def _khong_tim_thay(self, query: str, bi_loai: dict[str, int]) -> dict[str, Any]:
        logger.info(f"Media text retrieval: không nguồn nào vượt ngưỡng cho '{query}'")
        return {
            "query": query,
            "found": False,
            "results": [],
            "notes": [
                "Không nguồn nào (caption/OCR/transcript) có kết quả vượt ngưỡng.",
                f"Số ứng viên bị ngưỡng loại: {dict(sorted(bi_loai.items()))}.",
                "Không gọi LLM cho trường hợp này — không có bằng chứng thì không có câu trả lời.",
            ],
        }

    def _ghi_chu(self, theo_nguon: dict[str, list], bi_loai: dict[str, int]) -> list[str]:
        rong = sorted(n for n, v in theo_nguon.items() if not v)
        ghi = [f"Ứng viên sau ngưỡng: { {n: len(v) for n, v in sorted(theo_nguon.items())} }."]
        if rong:
            # Nói rõ nhánh nào rỗng: coi 'không có transcript nào' là 'không ai
            # nói gì về việc này' là hiểu sai, và đây là chỗ chặn hiểu sai đó.
            ghi.append(f"Không có kết quả nào từ: {', '.join(rong)}.")
        if any(bi_loai.values()):
            ghi.append(f"Bị ngưỡng loại: {dict(sorted(bi_loai.items()))}.")
        return ghi


__all__ = [
    "DEFAULT_CANDIDATES",
    "DEFAULT_THRESHOLDS",
    "DEFAULT_WEIGHTS",
    "MAX_TRANSCRIPT_MOMENTS",
    "RRF_K",
    "Evidence",
    "MediaResult",
    "MediaTextRetrievalService",
]
