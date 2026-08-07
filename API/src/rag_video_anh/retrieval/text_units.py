"""Đơn vị nhúng **văn bản** cho nhánh media: caption, OCR, transcript.

Mỗi point chỉ mang **một** loại nội dung. Không nối `caption + OCR + transcript`
thành một chuỗi, vì:

* OCR thường ngắn và toàn tên riêng/mã số; caption dài sẽ làm loãng nó;
* transcript lặp lại trên nhiều keyframe lân cận, nối vào là nhân bản vector;
* đổi một trường phải nhúng lại cả cụm;
* không cân được trọng số riêng cho từng nguồn về sau;
* và quan trọng nhất: **không biết hit đến từ đâu**. Trích dẫn phải nói được
  "chữ này nằm trên ảnh" hay "câu này có người nói", chứ không thể nói chung chung.

Không thêm tiền tố mô tả kiểu "Đây là caption của hình ảnh…" vào text đem nhúng.
Nó có mặt ở **mọi** point nên không phân biệt được gì, chỉ tốn token. Loại nội
dung nằm ở payload `source_type`.

Ảnh và keyframe **không** được nhúng ở đây; chúng chỉ còn là metadata cha để
giao diện hiển thị được và để trích dẫn trỏ đúng chỗ.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

SOURCE_CAPTION = "caption"
SOURCE_OCR = "ocr"
SOURCE_TRANSCRIPT = "transcript"
SOURCE_TYPES = (SOURCE_CAPTION, SOURCE_OCR, SOURCE_TRANSCRIPT)

# Đổi khi quy tắc chuẩn hoá hoặc cắt đoạn đổi. Nằm trong payload và trong khoá
# cache, nên vector cũ không bị dùng nhầm cho văn bản đã được cắt theo luật mới.
EMBEDDING_FORMAT_VERSION = "text-atomic-v1"

# OCR: giữ nguyên cả những mẩu rất ngắn — "HIT", "GEN 15" là đúng thứ người ta
# tìm. Chỉ cắt nhỏ khi một khối dài quá mức.
OCR_MAX_TOKENS = 200
OCR_OVERLAP_TOKENS = 20
# Transcript: ASR đang cắt audio theo đoạn ~60 giây.
TRANSCRIPT_MIN_TOKENS = 30
TRANSCRIPT_MAX_TOKENS = 300
TRANSCRIPT_WINDOW_TOKENS = 200
TRANSCRIPT_OVERLAP_TOKENS = 40

_KY_TU_DIEU_KHIEN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]+")


@lru_cache(maxsize=1)
def _bo_dem():
    """`cl100k_base` là bộ token của text-embedding-3-*; thiếu thì đếm theo từ."""
    try:
        import tiktoken

        return tiktoken.get_encoding("cl100k_base")
    except Exception:  # pragma: no cover - chỉ chạy khi môi trường thiếu tiktoken
        return None


def dem_token(text: str) -> int:
    bo = _bo_dem()
    if bo is None:
        return len(text.split())
    return len(bo.encode(text))


def chuan_hoa_caption(value: Any) -> str:
    """Caption là văn xuôi nên gộp mọi khoảng trắng là an toàn."""
    text = _lam_sach_co_ban(value)
    return re.sub(r"\s+", " ", text).strip()


# Watermark của CLB nằm trên **mọi** ảnh, nên chữ lấy từ nó có giá trị phân biệt
# gần bằng không: tìm "HIT" sẽ khớp cả kho. Kèm cả các biến thể VLM đọc sai đã
# quan sát được (HTC/HITC/HTS) — nhưng chỉ dùng để nhận diện *dòng watermark*,
# KHÔNG thay thế toàn cục, vì "HTC" có thể là nội dung thật ở ảnh khác.
WATERMARK = frozenset({
    "hit club", "hitclub", "hit's club", "clb hit club", "clb tin hoc dh cnhn",
    "clb tin hoc dh cn-hn", "hit clb tin hoc dh cnhn", "htc", "hitc", "hts",
})
# Hoa văn nhị phân trong logo: dãy 0/1 dài, có thể xen dấu phân cách. Ngưỡng 8
# chữ số để "1010" (mã lớp, số hiệu) không bị đụng tới.
_DAI_NHI_PHAN = re.compile(r"(?<![0-9A-Za-z])(?:[01][\s.,\-|/]{0,2}){8,}(?![0-9A-Za-z])")
_TOKEN_NHI_PHAN = re.compile(r"^[01]+$")
# Luật thứ hai cho dãy ngắn: một dòng mà phần lớn token là 0/1 thì chúng là hoa
# văn chứ không phải dữ liệu. Đòi tối thiểu 4 token để "Mã lớp 1010" không dính.
TY_LE_NHI_PHAN = 0.4
SO_TOKEN_NHI_PHAN_TOI_THIEU = 4


def _bo_hoa_van_nhi_phan(dong: str) -> str:
    """Bỏ hoa văn 0/1, giữ nguyên mã số và ngày tháng thật."""
    con_lai = _DAI_NHI_PHAN.sub(" ", dong)
    token = con_lai.split()
    nhi_phan = [t for t in token if _TOKEN_NHI_PHAN.match(t)]
    if len(nhi_phan) >= SO_TOKEN_NHI_PHAN_TOI_THIEU and len(nhi_phan) / len(token) >= TY_LE_NHI_PHAN:
        token = [t for t in token if not _TOKEN_NHI_PHAN.match(t)]
    return " ".join(token)


def _chuan_de_so_sanh(text: str) -> str:
    """Hạ chữ, bỏ dấu và dấu câu — CHỈ để đối chiếu watermark, không để lưu."""
    # "đ"/"Đ" là ký tự riêng, KHÔNG phân rã trong NFD, nên phải map tay — nếu
    # không thì "ĐH CNHN" thành "H CNHN" và không khớp watermark nào cả.
    thap = text.lower().replace("đ", "d")
    khong_dau = "".join(
        c for c in unicodedata.normalize("NFD", thap) if unicodedata.category(c) != "Mn"
    )
    return re.sub(r"[^a-z0-9\s-]+", " ", khong_dau).strip()


def _la_watermark(text: str) -> bool:
    chuan = re.sub(r"\s+", " ", _chuan_de_so_sanh(text))
    if not chuan:
        return True
    # Gỡ dần từng cụm watermark (cụm dài trước), đệm khoảng trắng để chỉ khớp
    # trọn token. Hết sạch nghĩa là dòng đó không còn gì ngoài watermark.
    con_lai = f" {chuan} "
    for cum in sorted(WATERMARK, key=len, reverse=True):
        while f" {cum} " in con_lai:
            con_lai = con_lai.replace(f" {cum} ", " ")
    return not con_lai.strip()


def loc_watermark(text: str) -> str:
    """Bỏ hoa văn nhị phân và dòng watermark; giữ nguyên nội dung sự kiện.

    Không thay thế toàn cục `HTC -> HIT`: sửa như vậy có ngày sửa nhầm một mã
    thật ở ảnh khác. Chỉ bỏ khi cả **dòng** là watermark, hoặc khi **toàn bộ**
    OCR sau khi làm sạch không còn gì ngoài các biến thể watermark.
    """
    dong = [_bo_hoa_van_nhi_phan(d) for d in text.split("\n")]
    giu = [d for d in dong if d and not _la_watermark(d)]
    if not giu:
        return ""      # không còn gì phân biệt được -> không sinh point OCR
    ket = "\n".join(giu)
    return "" if _la_watermark(ket) else ket


def chuan_hoa_ocr(value: Any) -> str:
    """OCR **giữ ranh giới dòng**, vì bố cục là thông tin.

    `clean_text` dùng cho payload gộp mọi khoảng trắng thành một dấu cách, làm
    dính "12h30 - 20/04/2024" với dòng kế tiếp thành một câu vô nghĩa. Ở đây chỉ
    gộp khoảng trắng **trong cùng một dòng**, và bỏ dòng rỗng.

    Không bỏ dấu, không hạ chữ thường, không dịch: mã, số hiệu và tên riêng là
    thứ duy nhất OCR mang lại mà caption không có.
    """
    text = _lam_sach_co_ban(value)
    dong = [re.sub(r"[ \t ]+", " ", d).strip() for d in text.split("\n")]
    # Lọc hoa văn/watermark ở ĐÂY, không ở khâu caption: OCR thô vẫn nằm
    # nguyên trong PostgreSQL, nên đổi luật lọc không phải gọi lại VLM.
    return loc_watermark("\n".join(d for d in dong if d))


def _lam_sach_co_ban(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    text = unicodedata.normalize("NFC", value)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return _KY_TU_DIEU_KHIEN.sub(" ", text)


def bam_noi_dung(text: str) -> str:
    """Băm văn bản đã chuẩn hoá — caption trùng nhau giữa nhiều frame chỉ gọi API một lần."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _cat_theo_token(text: str, cua_so: int, chong_lan: int) -> list[str]:
    """Cắt theo token nhưng **ưu tiên ranh giới dòng**, chỉ cắt giữa dòng khi buộc phải."""
    if dem_token(text) <= cua_so:
        return [text]
    bo = _bo_dem()
    if bo is None:  # pragma: no cover
        return [text]

    ra: list[str] = []
    hien_tai: list[str] = []
    for dong in text.split("\n"):
        thu = "\n".join([*hien_tai, dong]) if hien_tai else dong
        if hien_tai and dem_token(thu) > cua_so:
            ra.append("\n".join(hien_tai))
            hien_tai = [dong]
        else:
            hien_tai = [*hien_tai, dong] if hien_tai else [dong]
        # Một dòng đơn lẻ đã vượt cửa sổ thì không còn ranh giới nào để bám.
        while dem_token("\n".join(hien_tai)) > cua_so:
            ma = bo.encode("\n".join(hien_tai))
            ra.append(bo.decode(ma[:cua_so]))
            con_lai = bo.decode(ma[max(0, cua_so - chong_lan):])
            hien_tai = [con_lai]
            if dem_token(con_lai) <= cua_so:
                break
    if hien_tai and "\n".join(hien_tai).strip():
        ra.append("\n".join(hien_tai))
    return [d for d in (x.strip() for x in ra) if d]


@dataclass(frozen=True)
class TextRetrievalUnit:
    """Một point trong collection text. `text` là thứ duy nhất đem đi nhúng."""

    unit_id: str
    source_type: str
    text: str
    content_hash: str
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "source_type": self.source_type,
            "text": self.text,
            "content_hash": self.content_hash,
            "embedding_format_version": EMBEDDING_FORMAT_VERSION,
            **self.payload,
        }


def _dung_unit(unit_id: str, source_type: str, text: str, payload: dict[str, Any]) -> TextRetrievalUnit:
    return TextRetrievalUnit(
        unit_id=unit_id,
        source_type=source_type,
        text=text,
        content_hash=bam_noi_dung(text),
        payload={k: v for k, v in payload.items() if v is not None},
    )


def units_tu_anh(unit: Any) -> list[TextRetrievalUnit]:
    """Text unit của một ảnh tĩnh.

    Ảnh **không có** `video_id` hay `timestamp_sec` — giữ nguyên nguyên tắc cũ:
    không được để trích dẫn bịa ra một khoảnh khắc trong tấm ảnh tĩnh.
    """
    cha = {
        "media_kind": "image",
        "image_media_id": unit.image_media_id,
        "post_id": unit.post_id,
        "bucket_name": unit.bucket_name,
        "object_key": unit.object_key,
        "source_url": unit.source_url,
        # Metadata cấp BÀI, giống hệt trên mọi point sinh ra từ cùng một bài —
        # đó là thứ khiến `event_key == "..."` lọc được cả ba nguồn cùng lúc.
        "post_created_at": getattr(unit, "post_created_at", None),
        "event_key": getattr(unit, "event_key", None),
    }
    return _units_caption_ocr(f"image:{unit.image_media_id}", unit.caption, _ocr_con_bo_cuc(unit), cha)


def units_tu_keyframe(unit: Any) -> list[TextRetrievalUnit]:
    """Text unit của một keyframe video. Transcript của video đi đường riêng."""
    cha = {
        "media_kind": "video_frame",
        "video_id": unit.video_id,
        "video_media_id": unit.video_media_id,
        "frame_media_id": unit.frame_media_id,
        "frame_index": unit.frame_index,
        "timestamp_sec": unit.timestamp_sec,
        "post_id": unit.post_id,
        "bucket_name": unit.bucket_name,
        "object_key": unit.frame_object_key,
        "source_url": unit.source_url,
        # Metadata cấp BÀI, giống hệt trên mọi point sinh ra từ cùng một bài —
        # đó là thứ khiến `event_key == "..."` lọc được cả ba nguồn cùng lúc.
        "post_created_at": getattr(unit, "post_created_at", None),
        "event_key": getattr(unit, "event_key", None),
    }
    return _units_caption_ocr(f"frame:{unit.frame_media_id}", unit.caption, _ocr_con_bo_cuc(unit), cha)


def _ocr_con_bo_cuc(unit: Any) -> str:
    """Ưu tiên bản OCR còn xuống dòng; `ocr_text` đã bị `clean_text` làm phẳng.

    Không có `ocr_text_layout` thì vẫn dùng `ocr_text` — mất bố cục còn hơn mất
    chữ, và unit dựng tay trong test không phải khai thêm trường.
    """
    return getattr(unit, "ocr_text_layout", "") or getattr(unit, "ocr_text", "") or ""


def _units_caption_ocr(goc: str, caption: Any, ocr: Any, cha: dict[str, Any]) -> list[TextRetrievalUnit]:
    ra: list[TextRetrievalUnit] = []

    van_ban_caption = chuan_hoa_caption(caption)
    if van_ban_caption:
        ra.append(_dung_unit(f"{SOURCE_CAPTION}:{goc}", SOURCE_CAPTION, van_ban_caption, cha))

    van_ban_ocr = chuan_hoa_ocr(ocr)
    if van_ban_ocr:
        manh = _cat_theo_token(van_ban_ocr, OCR_MAX_TOKENS, OCR_OVERLAP_TOKENS)
        for i, doan in enumerate(manh):
            hau_to = f"#{i}" if len(manh) > 1 else ""
            ra.append(_dung_unit(f"{SOURCE_OCR}:{goc}{hau_to}", SOURCE_OCR, doan,
                                 {**cha, "ocr_chunk_index": i, "ocr_chunk_total": len(manh)}))
    return ra


def units_tu_transcript(units: list[Any]) -> list[TextRetrievalUnit]:
    """Gộp đoạn quá ngắn, cắt đoạn quá dài, giữ nguyên phần ở giữa.

    ASR hiện không có mốc thời gian theo từng từ, nên các đoạn con **dùng chung
    khoảng thời gian của đoạn cha**. Nội suy ra mốc riêng cho từng đoạn con là
    bịa, và trích dẫn video sai mốc còn tệ hơn không có mốc.
    """
    gop = _gop_doan_ngan(units)
    ra: list[TextRetrievalUnit] = []
    for doan in gop:
        text = chuan_hoa_caption(doan["text"])   # transcript là văn nói, gộp khoảng trắng được
        if not text:
            continue
        cha = {
            "media_kind": "transcript",
            "video_id": doan["video_id"],
            "post_id": doan["post_id"],
            "start_sec": doan["start_sec"],
            "end_sec": doan["end_sec"],
            "language": doan.get("language"),
            "source_segment_ids": doan["source_segment_ids"],
            "source_url": doan.get("source_url"),
            # Thiếu hai trường này ở transcript thì `event_key == "..."` tìm được
            # caption/OCR nhưng loại sạch nhánh lời thoại — im lặng và khó lần.
            "post_created_at": doan.get("post_created_at"),
            "event_key": doan.get("event_key"),
        }
        goc = f"{doan['video_id']}:{doan['start_sec']:.3f}-{doan['end_sec']:.3f}"
        manh = ([text] if dem_token(text) <= TRANSCRIPT_MAX_TOKENS
                else _cat_theo_token(text, TRANSCRIPT_WINDOW_TOKENS, TRANSCRIPT_OVERLAP_TOKENS))
        for i, phan in enumerate(manh):
            hau_to = f"#{i}" if len(manh) > 1 else ""
            ra.append(_dung_unit(f"{SOURCE_TRANSCRIPT}:{goc}{hau_to}", SOURCE_TRANSCRIPT, phan,
                                 {**cha, "chunk_index": i, "chunk_total": len(manh)}))
    return ra


def _gop_doan_ngan(units: list[Any]) -> list[dict[str, Any]]:
    """Đoạn dưới ngưỡng được dán vào đoạn kế tiếp; start/end là hợp của chúng."""
    ra: list[dict[str, Any]] = []
    cho: dict[str, Any] | None = None
    for u in units:
        hien = {
            "video_id": u.video_id,
            "post_id": u.post_id,
            "start_sec": u.start_sec,
            "end_sec": u.end_sec,
            "text": u.text,
            "language": getattr(u, "language", None),
            "source_segment_ids": list(u.source_segment_ids or []),
            "source_url": getattr(u, "source_url", None),
            "post_created_at": getattr(u, "post_created_at", None),
            "event_key": getattr(u, "event_key", None),
        }
        if cho is not None and cho["video_id"] == hien["video_id"]:
            hien = {
                **hien,
                "start_sec": cho["start_sec"],
                "end_sec": hien["end_sec"],
                "text": f"{cho['text']} {hien['text']}".strip(),
                "source_segment_ids": cho["source_segment_ids"] + hien["source_segment_ids"],
            }
            cho = None
        elif cho is not None:
            ra.append(cho)      # đổi video: đoạn ngắn đứng một mình còn hơn dán nhầm sang video khác
            cho = None

        if dem_token(chuan_hoa_caption(hien["text"])) < TRANSCRIPT_MIN_TOKENS:
            cho = hien
            continue
        ra.append(hien)
    if cho is not None:
        ra.append(cho)
    return ra
