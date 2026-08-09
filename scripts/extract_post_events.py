#!/usr/bin/env python3
"""Trích sự kiện từ nội dung bài đăng và ghi vào bốn bảng sự kiện.

Đây là thứ còn thiếu để `event_key` có giá trị: bốn bảng `event_*` đã có trong
schema từ đầu nhưng rỗng, nên mọi point Qdrant đều không có khoá sự kiện.

**Hai lượt, và lượt thứ hai mới là phần quan trọng.**

1. *Trích* — mỗi bài một lời gọi LLM, trả về tên sự kiện + nhãn kỳ + trích dẫn.
2. *Gom* — một lời gọi LLM trên DANH SÁCH TÊN đã thấy, gom các cách viết của
   cùng một sự kiện về một chuỗi.

Bỏ lượt 2 thì "HIT Contest Series", "HIT CONTEST SERIES 2021 SEASON 2" và
"Contest Series" thành ba `event_key` khác nhau, và lọc theo sự kiện chỉ trả về
một phần ba kho — hỏng theo kiểu vẫn ra kết quả nên không ai nhận ra.

**Chống bịa.** LLM phải trả về một `trich_dan` COPY NGUYÊN VĂN từ bài. Trích dẫn
không tìm thấy trong bài thì kết quả bị loại, dù nghe hợp lý tới đâu — một sự
kiện bịa sẽ kéo cả chùm ảnh của bài đó vào một sự kiện chưa từng xảy ra.

MẶC ĐỊNH LÀ CHẠY THỬ. Không gọi LLM, không ghi DB:

    python scripts/extract_post_events.py                    # xem kế hoạch
    python scripts/extract_post_events.py --extract-only      # gọi LLM, chỉ ghi cache
    python scripts/extract_post_events.py --apply --limit 20  # thử 20 bài trước
    python scripts/extract_post_events.py --apply             # chạy hết

Chạy lại chỉ làm phần còn thiếu: bài nào đã có trong cache thì không gọi LLM lại,
bài nào đã gắn sự kiện thì bỏ qua (trừ khi `--redo`).
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "API"))
load_dotenv(REPO_ROOT / ".env")

from src.configuration import AppConfig  # noqa: E402
from src.rag_video_anh.repository import RepositoryUnitOfWork  # noqa: E402
from src.rag_video_anh.repository.events import normalize_alias, slugify  # noqa: E402
from src.rag_video_anh.repository.models import PostModel  # noqa: E402

DEFAULT_CACHE_DIR = REPO_ROOT / "data/events"
DEFAULT_WORKERS = 6
DEFAULT_MIN_CONFIDENCE = 0.6
PROGRESS_EVERY = 25
# Hết credit hay sai khoá thì mọi bài đều hỏng tức thì. Không chặn sớm thì cả
# kho bị đánh dấu "không có sự kiện" trong vài phút mà không ai kịp nhận ra.
DEFAULT_MAX_CONSECUTIVE_ERRORS = 10
# Bài dài nhất trong kho ~4k ký tự; sự kiện luôn nằm ở đầu bài (tiêu đề trong
# ngoặc vuông), phần đuôi là hashtag và thông tin liên hệ.
MAX_CONTENT_CHARS = 2500
# Số tên tối đa trong một lời gọi gom. Đo 07/08/2026: gửi cả 142 tên thì
# `cx/gpt-5.5` timeout, vì câu trả lời dài vài nghìn token.
KICH_THUOC_LO_GOM = 40
# Lượt gom trả lời dài hơn hẳn lượt trích, nên nó cần hạn giờ riêng — timeout
# mặc định 30 giây của LLM là đủ cho một bài, không đủ cho một lô tên.
TIMEOUT_GOM_GIAY = 600.0

PROMPT_TRICH = """Bạn đọc một bài đăng Facebook của CLB Tin học HIT (trường đại học Việt Nam).
Nhiệm vụ: xác định bài này có nói về một HOẠT ĐỘNG/SỰ KIỆN CỤ THỂ của CLB hay không.

CÓ sự kiện, ví dụ: cuộc thi, buổi offline, lớp học, chuyến du lịch, đêm gala,
lễ kỷ niệm, đợt tuyển thành viên, workshop, seminar, chương trình thiện nguyện.
KHÔNG có sự kiện, ví dụ: lời chúc mừng ngày lễ, câu trích dẫn truyền cảm hứng,
bài chia sẻ kiến thức thuần tuý, thông báo hành chính không gắn hoạt động nào.

Trả về DUY NHẤT một object JSON, không giải thích, không bọc trong markdown:
{{"co_su_kien": true/false, "ten_su_kien": "...", "nhan_ky": "...", "nam": 2024, "do_tin_cay": 0.0, "trich_dan": "..."}}

Quy tắc bắt buộc:
- "ten_su_kien": tên CHUNG của chuỗi sự kiện, KHÔNG kèm năm/mùa/đợt.
  "HIT CONTEST SERIES 2021 SEASON 2" -> "HIT Contest Series".
  "THÔNG BÁO OFFLINE THÁNG 3" -> "Offline hàng tháng".
  "TUYỂN THÀNH VIÊN BAN QUẢN TRỊ KHÓA 12" -> "Tuyển thành viên Ban Quản trị".
- "nhan_ky": phần phân biệt kỳ này với kỳ khác ("2021 Season 2", "Khóa 12",
  "Tháng 3/2024"). Không xác định được thì để null.
- "nam": năm sự kiện diễn ra nếu bài nói rõ, ngược lại null. KHÔNG suy từ ngày đăng.
- "trich_dan": chép NGUYÊN VĂN một đoạn ngắn (dưới 25 từ) trong bài đã khiến bạn
  kết luận như vậy. Chép sai một chữ cũng bị loại. Không được tự viết lại.
- "do_tin_cay": 0.0-1.0. Không chắc thì để thấp, đừng đoán bừa.
- Không có sự kiện thì trả {{"co_su_kien": false}} và bỏ trống các trường khác.

Ngày đăng: {ngay_dang}
Nội dung bài:
---
{noi_dung}
---"""

PROMPT_GOM = """Dưới đây là các tên sự kiện đã trích được từ bài đăng của CLB Tin học HIT.
Nhiều tên là CÙNG MỘT chuỗi sự kiện viết khác nhau. Hãy gom chúng lại.

Trả về DUY NHẤT một mảng JSON, không giải thích, không bọc markdown:
[{{"ten_chuan": "...", "cac_ten_goc": ["...", "..."]}}]

Quy tắc bắt buộc:
- Mỗi tên gốc phải xuất hiện ở ĐÚNG MỘT nhóm. Không bỏ sót tên nào.
- "cac_ten_goc" phải chép chính xác chuỗi trong danh sách đầu vào.
- "ten_chuan": cách gọi tự nhiên nhất, viết hoa đúng chuẩn tiếng Việt.
- CHỈ gom khi chắc chắn là cùng một chuỗi hoạt động. Hai hoạt động khác nhau mà
  gộp lại thì bộ lọc sẽ trả về ảnh của sự kiện khác — tệ hơn là không gom.
- Không thêm tên mới không có trong danh sách.

Danh sách (kèm số bài dùng tên đó):
{danh_sach}"""


# ── Gọi LLM ──────────────────────────────────────────────────────────────────


def tao_llm(config: AppConfig, timeout: float | None = None):
    from langchain.chat_models import init_chat_model

    llm_config = config.llm
    kwargs: dict[str, Any] = {
        "temperature": 0,
        "timeout": timeout or llm_config.timeout,
        "max_retries": 2,
    }
    if llm_config.base_url:
        kwargs["base_url"] = llm_config.base_url
    if llm_config.api_key:
        kwargs["api_key"] = llm_config.api_key
    return init_chat_model(llm_config.model_name, **kwargs)


def doc_json(raw: str) -> Any:
    """Gỡ rào ```json mà mọi model đều thỉnh thoảng thêm vào."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0]
    return json.loads(text.strip())


# ── Lượt 1: trích từng bài ───────────────────────────────────────────────────


def chuan_hoa_de_so_khop(text: str) -> str:
    """Bỏ dấu, bỏ mọi thứ không phải chữ/số — để so trích dẫn với nội dung bài.

    So thô sẽ hỏng vì emoji, dấu cách lạ và xuống dòng: LLM chép đúng chữ nhưng
    `"🔥 [HIT CONTEST]"` với `"[HIT CONTEST]"` vẫn khác nhau về byte.
    """
    khong_dau = normalize_alias(text)
    return "".join(c for c in khong_dau if c.isalnum())


def trich_mot_bai(llm, post_id: str, noi_dung: str, ngay_dang: str) -> dict[str, Any]:
    prompt = PROMPT_TRICH.format(ngay_dang=ngay_dang, noi_dung=noi_dung[:MAX_CONTENT_CHARS])
    ket_qua = doc_json(llm.invoke(prompt).content)
    if not isinstance(ket_qua, dict):
        raise ValueError(f"LLM trả về {type(ket_qua).__name__}, cần object")
    ket_qua["post_id"] = post_id
    return ket_qua


def hop_le(ban_ghi: dict[str, Any], noi_dung: str, min_confidence: float) -> tuple[bool, str]:
    """Kiểm tra một kết quả trích trước khi cho nó vào DB.

    Trả về (nhận, lý do loại). Cửa quan trọng nhất là trích dẫn: nó là bằng chứng
    duy nhất cho thấy LLM đọc bài chứ không nhớ ra một sự kiện quen thuộc.
    """
    if not ban_ghi.get("co_su_kien"):
        return False, "khong_co_su_kien"
    ten = (ban_ghi.get("ten_su_kien") or "").strip()
    if not ten:
        return False, "thieu_ten"
    if not slugify(ten):
        return False, "ten_khong_tao_duoc_slug"
    try:
        do_tin_cay = float(ban_ghi.get("do_tin_cay") or 0.0)
    except (TypeError, ValueError):
        return False, "do_tin_cay_khong_doc_duoc"
    if do_tin_cay < min_confidence:
        return False, "do_tin_cay_thap"
    trich = (ban_ghi.get("trich_dan") or "").strip()
    if not trich:
        return False, "thieu_trich_dan"
    if chuan_hoa_de_so_khop(trich) not in chuan_hoa_de_so_khop(noi_dung):
        return False, "trich_dan_khong_co_trong_bai"
    return True, ""


# ── Lượt 2: gom tên về chuỗi ─────────────────────────────────────────────────


def gom_mot_lo(llm, dem_ten: Counter, ten_can_gom: list[str]) -> list[dict[str, Any]]:
    """Một lời gọi LLM cho một lô tên. Tên bị bỏ sót được vá lại thành chuỗi riêng."""
    danh_sach = "\n".join(f"- {ten} ({dem_ten[ten]} bài)" for ten in ten_can_gom)
    cum = doc_json(llm.invoke(PROMPT_GOM.format(danh_sach=danh_sach)).content)
    if not isinstance(cum, list):
        raise ValueError("Lượt gom phải trả về một mảng JSON")

    con_lai = set(ten_can_gom)
    ket_qua: list[dict[str, Any]] = []
    for nhom in cum:
        if not isinstance(nhom, dict):
            continue
        goc = [t for t in (nhom.get("cac_ten_goc") or []) if t in con_lai]
        if not goc:
            continue
        con_lai.difference_update(goc)
        ket_qua.append({"ten_chuan": (nhom.get("ten_chuan") or "").strip() or goc[0], "cac_ten_goc": goc})

    # Giữ thứ tự đầu vào cho phần LLM bỏ sót: chạy lại phải ra cùng kết quả.
    ket_qua.extend({"ten_chuan": ten, "cac_ten_goc": [ten]} for ten in ten_can_gom if ten in con_lai)
    return ket_qua


def doc_cache_gom(duong_dan: Path, dem_ten: Counter) -> list[dict[str, Any]]:
    """Dùng lại kết quả gom cũ, nhưng CHỈ KHI nó phủ hết các tên hiện có.

    Lượt gom tốn vài phút và cả chục lời gọi LLM, nên chạy `--apply` sau một lượt
    `--extract-only` không được bắt làm lại từ đầu. Nhưng dùng lại một cách gom
    thiếu tên còn tệ hơn làm lại: những tên vắng mặt sẽ lặng lẽ thành chuỗi
    riêng, đúng thứ lượt gom sinh ra để tránh.
    """
    if not duong_dan.exists():
        return []
    try:
        cum = json.loads(duong_dan.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    da_co = {goc for nhom in cum for goc in nhom.get("cac_ten_goc", [])}
    thieu = set(dem_ten) - da_co
    if thieu:
        print(f"Cache gom thiếu {len(thieu)} tên mới, sẽ gom lại.", flush=True)
        return []
    print(f"Dùng lại kết quả gom đã có: {duong_dan}")
    return cum


def gom_ten(llm, dem_ten: Counter, kich_thuoc_lo: int = KICH_THUOC_LO_GOM) -> list[dict[str, Any]]:
    """Gom các cách viết về một chuỗi, chia lô rồi gom lại lần nữa.

    Gửi cả 142 tên trong một lời gọi đã ĐO ĐƯỢC là timeout (07/08/2026): câu trả
    lời dài vài nghìn token, và proxy `cx/gpt-5.5` mất hơn 30 giây chỉ cho một
    câu ngắn. Chia lô làm mỗi lời gọi nhỏ lại, nhưng sinh ra vấn đề mới — hai
    cách viết của cùng một sự kiện có thể rơi vào hai lô khác nhau. Hai thứ chữa
    nó: **sắp xếp theo tên đã chuẩn hoá** ("team building", "team building with
    hit", "teambuilding" nằm cạnh nhau), và **một lượt gom thứ hai** trên các
    tên chuẩn của lượt đầu, nơi số lượng đã nhỏ hơn nhiều.
    """
    if not dem_ten:
        return []
    theo_thu_tu = sorted(dem_ten, key=normalize_alias)
    if len(theo_thu_tu) <= kich_thuoc_lo:
        return gom_mot_lo(llm, dem_ten, theo_thu_tu)

    lo = [theo_thu_tu[i : i + kich_thuoc_lo] for i in range(0, len(theo_thu_tu), kich_thuoc_lo)]
    print(f"  chia {len(theo_thu_tu)} tên thành {len(lo)} lô...", flush=True)
    vong_dau: list[dict[str, Any]] = []
    for i, phan in enumerate(lo, start=1):
        vong_dau.extend(gom_mot_lo(llm, dem_ten, phan))
        print(f"  ... lô {i}/{len(lo)}", flush=True)

    # Lượt hai: gom chính các tên chuẩn vừa sinh ra, rồi trải ngược về tên gốc.
    theo_ten_chuan: dict[str, list[str]] = {}
    for nhom in vong_dau:
        theo_ten_chuan.setdefault(nhom["ten_chuan"], []).extend(nhom["cac_ten_goc"])
    if len(theo_ten_chuan) <= 1:
        return [{"ten_chuan": k, "cac_ten_goc": v} for k, v in theo_ten_chuan.items()]

    # ĐÚNG MỘT lời gọi, không chia lô và không đệ quy: nhiệm vụ của lượt này là
    # nhìn thấy toàn cảnh, chia lô lại thì hai biến thể vẫn có thể lạc nhau lần
    # nữa. Danh sách ở đây đã ngắn hơn lượt đầu, và LLM của lượt gom chạy với
    # hạn giờ riêng nên một lời gọi dài không còn là vấn đề.
    dem_chuan = Counter({ten: sum(dem_ten[g] for g in goc) for ten, goc in theo_ten_chuan.items()})
    print(f"  gom lần hai trên {len(dem_chuan)} tên chuẩn...", flush=True)
    vong_hai = gom_mot_lo(llm, dem_chuan, sorted(dem_chuan, key=normalize_alias))

    return [
        {
            "ten_chuan": nhom["ten_chuan"],
            "cac_ten_goc": [goc for ten in nhom["cac_ten_goc"] for goc in theo_ten_chuan[ten]],
        }
        for nhom in vong_hai
    ]


# ── Đọc bài từ PostgreSQL ────────────────────────────────────────────────────


def doc_bai(limit: int | None, redo: bool) -> tuple[list[dict[str, str]], int]:
    from sqlalchemy import select

    with RepositoryUnitOfWork() as uow:
        rows = uow.session.execute(
            select(PostModel.post_id, PostModel.content, PostModel.created_time).order_by(
                PostModel.created_time
            )
        ).all()
        da_gan = set() if redo else uow.events.linked_post_ids()

    bai = [
        {
            "post_id": str(pid),
            "noi_dung": content,
            "ngay_dang": created.date().isoformat() if created else "không rõ",
        }
        for pid, content, created in rows
        if content and content.strip() and pid not in da_gan
    ]
    return (bai[:limit] if limit else bai), len(rows)


# ── Ghi vào bốn bảng ─────────────────────────────────────────────────────────


def ghi_db(nhan: list[dict[str, Any]], cum: list[dict[str, Any]], redo: bool) -> dict[str, int]:
    """Ghi series/occurrence/alias/liên kết. Một transaction cho cả mẻ."""
    ten_goc_toi_chuan = {
        goc: nhom["ten_chuan"] for nhom in cum for goc in nhom["cac_ten_goc"]
    }
    thong_ke = Counter()

    with RepositoryUnitOfWork() as uow:
        events = uow.events
        for ban_ghi in nhan:
            ten_goc = ban_ghi["ten_su_kien"].strip()
            ten_chuan = ten_goc_toi_chuan.get(ten_goc, ten_goc)
            slug = slugify(ten_chuan)
            if not slug:
                thong_ke["bo_qua_slug_rong"] += 1
                continue

            # Lượt gom chỉ nhìn thấy các tên của MẺ NÀY. Chạy tăng tiến mà bỏ qua
            # alias đã có thì "HIT Contest Series" của mẻ trước và "Contest
            # Series" của mẻ này thành hai chuỗi — đúng cái lượt gom sinh ra để
            # tránh. Tra alias trước, chỉ tạo series mới khi thật sự chưa từng thấy.
            da_biet = events.series_by_alias(ten_chuan) or events.series_by_alias(ten_goc)
            if da_biet is not None:
                series = da_biet
                thong_ke["nhap_vao_series_da_co"] += 1
            else:
                series = events.upsert_series(slug, ten_chuan)
            # Cả tên chuẩn lẫn cách viết trong bài đều thành alias: lần chạy sau
            # tra ra series cũ thay vì đẻ thêm một chuỗi trùng nghĩa.
            for alias in {ten_chuan, ten_goc}:
                if events.add_alias(alias, series_id=series.series_id):
                    thong_ke["alias"] += 1
                else:
                    thong_ke["alias_xung_dot"] += 1

            nhan_ky = (ban_ghi.get("nhan_ky") or "").strip()
            nam = ban_ghi.get("nam")
            nam = int(nam) if isinstance(nam, (int, float)) and 2000 <= int(nam) <= 2200 else None
            # Không có nhãn kỳ thì lấy năm đăng làm nhãn: mỗi kỳ vẫn tách được,
            # và `(series, label)` vẫn ổn định qua các lần chạy lại.
            label = nhan_ky or (str(nam) if nam else ban_ghi["ngay_dang"][:4])
            ky = events.upsert_occurrence(
                series.series_id,
                label,
                display_name=f"{ten_chuan} {label}".strip(),
                event_year=nam,
            )

            if redo:
                events.unlink_post(ban_ghi["post_id"])
            events.link_post(
                ban_ghi["post_id"],
                ky.occurrence_id,
                confidence=float(ban_ghi.get("do_tin_cay") or 0.0),
                assigned_by="llm",
                evidence={"trich_dan": ban_ghi.get("trich_dan"), "ten_goc": ten_goc},
                is_primary=True,
            )
            thong_ke["lien_ket"] += 1

        thong_ke.update(uow.events.counts())
    return dict(thong_ke)


# ── Điều phối ────────────────────────────────────────────────────────────────


def chay_luot_trich(llm, bai: list[dict[str, str]], cache: dict[str, Any], workers: int, max_loi: int) -> int:
    """Gọi LLM cho các bài chưa có trong cache. Trả về số bài lỗi."""
    con_thieu = [b for b in bai if b["post_id"] not in cache]
    if not con_thieu:
        print(f"Cache đã đủ {len(bai)} bài, không gọi LLM.")
        return 0

    print(f"Lượt 1: gọi LLM cho {len(con_thieu)}/{len(bai)} bài ({workers} luồng)...", flush=True)
    khoa = threading.Lock()
    trang_thai = {"xong": 0, "loi": 0, "loi_lien_tiep": 0}

    def lam(b: dict[str, str]):
        return trich_mot_bai(llm, b["post_id"], b["noi_dung"], b["ngay_dang"])

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(lam, b): b for b in con_thieu}
        for fut in as_completed(futures):
            b = futures[fut]
            try:
                cache[b["post_id"]] = fut.result()
                with khoa:
                    trang_thai["loi_lien_tiep"] = 0
            except Exception as exc:
                with khoa:
                    trang_thai["loi"] += 1
                    trang_thai["loi_lien_tiep"] += 1
                print(f"  LỖI bài {b['post_id']}: {exc.__class__.__name__}: {exc}", flush=True)
                if trang_thai["loi_lien_tiep"] >= max_loi:
                    print(f"\nDỪNG: {max_loi} lỗi liên tiếp. Kiểm tra endpoint/credit LLM.", flush=True)
                    for khac in futures:
                        khac.cancel()
                    break
            with khoa:
                trang_thai["xong"] += 1
                if trang_thai["xong"] % PROGRESS_EVERY == 0:
                    print(f"  ... {trang_thai['xong']}/{len(con_thieu)} bài", flush=True)
    return trang_thai["loi"]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--limit", type=int, default=None, help="Chỉ xử lý N bài đầu tiên.")
    p.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help=f"Số luồng. Mặc định: {DEFAULT_WORKERS}")
    p.add_argument("--redo", action="store_true", help="Làm lại cả bài đã gắn sự kiện.")
    p.add_argument("--refresh-cache", action="store_true", help="Bỏ cache lượt 1, gọi LLM lại từ đầu.")
    p.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    p.add_argument("--min-confidence", type=float, default=DEFAULT_MIN_CONFIDENCE)
    p.add_argument("--max-consecutive-errors", type=int, default=DEFAULT_MAX_CONSECUTIVE_ERRORS)
    p.add_argument("--extract-only", action="store_true", help="Gọi LLM và ghi cache, KHÔNG ghi DB.")
    p.add_argument("--apply", action="store_true", help="Gọi LLM và ghi vào PostgreSQL.")
    args = p.parse_args()

    bai, tong_bai = doc_bai(args.limit, args.redo)
    goi_llm = args.apply or args.extract_only

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    duong_cache = args.cache_dir / "pass1_extractions.json"
    cache: dict[str, Any] = {}
    if duong_cache.exists() and not args.refresh_cache:
        cache = json.loads(duong_cache.read_text(encoding="utf-8"))

    print(json.dumps({
        "tong_bai_trong_db": tong_bai,
        "bai_se_xu_ly": len(bai),
        "da_co_trong_cache": sum(1 for b in bai if b["post_id"] in cache),
        "min_confidence": args.min_confidence,
        "goi_llm": goi_llm,
        "ghi_db": args.apply,
    }, ensure_ascii=False, indent=2))
    if not bai:
        print("Không có bài nào cần xử lý.")
        return 0

    so_loi = 0
    if goi_llm:
        llm = tao_llm(AppConfig())
        so_loi = chay_luot_trich(llm, bai, cache, args.workers, args.max_consecutive_errors)
        duong_cache.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Đã lưu cache lượt 1: {duong_cache}")

    # Lọc: mỗi bài phải qua cửa trích dẫn + độ tin cậy mới được vào DB.
    noi_dung_theo_id = {b["post_id"]: b for b in bai}
    nhan, ly_do = [], Counter()
    for post_id, ban_ghi in cache.items():
        b = noi_dung_theo_id.get(post_id)
        if b is None:
            continue
        ok, vi_sao = hop_le(ban_ghi, b["noi_dung"], args.min_confidence)
        if ok:
            nhan.append({**ban_ghi, "ngay_dang": b["ngay_dang"]})
        else:
            ly_do[vi_sao] += 1

    dem_ten = Counter(b["ten_su_kien"].strip() for b in nhan)
    print(json.dumps({
        "bai_co_su_kien": len(nhan),
        "bai_bi_loai": dict(ly_do),
        "so_ten_rieng_biet": len(dem_ten),
        "loi_goi_llm": so_loi,
    }, ensure_ascii=False, indent=2))

    if not nhan:
        print("Chưa có bài nào qua cửa lọc.")
        return 0

    duong_cum = args.cache_dir / "pass2_clusters.json"
    cum = doc_cache_gom(duong_cum, dem_ten) if not args.refresh_cache else []
    if not cum and goi_llm:
        print(f"\nLượt 2: gom {len(dem_ten)} tên về các chuỗi sự kiện...", flush=True)
        cum = gom_ten(tao_llm(AppConfig(), timeout=TIMEOUT_GOM_GIAY), dem_ten)
        duong_cum.write_text(json.dumps(cum, ensure_ascii=False, indent=2), encoding="utf-8")

    gop_lai = [n for n in cum if len(n["cac_ten_goc"]) > 1]
    print(json.dumps({
        "so_chuoi_su_kien": len(cum),
        "so_chuoi_gom_tu_nhieu_ten": len(gop_lai),
        "vi_du_gom": [
            {"ten_chuan": n["ten_chuan"], "cac_ten_goc": n["cac_ten_goc"]}
            for n in sorted(gop_lai, key=lambda n: -len(n["cac_ten_goc"]))[:5]
        ],
    }, ensure_ascii=False, indent=2))

    if not args.apply:
        print("\nDry-run: chưa ghi PostgreSQL. Thêm --apply để thực hiện.")
        return 0

    print(json.dumps(ghi_db(nhan, cum, args.redo), ensure_ascii=False, indent=2))
    print(
        "\nĐã ghi DB, nhưng payload Qdrant CHƯA đổi — `event_key` chỉ tới được bộ\n"
        "lọc sau khi chạy backfill cho từng collection đang có:\n"
        "  python scripts/backfill_media_text_post_metadata.py --collection media_clip --apply\n"
        "  python scripts/backfill_media_text_post_metadata.py --collection video_transcript --apply"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
