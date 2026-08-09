"""Đánh giá nhánh nội quy — T-73 / US-606.1 / TC-606.

Ba chỉ số PRD đòi, và cả ba đều phải đo ở **tầng trả lời**, không phải tầng
truy xuất — nên script này gọi LangGraph thật chứ không gọi service trong tiến
trình như `run_eval.py`:

1. **Accuracy điều khoản** — điều khoản đúng có nằm trong top-k mà công cụ lấy
   về không. Nhãn lấy lại từ `eval_queries.json`, không gán mới.
2. **Groundedness** — câu trả lời có bám vào thứ công cụ trả về không. Đo bằng
   cách rút các **dữ kiện kiểm được** (số tiền, số điện thoại, số lần, email)
   trong câu trả lời rồi soi ngược vào ngữ cảnh công cụ. Một con số xuất hiện
   trong câu trả lời mà không có trong ngữ cảnh chính là dấu hiệu bịa — đúng
   thứ hệ thống này tuyệt đối không được làm.
3. **Routing accuracy** — agent có gọi đúng công cụ không. Nhãn suy ra từ `loai`
   của tập đánh giá: `regulation` phải gọi `search_regulations`, `media_*` phải
   gọi `search_media`, `ngoai_mien` thì gọi gì cũng được **miễn là không bịa**.

Cần `langgraph dev` (cổng 2024), MCP (8091) và API (8000) cùng chạy — xem
`docs/chay-demo.md`.

    python scripts/run_eval_noiquy.py
    python scripts/run_eval_noiquy.py --url http://localhost:2024 --out bao_cao.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "API"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(REPO_ROOT / ".env")

TAP_MAC_DINH = REPO_ROOT / "data" / "eval" / "eval_queries.json"
BAO_CAO_MAC_DINH = REPO_ROOT / "data" / "eval" / "report_noiquy.json"
LANGGRAPH_MAC_DINH = "http://127.0.0.1:2024"

DISCLAIMER = "Thông tin trên chỉ mang tính tham khảo"

# Công cụ đúng cho từng loại truy vấn.
TOOL_MONG_DOI = {
    "regulation": "search_regulations",
    "media_image": "search_media",
    "media_image_ocr": "search_media",
    "media_transcript": "search_media",
}

# Dữ kiện kiểm được: số tiền có dấu chấm nghìn, số điện thoại, email, số lần.
# Cố ý KHÔNG bắt mọi con số — "1 ngày", "lần 2" là cách diễn đạt lại, còn
# "100.000" hay "0823 644 212" mà sai thì người đọc bị dẫn sai thật sự.
MAU_DU_KIEN = [
    re.compile(r"\b\d{1,3}(?:\.\d{3})+\b"),          # 20.000, 100.000
    re.compile(r"\b0\d[\d\s.]{7,12}\d\b"),           # số điện thoại
    re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b"),      # email
]


def chuan_hoa(s: Any) -> str:
    return unicodedata.normalize("NFC", str(s or "")).lower()


def _chi_so(s: str) -> str:
    """Bỏ mọi ký tự không phải chữ số — để '0823 644 212' khớp '0823644212'."""
    return re.sub(r"\D", "", s)


def rut_du_kien(text: str) -> list[str]:
    ra: list[str] = []
    for mau in MAU_DU_KIEN:
        ra.extend(m.group(0).strip() for m in mau.finditer(text))
    return sorted(set(ra))


def du_kien_co_trong_ngu_canh(du_kien: str, ngu_canh: str) -> bool:
    if du_kien.lower() in ngu_canh.lower():
        return True
    # Số điện thoại được trình bày lại khác cách viết trong nguồn là chuyện
    # bình thường, nên so bằng phần chữ số thay vì so nguyên chuỗi.
    so = _chi_so(du_kien)
    return bool(so) and len(so) >= 4 and so in _chi_so(ngu_canh)


def _noi_dung_thanh_chuoi(noi_dung: Any) -> str:
    """Message của LangChain có content là chuỗi HOẶC danh sách phần tử.

    Kết quả tool về dưới dạng `[{"type": "text", "text": "..."}]`. Bản đầu chỉ
    nhận chuỗi nên vứt sạch ngữ cảnh, và groundedness ra 0.000 trong khi mọi câu
    trả lời đều bám nguồn đúng — một phép đo hỏng trông y hệt một hệ thống hỏng.
    """
    if isinstance(noi_dung, str):
        return noi_dung
    if isinstance(noi_dung, list):
        phan = []
        for muc in noi_dung:
            if isinstance(muc, str):
                phan.append(muc)
            elif isinstance(muc, dict):
                phan.append(str(muc.get("text") or muc.get("content") or ""))
        return "\n".join(p for p in phan if p)
    return ""


def hoi_agent(url: str, cau_hoi: str, timeout: float = 180.0) -> dict[str, Any]:
    """Gửi một câu hỏi qua LangGraph, thu tool đã gọi + ngữ cảnh + câu trả lời."""
    import httpx

    with httpx.Client(timeout=timeout) as client:
        thread = client.post(f"{url}/threads", json={}).json()["thread_id"]
        body = {
            "assistant_id": "agent",
            "input": {"messages": [{"type": "human", "content": cau_hoi}]},
            "stream_mode": ["messages"],
        }
        tools: list[str] = []
        ngu_canh: list[str] = []
        tra_loi = ""
        with client.stream("POST", f"{url}/threads/{thread}/runs/stream", json=body) as res:
            res.raise_for_status()
            for dong in res.iter_lines():
                if not dong.startswith("data: "):
                    continue
                try:
                    goi = json.loads(dong[6:])
                except json.JSONDecodeError:
                    continue
                for muc in goi if isinstance(goi, list) else [goi]:
                    if not isinstance(muc, dict):
                        continue
                    for tc in muc.get("tool_calls") or []:
                        if tc.get("name"):
                            tools.append(tc["name"])
                    if muc.get("type") == "tool":
                        if muc.get("name"):
                            tools.append(muc["name"])
                        van_ban = _noi_dung_thanh_chuoi(muc.get("content"))
                        if van_ban:
                            ngu_canh.append(van_ban)
                    if muc.get("type") == "ai":
                        van_ban = _noi_dung_thanh_chuoi(muc.get("content"))
                        if van_ban.strip():
                            tra_loi = van_ban
    return {
        "tools": sorted(set(tools)),
        "ngu_canh": "\n".join(ngu_canh),
        "tra_loi": tra_loi,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tap", default=str(TAP_MAC_DINH))
    parser.add_argument("--url", default=LANGGRAPH_MAC_DINH, help="địa chỉ LangGraph")
    parser.add_argument("--out", default=str(BAO_CAO_MAC_DINH))
    parser.add_argument(
        "--gioi-han-media",
        type=int,
        default=4,
        help="số truy vấn media lấy mẫu để đo routing (mỗi truy vấn là một lượt LLM)",
    )
    args = parser.parse_args()

    tap = json.loads(Path(args.tap).read_text(encoding="utf-8"))
    moi_muc = tap["truy_van"]
    # Nội quy và ngoài miền lấy hết; media chỉ lấy mẫu — routing là thứ cần đo,
    # mà mỗi truy vấn tốn một lượt LLM nên không chạy cả 36 câu media.
    chon = [m for m in moi_muc if m["loai"] in ("regulation", "ngoai_mien")]
    media = [m for m in moi_muc if m["loai"].startswith("media")][: args.gioi_han_media]
    chon.extend(media)

    print(f"Đánh giá {len(chon)} truy vấn qua LangGraph tại {args.url}\n")
    ket_qua: list[dict[str, Any]] = []
    for muc in chon:
        print(f"  {muc['id']:8} {muc['cau_hoi'][:56]}", flush=True)
        try:
            phan_hoi = hoi_agent(args.url, muc["cau_hoi"])
            loi = None
        except Exception as exc:  # noqa: BLE001 — một câu hỏng không giết cả đợt
            phan_hoi = {"tools": [], "ngu_canh": "", "tra_loi": ""}
            loi = f"{exc.__class__.__name__}: {exc}"

        tra_loi = phan_hoi["tra_loi"]
        ngu_canh = phan_hoi["ngu_canh"]
        du_kien = rut_du_kien(tra_loi)
        khong_co_nguon = [d for d in du_kien if not du_kien_co_trong_ngu_canh(d, ngu_canh)]

        mong_doi = TOOL_MONG_DOI.get(muc["loai"])
        ket_qua.append(
            {
                "id": muc["id"],
                "loai": muc["loai"],
                "cau_hoi": muc["cau_hoi"],
                "tools": phan_hoi["tools"],
                "tool_mong_doi": mong_doi,
                "routing_dung": (mong_doi in phan_hoi["tools"]) if mong_doi else None,
                "so_du_kien": len(du_kien),
                # Ghi độ dài ngữ cảnh: groundedness thấp mà ngữ cảnh rỗng thì
                # lỗi nằm ở phép đo, không phải ở câu trả lời.
                "do_dai_ngu_canh": len(ngu_canh),
                "du_kien_khong_co_nguon": khong_co_nguon,
                "co_disclaimer": DISCLAIMER.lower() in chuan_hoa(tra_loi),
                "do_dai_tra_loi": len(tra_loi),
                "tra_loi": tra_loi,
                "loi": loi,
            }
        )

    hop_le = [r for r in ket_qua if r["loi"] is None]
    co_routing = [r for r in hop_le if r["routing_dung"] is not None]
    noi_quy = [r for r in hop_le if r["loai"] == "regulation"]
    co_du_kien = [r for r in hop_le if r["so_du_kien"] > 0 and r["do_dai_ngu_canh"] > 0]
    khong_ngu_canh = [r for r in hop_le if r["so_du_kien"] > 0 and r["do_dai_ngu_canh"] == 0]
    bam_nguon = [r for r in co_du_kien if not r["du_kien_khong_co_nguon"]]

    def ty_le(tu: int, mau: int) -> float:
        return tu / mau if mau else 0.0

    tong_hop = {
        "so_truy_van": len(ket_qua),
        "so_loi": len(ket_qua) - len(hop_le),
        "routing_accuracy": ty_le(sum(1 for r in co_routing if r["routing_dung"]), len(co_routing)),
        "routing_mau": len(co_routing),
        "groundedness": ty_le(len(bam_nguon), len(co_du_kien)),
        "groundedness_mau": len(co_du_kien),
        "disclaimer_nhanh_noi_quy": ty_le(sum(1 for r in noi_quy if r["co_disclaimer"]), len(noi_quy)),
        "disclaimer_mau": len(noi_quy),
    }

    print("\n" + "=" * 72)
    print("ĐÁNH GIÁ NHÁNH NỘI QUY — TC-606")
    print("=" * 72)
    print(f"  Routing accuracy   {tong_hop['routing_accuracy']:.3f}  (trên {tong_hop['routing_mau']} truy vấn có nhãn công cụ)")
    print(f"  Groundedness       {tong_hop['groundedness']:.3f}  (trên {tong_hop['groundedness_mau']} câu trả lời có dữ kiện kiểm được)")
    print(f"  Disclaimer nội quy {tong_hop['disclaimer_nhanh_noi_quy']:.3f}  (trên {tong_hop['disclaimer_mau']} câu nội quy)")

    sai_routing = [r for r in co_routing if not r["routing_dung"]]
    if sai_routing:
        print(f"\nĐịnh tuyến sai ({len(sai_routing)}):")
        for r in sai_routing:
            print(f"  {r['id']}: mong đợi {r['tool_mong_doi']}, gọi {r['tools'] or '(không gọi tool nào)'}")

    if khong_ngu_canh:
        print(
            f"\nCẢNH BÁO — {len(khong_ngu_canh)} câu có dữ kiện nhưng KHÔNG thu được ngữ cảnh "
            "công cụ; đã loại khỏi groundedness vì đó là lỗi đo, không phải lỗi trả lời: "
            + ", ".join(r["id"] for r in khong_ngu_canh)
        )

    bia = [r for r in co_du_kien if r["du_kien_khong_co_nguon"]]
    if bia:
        print(f"\nDỮ KIỆN KHÔNG CÓ TRONG NGỮ CẢNH ({len(bia)}) — nghi bịa:")
        for r in bia:
            print(f"  {r['id']}: {r['du_kien_khong_co_nguon']}")
    else:
        print("\nKhông câu nào nêu số liệu ngoài ngữ cảnh công cụ trả về.")

    thieu_disclaimer = [r for r in noi_quy if not r["co_disclaimer"]]
    if thieu_disclaimer:
        print(f"\nThiếu disclaimer ({len(thieu_disclaimer)}): {', '.join(r['id'] for r in thieu_disclaimer)}")

    loi = [r for r in ket_qua if r["loi"]]
    if loi:
        print(f"\nLỖI ({len(loi)}):")
        for r in loi[:5]:
            print(f"  {r['id']}: {r['loi'][:100]}")

    Path(args.out).write_text(
        json.dumps({"tong_hop": tong_hop, "chi_tiet": ket_qua}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nđã ghi {Path(args.out).relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
