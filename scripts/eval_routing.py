#!/usr/bin/env python3
"""Đo tỉ lệ định tuyến đúng của router — US-606.1 / BR-507 / TC-507.

US-606.1 đòi báo cáo "% định tuyến đúng", và đó là con số duy nhất chưa ai đo.
Thiếu nó thì tranh luận "có nên giữ nút chọn chế độ thủ công không" chỉ dựa vào
cảm giác: router đúng ~95% thì nút là phao dự phòng hiếm dùng, còn ~70% thì nó
phải nằm ngay trước mắt.

**Nhãn không phải do người chạy tự nghĩ ra.** `data/eval/eval_queries.yaml` đã
mang sẵn trường `loai` cho từng câu hỏi, gán từ trước và vì mục đích khác (đo
Recall của truy xuất). Đích định tuyến suy ra từ đó theo một luật cố định — xem
DICH_MONG_DOI. Tự gán nhãn định tuyến lúc này là tự viết đáp án sau khi đã biết
hệ thống trả lời thế nào.

**Đo ở tầng tool call, không đo ở câu chữ.** Router "đúng" nghĩa là nó gọi đúng
công cụ; đọc câu trả lời rồi đoán xem nó đã tra nguồn nào là thêm một tầng suy
diễn có thể sai. Tool call là quan sát trực tiếp.

Chỉ ĐỌC: script không ghi DB, không sửa Qdrant. Nhưng nó gọi LLM thật (mỗi câu
hỏi một lượt agent), nên `--gioi-han` có mặt để chạy thử vài câu trước.

    python scripts/eval_routing.py --gioi-han 5     # thử nhanh
    python scripts/eval_routing.py                  # chạy cả 50 câu
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import yaml

GOC = Path(__file__).resolve().parent.parent
DUONG_EVAL = GOC / "data" / "eval" / "eval_queries.yaml"
DUONG_KET_QUA = GOC / "data" / "eval" / "results"

TOOL_MEDIA = "search_media"
TOOL_NOI_QUY = "search_regulations"
TOOL_TRUY_XUAT = {TOOL_MEDIA, TOOL_NOI_QUY}

# `loai` trong bộ eval → tool router LẼ RA phải gọi.
#
# `ngoai_mien` kỳ vọng KHÔNG gọi tool nào: BR-704 nói truy vấn ngoài miền phải
# nhận thông báo giới hạn phạm vi. Tra kho rồi mới từ chối vẫn ra câu trả lời
# đúng, nhưng đó là định tuyến sai — và nó tốn một lượt nhúng cho mỗi câu lạc đề.
DICH_MONG_DOI: dict[str, set[str]] = {
    "media_image": {TOOL_MEDIA},
    "media_image_ocr": {TOOL_MEDIA},
    "media_transcript": {TOOL_MEDIA},
    "regulation": {TOOL_NOI_QUY},
    "ngoai_mien": set(),
}


def doc_bo_eval(gioi_han: int | None) -> list[dict]:
    du_lieu = yaml.safe_load(DUONG_EVAL.read_text(encoding="utf-8"))
    muc = [m for m in du_lieu["truy_van"] if m.get("loai") in DICH_MONG_DOI]
    bo_qua = len(du_lieu["truy_van"]) - len(muc)
    if bo_qua:
        print(f"CẢNH BÁO: bỏ {bo_qua} mục có `loai` không nằm trong DICH_MONG_DOI")
    return muc[:gioi_han] if gioi_han else muc


def hoi_agent(client: httpx.Client, url_lg: str, cau_hoi: str) -> tuple[list[str], str | None]:
    """Chạy một câu qua agent, trả về (tên các tool đã gọi, lỗi nếu có)."""
    try:
        thread = client.post(f"{url_lg}/threads", json={}).raise_for_status().json()
        res = client.post(
            f"{url_lg}/threads/{thread['thread_id']}/runs/stream",
            json={
                "assistant_id": "agent",
                "input": {"messages": [{"role": "human", "content": cau_hoi}]},
                "stream_mode": ["messages"],
            },
        )
        res.raise_for_status()
    except Exception as exc:
        return [], f"{type(exc).__name__}: {exc}"

    # SSE: dòng kết thúc bằng \r, không strip thì tên sự kiện không khớp gì cả.
    su_kien = ""
    da_goi: list[str] = []
    for dong in res.text.splitlines():
        dong = dong.strip()
        if dong.startswith("event:"):
            su_kien = dong[6:].strip()
        elif dong.startswith("data:") and su_kien == "messages/complete":
            try:
                goi = json.loads(dong[5:].strip())
            except json.JSONDecodeError:
                continue
            for tin in goi if isinstance(goi, list) else [goi]:
                if isinstance(tin, dict):
                    for tc in tin.get("tool_calls") or []:
                        da_goi.append(tc.get("name", "?"))
    return da_goi, None


def cham_mot_muc(muc: dict, da_goi: list[str], loi: str | None) -> dict:
    """Chấm một câu.

    **`dung` một mình là chỉ số nói dối.** Nó chỉ hỏi "có gọi đúng tool ít nhất
    một lần không", nên một lượt gọi 1 lần và một lượt gọi 9 lần (trong đó 4 lần
    vào kho không thể trả lời) chấm điểm y hệt nhau. Đo thật ngày 08/08/2026 trên
    câu "chủ nhiệm năm nay là ai": 9 lời gọi, 4 lần `search_regulations` vào kho
    nội quy 4 đoạn chỉ nói về sử dụng phòng.
    Nên phải đếm cả **số lời gọi** và **số lời gọi phí**, xem `bao_cao`.
    """
    mong_doi = DICH_MONG_DOI[muc["loai"]]
    goi_truy_xuat = [t for t in da_goi if t in TOOL_TRUY_XUAT]
    thuc_te = set(goi_truy_xuat)
    dung = (thuc_te >= mong_doi and bool(mong_doi)) or (not mong_doi and not thuc_te)
    return {
        "id": muc["id"],
        "loai": muc["loai"],
        "cau_hoi": muc["cau_hoi"],
        "mong_doi": sorted(mong_doi),
        "thuc_te": sorted(thuc_te),
        "loi": loi,
        # Đúng = gọi đủ tool cần. Câu ngoài miền thì "đủ" nghĩa là không gọi gì.
        "dung": dung,
        "so_lan_goi": len(goi_truy_xuat),
        "chuoi_goi": goi_truy_xuat,
        # Lời gọi phí = gọi vào tool KHÔNG nằm trong kỳ vọng, cộng với mọi lần
        # gọi lại cùng một tool. Cả hai đều là một lượt nhúng bỏ đi.
        "goi_phi": len(goi_truy_xuat) - len(thuc_te & mong_doi),
        # Gọn = đúng tool, mỗi tool đúng một lần, không dư gì.
        "gon": dung and goi_truy_xuat == sorted(mong_doi),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--langgraph-url", default="http://localhost:2024")
    parser.add_argument("--gioi-han", type=int, default=None, help="chỉ chạy N câu đầu")
    parser.add_argument("--song-song", type=int, default=4, help="số câu chạy đồng thời")
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--ghi", action="store_true", help="ghi báo cáo JSON vào data/eval/results/")
    args = parser.parse_args()

    muc_can_do = doc_bo_eval(args.gioi_han)
    print(f"Đo định tuyến trên {len(muc_can_do)} câu, {args.song_song} luồng song song.\n")

    with httpx.Client(timeout=args.timeout) as client:
        def chay(muc: dict) -> dict:
            da_goi, loi = hoi_agent(client, args.langgraph_url, muc["cau_hoi"])
            ket_qua = cham_mot_muc(muc, da_goi, loi)
            dau = "OK " if ket_qua["dung"] else "SAI"
            if loi:
                dau = "LỖI"
            print(
                f"  {dau} {ket_qua['id']:<8} {ket_qua['loai']:<17} "
                f"mong={ket_qua['mong_doi'] or '(không tra)'} "
                f"thật={ket_qua['thuc_te'] or '(không tra)'}"
            )
            return ket_qua

        with ThreadPoolExecutor(max_workers=args.song_song) as pool:
            ket_qua = list(pool.map(chay, muc_can_do))

    return bao_cao(ket_qua, ghi=args.ghi)


def bao_cao(ket_qua: list[dict], *, ghi: bool) -> int:
    loi = [k for k in ket_qua if k["loi"]]
    do_duoc = [k for k in ket_qua if not k["loi"]]
    if not do_duoc:
        print("\nKhông câu nào chạy được — không có số nào để báo cáo.")
        return 1

    print("\n" + "=" * 72)
    print("TỈ LỆ ĐỊNH TUYẾN ĐÚNG THEO NHÓM (US-606.1)")
    print("=" * 72)
    theo_loai: dict[str, list[dict]] = {}
    for k in do_duoc:
        theo_loai.setdefault(k["loai"], []).append(k)
    for loai in sorted(theo_loai):
        nhom = theo_loai[loai]
        dung = sum(1 for k in nhom if k["dung"])
        print(f"  {loai:<18} {dung}/{len(nhom)}  ({dung / len(nhom):.0%})")

    tong_dung = sum(1 for k in do_duoc if k["dung"])
    tong_gon = sum(1 for k in do_duoc if k["gon"])
    print("-" * 72)
    print(f"  {'TỔNG (đúng)':<18} {tong_dung}/{len(do_duoc)}  ({tong_dung / len(do_duoc):.0%})")
    print(f"  {'TỔNG (gọn)':<18} {tong_gon}/{len(do_duoc)}  ({tong_gon / len(do_duoc):.0%})"
          "   ← đúng tool, mỗi tool đúng một lần")

    # Số lời gọi mới là thứ phơi ra chuyện router mò mẫm. "Đúng" mà gọi 9 lần
    # thì vẫn là 9 lượt nhúng cho một câu hỏi.
    goi = sorted(k["so_lan_goi"] for k in do_duoc)
    tong_goi = sum(goi)
    giua = goi[len(goi) // 2]
    phi = sum(k["goi_phi"] for k in do_duoc)
    print("\n" + "=" * 72)
    print("SỐ LỜI GỌI TOOL MỖI LƯỢT")
    print("=" * 72)
    print(f"  tổng {tong_goi} lời gọi cho {len(do_duoc)} câu — trung vị {giua}, cao nhất {goi[-1]}")
    print(f"  lời gọi phí (sai tool, hoặc gọi lại cùng tool): {phi}/{tong_goi} ({phi / max(1, tong_goi):.0%})")
    phan_bo = Counter(goi)
    print("  phân bố: " + "  ".join(f"{n} lần×{c} câu" for n, c in sorted(phan_bo.items())))

    nhieu = [k for k in do_duoc if k["so_lan_goi"] >= 3]
    if nhieu:
        print(f"\n  {len(nhieu)} câu gọi từ 3 lời gọi trở lên:")
        for k in sorted(nhieu, key=lambda x: -x["so_lan_goi"])[:8]:
            print(f"    {k['so_lan_goi']:2d}× {k['id']:<8} [{k['loai']}] {k['cau_hoi'][:52]}")
            print(f"        {' → '.join(k['chuoi_goi'])}")

    sai = [k for k in do_duoc if not k["dung"]]
    if sai:
        print(f"\n{len(sai)} câu định tuyến SAI:")
        for k in sai:
            print(f"  {k['id']:<8} [{k['loai']}] {k['cau_hoi']}")
            print(f"           mong={k['mong_doi'] or '(không tra)'} thật={k['thuc_te'] or '(không tra)'}")
    if loi:
        print(f"\n{len(loi)} câu KHÔNG chạy được (loại khỏi phép tính — US-601.1 AC-2):")
        for k in loi:
            print(f"  {k['id']:<8} {k['loi']}")

    if ghi:
        DUONG_KET_QUA.mkdir(parents=True, exist_ok=True)
        dich = DUONG_KET_QUA / "routing_accuracy.json"
        dich.write_text(
            json.dumps(
                {
                    "tong_so_do_duoc": len(do_duoc),
                    "so_dung": tong_dung,
                    "ti_le_dung": round(tong_dung / len(do_duoc), 4),
                    "so_gon": tong_gon,
                    "ti_le_gon": round(tong_gon / len(do_duoc), 4),
                    "tong_loi_goi": tong_goi,
                    "loi_goi_phi": phi,
                    "loi_goi_trung_vi": giua,
                    "loi_goi_cao_nhat": goi[-1],
                    "theo_loai": {
                        loai: {
                            "tong": len(nhom),
                            "dung": sum(1 for k in nhom if k["dung"]),
                        }
                        for loai, nhom in theo_loai.items()
                    },
                    "chi_tiet": ket_qua,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\nĐã ghi {dich}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
