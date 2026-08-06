"""Kiểm kê bộ dữ liệu thô trên MinIO — đọc thuần, không ghi gì.

Sinh lại đúng những con số trong `docs/data-audit.md` và file mốc hồi quy
`API/tests/fixtures/minio_dataset_census.txt`. Script này **không có --apply**
vì nó không sửa gì: chỉ `list_objects` và ranged GET.

    python scripts/audit_minio_dataset.py
    python scripts/audit_minio_dataset.py --deep          # đọc thêm header ảnh/video
    python scripts/audit_minio_dataset.py --census-out API/tests/fixtures/minio_dataset_census.txt

Hai bẫy đã dẫm, đừng dẫm lại:

- `posts.jsonl` phải tách bằng `split("\\n")`. Có một ký tự U+2028 trong
  `message`; `str.splitlines()` cắt luôn ở đó, tạo ra 516 mảnh và làm hỏng 2
  mảnh JSON.
- `width`/`height` trong `posts.jsonl` là số Facebook khai, không phải kích
  thước file đã tải. 77% lệch, và file thật thường **lớn hơn**. Cần số thật thì
  đọc header (`--deep`).
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import struct
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
API_ROOT = PROJECT_ROOT / "API"
sys.path.insert(0, str(API_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
load_dotenv(PROJECT_ROOT / ".env")

from minio_registration import (  # noqa: E402
    IMAGE_EXTENSIONS,
    VIDEO_EXTENSIONS,
    ObjectKeyError,
    event_id_from_object_key,
    normalize_prefix,
)

from src.rag_video_anh.pipeline.minio_storage import MinioStorage  # noqa: E402

DATASET_FILES = ("posts.jsonl", "manifest.json", "crawl.log")
# Ký tự mà str.splitlines() cắt còn "\n" thì không.
KY_TU_TACH_DONG_LA = "\v\f\x1c\x1d\x1e\x85  "


def doc_object(storage: MinioStorage, key: str, offset: int = 0, length: int | None = None) -> bytes:
    kwargs = {"offset": offset}
    if length is not None:
        kwargs["length"] = length
    response = storage.client.get_object(storage.bucket_name, key, **kwargs)
    try:
        return response.read()
    finally:
        response.close()
        response.release_conn()


def bam_sha256(storage: MinioStorage, key: str) -> str:
    """Băm nội dung thật của object, đọc theo khối để không nuốt 121 MB vào RAM.

    Cần thiết vì **ETag không đủ**: object tải lên nhiều phần mang ETag dạng
    `<md5>-<số phần>`, không phải MD5 của nội dung. Trên bộ này 41 video rơi
    vào diện đó, nên gom trùng bằng ETag sẽ bỏ sót đúng nhóm file lớn.
    """
    bam = hashlib.sha256()
    response = storage.client.get_object(storage.bucket_name, key)
    try:
        for khoi in response.stream(1024 * 1024):
            bam.update(khoi)
    finally:
        response.close()
        response.release_conn()
    return bam.hexdigest()


def kich_thuoc_jpeg(data: bytes) -> tuple[int, int] | None:
    i = 2
    while i < len(data) - 9:
        if data[i] != 0xFF:
            i += 1
            continue
        marker = data[i + 1]
        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
            i += 2
            continue
        seglen = struct.unpack(">H", data[i + 2 : i + 4])[0]
        if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
            cao, rong = struct.unpack(">HH", data[i + 5 : i + 9])
            return rong, cao
        i += 2 + seglen
    return None


def kich_thuoc_png(data: bytes) -> tuple[int, int] | None:
    if data[12:16] != b"IHDR":
        return None
    rong, cao = struct.unpack(">II", data[16:24])
    return rong, cao


def kich_thuoc_anh(storage: MinioStorage, key: str) -> tuple[int, int] | None:
    data = doc_object(storage, key, length=65536)
    return kich_thuoc_png(data) if key.lower().endswith(".png") else kich_thuoc_jpeg(data)


def thong_tin_video(storage: MinioStorage, key: str, size: int) -> tuple[float | None, bool] | None:
    """Duyệt atom top-level tới `moov` để lấy thời lượng và biết có track audio không."""
    offset = 0
    moov = None
    while offset < size - 8:
        header = doc_object(storage, key, offset=offset, length=8)
        if len(header) < 8:
            break
        length = struct.unpack(">I", header[:4])[0]
        atom = header[4:8]
        if length == 1:
            length = struct.unpack(">Q", doc_object(storage, key, offset=offset + 8, length=8))[0]
        if length < 8:
            break
        if atom == b"moov":
            moov = doc_object(storage, key, offset=offset, length=min(length, 8_000_000))
            break
        offset += length
    if moov is None:
        return None

    thoi_luong = None
    i = moov.find(b"mvhd")
    if i > 0:
        if moov[i + 4] == 0:
            timescale, duration = struct.unpack(">II", moov[i + 16 : i + 24])
        else:
            timescale, duration = struct.unpack(">IQ", moov[i + 24 : i + 36])
        thoi_luong = duration / timescale if timescale else None

    handlers = set()
    j = 0
    while (j := moov.find(b"hdlr", j)) >= 0:
        handlers.add(moov[j + 12 : j + 16].decode("latin1"))
        j += 4
    return thoi_luong, "soun" in handlers


def kiem_ke(prefix: str, deep: bool = False, census_out: str | None = None) -> dict:
    storage = MinioStorage()
    base = normalize_prefix(prefix)
    search = f"{base}/" if base else ""

    tat_ca = {obj.object_name: obj for obj in storage.client.list_objects(storage.bucket_name, prefix=search, recursive=True)}
    tong_byte = sum(obj.size or 0 for obj in tat_ca.values())
    theo_duoi = collections.Counter(Path(k).suffix.lower() or "(không đuôi)" for k in tat_ca)

    census: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])
    khong_dung_quy_uoc: list[str] = []
    for key in tat_ca:
        suffix = Path(key).suffix.lower()
        if suffix not in IMAGE_EXTENSIONS | VIDEO_EXTENSIONS:
            continue
        try:
            post_id = event_id_from_object_key(key, base)
        except ObjectKeyError:
            khong_dung_quy_uoc.append(key)
            continue
        census[post_id][0 if suffix in IMAGE_EXTENSIONS else 1] += 1

    file_media = [k for k in tat_ca if Path(k).suffix.lower() in IMAGE_EXTENSIONS | VIDEO_EXTENSIONS]
    trung_etag = collections.defaultdict(list)
    for key in file_media:
        trung_etag[tat_ca[key].etag].append(key)
    nhom_trung = {etag: keys for etag, keys in trung_etag.items() if len(keys) > 1}
    etag_multipart = [k for k in file_media if "-" in (tat_ca[k].etag or "")]

    ket_qua = {
        "prefix": base,
        "object": len(tat_ca),
        "byte": tong_byte,
        "theo_duoi": dict(theo_duoi.most_common()),
        "bai_co_media": len(census),
        "bai_co_anh": sum(1 for anh, _ in census.values() if anh),
        "bai_co_video": sum(1 for _, video in census.values() if video),
        "tong_anh": sum(anh for anh, _ in census.values()),
        "tong_video": sum(video for _, video in census.values()),
        "key_sai_quy_uoc": khong_dung_quy_uoc,
        "etag_nhom_trung": len(nhom_trung),
        "etag_file_trung": sum(len(v) for v in nhom_trung.values()),
        "etag_ban_du": sum(len(v) - 1 for v in nhom_trung.values()),
        # ETag của object tải nhiều phần không phải MD5 nội dung -> gom trùng
        # bằng ETag không đáng tin cho đúng nhóm này. Xác nhận bằng --deep.
        "etag_multipart": len(etag_multipart),
    }

    # ---- posts.jsonl: trùng ID, ký tự tách dòng lạ, lỗi tải media ----
    jsonl_key = f"{search}posts.jsonl"
    try:
        raw = doc_object(storage, jsonl_key).decode("utf-8")
    except Exception as exc:
        print(f"  ! bỏ qua {jsonl_key}: {exc.__class__.__name__}")
        raw = ""
    if raw:
        docs = [json.loads(dong) for dong in raw.split("\n") if dong.strip()]
        ids = [d["id"] for d in docs]
        trung = sorted({i for i, n in collections.Counter(ids).items() if n > 1})
        theo_id = {}
        for d in docs:
            theo_id.setdefault(d["id"], []).append(d)
        loi_theo_dong = sum(
            1
            for d in docs
            for m in (d.get("media") or [])
            if any(k.endswith("_error") and v for k, v in m.items())
        )
        loi_sau_dedupe = sum(
            1
            for id_, ban in theo_id.items()
            for m in (ban[0].get("media") or [])
            if any(k.endswith("_error") and v for k, v in m.items())
        )
        ket_qua |= {
            "jsonl_dong": len(docs),
            "jsonl_id_duy_nhat": len(set(ids)),
            "jsonl_id_trung": trung,
            "jsonl_conflict_created_time": {
                id_: sorted({b.get("created_time") for b in ban}) for id_, ban in theo_id.items() if len(ban) > 1
            },
            "media_loi_theo_dong": loi_theo_dong,
            "media_loi_sau_dedupe": loi_sau_dedupe,
            "ky_tu_tach_dong_la": dict(
                collections.Counter(
                    f"U+{ord(ch):04X}"
                    for d in docs
                    for ch in (d.get("message") or "")
                    if ch in KY_TU_TACH_DONG_LA
                )
            ),
            "phan_bo_media_moi_bai_sau_dedupe": dict(
                sorted(collections.Counter(len(ban[0].get("media") or []) for ban in theo_id.values()).items())
            ),
        }

    if deep:
        # Gom trùng bằng nội dung thật, không tin ETag (xem bam_sha256).
        trung_sha = collections.defaultdict(list)
        for key in sorted(file_media):
            trung_sha[bam_sha256(storage, key)].append(key)
        nhom_sha = {bam: keys for bam, keys in trung_sha.items() if len(keys) > 1}
        ket_qua |= {
            "sha256_nhom_trung": len(nhom_sha),
            "sha256_file_trung": sum(len(v) for v in nhom_sha.values()),
            "sha256_ban_du": sum(len(v) - 1 for v in nhom_sha.values()),
            "sha256_nhom_toan_anh": all(
                Path(k).suffix.lower() in IMAGE_EXTENSIONS for keys in nhom_sha.values() for k in keys
            ),
            "sha256_khop_etag": sorted(tuple(sorted(v)) for v in nhom_sha.values())
            == sorted(tuple(sorted(v)) for v in nhom_trung.values()),
            # Nhóm mà SHA-256 thấy còn ETag bỏ sót — chỗ này mới trả lời được
            # câu "có video trùng nào bị ETag multipart che mất không".
            "sha256_nhom_etag_bo_sot": [
                keys
                for keys in nhom_sha.values()
                if tuple(sorted(keys)) not in {tuple(sorted(v)) for v in nhom_trung.values()}
            ],
        }

        lech = 0
        that_lon_hon = 0
        canh_lon: list[int] = []
        khai_bao: dict[str, tuple[int, int]] = {}
        co_loi: set[str] = set()
        if raw:
            for d in {d["id"]: d for d in docs}.values():
                for m in d.get("media") or []:
                    if m.get("kind") != "image" or not m.get("local"):
                        continue
                    key = f"{search}{d['id']}/{m['local']}"
                    if m.get("width"):
                        khai_bao[key] = (m["width"], m["height"])
                    if any(k.endswith("_error") and v for k, v in m.items()):
                        co_loi.add(key)
        # Tách hai nhóm để trả lời đúng câu hỏi "ảnh tải lỗi có bị nhỏ hơn
        # không" bằng số đo file thật, chứ không bằng width/height khai báo.
        canh_lon_nhom: dict[str, list[int]] = {"sach": [], "co_loi": []}
        for key in sorted(k for k in tat_ca if Path(k).suffix.lower() in IMAGE_EXTENSIONS):
            that = kich_thuoc_anh(storage, key)
            if that is None:
                continue
            canh_lon.append(max(that))
            canh_lon_nhom["co_loi" if key in co_loi else "sach"].append(max(that))
            if key in khai_bao and khai_bao[key] != that:
                lech += 1
                if that[0] * that[1] > khai_bao[key][0] * khai_bao[key][1]:
                    that_lon_hon += 1
        canh_lon.sort()
        for nhom in canh_lon_nhom.values():
            nhom.sort()
        ket_qua |= {
            "anh_doc_duoc_header": len(canh_lon),
            "anh_lech_metadata": lech,
            "anh_that_lon_hon_khai_bao": that_lon_hon,
            "anh_theo_nhom": {
                ten: {
                    "n": len(nhom),
                    "p50_canh_lon": nhom[len(nhom) // 2] if nhom else None,
                    "toi_512px": sum(1 for x in nhom if x <= 512),
                    "ty_le_toi_512px": round(100 * sum(1 for x in nhom if x <= 512) / len(nhom), 2) if nhom else None,
                }
                for ten, nhom in canh_lon_nhom.items()
            },
            "anh_canh_lon_p50": canh_lon[len(canh_lon) // 2] if canh_lon else None,
            "anh_canh_lon_toi_512": sum(1 for x in canh_lon if x <= 512),
        }

        thoi_luong: list[float] = []
        khong_audio: list[str] = []
        for key in sorted(k for k in tat_ca if Path(k).suffix.lower() in VIDEO_EXTENSIONS):
            info = thong_tin_video(storage, key, tat_ca[key].size or 0)
            if info is None:
                print(f"  ! không thấy atom moov: {key}")
                continue
            giay, co_audio = info
            if giay:
                thoi_luong.append(giay)
            if not co_audio:
                khong_audio.append(key)
        thoi_luong.sort()
        ket_qua |= {
            "video_tong_giay": round(sum(thoi_luong)),
            "video_p50_giay": round(thoi_luong[len(thoi_luong) // 2]) if thoi_luong else None,
            "video_dai_nhat_giay": round(thoi_luong[-1]) if thoi_luong else None,
            "video_khong_co_audio": khong_audio,
        }

    if census_out:
        dich = Path(census_out)
        dich.parent.mkdir(parents=True, exist_ok=True)
        dong = [
            "# Điều tra dân số corpus thật, sinh bằng scripts/audit_minio_dataset.py --census-out.",
            f"# Bucket {storage.bucket_name}, prefix {base}/, full scan qua list_objects(recursive=True).",
            "# Định dạng: <facebook_post_id> <số ảnh> <số video>. Chỉ liệt kê bài có ít nhất một file media.",
            "# Dùng làm mốc hồi quy cho API/tests/test_minio_registration.py.",
        ]
        dong += [f"{post_id} {anh} {video}" for post_id, (anh, video) in sorted(census.items())]
        dich.write_text("\n".join(dong) + "\n", encoding="utf-8")
        print(f"Đã ghi census: {dich} ({len(census)} bài)")

    return ket_qua


def main() -> int:
    parser = argparse.ArgumentParser(description="Kiểm kê bộ dữ liệu thô trên MinIO (chỉ đọc).")
    parser.add_argument("--prefix", default="raw/google-drive/data", help="Thư mục gốc của bộ dữ liệu.")
    parser.add_argument("--deep", action="store_true", help="Đọc thêm header ảnh và atom moov của video (chậm hơn).")
    parser.add_argument("--census-out", default=None, help="Ghi file census <post_id> <số ảnh> <số video>.")
    args = parser.parse_args()

    ket_qua = kiem_ke(args.prefix, deep=args.deep, census_out=args.census_out)
    print(json.dumps(ket_qua, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
