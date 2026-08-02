"""Dựng / xem / tắt pod GPU chạy `embedding_server/` trên RunPod.

**Pod GPU tính tiền theo giờ**, nên script theo đúng quy ước của repo: không
làm gì cho tới khi có `--apply`. Chạy trần chỉ in ra kế hoạch và giá.

Không cần container registry: dùng ảnh PyTorch công khai, còn hai tệp code thì
gói lại, đẩy lên MinIO rồi cấp presigned URL cho pod tự tải — đúng cách
`runpod_worker/` vẫn nhận video. Build một ảnh CUDA ~12GB rồi push cũng được,
và sạch hơn cho môi trường thật (`embedding_server/Dockerfile`), nhưng cần
credential registry mà máy này không có.

    python scripts/deploy_embedding_pod.py                 # xem kế hoạch, không tạo gì
    python scripts/deploy_embedding_pod.py --apply         # tạo pod
    python scripts/deploy_embedding_pod.py --status        # xem pod đang chạy
    python scripts/deploy_embedding_pod.py --terminate ID  # TẮT (nhớ tắt, tiền tính theo giờ)

Sau khi pod lên, kiểm parity với chính nó trước khi tin:

    python scripts/check_embedding_parity.py --url https://<id>-8100.proxy.runpod.net/v1/embeddings \\
        --api-key "$EMBED_SERVER_API_KEY"
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import secrets
import sys
import tarfile
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "API"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(REPO_ROOT / ".env")

GRAPHQL = "https://api.runpod.io/graphql"
# Ảnh công khai trên Docker Hub, đã có sẵn torch + CUDA. Phần còn lại cài lúc
# khởi động (~200MB) rồi tải trọng số từ HuggingFace (~3,5GB).
BASE_IMAGE = "pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime"
POD_NAME = "hit-mira-embed"
PORT = 8100
# 16GB VRAM là thừa cho model 865M tham số ở fp16; chọn theo giá rẻ nhất.
GPU_UU_TIEN = [
    "NVIDIA RTX A5000",
    "NVIDIA RTX A4000",
    "NVIDIA RTX 4000 SFF Ada Generation",
    "NVIDIA RTX A4500",
    "NVIDIA GeForce RTX 3090",
]
DOI_TOI_DA_GIAY = 900
OBJECT_KEY = "runtime/embedding_server.tar.gz"
PRESIGN_GIO = 24


def _graphql(query: str) -> dict[str, Any]:
    import httpx

    api_key = os.environ["RUNPOD_API_KEY"]
    response = httpx.post(
        f"{GRAPHQL}?api_key={api_key}",
        json={"query": query},
        headers={"Content-Type": "application/json"},
        timeout=120.0,
    )
    response.raise_for_status()
    payload = response.json()
    if payload.get("errors"):
        raise RuntimeError(json.dumps(payload["errors"], ensure_ascii=False)[:500])
    return payload["data"]


def _gia_gpu(community: bool = False) -> list[tuple[str, float]]:
    """Giá mỗi GPU trong danh sách ưu tiên, **lọc đúng loại cloud sắp thuê**.

    Bản đầu hỏi `lowestPrice` không lọc, mà giá không lọc là giá community — rồi
    lại deploy với `cloudType: ALL` và rơi vào secure cloud. Kết quả: script in
    $0,160/giờ nhưng hoá đơn thật là $0,270/giờ, lệch 69% và chỉ lộ ra ở dòng
    `costPerHr` sau khi pod đã được tạo. In một giá mà mua giá khác là cách
    nhanh nhất để đốt hết số dư mà không hiểu vì sao.
    """
    loc = "secureCloud:false" if community else "secureCloud:true"
    data = _graphql(
        f"query {{ gpuTypes {{ id lowestPrice(input:{{gpuCount:1, {loc}}}) "
        "{ uninterruptablePrice } } }"
    )
    gia = {
        g["id"]: (g.get("lowestPrice") or {}).get("uninterruptablePrice")
        for g in data["gpuTypes"]
    }
    return [(gpu, gia[gpu]) for gpu in GPU_UU_TIEN if gia.get(gpu)]


def _so_du() -> float:
    return float(_graphql("query { myself { clientBalance } }")["myself"]["clientBalance"])


def _dong_goi_code() -> bytes:
    """Gói encoder.py + server.py thành tar.gz trong bộ nhớ."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for ten in ("encoder.py", "server.py"):
            tar.add(REPO_ROOT / "embedding_server" / ten, arcname=ten)
    return buffer.getvalue()


def _tai_len_minio(du_lieu: bytes) -> str:
    """Đẩy gói code lên MinIO, trả về presigned URL mà pod tải được từ ngoài."""
    from src.rag_video_anh.pipeline.minio_storage import MinioStorage

    storage = MinioStorage()
    storage.client.put_object(
        storage.bucket_name,
        OBJECT_KEY,
        io.BytesIO(du_lieu),
        length=len(du_lieu),
        content_type="application/gzip",
    )
    # `presigned_download_url` ký bằng MINIO_PUBLIC_ENDPOINT. Bắt buộc: URL ký
    # theo "localhost:9000" thì pod trên RunPod tải về chính nó rồi fail một
    # cách rất khó hiểu.
    return storage.presigned_download_url(OBJECT_KEY, expires_seconds=PRESIGN_GIO * 3600)


def _lenh_khoi_dong(url_code: str, khoa: str) -> str:
    """Lệnh chạy trong pod: tải code, cài dependency, phục vụ.

    Toàn bộ script được bọc base64 **có lý do**, không phải cho gọn. Bản đầu
    viết thẳng `bash -c '... urlretrieve("url") ...'` và chết ngay: presigned
    URL của MinIO chứa dấu nháy đơn lồng nhau làm đóng sớm chuỗi `bash -c`, mà
    URL lại đầy ký tự `&` nên phần rơi ra ngoài bị shell hiểu là chạy nền. Lệnh
    hỏng, container không bao giờ lên, và triệu chứng duy nhất là pod báo
    RUNNING với `uptimeInSeconds` âm — chẩn đoán rất mất thời gian.

    Base64 chỉ gồm [A-Za-z0-9+/=] nên không còn ký tự nào để shell diễn giải,
    và script bên trong tha hồ dùng nháy.

    `transformers<5` là bắt buộc — bản 5.x gỡ `clip_loss` mà remote code của
    jina-clip-v2 import ở mức module. Xem embedding_server/README.md.
    """
    script = "\n".join(
        [
            "set -euo pipefail",
            "cd /workspace",
            f'python -c "import urllib.request;urllib.request.urlretrieve(\'{url_code}\', \'/workspace/app.tar.gz\')"',
            "tar xzf app.tar.gz",
            "pip install --no-cache-dir 'transformers>=4.45,<5' accelerate timm einops "
            "'fastapi>=0.115' 'uvicorn[standard]>=0.30' 'pydantic>=2.7' pillow numpy",
            "export EMBED_PRELOAD=1",
            f"export EMBED_SERVER_API_KEY={khoa}",
            f"exec uvicorn server:app --host 0.0.0.0 --port {PORT}",
        ]
    )
    goi = base64.b64encode(script.encode()).decode()
    return f'bash -c "echo {goi} | base64 -d > /boot.sh && bash /boot.sh"'


def _kiem_lenh(lenh: str) -> None:
    """Chặn lệnh khởi động hỏng TRƯỚC khi thuê máy.

    Pod báo RUNNING kể cả khi lệnh khởi động vô nghĩa — nó vẫn tính tiền, vẫn
    không bao giờ trả lời, và triệu chứng duy nhất là `uptimeInSeconds` âm.
    Vài phép kiểm chuỗi ở đây rẻ hơn nhiều so với chẩn đoán chuyện đó.
    """
    phan_trong = lenh.removeprefix('bash -c "').removesuffix('"')
    if '"' in phan_trong or "'" in phan_trong:
        raise RuntimeError(f"Lệnh khởi động còn nháy lồng nhau, sẽ vỡ khi shell tách: {lenh[:120]}")
    if "&" in phan_trong.split("|")[0]:
        raise RuntimeError("Có ký tự '&' ngoài vùng an toàn — shell sẽ hiểu là chạy nền")


def _tao_pod(gpu_id: str, lenh: str, community: bool = False) -> dict[str, Any]:
    # containerDisk 40GB: 3,5GB trọng số + ~10GB ảnh + chỗ cho pip.
    # cloudType phải khớp với loại đã hỏi giá ở `_gia_gpu`, nếu không lại rơi
    # vào cảnh in một giá mua một giá.
    cloud = "COMMUNITY" if community else "SECURE"
    mutation = f"""
    mutation {{
      podFindAndDeployOnDemand(input: {{
        cloudType: {cloud}
        gpuCount: 1
        volumeInGb: 0
        containerDiskInGb: 40
        minVcpuCount: 4
        minMemoryInGb: 16
        gpuTypeId: "{gpu_id}"
        name: "{POD_NAME}"
        imageName: "{BASE_IMAGE}"
        ports: "{PORT}/http"
        dockerArgs: {json.dumps(lenh)}
      }}) {{ id costPerHr machineId }}
    }}
    """
    return _graphql(mutation)["podFindAndDeployOnDemand"]


def _tao_pod_co_du_phong(
    gia: list[tuple[str, float]], lenh: str, community: bool
) -> dict[str, Any] | None:
    """Thử lần lượt từng GPU trong danh sách ưu tiên cho tới khi thuê được.

    `stockStatus` chỉ nói "Low", không nói còn hay hết — cách duy nhất biết chắc
    là thử thuê. Community cloud hết máy khá thường xuyên: lần chuyển pod sang
    community đầu tiên, A5000 trả về `SUPPLY_CONSTRAINT` và script bỏ cuộc ngay
    dù còn bốn GPU nữa trong danh sách, con đắt nhất cũng chỉ hơn $0,06/giờ.

    Chỉ nuốt đúng lỗi hết máy. Lỗi khác (sai khoá, lệnh khởi động hỏng) mà cũng
    thử tiếp thì thành thuê năm cái máy hỏng liên tiếp, và vẫn tính tiền.
    """
    for gpu_id, gia_gio in gia:
        try:
            return _tao_pod(gpu_id, lenh, community)
        except RuntimeError as loi:
            if "SUPPLY_CONSTRAINT" not in str(loi):
                raise
            print(f"    {gpu_id} (${gia_gio:.3f}/giờ) hết máy, thử con tiếp theo...")
    return None


def _cho_san_sang(pod_id: str, khoa: str) -> bool:
    import httpx

    url = f"https://{pod_id}-{PORT}.proxy.runpod.net/health"
    bat_dau = time.monotonic()
    while time.monotonic() - bat_dau < DOI_TOI_DA_GIAY:
        try:
            phan_hoi = httpx.get(url, timeout=20.0)
            if phan_hoi.status_code == 200 and phan_hoi.json().get("ready"):
                print(f"  pod sẵn sàng sau {time.monotonic() - bat_dau:.0f}s: {phan_hoi.json()}")
                return True
        except Exception:
            pass  # pod chưa lên là chuyện bình thường trong lúc cài đặt
        print(f"  ...chờ ({time.monotonic() - bat_dau:.0f}s)", flush=True)
        time.sleep(20)
    print(f"  QUÁ {DOI_TOI_DA_GIAY}s mà pod chưa sẵn sàng — xem log trên console RunPod")
    return False


def lenh_status() -> int:
    data = _graphql(
        "query { myself { clientBalance currentSpendPerHr "
        "pods { id name desiredStatus costPerHr runtime { uptimeInSeconds } } } }"
    )["myself"]
    print(f"Số dư: ${data['clientBalance']:.2f}   đang tiêu: ${data['currentSpendPerHr']:.3f}/giờ")
    if not data["pods"]:
        print("Không có pod nào.")
        return 0
    for pod in data["pods"]:
        uptime = (pod.get("runtime") or {}).get("uptimeInSeconds") or 0
        print(
            f"  {pod['id']}  {pod['name']:20} {pod['desiredStatus']:10} "
            f"${pod['costPerHr']:.3f}/giờ  chạy {uptime / 3600:.1f}h"
        )
    return 0


def lenh_terminate(pod_id: str) -> int:
    _graphql(f'mutation {{ podTerminate(input: {{podId: "{pod_id}"}}) }}')
    print(f"Đã tắt pod {pod_id}. Kiểm lại bằng --status.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--apply", action="store_true", help="thật sự tạo pod (mặc định chỉ in kế hoạch)")
    parser.add_argument("--status", action="store_true", help="xem số dư và pod đang chạy")
    parser.add_argument("--terminate", metavar="POD_ID", help="tắt một pod")
    parser.add_argument(
        "--community",
        action="store_true",
        help="thuê community cloud (rẻ hơn ~40%%: A5000 $0,16 thay vì $0,27/giờ, "
        "đổi lại máy của bên thứ ba nên kém ổn định hơn)",
    )
    args = parser.parse_args()

    if args.status:
        return lenh_status()
    if args.terminate:
        return lenh_terminate(args.terminate)

    gia = _gia_gpu(args.community)
    if not gia:
        print("Không GPU nào trong danh sách ưu tiên còn chỗ.")
        return 1
    gpu_id, gia_gio = gia[0]
    so_du = _so_du()

    print(f"Cloud   : {'COMMUNITY (rẻ hơn, kém ổn định hơn)' if args.community else 'SECURE'}")
    print(f"GPU     : {gpu_id}  ${gia_gio:.3f}/giờ")
    if len(gia) > 1:
        du_phong = ", ".join(f"{g} (${p:.3f})" for g, p in gia[1:])
        print(f"Dự phòng: {du_phong}")
    print(f"Số dư   : ${so_du:.2f}  → chạy được ~{so_du / gia_gio:.0f} giờ")
    print(f"Ảnh     : {BASE_IMAGE}")
    print(f"Cổng    : https://<pod-id>-{PORT}.proxy.runpod.net/v1/embeddings")
    print("\nPod tính tiền THEO GIỜ kể cả lúc không dùng. Index xong nhớ --terminate.")

    if not args.apply:
        print("\n(chạy thử — thêm --apply để tạo thật)")
        return 0

    khoa = os.getenv("EMBED_SERVER_API_KEY") or secrets.token_urlsafe(24)
    # In khoá NGAY, trước khi làm bất cứ việc gì có thể hỏng. Khoá sinh ngẫu
    # nhiên mà chỉ in ở cuối thì script chết giữa chừng là mất luôn, còn pod
    # thì vẫn chạy và vẫn tính tiền — không gọi được, cũng không biết đường tắt.
    print(f"\nEMBED_SERVER_API_KEY={khoa}   <-- lưu lại ngay")
    print("\n1/4 gói code và đẩy lên MinIO...")
    url_code = _tai_len_minio(_dong_goi_code())
    print("2/4 tạo pod...")
    lenh = _lenh_khoi_dong(url_code, khoa)
    _kiem_lenh(lenh)
    pod = _tao_pod_co_du_phong(gia, lenh, args.community)
    if pod is None:
        print("\nKhông GPU nào trong danh sách còn máy. Thử lại sau, hoặc bỏ")
        print("--community để thuê secure cloud (đắt hơn nhưng thường còn chỗ).")
        return 1
    pod_id = pod["id"]
    print(f"    pod {pod_id}  ${pod['costPerHr']:.3f}/giờ")
    print("3/4 chờ cài đặt và nạp model (vài phút)...")
    san_sang = _cho_san_sang(pod_id, khoa)

    print("\n4/4 xong. Đặt vào .env:")
    print(f"    MEDIA_IMAGE_EMBEDDING_BASE_URL=https://{pod_id}-{PORT}.proxy.runpod.net/v1/embeddings")
    # JINA_API_KEY chính là thứ client gửi trong header `Authorization: Bearer`,
    # nên nó phải BẰNG khoá của pod. Đặt "local" ở đây là ăn 401 hàng loạt.
    print(f"    JINA_API_KEY={khoa}")
    print("    MEDIA_EMBEDDING_TOKENS_PER_MINUTE=0")
    print(f"\nTắt khi xong:  python scripts/deploy_embedding_pod.py --terminate {pod_id}")
    return 0 if san_sang else 1


if __name__ == "__main__":
    raise SystemExit(main())
