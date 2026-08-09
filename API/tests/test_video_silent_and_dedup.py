"""Hai lỗi lộ ra khi chạy thử video đầu tiên qua worker RunPod (07/08/2026).

Cả hai đều thuộc loại nguy hiểm nhất của dự án: **không ném lỗi, chỉ ghi ra dữ
liệu sai**. Lượt chạy báo thành công, hàng trong PostgreSQL đầy đủ, chỉ có nội
dung là bịa hoặc thiếu.

1. Video câm → ASR nhả ra đúng một ký tự `"<"`, được ghi status DONE. Đo trên
   cả 6 video câm của kho: cả 6 đều ra `"<"`. Nó sẽ thành một đơn vị
   `video_transcript` trích dẫn được — tức hệ thống khẳng định có người nói
   trong một video im lặng.
2. Video nền sáng → phép đo trùng bằng pixel 32×32 coi mọi frame là trùng nhau,
   cả video rút xuống 1 keyframe. Nội dung không mất theo kiểu báo lỗi, nó chỉ
   lặng lẽ không được index.

Số liệu kho: 52/59 video có tiếng thật, 6 câm (-91 dB), 1 không có track audio.
"""

from __future__ import annotations

import subprocess
from types import SimpleNamespace

import pytest

from src.rag_video_anh.pipeline.asr_service import AsrService
from src.rag_video_anh.pipeline.keyframe_extractor import KeyframeExtractorService

# ── Cổng lọc lời thoại rác ───────────────────────────────────────────────────


@pytest.mark.parametrize("rac", ["<", "...", "!!!", "-", "  ", "、。", "??"])
def test_doan_khong_co_chu_hay_so_bi_loai(rac):
    """`"<"` là thứ Zipformer thật sự trả về cho cả 6 video câm của kho."""
    assert AsrService._is_boilerplate_hallucination(rac) is True


@pytest.mark.parametrize(
    "that",
    [
        "xin chào các bạn",
        "vâng",
        "2024 là một năm đáng nhớ",
        "HIT Open Day",
    ],
)
def test_loi_noi_that_van_di_qua(that):
    """Cổng lọc siết quá tay còn tệ hơn: mất lời thoại thật thì không ai biết."""
    assert AsrService._is_boilerplate_hallucination(that) is False


@pytest.mark.parametrize(
    "cau",
    [
        "hãy subscribe cho kênh nhé",
        "nhớ đăng ký kênh nhé",
        "đừng quên like video này",
        "cảm ơn các bạn đã xem",
    ],
)
def test_van_loai_boilerplate(cau):
    """Ba câu cuối có chữ `đ` — trước khi sửa `_normalize_transcript_text` thì
    chúng lọt qua hết, vì NFKD xoá `đ` chứ không đổi thành `d`."""
    assert AsrService._is_boilerplate_hallucination(cau) is True


def test_bo_dau_giu_duoc_chu_d_gach():
    assert AsrService._normalize_transcript_text("Đăng ký kênh") == "dang ky kenh"


def test_bo_loc_go_han_doan_rac_khoi_ket_qua():
    """Ghim ở tầng lọc chứ không chỉ ở hàm kiểm tra: chỗ hỏng cũ nằm ở đây —
    hàm kiểm tra trả về đúng nhưng đoạn rác vẫn lọt vào `segments`."""
    svc = AsrService.__new__(AsrService)
    segments = [
        SimpleNamespace(text="<"),
        SimpleNamespace(text="chào mọi người"),
        SimpleNamespace(text="..."),
    ]

    giu_lai, so_bi_loai = svc._filter_hallucinated_segments(segments)

    assert [s.text for s in giu_lai] == ["chào mọi người"]
    assert so_bi_loai == 2


# ── Video không có track audio ───────────────────────────────────────────────


def test_khong_co_track_audio_khong_phai_la_loi():
    loi = subprocess.CalledProcessError(1, ["ffmpeg"])
    loi.stderr = "[out#0/wav] Output file does not contain any stream\nError opening output file"

    assert AsrService._la_video_khong_co_tieng(loi) is True


def test_file_hong_van_phai_bao_loi():
    """Bắt rộng tay thì file hỏng thật sẽ im lặng biến mất thành 'không có tiếng'."""
    loi = subprocess.CalledProcessError(1, ["ffmpeg"])
    loi.stderr = "moov atom not found\nInvalid data found when processing input"

    assert AsrService._la_video_khong_co_tieng(loi) is False


def test_stderr_rong_khong_duoc_doan_bua():
    loi = subprocess.CalledProcessError(1, ["ffmpeg"])
    loi.stderr = None

    assert AsrService._la_video_khong_co_tieng(loi) is False


# ── Cửa sổ thời gian khi loại frame trùng ────────────────────────────────────


class Cv2Gia:
    """`_deduplicate` chỉ dùng cv2 qua `_embedding`, mà test này thay hẳn `_embedding`."""


def build_extractor(max_gap_sec: float, duplicate_threshold: float = 0.90) -> KeyframeExtractorService:
    extractor = KeyframeExtractorService.__new__(KeyframeExtractorService)
    extractor.pipeline_config = SimpleNamespace(keyframe_max_gap_sec=max_gap_sec)
    extractor.model_config = SimpleNamespace(duplicate_threshold=duplicate_threshold)
    extractor.embedding_model = None
    # Mọi frame "giống hệt nhau" theo phép đo — đúng cái xảy ra thật với video
    # nền trắng: ba màn hình khác hẳn nhau vẫn cho cosine 0,98–0,99.
    extractor._embedding = lambda image, cv2: [1.0, 0.0, 0.0]
    return extractor


def candidates(*frame_indexes: int) -> list[tuple[int, object, dict[str, float]]]:
    return [(i, object(), {"quality_score": 0.8}) for i in frame_indexes]


def test_khong_co_cua_so_thi_moi_frame_bi_gop_thanh_mot():
    """Hành vi cũ, giữ lại để thấy rõ cửa sổ thời gian đang chữa cái gì."""
    extractor = build_extractor(max_gap_sec=0)

    giu = extractor._deduplicate(candidates(75, 250, 500, 800), Cv2Gia(), fps=30.0)

    assert len(giu) == 1


def test_cua_so_giu_do_phu_theo_thoi_gian():
    """Video 28,8s @30fps: các frame cách nhau >10s không được coi là trùng."""
    extractor = build_extractor(max_gap_sec=10.0)

    giu = extractor._deduplicate(candidates(75, 250, 500, 850), Cv2Gia(), fps=30.0)

    # 75 (2,5s) vs 250 (8,3s): cách 5,8s -> vẫn so nhau, 250 bị loại.
    # 500 (16,7s): cách 75 tới 14,2s -> không so, được giữ.
    # 850 (28,3s): cách 500 tới 11,7s -> không so, được giữ.
    assert [i for i, _, _, _ in giu] == [75, 500, 850]


def test_frame_trung_that_nam_sat_nhau_van_bi_loai():
    """Cửa sổ không được biến thành 'giữ tất': frame liền kề vẫn phải gộp."""
    extractor = build_extractor(max_gap_sec=10.0)

    giu = extractor._deduplicate(candidates(250, 251, 255, 260), Cv2Gia(), fps=30.0)

    assert [i for i, _, _, _ in giu] == [250]


def test_khong_biet_fps_thi_tat_cua_so():
    """fps=0 (không đọc được metadata) thì quy đổi giây sang frame là vô nghĩa —
    thà giữ hành vi cũ còn hơn dựng một cửa sổ dài tuỳ tiện."""
    extractor = build_extractor(max_gap_sec=10.0)

    giu = extractor._deduplicate(candidates(75, 500, 800), Cv2Gia(), fps=0.0)

    assert len(giu) == 1
