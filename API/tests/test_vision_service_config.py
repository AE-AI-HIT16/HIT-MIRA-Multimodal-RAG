"""Cấu hình VLM caption/OCR — TC-903 (không còn nhà cung cấp mặc định).

Trước đây thiếu `MEDIA_VISION_*` thì cấu hình rơi về nhóm `OPENROUTER_*`, nên
một khoá hết hạn ở nhà cung cấp mà không ai chọn vẫn giết được cả tầng caption.
Bỏ nhánh rơi đó rồi thì thiếu biến nào phải nói đúng biến ấy — đây là chỗ ghim.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.rag_video_anh.pipeline.qwen_vision_service import QwenVisionService
from src.rag_video_anh.schemas import StageStatus
from src.rag_video_anh.schemas.video_metadata import KeyFrame, KeyFrameSet

DAY_DU = {
    "vision_model_name": "cx/gpt-5.5",
    "vision_api_base_url": "http://10.0.0.5:20128/v1",
    "vision_api_key": "sk-test",
    "vision_api_timeout": 120.0,
    "vision_api_temperature": 0.0,
    "vision_max_tokens": 1024,
}


def cau_hinh(**ghi_de):
    """AppConfig giả, chỉ giữ nhóm trường mà nhánh vision đọc."""
    return SimpleNamespace(
        media_models=SimpleNamespace(**{**DAY_DU, **ghi_de}),
        media_prompts=SimpleNamespace(vision_system_prompt=None, vision_user_prompt=None),
    )


class ClientGia:
    """Bắt lại đúng kwargs gửi lên endpoint để soi `extra_body`."""

    def __init__(self) -> None:
        self.kwargs: dict = {}
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.kwargs = kwargs
        noi_dung = '{"ocr_text":"","ocr_blocks":[],"caption_text":"c","scene":"s","objects":[],"activities":[],"keywords":[]}'
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=noi_dung))])


def mot_keyframe() -> KeyFrameSet:
    frame = KeyFrame(
        frame_id="f0", media_id="m", frame_index=0, timestamp_ms=0, timestamp_sec=0.0, image_payload=object()
    )
    return KeyFrameSet(media_id="m", frames=[frame])


def mot_keyframe_ma_hoa_duoc() -> KeyFrameSet:
    """Payload `bytes` đi thẳng qua base64, không cần Pillow.

    Các test thiếu-cấu-hình ở trên dừng trước khi chạm ảnh nên `object()` là đủ,
    còn ở đây phải gọi được tới `chat.completions.create` mới soi được kwargs.
    """

    frame = KeyFrame(
        frame_id="f0", media_id="m", frame_index=0, timestamp_ms=0, timestamp_sec=0.0, image_payload=b"\xff\xd8\xff\xd9"
    )
    return KeyFrameSet(media_id="m", frames=[frame])


@pytest.mark.parametrize(
    "thieu, bien",
    [
        ("vision_api_key", "MEDIA_VISION_API_KEY"),
        ("vision_api_base_url", "MEDIA_VISION_API_BASE_URL"),
        ("vision_model_name", "MEDIA_VISION_MODEL_NAME"),
    ],
)
def test_thieu_bien_nao_thi_noi_dung_bien_do(thieu: str, bien: str) -> None:
    ocr, caption = QwenVisionService(config=cau_hinh(**{thieu: None})).analyze(
        media_id="m", keyframes=mot_keyframe()
    )

    assert ocr.status is StageStatus.SKIPPED
    assert caption.status is StageStatus.SKIPPED
    assert bien in (ocr.reason or "")
    # Không được nhắc tới nhà cung cấp nào: người vận hành chọn endpoint, không
    # phải đoán xem hệ thống đang lặng lẽ rơi về đâu.
    assert "OPENROUTER" not in (ocr.reason or "").upper()


@pytest.mark.parametrize("thieu", ["vision_api_key", "vision_api_base_url", "vision_model_name"])
def test_placeholder_chua_thay_cung_tinh_la_thieu(thieu: str) -> None:
    """`${...}` sót lại trong YAML từng lọt qua và biến thành lỗi 401 khó lần."""
    ocr, _ = QwenVisionService(config=cau_hinh(**{thieu: "${" + thieu.upper() + "}"})).analyze(
        media_id="m", keyframes=mot_keyframe()
    )

    assert ocr.status is StageStatus.SKIPPED
    assert "MEDIA_VISION_" in (ocr.reason or "")


def test_thieu_cau_hinh_khong_lam_hong_ca_lo() -> None:
    """Caption hỏng thì bỏ qua, không được ném lên chặn cả pipeline (BR: không bịa)."""
    ocr, caption = QwenVisionService(config=cau_hinh(vision_api_key=None)).analyze(
        media_id="m", keyframes=mot_keyframe()
    )

    assert (ocr.results, caption.results) == ([], [])
    assert ocr.status is not StageStatus.ERROR


def test_co_khai_repetition_penalty_thi_gui_qua_extra_body() -> None:
    """Backend vLLM cần tham số này, nếu không ảnh nhiều chữ lặp tới cụt token."""
    client = ClientGia()
    QwenVisionService(config=cau_hinh(vision_repetition_penalty=1.05), client=client).analyze(
        media_id="m", keyframes=mot_keyframe_ma_hoa_duoc()
    )

    assert client.kwargs["extra_body"] == {"repetition_penalty": 1.05}


@pytest.mark.parametrize("gia_tri", [None, "", "khong-phai-so", 0, 0.0])
def test_khong_khai_thi_khong_gui_tham_so_ngoai_chuan(gia_tri) -> None:
    """`repetition_penalty` không có trong chuẩn OpenAI.

    Gửi kèm tới endpoint không hiểu là ăn 400 và mất cả tầng caption, nên mọi
    giá trị chưa khai/không hợp lệ đều phải im lặng bỏ qua chứ không đoán bừa.
    """
    client = ClientGia()
    QwenVisionService(config=cau_hinh(vision_repetition_penalty=gia_tri), client=client).analyze(
        media_id="m", keyframes=mot_keyframe_ma_hoa_duoc()
    )

    # Không có dòng này thì một ảnh mã hoá hỏng cũng làm test xanh: `create`
    # không được gọi, `kwargs` rỗng, và "không có extra_body" đúng một cách vô nghĩa.
    assert client.kwargs, "chat.completions.create chưa từng được gọi"
    assert "extra_body" not in client.kwargs


def test_cau_hinh_cu_khong_co_truong_nay_van_chay() -> None:
    """AppConfig cũ (chưa có trường) không được vỡ — getattr phải có mặc định."""
    thieu_truong = SimpleNamespace(**DAY_DU)
    client = ClientGia()
    dich_vu = QwenVisionService(
        config=SimpleNamespace(
            media_models=thieu_truong,
            media_prompts=SimpleNamespace(vision_system_prompt=None, vision_user_prompt=None),
        ),
        client=client,
    )
    dich_vu.analyze(media_id="m", keyframes=mot_keyframe_ma_hoa_duoc())

    assert client.kwargs, "chat.completions.create chưa từng được gọi"
    assert "extra_body" not in client.kwargs


def test_json_cut_giua_chung_phai_bao_hong_chu_khong_thanh_caption() -> None:
    """Đo 07/08/2026: một ảnh poster trả về JSON cụt 9.095 ký tự.

    Nhánh fallback cũ ghi nguyên khối đó vào `caption_text` với trạng thái DONE,
    nên lượt chạy lại bỏ qua ảnh này vĩnh viễn và payload Qdrant nhiễm rác.
    """
    client = ClientGia()
    client._create = lambda **kw: SimpleNamespace(  # type: ignore[method-assign]
        choices=[SimpleNamespace(message=SimpleNamespace(content='{"ocr_text":"a","objects":["x","x","x'))]
    )
    client.chat = SimpleNamespace(completions=SimpleNamespace(create=client._create))

    ocr, caption = QwenVisionService(config=cau_hinh(), client=client).analyze(
        media_id="m", keyframes=mot_keyframe_ma_hoa_duoc()
    )

    assert caption.results[0].status is StageStatus.ERROR
    assert ocr.results[0].status is StageStatus.ERROR
    assert not caption.results[0].caption_text
    assert "JSON hỏng" in (caption.results[0].reason or "")


def test_van_xuoi_khong_phai_json_van_duoc_giu_lam_caption() -> None:
    """Model nào trả thẳng câu mô tả thì vẫn dùng được — đừng vạ lây."""
    client = ClientGia()
    client._create = lambda **kw: SimpleNamespace(  # type: ignore[method-assign]
        choices=[SimpleNamespace(message=SimpleNamespace(content="Một nhóm sinh viên đang chụp ảnh."))]
    )
    client.chat = SimpleNamespace(completions=SimpleNamespace(create=client._create))

    _, caption = QwenVisionService(config=cau_hinh(), client=client).analyze(
        media_id="m", keyframes=mot_keyframe_ma_hoa_duoc()
    )

    assert caption.results[0].status is StageStatus.DONE
    assert caption.results[0].caption_text == "Một nhóm sinh viên đang chụp ảnh."
