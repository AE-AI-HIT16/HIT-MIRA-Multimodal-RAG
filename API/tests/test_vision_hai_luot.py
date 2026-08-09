"""Tách OCR và caption thành hai lời gọi, lượt 2 nhận bối cảnh của lượt 1.

Mặc định vẫn là MỘT lời gọi — hai lượt nhân đôi chi phí VLM (đo trên kho này:
~1,7 giờ lên ~3,5 giờ cho 4.986 media). Bộ test khoá lại cả hai chế độ, và khoá
hình dạng dict trả về phải giống nhau để mọi hàm ghi DB phía sau không phải biết
đã chạy mấy lượt.
"""

from __future__ import annotations

import json

import pytest

from src.rag_video_anh.pipeline.qwen_vision_service import QwenVisionService
from src.rag_video_anh.schemas import KeyFrame, KeyFrameSet, StageStatus


class FakeClient:
    """Client OpenAI giả, trả lần lượt các phản hồi đã dựng sẵn."""

    def __init__(self, phan_hoi: list[dict]) -> None:
        self.phan_hoi = list(phan_hoi)
        self.messages_da_gui: list[list] = []
        self.chat = self

    @property
    def completions(self):
        return self

    def create(self, *, model, messages, **kwargs):
        self.messages_da_gui.append(messages)
        noi_dung = json.dumps(self.phan_hoi.pop(0), ensure_ascii=False)
        return type(
            "Resp",
            (),
            {"choices": [type("C", (), {"message": type("M", (), {"content": noi_dung})()})()]},
        )()


def _service(client, hai_luot: bool) -> QwenVisionService:
    service = QwenVisionService(client=client)
    service.model_config = service.model_config.model_copy(update={"vision_two_pass": hai_luot})
    return service


def _keyframes() -> KeyFrameSet:
    return KeyFrameSet(
        media_id="m1",
        frames=[
            KeyFrame(
                media_id="m1",
                frame_id="f1",
                frame_index=0,
                timestamp_sec=0.0,
                timestamp_ms=0,
                image_payload=b"\xff\xd8\xff",
            )
        ],
        status=StageStatus.DONE,
    )


def test_mot_luot_van_la_mac_dinh_va_chi_goi_mot_lan():
    client = FakeClient([{"ocr_text": "HIT CLUB", "caption_text": "Một tấm poster"}])
    service = _service(client, hai_luot=False)

    ocr_set, caption_set = service.analyze(media_id="m1", keyframes=_keyframes())

    assert len(client.messages_da_gui) == 1
    assert ocr_set.results[0].full_text == "HIT CLUB"
    assert caption_set.results[0].caption_text == "Một tấm poster"


def test_hai_luot_goi_dung_hai_lan_va_giu_nguyen_hinh_dang_ket_qua():
    client = FakeClient(
        [
            {"ocr_text": "HIT CLUB 2024", "ocr_blocks": ["HIT CLUB", "2024"]},
            {"caption_text": "Poster tuyển thành viên CLB HIT năm 2024", "scene": "indoor", "keywords": ["poster"]},
        ]
    )
    service = _service(client, hai_luot=True)

    ocr_set, caption_set = service.analyze(media_id="m1", keyframes=_keyframes())

    assert len(client.messages_da_gui) == 2
    # OCR lấy của lượt 1, caption lấy của lượt 2 — không trộn nguồn.
    assert ocr_set.results[0].full_text == "HIT CLUB 2024"
    assert caption_set.results[0].caption_text == "Poster tuyển thành viên CLB HIT năm 2024"
    assert caption_set.results[0].generation_meta["keywords"] == ["poster"]


def test_luot_hai_nhan_duoc_chu_ocr_cua_luot_mot():
    client = FakeClient([{"ocr_text": "ĐẠI HỘI CLB"}, {"caption_text": "Ảnh đại hội"}])
    service = _service(client, hai_luot=True)

    service.analyze(media_id="m1", keyframes=_keyframes())

    van_ban_luot_hai = client.messages_da_gui[1][1]["content"][0]["text"]
    assert "ĐẠI HỘI CLB" in van_ban_luot_hai
    # Lượt 1 không được biết gì về bối cảnh — nó chỉ đọc chữ.
    assert "BỐI CẢNH" not in client.messages_da_gui[0][1]["content"][0]["text"]


def test_luot_hai_nhan_lời_thoai_va_nhan_vat_the():
    client = FakeClient([{"ocr_text": ""}, {"caption_text": "x"}])
    service = _service(client, hai_luot=True)

    service.analyze(
        media_id="m1",
        keyframes=_keyframes(),
        caption_policy={"context": {"asr": "chào mừng các bạn", "object_counts": {"person": 3, "chair": 1}}},
    )

    van_ban = client.messages_da_gui[1][1]["content"][0]["text"]
    assert "chào mừng các bạn" in van_ban
    assert "person x3" in van_ban
    assert "chair x1" in van_ban
    # Hàng rào chống bịa phải đi kèm bối cảnh, nếu không caption sẽ kể việc chỉ
    # nghe thấy mà không nhìn thấy.
    assert "TUYỆT ĐỐI không mô tả điều chỉ nghe thấy" in van_ban


def test_khong_co_boi_canh_thi_khong_chen_khoi_rong():
    client = FakeClient([{"ocr_text": ""}, {"caption_text": "x"}])
    service = _service(client, hai_luot=True)

    service.analyze(media_id="m1", keyframes=_keyframes())

    assert "BỐI CẢNH" not in client.messages_da_gui[1][1]["content"][0]["text"]


@pytest.mark.parametrize("hai_luot, hau_to", [(False, False), (True, True)])
def test_prompt_version_ghi_ro_da_chay_may_luot(hai_luot, hau_to):
    """Hai cách sinh khác hẳn nhau thì không được mang cùng một nhãn."""
    client = FakeClient([{"ocr_text": ""}, {"caption_text": "x"}])
    service = _service(client, hai_luot=hai_luot)

    _, caption_set = service.analyze(media_id="m1", keyframes=_keyframes())

    assert caption_set.results[0].prompt_version.endswith("-2pass") is hau_to


# --------------------------------------------------------------------------
# Trần cho khối lời thoại — xem ImageProcessingWorker.GIOI_HAN_KY_TU_ASR
# --------------------------------------------------------------------------


class _Doan:
    def __init__(self, start_time, end_time, text):
        self.start_time, self.end_time, self.text = start_time, end_time, text


class _Uow:
    def __init__(self, transcript):
        self.results = type("R", (), {"get_transcript_by_video_media_id": lambda _s, _v: transcript})()


class _Frame:
    video_media_id = "v1"

    def __init__(self, timestamp):
        self.timestamp = timestamp


def test_doan_asr_dai_bi_cat_quanh_vi_tri_khung_hinh():
    """Kho này gộp cả bài nói vào 3-4 đoạn; lấy nguyên đoạn là mời model bịa."""
    from src.rag_video_anh.pipeline.image_processing_worker import (
        GIOI_HAN_KY_TU_ASR,
        ImageProcessingWorker,
    )

    # Mốc đặt đúng giữa đoạn: khung hình ở giây 50/100 thì cửa sổ phải trùm nó.
    # (Chuỗi lặp đều không dùng được — mọi lát cắt sẽ trông y hệt nhau.)
    chu = "a" * 2000 + "MOC-GIUA-DOAN" + "b" * 2000
    transcript = type("T", (), {"segments": [_Doan(0.0, 100.0, chu)]})()

    ket_qua = ImageProcessingWorker._loi_noi_quanh_khung_hinh(_Uow(transcript), _Frame(50.0))

    assert len(ket_qua) == GIOI_HAN_KY_TU_ASR
    assert "MOC-GIUA-DOAN" in ket_qua
    # Cắt quanh vị trí khung hình, không phải từ đầu đoạn.
    assert ket_qua != chu[:GIOI_HAN_KY_TU_ASR]


def test_khung_hinh_ngoai_moi_doan_thi_khong_co_loi_thoai():
    from src.rag_video_anh.pipeline.image_processing_worker import ImageProcessingWorker

    transcript = type("T", (), {"segments": [_Doan(0.0, 10.0, "xin chào")]})()

    assert ImageProcessingWorker._loi_noi_quanh_khung_hinh(_Uow(transcript), _Frame(999.0)) == ""
