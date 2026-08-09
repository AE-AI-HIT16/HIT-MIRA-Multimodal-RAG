"""Cổng chặn ghi đè cho những khâu KHÔNG HỀ CHẠY.

Mọi `upsert_*` của repository đều ghi đè vô điều kiện, nên một khâu bị tắt theo
route mà vẫn gọi upsert sẽ xoá dữ liệu thật của lượt chạy trước. Đo được ngày
08/08/2026 trên chính kho này:

  * Một lượt caption với detection tắt đã xoá **17.262** object YOLO, để lại
    4.986 hàng `object_results` mang trạng thái `DONE` — nhìn từ mọi phía đều
    giống "YOLO chạy xong và không thấy vật gì".
  * `_persist_ocr`/`_persist_captions` còn ghi thẳng chuỗi chẩn đoán
    ("OCR disabled by route") vào đúng ô nội dung, rồi chuỗi đó đi tiếp vào
    payload Qdrant và hiện ra như một trích dẫn.

Điểm mấu chốt: **"chạy rồi mà không thấy gì" và "không hề chạy" là hai chuyện
khác nhau**, và chỉ `reason` phân biệt được. Vì vậy dấu hiệu này là hợp đồng —
đúng dấu hiệu mà `PipelineService._pipeline_status` đang dùng để khỏi báo
`partial_success` oan cho các khâu tắt theo route.

Hệ quả cho người viết khâu mới: khâu nào tắt theo route thì `reason` BẮT BUỘC
phải chứa cụm dưới đây, nếu không dữ liệu của khâu đó sẽ bị xoá âm thầm.
"""

from __future__ import annotations

DAU_HIEU_TAT_THEO_ROUTE = "disabled by route"


def noi_dung_ghi_duoc(noi_dung: str | None, status: object) -> str | None:
    """Nội dung để đưa xuống DB, hoặc None nếu phải giữ nguyên cái cũ.

    Khung hình chạy hỏng thì `full_text`/`caption_text` mang giá trị mặc định
    `""` chứ không phải None — ghi thẳng nó xuống là xoá trắng kết quả tốt của
    lượt trước chỉ vì lượt này gặp lỗi mạng. Trạng thái vẫn được ghi, nên lượt
    sau vẫn biết đường thử lại.

    Và tuyệt đối không lấy `reason` làm nội dung: đó là câu chẩn đoán
    ("Qwen vision analysis failed: ..."), nó sẽ đi tiếp vào payload Qdrant rồi
    hiện ra như một trích dẫn của chính bức ảnh.
    """
    return None if str(getattr(status, "value", status)).strip().lower() == "error" else noi_dung


def khau_da_chay(reason: str | None) -> bool:
    """False khi khâu bị tắt theo route — lúc đó tuyệt đối không ghi gì vào DB.

    Cố ý chỉ chặn đúng trường hợp "tắt theo route" chứ không chặn mọi `SKIPPED`:
    một khâu chạy rồi hỏng vẫn cần ghi lại trạng thái hỏng, nếu không thì lượt
    sau không biết đường thử lại.
    """
    return DAU_HIEU_TAT_THEO_ROUTE not in str(reason or "").strip().lower()
