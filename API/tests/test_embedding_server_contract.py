"""`ImageEmbeddingService` phải nói chuyện được với `embedding_server` y như với Jina.

Cả việc tự host đứng hay đổ đều nằm ở chỗ này: nếu server tự viết lệch giao
thức dù chỉ một trường, thì hoặc là vỡ ngay (còn may), hoặc là trả về vector
xếp sai thứ tự — lúc đó ảnh A mang vector của ảnh B, Qdrant nhận hết, và không
có lỗi nào được ném ra. Chỉ tới khi ai đó hỏi chatbot mới thấy kết quả vô lý.

Test chạy hoàn toàn offline: encoder thật bị thay bằng bản giả trả vector đếm
được, nên không cần GPU, không cần 3,5GB trọng số, không cần mạng.
"""

from __future__ import annotations

import base64
import io
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "embedding_server"))

from src.rag_video_anh.embedding.embedding_service import (  # noqa: E402
    ImageEmbeddingService,
)

fastapi_testclient = pytest.importorskip("fastapi.testclient")

DIM = 1024


class _EncoderGia:
    """Trả vector nhận dạng được: phần tử đầu là chỉ số, để bắt lỗi xếp sai thứ tự."""

    def __init__(self) -> None:
        self.lan_goi: list[tuple[str, int]] = []

    def _vectors(self, n: int) -> list[list[float]]:
        return [[float(i)] + [0.0] * (DIM - 1) for i in range(n)]

    def encode_texts(self, texts):
        self.lan_goi.append(("text", len(texts)))
        return self._vectors(len(texts))

    def encode_images(self, images):
        self.lan_goi.append(("image", len(images)))
        return self._vectors(len(images))

    # `server.health` đọc hai thuộc tính này.
    device = "cpu"
    ready = True


@pytest.fixture
def app_va_encoder(monkeypatch):
    import server

    encoder_gia = _EncoderGia()
    monkeypatch.setattr(server, "encoder", encoder_gia)
    monkeypatch.setattr(server, "API_KEY", "", raising=False)
    return server.app, encoder_gia


@pytest.fixture
def service_noi_vao(app_va_encoder):
    """Dựng service thật, nhưng đẩy HTTP của nó vào thẳng app trong tiến trình."""
    app, encoder_gia = app_va_encoder
    client = fastapi_testclient.TestClient(app)

    def post_gia(url, **kwargs):
        return client.post("/v1/embeddings", json=kwargs.get("json"), headers=kwargs.get("headers"))

    service = ImageEmbeddingService(
        api_key="local",
        base_url="http://noi-bo/v1/embeddings",
        dimensions=DIM,
        http_post=post_gia,
        tokens_per_minute=0,  # tắt giữ nhịp, test không cần chờ
    )
    return service, encoder_gia


def _anh_jpeg_tam(thu_muc: Path, mau: tuple[int, int, int]) -> Path:
    from PIL import Image

    duong_dan = thu_muc / f"anh_{mau[0]}.jpg"
    Image.new("RGB", (64, 64), mau).save(duong_dan, format="JPEG")
    return duong_dan


def test_nhung_text_qua_duoc_server(service_noi_vao):
    """Đường lời thoại và câu hỏi người dùng."""
    service, encoder_gia = service_noi_vao
    vectors = service.embed_texts(["xin chào", "hội thảo CLB"])

    assert len(vectors) == 2
    assert all(len(v) == DIM for v in vectors)
    assert encoder_gia.lan_goi == [("text", 2)]


def test_nhung_anh_qua_duoc_server(service_noi_vao, tmp_path):
    """Đường keyframe và ảnh tĩnh."""
    service, encoder_gia = service_noi_vao
    paths = [_anh_jpeg_tam(tmp_path, (10, 20, 30)), _anh_jpeg_tam(tmp_path, (200, 100, 50))]

    vectors = service.embed_images([str(p) for p in paths])

    assert len(vectors) == 2
    assert all(len(v) == DIM for v in vectors)
    assert encoder_gia.lan_goi == [("image", 2)]


def test_giu_dung_thu_tu_theo_index(service_noi_vao):
    """Vector thứ i phải là của input thứ i.

    Client sắp lại theo trường `index`, nên server trả sai thứ tự vẫn "chạy" —
    chỉ là mọi vector gắn nhầm chủ. Vector giả mang chính chỉ số ở phần tử đầu
    nên chỗ lệch lộ ra ngay.
    """
    service, _ = service_noi_vao
    vectors = service.embed_texts(["một", "hai", "ba", "bốn"])
    assert [v[0] for v in vectors] == [0.0, 1.0, 2.0, 3.0]


def test_sai_so_chieu_bi_tu_choi_ngay(app_va_encoder):
    """Số chiều lệch mà lọt qua thì Qdrant mới báo, sau khi đã tốn cả mẻ nhúng."""
    app, _ = app_va_encoder
    client = fastapi_testclient.TestClient(app)

    phan_hoi = client.post(
        "/v1/embeddings",
        json={"model": "jina-clip-v2", "input": [{"text": "a"}], "dimensions": 512},
    )
    assert phan_hoi.status_code == 422
    assert "512" in phan_hoi.json()["detail"]


def test_tron_text_va_anh_bi_tu_choi(app_va_encoder):
    """Trộn hai loại thì `index` trả về không còn khớp input của client."""
    app, _ = app_va_encoder
    client = fastapi_testclient.TestClient(app)

    anh = base64.b64encode(_jpeg_bytes()).decode()
    phan_hoi = client.post(
        "/v1/embeddings",
        json={"model": "jina-clip-v2", "input": [{"text": "a"}, {"image": anh}]},
    )
    assert phan_hoi.status_code == 422


def test_sai_khoa_thi_401_va_client_coi_la_loi_chet(app_va_encoder, monkeypatch):
    """401/402/403 phải nổi lên thành `ImageEmbeddingProviderFatalError`.

    Đây chính là lưới đã bắt được vụ hết số dư: nếu server tự host trả 401 mà
    client coi là "ảnh này hỏng" thì mẻ index lại chạy tiếp hàng chục video,
    nhúng được số không, rồi báo thành công.
    """
    import server
    from src.rag_video_anh.embedding.embedding_service import (
        ImageEmbeddingProviderFatalError,
    )

    monkeypatch.setattr(server, "API_KEY", "khoa-that", raising=False)
    client = fastapi_testclient.TestClient(server.app)

    def post_gia(url, **kwargs):
        return client.post("/v1/embeddings", json=kwargs.get("json"), headers=kwargs.get("headers"))

    service = ImageEmbeddingService(
        api_key="khoa-sai",
        base_url="http://noi-bo/v1/embeddings",
        dimensions=DIM,
        http_post=post_gia,
        tokens_per_minute=0,
    )
    with pytest.raises(ImageEmbeddingProviderFatalError):
        service.embed_texts(["xin chào"])


def test_health_khong_can_model_da_nap(app_va_encoder):
    """Health check phải trả lời được trong lúc model còn đang nạp."""
    app, _ = app_va_encoder
    phan_hoi = fastapi_testclient.TestClient(app).get("/health")
    assert phan_hoi.status_code == 200
    assert phan_hoi.json()["status"] == "ok"


def _jpeg_bytes() -> bytes:
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (16, 16), (1, 2, 3)).save(buffer, format="JPEG")
    return buffer.getvalue()
