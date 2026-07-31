"""US-405.1 — link bài gốc phải đi hết đường từ PostgreSQL tới câu trích dẫn.

AC-1: có `source_url` thì hiển thị link tới bài gốc.
AC-2: không có thì hiển thị "nguồn nội bộ", **không** dựng link gãy.

Cột dữ liệu vốn đã có sẵn: `posts.post_url`, 496/496 bài đều có. Cái thiếu chỉ là
đường ống — nó không hề được đưa vào payload Qdrant, nên tầng trả lời không có gì
để trích dẫn ngoài `post_id`, một chuỗi UUID vô nghĩa với người đọc.

Chỗ nguy hiểm nằm ở AC-2: nếu context im lặng khi thiếu link, LLM rất dễ tự dựng
một URL trông hợp lý. Nói thẳng "nguồn nội bộ" là cách rẻ nhất để chặn.
"""

from __future__ import annotations

from types import SimpleNamespace
from uuid import uuid4

from src.rag_video_anh.repository.schemas import (
    MediaRecord,
    MediaType,
)
from src.rag_video_anh.retrieval.retrieval_service import VideoRetrievalService
from src.rag_video_anh.retrieval.retrieval_units import ImageRetrievalUnitBuilder, _post_source_url

POST_URL = "https://www.facebook.com/1679787570636388/posts/1001928175089001"


# ---------- Đọc link từ kho ----------


class FakePostRepository:
    def __init__(self, post_url: str | None) -> None:
        self.post_url = post_url
        self.asked: list[str] = []

    def get(self, post_id):
        self.asked.append(str(post_id))
        return SimpleNamespace(post_url=self.post_url)


def test_doc_duoc_link_bai_goc_tu_bang_posts() -> None:
    posts = FakePostRepository(POST_URL)
    uow = SimpleNamespace(posts=posts)
    post_id = uuid4()

    assert _post_source_url(uow, post_id) == POST_URL
    assert posts.asked == [str(post_id)]


def test_khong_co_link_thi_tra_none_chu_khong_tra_chuoi_rong() -> None:
    """`None` và `""` phân biệt được ở tầng hiển thị; đừng làm nhoè."""
    assert _post_source_url(SimpleNamespace(posts=FakePostRepository(None)), uuid4()) is None
    assert _post_source_url(SimpleNamespace(posts=FakePostRepository("   ")), uuid4()) is None


def test_kho_khong_co_repo_posts_thi_bo_qua_chu_khong_no() -> None:
    """Thiếu link không đáng để chặn cả mẻ index."""
    assert _post_source_url(SimpleNamespace(), uuid4()) is None
    assert _post_source_url(SimpleNamespace(posts=None), uuid4()) is None


def test_loi_khi_doc_posts_khong_lam_hong_ca_unit() -> None:
    class RepoHong:
        def get(self, post_id):
            raise RuntimeError("mất kết nối")

    assert _post_source_url(SimpleNamespace(posts=RepoHong()), uuid4()) is None


# ---------- Unit ảnh mang theo link ----------


class FakeMediaRepository:
    def __init__(self, media: MediaRecord) -> None:
        self.media = media

    def get_media(self, media_id):
        return self.media if str(media_id) == str(self.media.media_id) else None


class FakeResultRepository:
    def get_caption_result(self, media_id):
        return None

    def get_object_result(self, media_id):
        return None

    def get_ocr_result(self, media_id):
        return None


class FakeUnitOfWork:
    def __init__(self, media_repo, result_repo, posts_repo) -> None:
        self.media = media_repo
        self.results = result_repo
        self.posts = posts_repo

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return None


def test_unit_anh_mang_theo_source_url() -> None:
    media_id = uuid4()
    post_id = uuid4()
    media = MediaRecord(
        media_id=media_id,
        post_id=post_id,
        media_type=MediaType.IMAGE.value,
        bucket_name="mira-data",
        object_key="images/anh.jpg",
    )
    uow = FakeUnitOfWork(FakeMediaRepository(media), FakeResultRepository(), FakePostRepository(POST_URL))

    builder = ImageRetrievalUnitBuilder(uow_factory=lambda: uow, check_image_exists=False)
    unit = builder.build(media_id)

    assert unit is not None
    assert unit.to_dict()["source_url"] == POST_URL


# ---------- Chuỗi trích dẫn ----------


def test_context_gan_link_bai_goc() -> None:
    """AC-1: có link thì phải thấy link trong chuỗi mà tầng trả lời đọc."""
    from src.rag_video_anh.retrieval.retriever import MediaClipHit

    clip = MediaClipHit(
        score=0.8,
        video_id=None,
        unit_id="image:1",
        timestamp_sec=None,
        caption="Thành viên CLB chụp ảnh tập thể",
        ocr_text="",
        bucket_name="mira-data",
        frame_object_key="images/anh.jpg",
        media_kind="image",
        payload={"post_id": "post-1", "source_url": POST_URL},
    )

    context = VideoRetrievalService._format_context([clip], [])

    assert POST_URL in context
    assert "nguồn nội bộ" not in context


def test_context_noi_ro_nguon_noi_bo_khi_thieu_link() -> None:
    """AC-2: im lặng là mời LLM bịa ra một URL trông hợp lý."""
    from src.rag_video_anh.retrieval.retriever import MediaClipHit, TranscriptMoment, TranscriptVideoHit

    clip = MediaClipHit(
        score=0.8,
        video_id="video-1",
        unit_id="clip-1",
        timestamp_sec=75.0,
        caption="Sinh viên thuyết trình",
        ocr_text="",
        bucket_name="mira-data",
        frame_object_key="frames/1.jpg",
        payload={"post_id": "post-1"},
    )
    video = TranscriptVideoHit(
        video_id="video-1",
        score=0.7,
        moments=[TranscriptMoment(score=0.7, start_sec=10.0, end_sec=20.0, text="XIN CHÀO", unit_id="tr-1")],
    )

    context = VideoRetrievalService._format_context([clip], [video])

    assert context.count("nguồn nội bộ") == 2
    assert "http" not in context
