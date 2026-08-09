"""Bake các dependency code/config nhỏ của jina-clip-v2 vào Docker image.

RunPod chỉ cache một model cho mỗi endpoint. Weights lớn của jina-clip-v2 ở
volume đó, còn các repo được `auto_map` tham chiếu phải có sẵn trong image để
worker chạy hoàn toàn offline.
"""

from pathlib import Path

from huggingface_hub import snapshot_download

DEPENDENCIES = (
    (
        "jinaai/jina-clip-implementation",
        "39e6a55ae971b59bea6e44675d237c99762e7ee2",
        ["*.py"],
    ),
    (
        "jinaai/jina-embeddings-v3",
        "ab036b023d30b4d1138c4c3bfa9f0c445ab455d6",
        [
            "config.json",
            "special_tokens_map.json",
            "tokenizer.json",
            "tokenizer_config.json",
        ],
    ),
    (
        "jinaai/xlm-roberta-flash-implementation",
        "845308d0fd72a8406a3e378450e1a09522790419",
        ["*.py"],
    ),
)


def main() -> None:
    for repo_id, revision, allow_patterns in DEPENDENCIES:
        snapshot = Path(
            snapshot_download(
                repo_id,
                revision=revision,
                allow_patterns=allow_patterns,
            )
        )
        # Runtime gọi các repo này với revision mặc định `main`. Khi download
        # bằng commit hash, huggingface_hub không luôn tạo refs/main; tạo rõ để
        # offline lookup tìm đúng snapshot đã ghim.
        refs = snapshot.parent.parent / "refs"
        refs.mkdir(exist_ok=True)
        # huggingface_hub 0.36 đọc refs bằng ``f.read()`` mà không strip;
        # newline sẽ trở thành một phần commit hash và làm cache lookup trượt.
        (refs / "main").write_text(revision, encoding="utf-8")
        print(f"cached {repo_id}@{revision}")


if __name__ == "__main__":
    main()
