from rag.config.rag_config import rag_config
rag_config.CHUNK_SIZE_CHARS = 10
rag_config.CHUNK_OVERLAP_CHARS = 2

def _chunk_text(pages: list[str]) -> list[dict]:
    chunk_size = rag_config.CHUNK_SIZE_CHARS
    overlap = rag_config.CHUNK_OVERLAP_CHARS

    full_text = ""
    char_to_page = []
    for page_no, page_text in enumerate(pages, start=1):
        full_text += page_text
        char_to_page.extend([page_no] * len(page_text))

    full_text = full_text.strip()
    if not full_text:
        return []

    chunks = []
    start = 0
    while start < len(full_text):
        end = min(start + chunk_size, len(full_text))
        chunk_str = full_text[start:end].strip()
        if chunk_str:
            page_idx = min(start, len(char_to_page) - 1) if char_to_page else 0
            page_number = char_to_page[page_idx] if char_to_page else 1
            chunks.append({"text": chunk_str, "page_number": page_number})
        if end == len(full_text):
            break
        start = end - overlap
    return chunks

print(_chunk_text(["  abcde\n", "fghijk\n"]))
