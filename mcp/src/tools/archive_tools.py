from src.clients import archive_api_client as api

async def search_archive(keyword: str = None, status: str = None, warehouseId: str = None,
                          language: str = None, maintenance: str = None,
                          createdFrom: str = None, createdTo: str = None,
                          updatedFrom: str = None, updatedTo: str = None,
                          page: int = 1, size: int = 10) -> dict:
    """Tìm kiếm danh sách hồ sơ lưu trữ theo từ khóa/bộ lọc."""
    return await api.search_archives(keyword, status, warehouseId, language,
                                      maintenance, createdFrom, createdTo,
                                      updatedFrom, updatedTo, page, size)

async def get_archive_detail(archive_id: str) -> dict:
    """Lấy chi tiết một hồ sơ theo id (chỉ dùng khi đã có id, thường lấy từ kết quả search_archive)."""
    return await api.get_archive_detail(archive_id)

async def get_staff_archive_metadata(only_metadata: bool = True) -> dict:
    """Lấy cấu trúc / metadata hồ sơ cán bộ."""
    return await api.get_staff_archive_metadata(only_metadata)

async def get_archive_file(key: str, fileName: str) -> dict:
    """Trả về link để xem/tải file đính kèm trong hồ sơ."""
    url = await api.get_file_proxy_url(key, fileName)
    return {"fileUrl": url, "fileName": fileName}