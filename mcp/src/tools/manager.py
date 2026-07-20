import os
import yaml
from config.configs import config_object, resolve_project_path
from logger import get_logger

logger = get_logger(__name__)


class CustomToolManager:
    def __init__(self):
        self.tools_by_category = self._load_tools()

    @staticmethod
    def _load_tools() -> dict:
        tools_file = str(resolve_project_path(config_object.RESOURCES.TOOLS_PATH))

        if not os.path.exists(tools_file):
            logger.error("Không tìm thấy tệp cấu hình công cụ: %s", tools_file)
            raise FileNotFoundError(f"Không tìm thấy tệp cấu hình công cụ: {tools_file}")

        with open(tools_file, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        tools_by_category = {}
        for tool in data.get("tools", []):
            category = tool["category"]
            tools_by_category.setdefault(category, {})
            tools_by_category[category][tool["name_tool"]] = {
                "name_tool": tool["name_tool"],
                "description": tool["description"],
                "inputSchema": tool.get("inputSchema"),
            }

        total = sum(len(v) for v in tools_by_category.values())
        logger.info(
            "Đã tải %s công cụ từ tools.yaml; các nhóm: %s",
            total,
            list(tools_by_category.keys()),
        )
        return tools_by_category

    def get_tools_by_category(self, category: str) -> dict:
        return self.tools_by_category.get(category, {})

    def reload(self):
        """Cho phép reload tools.yaml khi runtime mà không cần restart server."""
        self.tools_by_category = self._load_tools()
