import json
import os
from pathlib import Path
from string import Template

import yaml
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(PROJECT_ROOT / ".env")


class ConfigObj:
    """Cho phép truy cập nested dictionary bằng dot notation."""

    def __init__(self, values: dict) -> None:
        self.__dict__.update(values)


def dict2obj(values: dict) -> ConfigObj:
    """Chuyển nested dictionary thành nested ConfigObj."""
    return json.loads(json.dumps(values), object_hook=ConfigObj)


def resolve_project_path(path: str | Path) -> Path:
    """Resolve đường dẫn tuyệt đối hoặc đường dẫn tương đối từ repository root."""
    resolved_path = Path(path).expanduser()
    if resolved_path.is_absolute():
        return resolved_path
    return PROJECT_ROOT / resolved_path


def yaml2obj(yaml_path: str | Path) -> ConfigObj:
    """Đọc YAML, thay ${ENV_VAR} bằng giá trị từ .env và trả về object."""
    config_path = resolve_project_path(yaml_path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy file cấu hình: {config_path}")

    raw = config_path.read_text(encoding="utf-8")
    substituted = Template(raw).safe_substitute(os.environ)
    data = yaml.safe_load(substituted) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Cấu hình YAML phải là object: {config_path}")
    return dict2obj(data)


def _config_path_from_env(env_var: str) -> Path:
    path = os.getenv(env_var)
    if not path:
        raise ValueError(f"Environment variable '{env_var}' is not set.")
    return resolve_project_path(path)


config_object = yaml2obj(_config_path_from_env("MCP_CONFIG_PATH"))
