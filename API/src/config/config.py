"""
Cấu hình tập trung cho ChatBot.

Sử dụng cùng pattern với chatbot_homepage:
- Đọc file YAML config
- Thay thế biến ${ENV_VAR} bằng giá trị thực từ .env
- Chuyển đổi dict -> object (dot notation access)
"""
import json
import os
from pathlib import Path

import yaml
from dotenv import load_dotenv
from string import Template

# Resolve project root: config.py -> config/ -> src/ -> API/ -> project root
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(_PROJECT_ROOT / ".env")


class ConfigObj:
    """Cho phép truy cập dict dưới dạng object (dot notation)."""
    def __init__(self, dict1):
        self.__dict__.update(dict1)


def dict2obj(dict1):
    """Chuyển nested dict thành nested object."""
    return json.loads(json.dumps(dict1), object_hook=ConfigObj)


def yaml2obj(yaml_path):
    """Đọc file YAML, thay thế biến môi trường, trả về object."""
    with open(yaml_path) as f:
        raw = f.read()
        template = Template(raw)
        substituted = template.safe_substitute(os.environ)
    data_load = yaml.safe_load(substituted)
    config_obj = dict2obj(data_load)
    return config_obj


def _resolve_path(env_var: str) -> str:
    """Resolve a path from env var relative to project root."""
    path = os.getenv(env_var)
    if path is None:
        raise ValueError(f"Environment variable '{env_var}' is not set.")
    return str(_PROJECT_ROOT / path)


config_object = yaml2obj(_resolve_path("CONFIG_PATH"))
config_prompts = yaml2obj(_resolve_path("PROMPTS_PATH"))
config_models = yaml2obj(_resolve_path("MODELS_PATH"))

