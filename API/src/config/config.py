import json
import os
from pathlib import Path

import yaml
from dotenv import load_dotenv
from string import Template

# Resolve paths: config.py -> config/ -> src/ -> API/ -> project root
_API_ROOT = Path(__file__).resolve().parents[2]
_PROJECT_ROOT = _API_ROOT.parent
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


DEFAULT_CONFIG_PATHS = {
    "CONFIG_PATH": _API_ROOT / "Resources" / "dev.yaml",
    "PROMPTS_PATH": _API_ROOT / "Resources" / "prompt.yaml",
    "MODELS_PATH": _API_ROOT / "Resources" / "model.yaml",
}


def resolve_project_path(value: str | Path) -> Path:
    """Resolve a YAML path across repo-root and API-root layouts."""
    path = Path(value)
    if path.is_absolute():
        return path

    candidates = [_PROJECT_ROOT / path, _API_ROOT / path]
    if path.parts and path.parts[0] == "API":
        candidates.append(_API_ROOT.joinpath(*path.parts[1:]))

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _resolve_path(env_var: str) -> str:
    """Resolve a config path from env var or the API defaults."""
    configured_path = os.getenv(env_var)
    if not configured_path:
        return str(DEFAULT_CONFIG_PATHS[env_var])
    return str(resolve_project_path(configured_path))


config_object = yaml2obj(_resolve_path("CONFIG_PATH"))
config_prompts = yaml2obj(_resolve_path("PROMPTS_PATH"))
config_models = yaml2obj(_resolve_path("MODELS_PATH"))

