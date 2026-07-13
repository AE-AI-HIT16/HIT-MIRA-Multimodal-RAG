"""
Cấu hình tập trung cho ChatBot.

Sử dụng cùng pattern với chatbot_homepage:
- Đọc file YAML config
- Thay thế biến ${ENV_VAR} bằng giá trị thực từ .env
- Chuyển đổi dict -> object (dot notation access)
"""
import json
import os

import yaml
from dotenv import load_dotenv
from string import Template

load_dotenv()


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


config_object = yaml2obj(os.getenv("CONFIG_PATH"))
config_prompts = yaml2obj(os.getenv("PROMPTS_PATH"))
config_models = yaml2obj(os.getenv("MODELS_PATH"))
config_messages = yaml2obj(os.getenv("MESSAGES_PATH"))
config_agents = yaml2obj(os.getenv("AGENTS_PATH"))
