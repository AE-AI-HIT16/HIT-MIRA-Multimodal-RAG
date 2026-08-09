"""Hằng số dùng chung trong ChatBot."""

from pydantic import Json

DAY_MAP = {
    "Monday": "Thứ 2",
    "Tuesday": "Thứ 3",
    "Wednesday": "Thứ 4",
    "Thursday": "Thứ 5",
    "Friday": "Thứ 6",
    "Saturday": "Thứ 7",
    "Sunday": "Chủ nhật"
}

TYPE_MAPPING = {
    'string': str,
    'integer': int,
    'float': float,
    'boolean': bool,
    'list': list,
    'array': list,
    'object': dict,
    'datetime': str,
    'uuid': str,
    'json': Json,
}
