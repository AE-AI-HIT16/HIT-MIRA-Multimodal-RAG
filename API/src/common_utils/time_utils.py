"""Tiện ích thời gian cho ChatBot."""

from datetime import datetime

from src.common_utils.constants import DAY_MAP


def get_system_time():
    systems_time = datetime.now().isoformat()
    return systems_time


def get_current_date_info():
    now = datetime.now()

    # Biến chứa ngày hiện tại theo định dạng YYYY-MM-DD %H:%M
    current_time = now.strftime("%Y-%m-%d %H:%M")

    day_of_week = DAY_MAP.get(now.strftime("%A"), "Unknown")
    return current_time, day_of_week
