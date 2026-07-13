import logging
from datetime import datetime
from pathlib import Path
import os
import shutil
import sys

from src.config import config_object


class OperationLog:
    def __init__(self, save_folder=None, log_life_circle=180, log_name="Operation Log"):
        self.log_life_circle = log_life_circle
        self.save_folder = save_folder
        self.log_name = log_name
        self.logger = None
        if not self.save_folder:
            self.save_folder = os.path.join(os.path.dirname(__file__), "Logs")

        # Check and create folder if not
        Path(self.save_folder).mkdir(parents=True, exist_ok=True)
        # Config Log
        self.config_log(flag=False)

    def config_log(self, flag=True):
        now_day = datetime.now().strftime('%Y-%m-%d')
        if now_day not in self.log_name or not flag:
            self.log_name = f"Log_{now_day}"
            self.logger = logging.getLogger(self.log_name)
            # Sử dụng caller_path và caller_line trong formatter
            formatter = logging.Formatter('%(asctime)s | %(caller_path)s:%(caller_line)d | %(message)s',
                                          datefmt='%Y-%m-%d %H:%M:%S')
            file_handler = logging.FileHandler(f"{self.save_folder}/{self.log_name}.log",
                                               mode='a', encoding="utf-8")
            file_handler.setFormatter(formatter)
            stream_handler = logging.StreamHandler(stream=sys.stdout)
            stream_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
            self.logger.addHandler(stream_handler)
            self.logger.propagate = False

    @staticmethod
    def _get_caller_info() -> dict:
        frame = sys._getframe(2)
        return {'caller_path': frame.f_code.co_filename, 'caller_line': frame.f_lineno}

    def debug(self, content):
        self.config_log()
        self.logger.setLevel(logging.DEBUG)
        extra = self._get_caller_info()
        self.logger.debug(f"[DEBUG]: {content}\n", extra=extra)

    def info(self, content):
        self.config_log()
        self.logger.setLevel(logging.INFO)
        extra = self._get_caller_info()

        self.logger.info(f"[INFO]: {content}\n", extra=extra)

    def warning(self, content):
        self.config_log()
        self.logger.setLevel(logging.WARNING)
        extra = self._get_caller_info()
        self.logger.warning(f"[WARNING]: {content}\n", extra=extra)

    def error(self, content):
        self.config_log()
        self.logger.setLevel(logging.ERROR)
        extra = self._get_caller_info()
        self.logger.error(f"[ERROR]: {content}\n", extra=extra)

    def delete_expired_folder(self):
        self.config_log()
        folder = os.path.dirname(self.save_folder)
        day = self.log_life_circle

        folder_paths_list = [f.path for f in os.scandir(folder) if f.is_dir()]
        for folder in folder_paths_list:
            folder_last_modified_time = max(os.stat(root).st_mtime for root, _, _ in os.walk(folder))
            modification_date = datetime.fromtimestamp(folder_last_modified_time)
            number_of_days = (datetime.now() - modification_date).days
            if number_of_days >= day:
                try:
                    shutil.rmtree(folder)
                    self.info(f"Logs folder: '{folder}' has been deleted")
                except Exception as err:
                    self.warning(f"Directory '%s' can not be removed. {err}" % folder)


logger = OperationLog(config_object.LOG.LOG_DIR)
