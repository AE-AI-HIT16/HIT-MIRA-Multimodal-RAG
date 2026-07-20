import logging
import sys
from functools import wraps
from inspect import iscoroutinefunction
from typing import Callable


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def log_exceptions(logger: logging.Logger) -> Callable:
    """Log full traceback and return a structured error for MCP tool calls."""

    def decorator(func: Callable) -> Callable:
        if iscoroutinefunction(func):

            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                try:
                    return await func(*args, **kwargs)
                except Exception as exc:
                    logger.exception("Lỗi khi thực thi công cụ '%s'", func.__name__)
                    return {
                        "success": False,
                        "error_type": exc.__class__.__name__,
                        "message": str(exc),
                    }

            return async_wrapper

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as exc:
                logger.exception("Lỗi khi thực thi công cụ '%s'", func.__name__)
                return {
                    "success": False,
                    "error_type": exc.__class__.__name__,
                    "message": str(exc),
                }

        return sync_wrapper

    return decorator
