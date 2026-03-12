"""Timing decorator for image-processing operations."""

from __future__ import annotations

import functools
import logging
import time


def log_operation(logger: logging.Logger):
    """Decorator that logs function name and execution time in milliseconds.

    Usage::

        @log_operation(logger)
        def expensive_function(img, ksize=5):
            ...
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.info(
                "%s completed in %.1f ms", func.__name__, elapsed_ms,
            )
            return result

        return wrapper

    return decorator
