"""Timing decorator for image-processing operations."""

from __future__ import annotations

import functools
import logging
import time


def log_operation(logger_or_name):
    """Decorator that logs function name and execution time in milliseconds.

    Accepts either a :class:`logging.Logger` instance or a string name.

    Usage::

        @log_operation(logger)
        def expensive_function(img, ksize=5):
            ...

        @log_operation("my_operation")
        def another_function(img):
            ...
    """
    if isinstance(logger_or_name, str):
        _logger = logging.getLogger(logger_or_name)
    else:
        _logger = logger_or_name

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed_ms = (time.perf_counter() - start) * 1000
            _logger.info(
                "%s completed in %.1f ms", func.__name__, elapsed_ms,
            )
            return result

        return wrapper

    return decorator
