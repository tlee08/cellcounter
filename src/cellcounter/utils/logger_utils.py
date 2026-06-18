import functools
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Literal

from loguru import logger

from cellcounter.constants import CACHE_DIR

type LogLevel = Literal[
    "TRACE",
    "DEBUG",
    "INFO",
    "SUCCESS",
    "WARNING",
    "ERROR",
    "CRITICAL",
]


def configure_logger(
    level: LogLevel = "INFO",
    log_file: Path | str | None = None,
    *,
    json_output: bool = False,
) -> None:
    """Configures logging for the cellcounter package."""
    # Clear defaults
    logger.remove()
    # Set configs
    logger.configure(extra={"func_name": "-"})
    # Console: human-readable format
    logger.add(
        sys.stderr,
        level=level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{extra[func_name]}</cyan> | "
            "<level>{message}</level>"
        ),
    )
    # File: structured (JSON if json_output, else detailed)
    log_file = log_file or CACHE_DIR / "cellcounter.log"
    logger.add(
        log_file,
        level="DEBUG",
        rotation="50 MB",
        retention="60 days",
        compression="gz",
        serialize=json_output,
        format=(
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
            "{level} | "
            "{name}:{function}:{line} | "
            "{extra[func_name]} | "
            "{message}"
        ),
    )


def trace(_func: Callable | None = None, *, level: LogLevel = "INFO") -> Callable:
    """Log function entry, exit, duration. Exception → full traceback.

    Usage:
    ```
    @trace
    def foo(): ...

    @trace(level="TRACE")
    def hot_path(): ...
    ```
    """

    def decorator(func: Callable) -> Callable:
        module_name = func.__module__
        qual_name = getattr(func, "__qualname__", "unknown_callable")
        func_name = f"{module_name}.{qual_name}"

        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> object:
            with logger.contextualize(func_name=func_name):
                logger.log(level, "→ called")
                t0 = time.perf_counter()
                try:
                    result = func(*args, **kwargs)
                    elapsed = time.perf_counter() - t0
                    logger.log(level, "← done ({:.2f}s)", elapsed)
                    return result
                except Exception:
                    elapsed = time.perf_counter() - t0
                    logger.exception("✗ FAILED after {:.2f}s", elapsed)
                    raise

        return wrapper

    # allows both @trace and @trace(level=...)
    return decorator(_func) if _func else decorator
