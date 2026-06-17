import functools
import sys
import time
from collections.abc import Callable
from pathlib import Path

from loguru import logger

from cellcounter.constants import CACHE_DIR


def configure_logger(
    level: str = "INFO",
    log_file: Path | str | None = None,
    *,
    json_output: bool = False,
) -> None:
    """Configures logging for the cellcounter package."""
    logger.remove()  # Clear defaults

    # Console: human-readable format
    logger.add(
        sys.stderr,
        level=level,
        format=(
            "<green>{time:HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | <cyan>{extra[run_id]}</cyan> | "
            "<level>{message}</level>"
        ),
    )

    # File: structured (JSON if json_output, else detailed)
    log_file = log_file or CACHE_DIR / "cellcounter.log"
    logger.add(
        log_file,
        level="DEBUG",
        rotation="10 MB",
        retention="7 days",
        compression="gz",
        serialize=json_output,
        format=(
            "{time} | {level} | {name}:{function}:{line} | {extra[run_id]} | {message}"
        ),
    )


def trace(_func: Callable | None = None, *, level: str = "DEBUG") -> Callable:
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
        name = f"{module_name}.{qual_name}"

        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> object:
            logger.info("→ {}() called", name)
            t0 = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                elapsed = time.perf_counter() - t0
                logger.log(level, "← {}() done ({:.2f}s)", name, elapsed)
                return result
            except Exception:
                elapsed = time.perf_counter() - t0
                logger.exception("✗ {}() FAILED after {:.2f}s", name, elapsed)
                raise

        return wrapper

    # allows both @trace and @trace(level=...)
    return decorator(_func) if _func else decorator
