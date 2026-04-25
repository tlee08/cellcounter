import logging
import logging.handlers
from pathlib import Path

from cellcounter.constants import CACHE_DIR

LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def setup_logging(
    level: int = logging.INFO, log_file: Path | str | None = None
) -> None:
    """Configure logging once at application startup.

    After calling this, use: logger = logging.getLogger(__name__)
    """
    root = logging.getLogger("cellcounter")
    if root.handlers:  # if already configured
        return

    # Set logging levels and formatter
    root.setLevel(logging.DEBUG)
    formatter = logging.Formatter(LOG_FORMAT)

    # Console
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formatter)
    root.addHandler(ch)

    # File (with rotation)
    log_file = Path(log_file) if log_file else CACHE_DIR / "debug.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=10_000_000, backupCount=3
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    root.addHandler(fh)
