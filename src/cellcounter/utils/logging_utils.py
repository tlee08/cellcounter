import io
import logging
import os
import sys

from cellcounter.constants import CACHE_DIR
from cellcounter.utils.misc_utils import get_func_name_in_stack

LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_IO_OBJ_FORMAT = "%(levelname)s - %(message)s"


def add_console_handler(logger: logging.Logger, level: int) -> None:
    """
    Add a console (stderr) handler to the logger if not already present.

    Parameters
    ----------
    logger : logging.Logger
        The logger to which the handler will be added.
    level : int
        Logging level for the console handler.

    Returns
    -------
    None
    """
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler) and handler.stream == sys.stderr:
            return
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    logger.addHandler(console_handler)


def add_log_file_handler(logger: logging.Logger, level: int, log_fp: str | None = None) -> str:
    """
    Add a file handler to the logger if not already present.

    Parameters
    ----------
    logger : logging.Logger
        The logger to which the handler will be added.
    level : int
        Logging level for the file handler.
    log_fp : str, optional
        Custom log file path. If None, uses default cache dir log file.

    Returns
    -------
    str
        The log file path used by the handler. Returns an empty string if handler setup fails.
    """
    log_fp = log_fp or os.path.join(CACHE_DIR, "debug.log")
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler) and handler.baseFilename == log_fp:
            return handler.baseFilename
    try:
        os.makedirs(os.path.dirname(log_fp), exist_ok=True)
        file_handler = logging.FileHandler(log_fp, mode="a")
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
        logger.addHandler(file_handler)
        return file_handler.baseFilename
    except Exception as e:
        logger.error(f"Failed to add file handler: {e}")
        return ""


def add_io_obj_handler(logger: logging.Logger, level: int) -> io.StringIO:
    """
    Add a StringIO handler to the logger if not already present.

    Parameters
    ----------
    logger : logging.Logger
        The logger to which the handler will be added.
    level : int
        Logging level for the StringIO handler.

    Returns
    -------
    io.StringIO
        The StringIO object used by the handler.
    """
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler) and isinstance(handler.stream, io.StringIO):
            return handler.stream
    io_obj = io.StringIO()
    io_obj_handler = logging.StreamHandler(io_obj)
    io_obj_handler.setLevel(level)
    io_obj_handler.setFormatter(logging.Formatter(LOG_IO_OBJ_FORMAT))
    logger.addHandler(io_obj_handler)
    return io_obj


def reset_handlers(logger: logging.Logger) -> None:
    """
    Remove all handlers from the logger.

    Parameters
    ----------
    logger : logging.Logger
        The logger whose handlers will be removed.

    Returns
    -------
    None
    """
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()


def init_logger(
    name: str | None = None,
    console_level: int | None = None,
    file_level: int | None = None,
    io_obj_level: int | None = None,
    log_fp: str | None = None,
) -> logging.Logger:
    """
    Set up and return a logger with configurable handlers and options.

    Parameters
    ----------
    name : str, optional
        Logger name. If None, uses the caller's function name.
    console_level : int, optional
        Logging level for the console handler. If None, not added.
    file_level : int, optional
        Logging level for the file handler. If None, not added.
    io_obj_level : int, optional
        Logging level for the StringIO handler. If None, not added.
    log_fp : str, optional
        Custom log file path. If None, uses default cache dir log file.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    logger = logging.getLogger(name or get_func_name_in_stack(2))
    logger.setLevel(logging.DEBUG)
    reset_handlers(logger)
    if console_level is not None:
        add_console_handler(logger, console_level)
    if file_level is not None:
        add_log_file_handler(logger, file_level, log_fp)
    if io_obj_level is not None:
        add_io_obj_handler(logger, io_obj_level)
    return logger


def init_logger_console(
    name: str | None = None,
    console_level: int = logging.INFO,
) -> logging.Logger:
    """
    Initialize a logger that logs to the console (stderr).

    Parameters
    ----------
    name : str, optional
        Logger name. If None, uses the caller's function name.
    console_level : int, default logging.INFO
        Logging level for the console handler.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    return init_logger(
        name=name or get_func_name_in_stack(2),
        console_level=console_level,
    )


def init_logger_file(
    name: str | None = None,
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
    log_fp: str | None = None,
) -> logging.Logger:
    """
    Initialize a logger that logs to both console and file.

    Parameters
    ----------
    name : str, optional
        Logger name. If None, uses the caller's function name.
    console_level : int, default logging.INFO
        Logging level for the console handler.
    file_level : int, default logging.DEBUG
        Logging level for the file handler.
    log_fp : str, optional
        Custom log file path. If None, uses default cache dir log file.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    return init_logger(
        name=name or get_func_name_in_stack(2),
        console_level=console_level,
        file_level=file_level,
        log_fp=log_fp,
    )


def init_logger_io_obj(
    name: str | None = None,
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
    io_obj_level: int = logging.INFO,
    log_fp: str | None = None,
) -> tuple[logging.Logger, io.StringIO]:
    """
    Initialize a logger that logs to console, file, and an in-memory StringIO object.

    Parameters
    ----------
    name : str, optional
        Logger name. If None, uses the caller's function name.
    console_level : int, default logging.INFO
        Logging level for the console handler.
    file_level : int, default logging.DEBUG
        Logging level for the file handler.
    io_obj_level : int, default logging.INFO
        Logging level for the StringIO handler.
    log_fp : str, optional
        Custom log file path. If None, uses default cache dir log file.

    Returns
    -------
    tuple[logging.Logger, io.StringIO]
        Tuple of (configured logger, StringIO object used by the handler).
    """
    logger = init_logger(
        name=name or get_func_name_in_stack(2),
        console_level=console_level,
        file_level=file_level,
        io_obj_level=io_obj_level,
        log_fp=log_fp,
    )
    io_obj = add_io_obj_handler(logger, io_obj_level)
    return logger, io_obj


def get_io_obj_content(io_obj: io.IOBase) -> str:
    """
    Read and return the content from an IOBase object, restoring cursor position.

    Parameters
    ----------
    io_obj : io.IOBase
        The IOBase object (e.g., StringIO) to read from.

    Returns
    -------
    str
        The content of the IOBase object.
    """
    cursor = io_obj.tell()
    io_obj.seek(0)
    msg = io_obj.read()
    io_obj.seek(cursor)
    return msg


def add_log_extra(logger: logging.Logger, extra: dict) -> logging.LoggerAdapter:
    """
    Return a LoggerAdapter that injects contextual information (extra fields) into log records.

    Parameters
    ----------
    logger : logging.Logger
        The logger to wrap with extra context.
    extra : dict
        Dictionary of extra fields to inject into log records.

    Returns
    -------
    logging.LoggerAdapter
        LoggerAdapter that injects the extra context.

    Examples
    --------
    >>> logger = add_log_extra(logger, {"user_id": 123})
    >>> logger.info("User logged in")
    """
    return logging.LoggerAdapter(logger, extra)
