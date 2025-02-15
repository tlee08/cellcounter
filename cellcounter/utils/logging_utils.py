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
    If logger does not have a console handler,
    create a console handler and add it to the logger.

    Returns nothing as the console handler is always sys,stderr.
    """
    # Checking if logger has a console handler
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler):
            if handler.stream == sys.stderr:
                return
    # Adding console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    logger.addHandler(console_handler)


def add_log_file_handler(logger: logging.Logger, level: int) -> str:
    """
    If logger does not have a file handler,
    create a file handler and add it to the logger.

    Returns the log filepath.
    """
    log_fp = os.path.join(CACHE_DIR, "debug.log")
    # Checking if logger has a file handler
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            if handler.baseFilename == log_fp:
                return handler.baseFilename
    # Adding file handler
    os.makedirs(os.path.dirname(log_fp), exist_ok=True)
    file_handler = logging.FileHandler(log_fp, mode="a")
    file_handler.setLevel(level)
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    logger.addHandler(file_handler)
    return file_handler.baseFilename


def add_io_obj_handler(logger: logging.Logger, level: int) -> io.StringIO:
    """
    If logger does not have a StringIO handler,
    create a StringIO handler and add it to the logger.

    Returns the StringIO object.
    """
    # Checking if logger has a StringIO handler
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler):
            if isinstance(handler.stream, io.StringIO):
                return handler.stream
    # Adding StringIO handler
    io_obj = io.StringIO()
    io_obj_handler = logging.StreamHandler(io_obj)
    io_obj_handler.setLevel(level)
    io_obj_handler.setFormatter(logging.Formatter(LOG_IO_OBJ_FORMAT))
    logger.addHandler(io_obj_handler)
    return io_obj


def init_logger(
    name: str | None = None,
    console_level: int | None = None,
    file_level: int | None = None,
    io_obj_level: int | None = None,
) -> logging.Logger:
    """
    Setup logging configuration.

    For each of the following levels,
    if the level argument is not None,
    then add the handler with that level:
    - console
    - file (<cache_dir>/debug.log)
    - io.StringIO object
    """
    # Creating logger
    logger = logging.getLogger(name or get_func_name_in_stack(2))
    logger.setLevel(logging.DEBUG)
    # Adding handlers
    if console_level is not None:
        add_console_handler(logger, console_level)
    if file_level is not None:
        add_log_file_handler(logger, file_level)
    if io_obj_level is not None:
        add_io_obj_handler(logger, io_obj_level)
    return logger


def init_logger_console(
    name: str | None = None,
    console_level: int = logging.INFO,
) -> logging.Logger:
    """
    Logs to:
    - console
    """
    return init_logger(
        name=name or get_func_name_in_stack(2),
        console_level=console_level,
    )


def init_logger_file(
    name: str | None = None,
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
) -> logging.Logger:
    """
    Logs to:
    - console
    - file (<cache_dir>/debug.log)
    """
    return init_logger(
        name=name or get_func_name_in_stack(2),
        console_level=console_level,
        file_level=file_level,
    )


def init_logger_io_obj(
    name: str | None = None,
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
    io_obj_level: int = logging.INFO,
) -> tuple[logging.Logger, io.StringIO]:
    """
    Logs to:
    - console
    - file (<cache_dir>/debug.log)
    - io.StringIO object (returned)
    """
    logger = init_logger(
        name=name or get_func_name_in_stack(2),
        console_level=console_level,
        file_level=file_level,
        io_obj_level=io_obj_level,
    )
    io_obj = add_io_obj_handler(logger)
    return logger, io_obj


def get_io_obj_content(io_obj: io.IOBase) -> str:
    """
    Reads and returns the content from the IOBase object.
    Also restores cursor position of the object.
    """
    cursor = io_obj.tell()
    io_obj.seek(0)
    msg = io_obj.read()
    io_obj.seek(cursor)
    return msg
