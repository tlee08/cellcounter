"""Miscellaneous utilities."""

from pathlib import Path

from loguru import logger


def has_output_files(*fp_ls: Path) -> bool:
    """Check if there are output files already (for overwrite risk).

    If any exist, logs warning and returns True.
    """
    exists_ls = [fp for fp in fp_ls if fp.exists()]
    if exists_ls:
        logger.warning(
            "File(s) already exists - not overwriting: {}",
            exists_ls,
        )
        return True
    return False


def missing_input_files(*fp_ls: Path) -> bool:
    """Check whether any input files are missing.

    If any are missing, logs warning and returns True.
    """
    missing_ls = [fp for fp in fp_ls if not fp.exists()]
    if missing_ls:
        logger.warning(
            "File(s) do not exist: {}",
            missing_ls,
        )
        return True
    return False
