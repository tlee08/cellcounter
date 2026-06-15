"""Filepath models to ensure we follow specific file structure."""

from pathlib import Path

from cellcounter.models.fp_models.proj_fp import ProjFp


def get_proj_fp(proj_dir: Path | str, *, tuning: bool = False) -> ProjFp:
    """Returns Project Filepath object, given context."""
    return ProjFp(proj_dir, tuning=tuning)
