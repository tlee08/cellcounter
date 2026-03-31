"""Filepath models to ensure we follow specific file structure."""

from pathlib import Path

from cellcounter.models.fp_models.proj_fp import ProjFp
from cellcounter.models.fp_models.proj_tuning_fp import ProjTuningFp


def get_proj_fp(proj_dir: Path | str, *, tuning: bool = False) -> ProjFp:
    """Returns Project Filepath object, given context."""
    return ProjTuningFp(proj_dir) if tuning else ProjFp(proj_dir)
