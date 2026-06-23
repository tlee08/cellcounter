"""Filepath models to ensure we follow specific file structure."""

from .proj_fp import ProjFp, get_proj_fp
from .ref_fp import RefFp

__all__ = [
    "ProjFp",
    "RefFp",
    "get_proj_fp",
]
