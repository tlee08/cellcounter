import functools
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from cellcounter.models.fp_models.proj_fp import ProjFp
from cellcounter.models.fp_models.proj_tuning_fp import ProjTuningFp

logger = logging.getLogger(__name__)


def get_proj_fm(proj_dir: Path | str, *, tuning: bool = False) -> ProjFp:
    """Returns Project Filepath object, given context."""
    return ProjTuningFp(proj_dir) if tuning else ProjFp(proj_dir)


def check_overwrite(*fp_attrs: str) -> Callable:
    """Decorator to check if output files exist before running a pipeline step.

    Args:
        *fp_attrs: Names of pfm attributes to check for existence.
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(self, *args, overwrite: bool = False, **kwargs) -> Any:
            if not overwrite:
                for attr in fp_attrs:
                    fp = getattr(self.pfm, attr)
                    if fp.exists():
                        logger.warning(
                            "WARNING: Output file, %s, already exists - "
                            "not overwriting file.\n"
                            "To overwrite, specify overwrite=True.\n",
                            fp,
                        )
                        return None
            return func(self, *args, overwrite=overwrite, **kwargs)

        return wrapper

    return decorator
