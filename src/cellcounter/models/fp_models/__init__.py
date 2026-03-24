import functools
import logging
from collections.abc import Callable
from pathlib import Path

from ty_extensions import Unknown

from cellcounter.models.fp_models.proj_fp import ProjFp
from cellcounter.models.fp_models.proj_tuning_fp import ProjTuningFp

logger = logging.getLogger(__name__)


@staticmethod
def get_proj_fm(proj_dir: Path | str, *, tuning: bool = False) -> ProjFp:
    """Returns Project Filepath object, given context."""
    return ProjTuningFp(proj_dir) if tuning else ProjFp(proj_dir)


@staticmethod
def check_overwrite(*fp_attrs: str) -> Callable:
    """Decorator to check if output files exist before running a pipeline step.

    Args:
        *fp_attrs: Names of ProjFp attributes to check for existence.

    Example:
        @_check_overwrite("bgrm", "dog")
        def cellc1(cls, proj_dir, *, overwrite=False, tuning=False): ...
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(
            proj_dir: Path | str,
            *args,
            overwrite: bool = False,
            tuning: bool = False,
            **kwargs,
        ) -> Unknown:
            # Check if file exists
            if not overwrite:
                pfm = get_proj_fm(proj_dir, tuning=tuning)
                for attr in fp_attrs:
                    fp = getattr(pfm, attr)
                    if fp.exists():
                        fp_str = f", {fp}, " if fp else " "
                        warning_msg = (
                            f"WARNING: Output file{fp_str}already exists - "
                            "not overwriting file.\n"
                            "To overwrite, specify overwrite=True`.\n"
                        )
                        logger.warning(warning_msg)
                        return None
            # Run func
            return func(
                proj_dir,
                *args,
                overwrite=overwrite,
                tuning=tuning,
                **kwargs,
            )

        return wrapper

    return decorator
