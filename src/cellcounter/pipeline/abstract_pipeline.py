"""Base pipeline class with GPU/CPU switching and Dask cluster management.

Provides the AbstractPipeline base class that handles:
- GPU/CPU mode switching via set_gpu()
- Dask cluster creation (heavy, busy, GPU clusters)
- Config and filepath model access
"""

import functools
import logging
from abc import ABC
from collections.abc import Callable
from pathlib import Path

from dask.distributed import LocalCluster, SpecCluster

from cellcounter.constants import CUPY_ENABLED, DASK_CUDA_ENABLED
from cellcounter.funcs.cpu_cellc_funcs import CpuCellcFuncs
from cellcounter.models.fp_models import get_proj_fp
from cellcounter.models.fp_models.proj_fp import ProjFp
from cellcounter.models.proj_config import ProjConfig

logger = logging.getLogger(__name__)


def _get_gpu_cluster_factory() -> Callable[..., SpecCluster]:
    """Get GPU cluster factory if available, else CPU fallback."""
    if DASK_CUDA_ENABLED:
        from dask_cuda import LocalCUDACluster  # noqa: PLC0415

        return LocalCUDACluster
    return lambda: LocalCluster(n_workers=1, threads_per_worker=1)


def _get_cellc_funcs() -> CpuCellcFuncs:
    """Get GPU cellc funcs if available, else CPU fallback."""
    if CUPY_ENABLED:
        from cellcounter.funcs.gpu_cellc_funcs import GpuCellcFuncs  # noqa: PLC0415

        return GpuCellcFuncs()
    return CpuCellcFuncs()


def _check_overwrite(*fp_attrs: str) -> Callable:
    """Decorator to check if output files exist before running a pipeline step.

    Args:
        *fp_attrs: Names of pfm attributes to check for existence.
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(
            self: AbstractPipeline, *args, overwrite: bool = False, **kwargs
        ) -> object:
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


class AbstractPipeline(ABC):
    """Base class for pipeline operations with GPU/CPU switching.

    Provides:
    - Runtime GPU/CPU mode switching via set_gpu()
    - Dask cluster factories (heavy, busy, GPU)
    - Config and filepath model access

    Attributes:
        cellc_funcs: Cell counting functions (CPU or GPU backend).
        pfm: Project filepath model.
        config: Project configuration.
    """

    cellc_funcs: CpuCellcFuncs
    _gpu_cluster: Callable[..., SpecCluster]
    _tuning: bool
    _pfm: ProjFp

    def __init__(self, proj_dir: Path | str, *, tuning: bool = False) -> None:
        """Initialize pipeline with project directory.

        Args:
            proj_dir: Path to project directory.
            tuning: If True, use tuning subdirectory for parameters.
        """
        self._tuning = tuning
        self._pfm = get_proj_fp(proj_dir, tuning=tuning)
        self.set_gpu(enabled=True)

    @property
    def pfm(self) -> ProjFp:
        """Project filepath model."""
        return self._pfm

    @property
    def tuning(self) -> bool:
        """Is project in tuning version or raw version."""
        return self._tuning

    @property
    def config(self) -> ProjConfig:
        """Project configuration (cached on pfm)."""
        return self._pfm.config

    def update_config(self, updates: dict) -> None:
        """Update project configuration with new values.

        Args:
            updates: Key-value pairs to update in config.
        """
        ProjConfig.ensure(self._pfm.config_fp, updates)

    def heavy_cluster(self) -> SpecCluster:
        """Create cluster with few workers and high memory per worker.

        Use for memory-intensive operations like watershed.
        """
        return LocalCluster(
            n_workers=self.config.cluster.heavy_n_workers,
            threads_per_worker=self.config.cluster.heavy_threads_per_worker,
        )

    def busy_cluster(self) -> SpecCluster:
        """Create cluster with many workers and low memory per worker.

        Use for I/O-bound or parallel operations.
        """
        return LocalCluster(
            n_workers=self.config.cluster.busy_n_workers,
            threads_per_worker=self.config.cluster.busy_threads_per_worker,
        )

    def gpu_cluster(self) -> SpecCluster:
        """Create GPU-enabled cluster for CUDA operations.

        Falls back to CPU cluster if CUDA unavailable.
        """
        return self._gpu_cluster()

    def set_gpu(self, *, enabled: bool = True) -> None:
        """Switch between GPU and CPU mode at runtime.

        Args:
            enabled: True for GPU mode, False for CPU mode.
        """
        if enabled:
            self._gpu_cluster = _get_gpu_cluster_factory()
            self.cellc_funcs = _get_cellc_funcs()
        else:
            self._gpu_cluster = lambda: LocalCluster(n_workers=2, threads_per_worker=1)
            self.cellc_funcs = CpuCellcFuncs()
