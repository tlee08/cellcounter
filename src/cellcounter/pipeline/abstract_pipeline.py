import logging
from abc import ABC
from collections.abc import Callable
from pathlib import Path

from dask.distributed import LocalCluster, SpecCluster

from cellcounter.constants import CUPY_ENABLED, DASK_CUDA_ENABLED
from cellcounter.funcs.cpu_cellc_funcs import CpuCellcFuncs
from cellcounter.models.fp_models import get_proj_fm
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


class AbstractPipeline(ABC):
    """Base class for pipeline operations."""

    cellc_funcs: CpuCellcFuncs
    _gpu_cluster: Callable[..., SpecCluster]
    _pfm: ProjFp
    _tuning: bool

    def __init__(self, proj_dir: Path | str, *, tuning: bool = False) -> None:
        """Init."""
        self._pfm = get_proj_fm(proj_dir, tuning=tuning)
        self._tuning = tuning
        self.set_gpu(enabled=True)

    @property
    def pfm(self) -> ProjFp:
        """Project filepath model."""
        return self._pfm

    @property
    def config(self) -> ProjConfig:
        """Project configuration (cached on pfm)."""
        return self._pfm.config

    def update_config(self, **updates) -> None:
        """Update configs."""
        ProjConfig.ensure(self._pfm.config_fp, **updates)

    def heavy_cluster(self) -> SpecCluster:
        """Make heavy cluster (few workers, high RAM)."""
        return LocalCluster(
            n_workers=self.config.heavy_n_workers,
            threads_per_worker=self.config.heavy_threads_per_worker,
        )

    def busy_cluster(self) -> SpecCluster:
        """Make busy cluster (many workers, low RAM)."""
        return LocalCluster(
            n_workers=self.config.busy_n_workers,
            threads_per_worker=self.config.busy_threads_per_worker,
        )

    def gpu_cluster(self) -> SpecCluster:
        """Make GPU cluster."""
        return self._gpu_cluster()

    def set_gpu(self, *, enabled: bool = True) -> None:
        """Force GPU or CPU mode at runtime."""
        if enabled:
            self._gpu_cluster = _get_gpu_cluster_factory()
            self.cellc_funcs = _get_cellc_funcs()
        else:
            self._gpu_cluster = lambda: LocalCluster(n_workers=2, threads_per_worker=1)
            self.cellc_funcs = CpuCellcFuncs()
