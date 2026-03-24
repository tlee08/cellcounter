import logging
from abc import ABC
from pathlib import Path

from dask.distributed import LocalCluster, SpecCluster

from cellcounter.constants import (
    DASK_CUDA_ENABLED,
    ELASTIX_ENABLED,
    GPU_ENABLED,
)
from cellcounter.funcs.cpu_cellc_funcs import CpuCellcFuncs
from cellcounter.models.fp_models import get_proj_fm
from cellcounter.models.fp_models.proj_fp import ProjFp
from cellcounter.models.proj_config import ProjConfig
from cellcounter.utils.misc_utils import import_extra_error_func

logger = logging.getLogger(__name__)


# Optional dependency: gpu (with dask-cuda)
if DASK_CUDA_ENABLED:
    from dask_cuda import LocalCUDACluster
else:
    LocalCUDACluster = lambda: LocalCluster(n_workers=1, threads_per_worker=1)  # noqa: E731
    logger.warning(
        "Warning Dask-Cuda functionality not installed.\n"
        "Using single GPU functionality instead (1 worker)\n"
        "Dask-Cuda currently only available on Linux"
    )
# Optional dependency: gpu
if GPU_ENABLED:
    from cellcounter.funcs.gpu_cellc_funcs import GpuCellcFuncs
else:
    LocalCUDACluster = lambda: LocalCluster(n_workers=2, threads_per_worker=1)  # noqa: E731
    GpuCellcFuncs = CpuCellcFuncs
    logger.warning(
        "Warning GPU functionality not installed.\n"
        "Using CPU functionality instead (much slower).\n"
        'Can install with `pip install "cellcounter[gpu]"`'
    )
# Optional dependency: elastix
if ELASTIX_ENABLED:
    from cellcounter.funcs.elastix_funcs import ElastixFuncs
else:
    ElastixFuncs = import_extra_error_func("elastix")
    logger.warning(
        "Warning Elastix functionality not installed and unavailable.\n"
        'Can install with `pip install "cellcounter[elastix]"`'
    )


class AbstractPipeline(ABC):
    """Base class for pipeline operations."""

    # GPU enabled cell funcs
    cellc_funcs: type[CpuCellcFuncs] = GpuCellcFuncs
    # GPU cluster factory
    _gpu_cluster = LocalCUDACluster
    _pfm: ProjFp
    _tuning: bool

    def __init__(self, proj_dir: Path | str, *, tuning: bool = False) -> None:
        """Init."""
        self._pfm = get_proj_fm(proj_dir, tuning=tuning)
        self._tuning = tuning

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

    @classmethod
    def set_gpu(cls, *, enabled: bool = True) -> None:
        """Set GPU cluster mode globally."""
        if enabled:
            cls._gpu_cluster = LocalCUDACluster
            cls.cellc_funcs = GpuCellcFuncs
        else:
            cls._gpu_cluster = lambda: LocalCluster(n_workers=2, threads_per_worker=1)
            cls.cellc_funcs = CpuCellcFuncs
