import importlib.util

import dask

from .constants import CACHE_DIR


# Checking if CPU or GPU version
def package_is_exists(package_name: str) -> bool:
    spec = importlib.util.find_spec(package_name)
    return spec is not None


def package_is_importable(pacakage_name: str) -> bool:
    try:
        importlib.import_module(pacakage_name)
        return True
    except ImportError:
        return False


# Checking whether dask_cuda works (i.e. is Linux and has CUDA)
DASK_CUDA_ENABLED = package_is_importable("dask_cuda")
# Checking whether gpu extra dependency (CuPy) is installed
GPU_ENABLED = package_is_importable("cupy")
# Checking whether elastix extra dependency is installed
ELASTIX_ENABLED = package_is_importable("SimpleITK")

# Setting Dask configuration
dask.config.set(
    {
        # "distributed.scheduler.active-memory-manager.measure": "managed",
        # "distributed.worker.memory.rebalance.measure": "managed",
        # "distributed.worker.memory.spill": False,
        # "distributed.worker.memory.pause": False,
        # "distributed.worker.memory.terminate": False,
        "temporary-directory": CACHE_DIR
    }
)


#####################################################################
# IMPORTING SUBMODULES
#####################################################################

from cellcounter.funcs.batch_combine_funcs import BatchCombineFuncs
from cellcounter.funcs.viewer_funcs import ViewerFuncs
from cellcounter.pipeline.pipeline import Pipeline
from cellcounter.pipeline.visual_check import VisualCheck
