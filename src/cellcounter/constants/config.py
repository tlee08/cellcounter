import importlib.util

import dask

from cellcounter.constants.paths import CACHE_DIR

# Checking whether dask_cuda works (i.e. is Linux and has CUDA)
DASK_CUDA_ENABLED = importlib.util.find_spec("dask_cuda") is not None

# Checking whether gpu extra dependency (CuPy) is installed
CUPY_ENABLED = importlib.util.find_spec("cupy") is not None

# Setting Dask configuration
dask.config.set(
    {
        # "distributed.scheduler.active-memory-manager.measure": "managed",
        # "distributed.worker.memory.rebalance.measure": "managed",
        # "distributed.worker.memory.spill": False,
        # "distributed.worker.memory.pause": False,
        # "distributed.worker.memory.terminate": False,
        "temporary-directory": str(CACHE_DIR),
        "array.rechunk.method": "p2p",
        # Prevent task fusion (which can cause large memory blowouts)
        "optimization.fuse.active": False,
        "distributed.worker.memory.target": False,
        "distributed.worker.memory.spill": False,
        "distributed.worker.memory.pause": 0.80,
        "distributed.worker.memory.terminate": 0.95,
    }
)
