import importlib.util

# Checking whether dask_cuda works (i.e. is Linux and has CUDA)
DASK_CUDA_ENABLED = importlib.util.find_spec("dask_cuda") is not None

# Checking whether gpu extra dependency (CuPy) is installed
CUPY_ENABLED = importlib.util.find_spec("cupy") is not None
