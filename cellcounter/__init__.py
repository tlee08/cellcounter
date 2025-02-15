import importlib.util


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
