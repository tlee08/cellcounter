"""GPU-accelerated cell counting with automatic memory management.

Inherits from CpuCellcFuncs and wraps all methods with:
- GPU memory cleanup before/after execution
- Automatic cupy→numpy conversion for outputs

Use GpuCellcFuncs() for GPU mode, CpuCellcFuncs() for CPU fallback.
"""

import functools
from collections.abc import Callable
from typing import ParamSpec, TypeVar

import cupy as cp
import numpy.typing as npt
from cupyx.scipy import ndimage as cp_ndimage

from cellcounter.funcs.cpu_cellc_funcs import CpuCellcFuncs

P = ParamSpec("P")
T = TypeVar("T")


# Methods that need GPU memory clearing + numpy conversion
_GPU_METHODS = {
    "tophat_filt",
    "dog_filt",
    "gauss_blur_filt",
    "gauss_subt_filt",
    "intensity_cutoff",
    "otsu_thresh",
    "mean_thresh",
    "manual_thresh",
    "_offset_labels_by_block",
    "map_values_to_arr",
    "volume_filter",
    "mask",
    "downsample",
    "reorient",
}

# Methods that need GPU memory clearing only (already convert to numpy internally)
_GPU_METHODS_NO_CONVERT = {
    "mask2label",
    "get_boundary_pairs",
    "get_label_sizemap",
    "label2volume",
    "get_local_maxima",
    "wshed_segm",
    "wshed_segm_volumes",
    "get_coords",
    "get_cells",
}


def _clear_gpu_memory[**P, T](func: Callable[P, T]) -> Callable[P, T]:
    """Decorator to clear GPU memory before and after function execution."""

    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
        result = func(*args, **kwargs)
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
        return result

    return wrapper


def _to_numpy[**P](func: Callable[P, cp.ndarray]) -> Callable[P, npt.NDArray]:
    """Decorator to convert cupy result to numpy."""

    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> npt.NDArray:
        return func(*args, **kwargs).get()

    return wrapper


def _create_gpu_method(name: str, convert_to_numpy: bool) -> Callable:
    """Create a GPU-wrapped method."""
    cpu_method = getattr(CpuCellcFuncs, name)
    if convert_to_numpy:
        return _clear_gpu_memory(_to_numpy(cpu_method))
    return _clear_gpu_memory(cpu_method)


class GpuCellcFuncs(CpuCellcFuncs):
    """GPU cell counting with automatic memory cleanup.

    Inherits from CpuCellcFuncs and wraps methods with GPU memory management.
    """

    def __init__(self) -> None:
        """Make GPU-enabled cellcounting functions object."""
        super().__init__(xp=cp, xdimage=cp_ndimage)


# Dynamically create wrapped methods
for _name in _GPU_METHODS:
    setattr(GpuCellcFuncs, _name, _create_gpu_method(_name, convert_to_numpy=True))
for _name in _GPU_METHODS_NO_CONVERT:
    setattr(GpuCellcFuncs, _name, _create_gpu_method(_name, convert_to_numpy=False))
