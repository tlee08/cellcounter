import functools
import logging

import numpy.typing as npt

from cellcounter.constants import GPU_ENABLED
from cellcounter.funcs.cpu_cellc_funcs import CpuCellcFuncs
from cellcounter.utils.misc_utils import import_extra_error_func

logger = logging.getLogger(__name__)

# Optional dependency: gpu
if GPU_ENABLED:
    import cupy as cp
    from cupyx.scipy import ndimage as cp_ndimage
else:
    import_extra_error_func("gpu")()


class GpuCellcFuncs(CpuCellcFuncs):
    """GPU cell counting funcs.

    Leverages CPU logic in numpy and GPU porting with cupy.
    """

    xp = cp
    xdimage = cp_ndimage

    ###################################################################################################
    # GPU HELPER FUNCTIONS
    ###################################################################################################

    @classmethod
    def _clear_cuda_mem(cls) -> None:
        """Clears ALL current cupy array and frees GPU memory.

        To avoid GPU memory buildup, you MUST run this after a
        GPU block of work.
        """
        # Also removing ALL references to the arguments
        logger.debug("Removing all cp arrays in program (global and local)")
        all_vars = {**globals(), **locals()}
        var_keys = set(all_vars.keys())
        for k in var_keys:
            if isinstance(all_vars[k], cp.ndarray):
                logger.debug("REMOVING: %s", k)
                exec("del k")
        logger.debug("Clearing CUDA memory")
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()

    @classmethod
    def _clear_cuda_mem_dec(cls, func):
        """Decorator helper function to clean GPU CUDA memory for `func`.

        Runs `cls._clear_cuda_mem` before and after `func`.
        """

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            cls._clear_cuda_mem()
            res = func(*args, **kwargs)
            cls._clear_cuda_mem()
            return res

        return wrapper

    @classmethod
    def _check_cuda_mem(cls):
        logger.info(cp.get_default_memory_pool().used_bytes())
        logger.info(cp.get_default_memory_pool().n_free_blocks())
        logger.info(cp.get_default_pinned_memory_pool().n_free_blocks())

    ###################################################################################################
    # PROCESSING FUNCTIONS
    ###################################################################################################

    @classmethod
    def tophat_filt(cls, *args, **kwargs):
        return cls._clear_cuda_mem_dec(super().tophat_filt)(*args, **kwargs).get()

    @classmethod
    def dog_filt(cls, *args, **kwargs):
        return cls._clear_cuda_mem_dec(super().dog_filt)(*args, **kwargs).get()

    @classmethod
    def gauss_blur_filt(cls, *args, **kwargs):
        return cls._clear_cuda_mem_dec(super().gauss_blur_filt)(*args, **kwargs).get()

    @classmethod
    def gauss_subt_filt(cls, *args, **kwargs):
        return cls._clear_cuda_mem_dec(super().gauss_subt_filt)(*args, **kwargs).get()

    @classmethod
    def intensity_cutoff(cls, *args, **kwargs):
        return cls._clear_cuda_mem_dec(super().intensity_cutoff)(*args, **kwargs).get()

    @classmethod
    def otsu_thresh(cls, *args, **kwargs):
        return cls._clear_cuda_mem_dec(super().otsu_thresh)(*args, **kwargs).get()

    @classmethod
    def mean_thresh(cls, *args, **kwargs):
        return cls._clear_cuda_mem_dec(super().mean_thresh)(*args, **kwargs).get()

    @classmethod
    def manual_thresh(cls, *args, **kwargs):
        return cls._clear_cuda_mem_dec(super().manual_thresh)(*args, **kwargs).get()

    @classmethod
    def offset_labels_by_block(
        cls,
        block: npt.NDArray,
        block_info: dict | None = None,
        max_labels_per_chunk: int | None = None,
    ):
        return cls._clear_cuda_mem_dec(super().offset_labels_by_block)(
            block, block_info, max_labels_per_chunk
        ).get()

    @classmethod
    def mask2label(
        cls,
        block: npt.NDArray,
        block_info: dict | None = None,
        max_labels_per_chunk: int | None = None,
    ) -> npt.NDArray:
        # NOTE: already converted to numpy inside
        return cls._clear_cuda_mem_dec(super().mask2label)(
            block, block_info, max_labels_per_chunk
        )

    @classmethod
    def get_boundary_pairs(cls, *args, **kwargs):
        return cls._clear_cuda_mem_dec(super().get_boundary_pairs)(*args, **kwargs)

    @classmethod
    def get_label_sizemap(cls, *args, **kwargs):
        return cls._clear_cuda_mem_dec(super().get_label_sizemap)(*args, **kwargs)

    @classmethod
    def map_values_to_arr(cls, *args, **kwargs):
        return cls._clear_cuda_mem_dec(super().map_values_to_arr)(*args, **kwargs).get()

    @classmethod
    def label2volume(cls, *args, **kwargs):
        # NOTE: already converted to numpy inside
        return cls._clear_cuda_mem_dec(super().label2volume)(*args, **kwargs)

    @classmethod
    def volume_filter(cls, *args, **kwargs):
        return cls._clear_cuda_mem_dec(super().volume_filter)(*args, **kwargs).get()

    @classmethod
    def get_local_maxima(
        cls,
        block: npt.NDArray,
        sigma: int = 10,
        mask_block: None | npt.NDArray = None,
        block_info: dict | None = None,
        max_labels_per_chunk: int | None = None,
    ) -> npt.NDArray:
        # NOTE: already converted to numpy inside
        return cls._clear_cuda_mem_dec(super().get_local_maxima)(
            block, sigma, mask_block, block_info, max_labels_per_chunk
        )

    @classmethod
    def mask(cls, *args, **kwargs):
        return cls._clear_cuda_mem_dec(super().mask)(*args, **kwargs).get()

    @classmethod
    def wshed_segm(cls, *args, **kwargs):
        # NOTE: This is a CPU function
        return cls._clear_cuda_mem_dec(super().wshed_segm)(*args, **kwargs)

    @classmethod
    def wshed_segm_volumes(cls, *args, **kwargs):
        # NOTE: This is a CPU function
        return cls._clear_cuda_mem_dec(super().wshed_segm_volumes)(*args, **kwargs)

    @classmethod
    def get_coords(cls, *args, **kwargs):
        return cls._clear_cuda_mem_dec(super().get_coords)(*args, **kwargs)

    @classmethod
    def get_cells(cls, *args, **kwargs):
        # NOTE: This is a CPU function
        return cls._clear_cuda_mem_dec(super().get_cells)(*args, **kwargs)
