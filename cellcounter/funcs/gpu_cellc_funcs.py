import functools

from cellcounter.constants import GPU_ENABLED
from cellcounter.funcs.cpu_cellc_funcs import CpuCellcFuncs
from cellcounter.utils.logging_utils import init_logger_file
from cellcounter.utils.misc_utils import import_extra_error_func

# Optional dependency: gpu
if GPU_ENABLED:
    import cupy as cp
    from cupyx.scipy import ndimage as cp_ndimage
else:
    import_extra_error_func("gpu")()


class GpuCellcFuncs(CpuCellcFuncs):
    xp = cp
    xdimage = cp_ndimage

    logger = init_logger_file(__name__)

    ###################################################################################################
    # GPU HELPER FUNCTIONS
    ###################################################################################################

    @classmethod
    def _clear_cuda_mem(cls):
        # Also removing ALL references to the arguments
        cls.logger.debug("Removing all cp arrays in program (global and local)")
        all_vars = {**globals(), **locals()}
        var_keys = set(all_vars.keys())
        for k in var_keys:
            if isinstance(all_vars[k], cp.ndarray):
                cls.logger.debug(f"REMOVING: {k}")
                exec("del k")
        cls.logger.debug("Clearing CUDA memory")
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()

    @classmethod
    def _clear_cuda_mem_dec(cls, func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            cls._clear_cuda_mem()
            res = func(*args, **kwargs)
            cls._clear_cuda_mem()
            return res

        return wrapper

    @classmethod
    def _check_cuda_mem(cls):
        cls.logger.info(cp.get_default_memory_pool().used_bytes())
        cls.logger.info(cp.get_default_memory_pool().n_free_blocks())
        cls.logger.info(cp.get_default_pinned_memory_pool().n_free_blocks())

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
    def mask2label(cls, *args, **kwargs):
        return cls._clear_cuda_mem_dec(super().mask2label)(*args, **kwargs).get()

    @classmethod
    def mask2volume(cls, *args, **kwargs):
        # NOTE: Already returns a numpy array
        return cls._clear_cuda_mem_dec(super().mask2volume)(*args, **kwargs)

    @classmethod
    def label2volume(cls, *args, **kwargs):
        return cls._clear_cuda_mem_dec(super().label2volume)(*args, **kwargs).get()

    @classmethod
    def visualise_stats(cls, *args, **kwargs):
        return cls._clear_cuda_mem_dec(super().visualise_stats)(*args, **kwargs)

    @classmethod
    def volume_filter(cls, *args, **kwargs):
        return cls._clear_cuda_mem_dec(super().volume_filter)(*args, **kwargs).get()

    @classmethod
    def get_local_maxima(cls, *args, **kwargs):
        return cls._clear_cuda_mem_dec(super().get_local_maxima)(*args, **kwargs).get()

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
