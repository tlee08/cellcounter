import os
from enum import Enum

import dask.array as da
import napari
import numpy as np
import tifffile
from dask.distributed import LocalCluster

from cellcounter.funcs.arr_io_funcs import ArrIOFuncs
from cellcounter.utils.dask_utils import cluster_process
from cellcounter.utils.io_utils import async_read_files_run
from cellcounter.utils.logging_utils import init_logger_file
from cellcounter.utils.misc_utils import dictlists2listdicts
from cellcounter.utils.proj_org_utils import ProjFpModel, ProjFpModelTuning

VRANGE = "vrange"
CMAP = "cmap"


class Colormaps(Enum):
    GRAY = "gray"
    RED = "red"
    GREEN = "green"
    BLUE = "blue"
    YELLOW = "yellow"
    VIRIDIS = "viridis"
    MAGMA = "magma"
    SET1 = "Set1"


VIEW_IMGS_PARAMS = {
    "ref": {VRANGE: (0, 10000), CMAP: Colormaps.GREEN.value},
    "annot": {VRANGE: (0, 10000), CMAP: Colormaps.SET1.value},
    "raw": {VRANGE: (0, 10000), CMAP: Colormaps.GRAY.value},
    "downsmpl1": {VRANGE: (0, 10000), CMAP: Colormaps.GRAY.value},
    "downsmpl2": {VRANGE: (0, 10000), CMAP: Colormaps.GRAY.value},
    "trimmed": {VRANGE: (0, 10000), CMAP: Colormaps.GRAY.value},
    "bounded": {VRANGE: (0, 10000), CMAP: Colormaps.GRAY.value},
    "regresult": {VRANGE: (0, 1000), CMAP: Colormaps.GREEN.value},
    "premask_blur": {VRANGE: (0, 10000), CMAP: Colormaps.RED.value},
    "mask": {VRANGE: (0, 5), CMAP: Colormaps.RED.value},
    "outline": {VRANGE: (0, 5), CMAP: Colormaps.RED.value},
    "mask_reg": {VRANGE: (0, 5), CMAP: Colormaps.RED.value},
    "overlap": {VRANGE: (0, 10000), CMAP: Colormaps.GRAY.value},
    "bgrm": {VRANGE: (0, 2000), CMAP: Colormaps.GRAY.value},
    "dog": {VRANGE: (0, 500), CMAP: Colormaps.GRAY.value},
    "adaptv": {VRANGE: (0, 500), CMAP: Colormaps.GRAY.value},
    "threshd": {VRANGE: (0, 5), CMAP: Colormaps.GRAY.value},
    "threshd_volumes": {VRANGE: (0, 10000), CMAP: Colormaps.GREEN.value},
    "threshd_filt": {VRANGE: (0, 10000), CMAP: Colormaps.GREEN.value},
    "maxima": {VRANGE: (0, 5), CMAP: Colormaps.GREEN.value},
    "wshed_volumes": {VRANGE: (0, 1000), CMAP: Colormaps.GREEN.value},
    "wshed_filt": {VRANGE: (0, 1000), CMAP: Colormaps.GREEN.value},
    "threshd_final": {VRANGE: (0, 10000), CMAP: Colormaps.GRAY.value},
    "maxima_final": {VRANGE: (0, 5), CMAP: Colormaps.RED.value},
    "wshed_final": {VRANGE: (0, 1000), CMAP: Colormaps.GREEN.value},
    "points_raw": {VRANGE: (0, 5), CMAP: Colormaps.GREEN.value},
    "heatmap_raw": {VRANGE: (0, 20), CMAP: Colormaps.RED.value},
    "points_trfm": {VRANGE: (0, 5), CMAP: Colormaps.GREEN.value},
    "heatmap_trfm": {VRANGE: (0, 100), CMAP: Colormaps.RED.value},
}


class ViewerFuncs:
    logger = init_logger_file(__name__)

    @classmethod
    def read_img(cls, fp, trimmer: None | tuple[slice, ...] = None):
        """
        Reading, trimming (if possible), and returning the array in memory.
        """
        if os.path.splitext(fp)[1] == ".zarr":
            arr = da.from_zarr(fp)
            if trimmer is not None:
                arr = arr[*trimmer]
            return arr.compute()
        elif os.path.splitext(fp)[1] == ".tif":
            arr = tifffile.imread(fp)
            if trimmer is not None:
                arr = arr[*trimmer]
            return arr
        else:
            raise NotImplementedError("Only .zarr and .tif files are supported.")

    @classmethod
    def view_arrs(cls, fp_ls: list[str], trimmer: tuple[slice, ...], **kwargs):
        # Asserting all kwargs_ls list lengths are equal to fp_ls length
        for k, v in kwargs.items():
            assert len(v) == len(fp_ls)
        # Reading arrays
        arr_ls = async_read_files_run(fp_ls, lambda fp: cls.read_img(fp, trimmer))
        # "Transposing" kwargs from dict of lists to list of dicts
        kwargs_ls = dictlists2listdicts(kwargs)
        # Making napari viewer
        viewer = napari.Viewer()
        # Adding image to napari viewer
        for i, arr in enumerate(arr_ls):
            cls.logger.info(f"Napari viewer - adding image # {i} / {len(arr_ls)}")
            # NOTE: best kwargs to use are name, contrast_limits, and colormap
            viewer.add_image(
                data=arr,
                blending="additive",
                **kwargs_ls[i],
            )
        # Running viewer
        napari.run()

    @classmethod
    def view_arrs_from_pfm(
        cls, proj_dir: str, imgs_to_view_ls: list[str], trimmer: tuple[slice, ...], tuning: bool = True
    ) -> None:
        pfm = ProjFpModelTuning(proj_dir) if tuning else ProjFpModel(proj_dir)
        return cls.view_arrs(
            fp_ls=[getattr(pfm, i) for i in imgs_to_view_ls],
            trimmer=trimmer,
            name=imgs_to_view_ls,
            contrast_limits=[VIEW_IMGS_PARAMS[i][VRANGE] for i in imgs_to_view_ls],
            colormap=[VIEW_IMGS_PARAMS[i][CMAP] for i in imgs_to_view_ls],
        )

    # TODO: implement elsewhere for usage examples
    @classmethod
    def save_arr(
        cls,
        fp_in: str,
        fp_out: str,
        trimmer: tuple[slice, ...] | None = None,
        **kwargs,
    ) -> np.ndarray:
        """
        NOTE: exports as tiff only.
        """
        with cluster_process(LocalCluster()):
            # Reading
            arr = cls.read_img(fp_in, trimmer)
            # Writing
            ArrIOFuncs.write_tiff(arr, fp_out)
            # Returning
            return arr

    @classmethod
    def combine_arrs(
        cls,
        fp_in_ls: tuple[str, ...],
        fp_out: str,
        trimmer: tuple[slice, ...] | None = None,
        **kwargs,
    ) -> np.ndarray:
        dtype = np.uint16
        # Reading arrays
        arr_ls = []
        for i in fp_in_ls:
            # Read image
            arr = cls.read_img(i, trimmer)
            # Convert to dtype (rounded and within clip bounds)
            arr = arr.round(0).clip(0, 2**16 - 1).astype(dtype)
            # Adding image to list
            arr_ls.append(arr)
        # Stacking arrays
        arr = np.stack(arr_ls, axis=-1, dtype=dtype)
        # Writing to file
        ArrIOFuncs.write_tiff(arr, fp_out)
        return arr
