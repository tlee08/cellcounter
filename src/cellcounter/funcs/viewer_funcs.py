import logging
from enum import Enum
from pathlib import Path

import dask.array as da
import napari
import numpy as np
import numpy.typing as npt
import tifffile
from dask.distributed import LocalCluster

from cellcounter.funcs.io_funcs import async_read_files_run, write_tiff
from cellcounter.models.fp_models import get_proj_fm
from cellcounter.utils.dask_utils import cluster_process
from cellcounter.utils.misc_utils import dictlists2listdicts

logger = logging.getLogger(__name__)

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


def read_img(fp: Path | str, trimmer: None | tuple[slice, ...] = None) -> npt.NDArray:
    """Reading, trimming (if possible), and returning the array in memory."""
    fp = Path(fp)
    if fp.suffix == ".zarr":
        arr = da.from_zarr(fp)
        if trimmer is not None:
            arr = arr[*trimmer]
        return arr.compute()
    if fp.suffix == ".tif":
        arr = tifffile.imread(fp)
        if trimmer is not None:
            arr = arr[*trimmer]
        return arr
    err_msg = "Only .zarr and .tif files are supported."
    raise NotImplementedError(err_msg)


def view_arrs(fp_ls: list[str], trimmer: tuple[slice, ...], **kwargs) -> None:
    """Run Napari viewer."""
    # Asserting all kwargs_ls list lengths are equal to fp_ls length
    for v in kwargs.values():
        assert len(v) == len(fp_ls)
    # Reading arrays
    arr_ls = async_read_files_run(fp_ls, lambda fp: read_img(fp, trimmer))
    # "Transposing" kwargs from dict of lists to list of dicts
    kwargs_ls = dictlists2listdicts(kwargs)
    # Making napari viewer
    viewer = napari.Viewer()
    # Adding image to napari viewer
    for i, arr in enumerate(arr_ls):
        logger.info("Napari viewer - adding image # %s / %s", i, len(arr_ls))
        # NOTE: best kwargs to use are name, contrast_limits, and colormap
        viewer.add_image(
            data=arr,
            blending="additive",
            **kwargs_ls[i],
        )
    # Running viewer
    napari.run()


def view_arrs_from_pfm(
    proj_dir: str,
    imgs_to_view_ls: list[str],
    trimmer: tuple[slice, ...],
    *,
    tuning: bool = True,
) -> None:
    """Run Napari viewer."""
    pfm = get_proj_fm(proj_dir, tuning=tuning)
    return view_arrs(
        fp_ls=[getattr(pfm, i) for i in imgs_to_view_ls],
        trimmer=trimmer,
        name=imgs_to_view_ls,
        contrast_limits=[VIEW_IMGS_PARAMS[i][VRANGE] for i in imgs_to_view_ls],
        colormap=[VIEW_IMGS_PARAMS[i][CMAP] for i in imgs_to_view_ls],
    )


def save_arr(
    fp_in: str,
    fp_out: str,
    trimmer: tuple[slice, ...] | None = None,
    **kwargs,
) -> npt.NDArray:
    """NOTE: exports as tiff only."""
    with cluster_process(LocalCluster()):
        # Reading
        arr = read_img(fp_in, trimmer)
        # Writing
        write_tiff(arr, fp_out)
        # Returning
        return arr


def combine_arrs(
    fp_in_ls: tuple[Path | str, ...],
    fp_out: Path | str,
    trimmer: tuple[slice, ...] | None = None,
    **kwargs,
) -> npt.NDArray:
    dtype = np.uint16
    # Reading arrays
    arr_ls = []
    for i in fp_in_ls:
        # Read image
        arr = read_img(i, trimmer)
        # Convert to dtype (rounded and within clip bounds)
        arr = arr.round(0).clip(0, 2**16 - 1).astype(dtype)
        # Adding image to list
        arr_ls.append(arr)
    # Stacking arrays
    arr = np.stack(arr_ls, axis=-1, dtype=dtype)
    # Writing to file
    write_tiff(arr, fp_out)
    return arr
