"""Napari visualization utilities for pipeline outputs.

Provides:
- read_img(): Load TIFF or Zarr arrays with optional trimming
- view_arrs(): Display multiple arrays in Napari with custom colormaps
- view_arrs_from_pfm(): Convenience wrapper using project filepath model
- combine_arrs(): Stack arrays into multi-channel TIFF for viewing
"""

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
    """Available colormaps for Napari visualization."""

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
    """Load TIFF or Zarr file into memory with optional trimming.

    Args:
        fp: Path to .tif or .zarr file.
        trimmer: Optional tuple of slices to extract region of interest.

    Returns:
        Numpy array loaded into memory.
    """
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


def view_arrs(fp_ls: list[Path | str], trimmer: tuple[slice, ...], **kwargs) -> None:
    """Display multiple arrays in Napari viewer.

    Args:
        fp_ls: List of file paths to load.
        trimmer: Tuple of slices to extract region of interest.
        **kwargs: Per-image options (name, contrast_limits, colormap as lists).
    """
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
    """Display project images in Napari using predefined display settings.

    Convenience wrapper that loads files from project filepath model
    and applies default colormaps/contrast limits.

    Args:
        proj_dir: Project directory path.
        imgs_to_view_ls: List of image attribute names from pfm.
        trimmer: Tuple of slices to extract region of interest.
        tuning: Use tuning subdirectory if True.
    """
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
    """Read and save array to TIFF format.

    Args:
        fp_in: Input file path (.tif or .zarr).
        fp_out: Output TIFF file path.
        trimmer: Optional tuple of slices to extract region of interest.

    Returns:
        The saved array.
    """
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
    """Stack multiple arrays into multi-channel TIFF.

    Args:
        fp_in_ls: Tuple of input file paths.
        fp_out: Output TIFF file path.
        trimmer: Optional tuple of slices to extract region of interest.

    Returns:
        Stacked array with shape (z, y, x, channels).
    """
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
