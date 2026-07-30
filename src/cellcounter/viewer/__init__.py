"""Interactive Napari viewer for pipeline outputs.

Provides:
- read_img(): Load TIFF or Zarr arrays with optional trimming
- view_images(): Display project images in Napari with sensible defaults
"""

import asyncio
from pathlib import Path

import napari
from loguru import logger

from cellcounter.funcs.io_funcs import async_read_files, read_img
from cellcounter.models.fp_models import ProjFp

# Display defaults per image type
pfm = ProjFp(Path.cwd())
DISPLAY_DEFAULTS = {
    pfm.ref.name: {"contrast_limits": (0, 10000), "colormap": "green"},
    pfm.annot.name: {"contrast_limits": (0, 10000), "colormap": "Set1"},
    pfm.raw.name: {"contrast_limits": (0, 10000), "colormap": "gray"},
    pfm.downsmpl1.name: {"contrast_limits": (0, 10000), "colormap": "gray"},
    pfm.downsmpl2.name: {"contrast_limits": (0, 10000), "colormap": "gray"},
    pfm.trimmed.name: {"contrast_limits": (0, 10000), "colormap": "gray"},
    pfm.bounded.name: {"contrast_limits": (0, 10000), "colormap": "gray"},
    pfm.regresult.name: {"contrast_limits": (0, 1000), "colormap": "green"},
    pfm.bgrm.name: {"contrast_limits": (0, 2000), "colormap": "gray"},
    pfm.dog.name: {"contrast_limits": (0, 500), "colormap": "gray"},
    pfm.adaptv.name: {"contrast_limits": (0, 500), "colormap": "gray"},
    pfm.threshd.name: {"contrast_limits": (0, 5), "colormap": "gray"},
    pfm.threshd_labels.name: {"contrast_limits": (0, 10000), "colormap": "green"},
    pfm.threshd_volumes.name: {"contrast_limits": (0, 10000), "colormap": "green"},
    pfm.threshd_filt.name: {"contrast_limits": (0, 10000), "colormap": "green"},
    pfm.maxima.name: {"contrast_limits": (0, 5), "colormap": "green"},
    pfm.maxima_labels.name: {"contrast_limits": (0, 1000), "colormap": "green"},
    pfm.wshed_labels.name: {"contrast_limits": (0, 1000), "colormap": "green"},
    pfm.wshed_volumes.name: {"contrast_limits": (0, 1000), "colormap": "green"},
    pfm.wshed_filt.name: {"contrast_limits": (0, 1000), "colormap": "green"},
    pfm.points_raw.name: {"contrast_limits": (0, 5), "colormap": "green"},
    pfm.heatmap_raw.name: {"contrast_limits": (0, 20), "colormap": "red"},
    pfm.points_trfm.name: {"contrast_limits": (0, 5), "colormap": "green"},
    pfm.heatmap_trfm.name: {"contrast_limits": (0, 100), "colormap": "red"},
}


def _build_display_kwargs(
    names_ls: list[str],
    **kwargs,
) -> tuple[list, list]:
    contrast_limits = []
    colormaps = []
    for name in names_ls:
        defaults = DISPLAY_DEFAULTS.get(
            str(name), {"contrast_limits": (0, 10000), "colormap": "gray"}
        )
        cl = kwargs.get("contrast_limits", {}).get(name, defaults["contrast_limits"])
        cm = kwargs.get("colormap", {}).get(name, defaults["colormap"])
        contrast_limits.append(cl)
        colormaps.append(cm)
    return contrast_limits, colormaps


def _show_in_napari(imgs_fp_ls, names_ls, arr_ls, contrast_limits, colormaps):
    kwargs_ls = [
        {
            "name": names_ls[i],
            "contrast_limits": contrast_limits[i],
            "colormap": colormaps[i],
        }
        for i in range(len(names_ls))
    ]
    viewer = napari.Viewer()
    for i, arr in enumerate(arr_ls):
        logger.info("Adding image {} / {}: {}", i + 1, len(arr_ls), imgs_fp_ls[i])
        viewer.add_image(data=arr, blending="additive", **kwargs_ls[i])
    napari.run()


def view_images(
    imgs_fp_ls: list[Path | str],
    trimmer: tuple[slice, ...] | None = None,
    **kwargs,
) -> None:
    """Display project images in Napari with sensible defaults (sync).

    Use in scripts or anywhere outside a running event loop.

    Args:
        imgs_fp_ls: List of image attribute names from pfm (e.g., ["bgrm", "dog"]).
        trimmer: Optional tuple of slices to crop region of interest.
        **kwargs: Override default display settings per image.
            e.g., contrast_limits={"bgrm": (0, 5000)}, colormap={"bgrm": "red"}
    """
    imgs_fp_ls = [Path(_i) for _i in imgs_fp_ls]
    names_ls = [Path(_i).name for _i in imgs_fp_ls]
    contrast_limits, colormaps = _build_display_kwargs(names_ls, **kwargs)
    arr_ls = asyncio.run(async_read_files(imgs_fp_ls, lambda fp: read_img(fp, trimmer)))
    _show_in_napari(imgs_fp_ls, names_ls, arr_ls, contrast_limits, colormaps)


async def async_view_images(
    imgs_fp_ls: list[Path | str],
    trimmer: tuple[slice, ...] | None = None,
    **kwargs,
) -> None:
    """Display project images in Napari with sensible defaults (async).

    Use in marimo notebooks or anywhere inside a running event loop.

    Args:
        imgs_fp_ls: List of image attribute names from pfm (e.g., ["bgrm", "dog"]).
        trimmer: Optional tuple of slices to crop region of interest.
        **kwargs: Override default display settings per image.
            e.g., contrast_limits={"bgrm": (0, 5000)}, colormap={"bgrm": "red"}
    """
    imgs_fp_ls = [Path(_i) for _i in imgs_fp_ls]
    names_ls = [Path(_i).name for _i in imgs_fp_ls]
    contrast_limits, colormaps = _build_display_kwargs(names_ls, **kwargs)
    arr_ls = await async_read_files(imgs_fp_ls, lambda fp: read_img(fp, trimmer))
    _show_in_napari(imgs_fp_ls, names_ls, arr_ls, contrast_limits, colormaps)
