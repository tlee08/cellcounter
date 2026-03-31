"""Interactive Napari viewer for pipeline outputs.

Provides:
- read_img(): Load TIFF or Zarr arrays with optional trimming
- view_images(): Display project images in Napari with sensible defaults
"""

import logging
from pathlib import Path

import napari

from cellcounter.funcs.io_funcs import async_read_files_run, read_img
from cellcounter.models.fp_models.proj_fp import ProjFp

logger = logging.getLogger(__name__)

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


def view_images(
    imgs_fp_ls: list[Path | str],
    trimmer: tuple[slice, ...] | None = None,
    *args,
    **display_overrides,
) -> None:
    """Display project images in Napari with sensible defaults.

    Args:
        imgs_fp_ls: List of image attribute names from pfm (e.g., ["bgrm", "dog"]).
        trimmer: Optional tuple of slices to crop region of interest.
        **display_overrides: Override default display settings per image.
            e.g., contrast_limits={"bgrm": (0, 5000)}, colormap={"bgrm": "red"}
    """
    imgs_fp_ls = [Path(_i) for _i in imgs_fp_ls]
    names_ls = [Path(_i).name for _i in imgs_fp_ls]
    # Build file paths and display settings
    contrast_limits = []
    colormaps = []
    for name in names_ls:
        defaults = DISPLAY_DEFAULTS.get(
            str(name), {"contrast_limits": (0, 10000), "colormap": "gray"}
        )
        cl = display_overrides.get("contrast_limits", {}).get(
            name, defaults["contrast_limits"]
        )
        cm = display_overrides.get("colormap", {}).get(name, defaults["colormap"])
        contrast_limits.append(cl)
        colormaps.append(cm)
    # Read arrays in parallel
    arr_ls = async_read_files_run(imgs_fp_ls, lambda fp: read_img(fp, trimmer))
    # Build napari kwargs per image
    kwargs_ls = [
        {
            "name": names_ls[i],
            "contrast_limits": contrast_limits[i],
            "colormap": colormaps[i],
        }
        for i in range(len(names_ls))
    ]
    # Create viewer and add images
    viewer = napari.Viewer()
    for i, arr in enumerate(arr_ls):
        logger.info("Adding image %d / %d: %s", i + 1, len(arr_ls), imgs_fp_ls[i])
        viewer.add_image(data=arr, blending="additive", **kwargs_ls[i])

    napari.run()
