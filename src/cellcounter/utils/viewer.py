"""Interactive Napari viewer for pipeline outputs.

Provides:
- read_img(): Load TIFF or Zarr arrays with optional trimming
- view_images(): Display project images in Napari with sensible defaults
"""

import logging
from pathlib import Path

import napari

from cellcounter.funcs.io_funcs import async_read_files_run, read_img
from cellcounter.models.fp_models import get_proj_fp

logger = logging.getLogger(__name__)

# Display defaults per image type
DISPLAY_DEFAULTS = {
    "ref": {"contrast_limits": (0, 10000), "colormap": "green"},
    "annot": {"contrast_limits": (0, 10000), "colormap": "Set1"},
    "raw": {"contrast_limits": (0, 10000), "colormap": "gray"},
    "downsmpl1": {"contrast_limits": (0, 10000), "colormap": "gray"},
    "downsmpl2": {"contrast_limits": (0, 10000), "colormap": "gray"},
    "trimmed": {"contrast_limits": (0, 10000), "colormap": "gray"},
    "bounded": {"contrast_limits": (0, 10000), "colormap": "gray"},
    "regresult": {"contrast_limits": (0, 1000), "colormap": "green"},
    "bgrm": {"contrast_limits": (0, 2000), "colormap": "gray"},
    "dog": {"contrast_limits": (0, 500), "colormap": "gray"},
    "adaptv": {"contrast_limits": (0, 500), "colormap": "gray"},
    "threshd": {"contrast_limits": (0, 5), "colormap": "gray"},
    "threshd_labels": {"contrast_limits": (0, 10000), "colormap": "green"},
    "threshd_volumes": {"contrast_limits": (0, 10000), "colormap": "green"},
    "threshd_filt": {"contrast_limits": (0, 10000), "colormap": "green"},
    "maxima": {"contrast_limits": (0, 5), "colormap": "green"},
    "maxima_labels": {"contrast_limits": (0, 1000), "colormap": "green"},
    "wshed_labels": {"contrast_limits": (0, 1000), "colormap": "green"},
    "wshed_volumes": {"contrast_limits": (0, 1000), "colormap": "green"},
    "wshed_filt": {"contrast_limits": (0, 1000), "colormap": "green"},
    "points_raw": {"contrast_limits": (0, 5), "colormap": "green"},
    "heatmap_raw": {"contrast_limits": (0, 20), "colormap": "red"},
    "points_trfm": {"contrast_limits": (0, 5), "colormap": "green"},
    "heatmap_trfm": {"contrast_limits": (0, 100), "colormap": "red"},
}


def view_images(
    proj_dir: str | Path,
    images: list[str],
    trimmer: tuple[slice, ...] | None = None,
    *,
    tuning: bool = True,
    **display_overrides,
) -> None:
    """Display project images in Napari with sensible defaults.

    Args:
        proj_dir: Project directory path.
        images: List of image attribute names from pfm (e.g., ["bgrm", "dog"]).
        trimmer: Optional tuple of slices to crop region of interest.
        tuning: Use tuning subdirectory if True.
        **display_overrides: Override default display settings per image.
            e.g., contrast_limits={"bgrm": (0, 5000)}, colormap={"bgrm": "red"}
    """
    pfm = get_proj_fp(proj_dir, tuning=tuning)
    # Build file paths and display settings
    fp_ls = [getattr(pfm, name) for name in images]
    names = images
    contrast_limits = []
    colormaps = []
    for name in images:
        defaults = DISPLAY_DEFAULTS.get(
            name, {"contrast_limits": (0, 10000), "colormap": "gray"}
        )
        cl = display_overrides.get("contrast_limits", {}).get(
            name, defaults["contrast_limits"]
        )
        cm = display_overrides.get("colormap", {}).get(name, defaults["colormap"])
        contrast_limits.append(cl)
        colormaps.append(cm)
    # Read arrays in parallel
    arr_ls = async_read_files_run(fp_ls, lambda fp: read_img(fp, trimmer))
    # Build napari kwargs per image
    kwargs_ls = [
        {
            "name": names[i],
            "contrast_limits": contrast_limits[i],
            "colormap": colormaps[i],
        }
        for i in range(len(images))
    ]
    # Create viewer and add images
    viewer = napari.Viewer()
    for i, arr in enumerate(arr_ls):
        logger.info("Adding image %d / %d: %s", i + 1, len(arr_ls), names[i])
        viewer.add_image(data=arr, blending="additive", **kwargs_ls[i])

    napari.run()
