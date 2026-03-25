"""View pipeline outputs in Napari."""

from pathlib import Path

from cellcounter.utils.viewer import view_images

if __name__ == "__main__":
    # =========================================
    # CHANGE PROJECT PATH HERE
    proj_dir = Path("/path/to/project")
    # =========================================

    view_images(
        proj_dir=proj_dir,
        images=[
            # Registration
            # "ref",
            # "annot",
            # "downsmpl1",
            # "downsmpl2",
            # "trimmed",
            # "bounded",
            # "regresult",
            # Cell counting
            "bgrm",
            "dog",
            "adaptv",
            "threshd",
            "threshd_filt",
            "maxima",
            "wshed_filt",
            # Visual check outputs
            # "points_raw",
            # "heatmap_raw",
            # "points_trfm",
            # "heatmap_trfm",
        ],
        trimmer=None,  # or: (slice(100, 200), slice(None), slice(None))
        tuning=True,
    )
