"""View pipeline outputs in Napari."""

from cellcounter.models.fp_models import get_proj_fp
from cellcounter.utils.viewer import view_images

if __name__ == "__main__":
    # =========================================
    # CHANGE PROJECT PATH HERE
    proj_dir = "/path/to/project"
    tuning = False
    # =========================================

    pfm = get_proj_fp(proj_dir, tuning=tuning)
    view_images(
        imgs_fp_ls=[
            # Raw
            pfm.raw,
            # Registration
            # "ref",
            # "annot",
            # "downsmpl1",
            # "downsmpl2",
            # "trimmed",
            # "bounded",
            # "regresult",
            # Cell counting
            pfm.bgrm,
            pfm.dog,
            pfm.adaptv,
            pfm.threshd,
            # pfm.threshd_labels,
            pfm.threshd_volumes,
            pfm.threshd_filt,
            pfm.maxima,
            # pfm.maxima_labels,
            # pfm.wshed_labels,
            pfm.wshed_volumes,
            pfm.wshed_filt,
            # Visual check outputs
            # "points_raw",
            # "heatmap_raw",
            # "points_trfm",
            # "heatmap_trfm",
        ],
        trimmer=(
            slice(500, 510),
            slice(1500, 3000),
            slice(1500, 3000),
        ),  # e.g.: (slice(100, 200), slice(None), slice(None))
    )
