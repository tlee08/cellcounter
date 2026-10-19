from pathlib import Path

CACHE_DIR = Path.home() / ".cellcounter"
ATLAS_DIR = CACHE_DIR / "atlas_resources"

IMAGE_CATEGORIES = {
    "Raw": [
        "raw",
    ],
    "Registration": [
        "ref",
        "annot",
        "downsmpl1",
        "downsmpl2",
        "trimmed",
        "bounded",
        "regresult",
    ],
    "Cell Counting": [
        "bgrm",
        "dog",
        "adaptv",
        "threshold",
        "threshold_labels",
        "threshold_volumes",
        "threshold_filt",
        "maxima",
        "maxima_labels",
        "watershed_labels",
        "watershed_volumes",
        "watershed_filt",
    ],
    "Visual QC": [
        "points_raw",
        "heatmap_raw",
        "points_trfm",
        "heatmap_trfm",
    ],
}
