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
        "threshd",
        "threshd_labels",
        "threshd_volumes",
        "threshd_filt",
        "maxima",
        "maxima_labels",
        "wshed_labels",
        "wshed_volumes",
        "wshed_filt",
    ],
    "Visual QC": [
        "points_raw",
        "heatmap_raw",
        "points_trfm",
        "heatmap_trfm",
    ],
}
