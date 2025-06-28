import os
import pathlib
from enum import Enum

import dask
import numpy as np

from cellcounter.utils import package_is_importable

PROC_CHUNKS = (500, 1000, 1000)
# PROC_CHUNKS = (500, 1200, 1200)

# DEPTH = 10
DEPTH = 50

ROWS_PARTITION = 10000000


class Coords(Enum):
    """
    NOTE: this is the de facto order of the 3d dimensions (for tiff and zarr)
    """

    Z = "z"
    Y = "y"
    X = "x"


TRFM = "trfm"

CELL_IDX_NAME = "label"


class AnnotColumns(Enum):
    ID = "id"
    ATLAS_ID = "atlas_id"
    ONTOLOGY_ID = "ontology_id"
    ACRONYM = "acronym"
    NAME = "name"
    COLOR_HEX_TRIPLET = "color_hex_triplet"
    GRAPH_ORDER = "graph_order"
    ST_LEVEL = "st_level"
    HEMISPHERE_ID = "hemisphere_id"
    PARENT_STRUCTURE_ID = "parent_structure_id"


class AnnotExtraColumns(Enum):
    PARENT_ID = "parent_id"
    PARENT_ACRONYM = "parent_acronym"
    CHILDREN = "children"


ANNOT_COLUMNS_TYPES = {
    AnnotColumns.ID.value: np.float64,
    AnnotColumns.ATLAS_ID.value: np.float64,
    AnnotColumns.ONTOLOGY_ID.value: np.float64,
    AnnotColumns.ACRONYM.value: str,
    AnnotColumns.NAME.value: str,
    AnnotColumns.COLOR_HEX_TRIPLET.value: str,
    AnnotColumns.GRAPH_ORDER.value: np.float64,
    AnnotColumns.ST_LEVEL.value: np.float64,
    AnnotColumns.HEMISPHERE_ID.value: np.float64,
    AnnotColumns.PARENT_STRUCTURE_ID.value: np.float64,
}

ANNOT_COLUMNS_FINAL = [
    AnnotColumns.NAME.value,
    AnnotColumns.ACRONYM.value,
    AnnotColumns.COLOR_HEX_TRIPLET.value,
    AnnotColumns.PARENT_STRUCTURE_ID.value,
    AnnotExtraColumns.PARENT_ACRONYM.value,
]


class CellColumns(Enum):
    COUNT = "count"
    VOLUME = "volume"
    SUM_INTENSITY = "sum_intensity"
    # MAX_INTENSITY = "max_intensity"
    IOV = "iov"


CELL_AGG_MAPPINGS = {
    CellColumns.COUNT.value: "sum",
    CellColumns.VOLUME.value: "sum",
    CellColumns.SUM_INTENSITY.value: "sum",
    # CellMeasures.MAX_INTENSITY.value: "max",
}


class SpecialRegions(Enum):
    INVALID = "invalid"
    UNIVERSE = "universe"
    NO_LABEL = "no_label"


MASK_VOLUME = "volume"


class MaskColumns(Enum):
    VOLUME_ANNOT = f"{MASK_VOLUME}_annot"
    VOLUME_MASK = f"{MASK_VOLUME}_mask"
    VOLUME_PROP = f"{MASK_VOLUME}_prop"


CACHE_DIR = os.path.join(pathlib.Path.home(), ".cellcounter")
ATLAS_DIR = os.path.join(CACHE_DIR, "atlas_resources")

# Checking whether dask_cuda works (i.e. is Linux and has CUDA)
DASK_CUDA_ENABLED = package_is_importable("dask_cuda")
# Checking whether gpu extra dependency (CuPy) is installed
GPU_ENABLED = package_is_importable("cupy")
# Checking whether elastix extra dependency is installed
ELASTIX_ENABLED = package_is_importable("SimpleITK")

# Setting Dask configuration
dask.config.set(
    {
        # "distributed.scheduler.active-memory-manager.measure": "managed",
        # "distributed.worker.memory.rebalance.measure": "managed",
        # "distributed.worker.memory.spill": False,
        # "distributed.worker.memory.pause": False,
        # "distributed.worker.memory.terminate": False,
        "temporary-directory": CACHE_DIR
    }
)
