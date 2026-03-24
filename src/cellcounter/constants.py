import importlib.util
from enum import Enum
from pathlib import Path

import dask
import numpy as np

PROC_CHUNKS = (500, 500, 500)

ROWS_PARTITION = 10_000_000


class Coords(Enum):
    """The de facto order of the 3d dimensions (for tiff and zarr)."""

    Z = "z"
    Y = "y"
    X = "x"


TRFM = "trfm"

CELL_IDX_NAME = "label"


class AnnotColumns(Enum):
    """Atlas annotation column names."""

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
    """Other annotation columns derived from the Atlas annotations."""

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


CACHE_DIR = Path.home() / ".cellcounter"
ATLAS_DIR = CACHE_DIR / "atlas_resources"

# Checking whether dask_cuda works (i.e. is Linux and has CUDA)
DASK_CUDA_ENABLED = importlib.util.find_spec("dask_cuda") is not None
# Checking whether gpu extra dependency (CuPy) is installed
CUPY_ENABLED = importlib.util.find_spec("cupy") is not None

# Setting Dask configuration
dask.config.set(
    {
        # "distributed.scheduler.active-memory-manager.measure": "managed",
        # "distributed.worker.memory.rebalance.measure": "managed",
        # "distributed.worker.memory.spill": False,
        # "distributed.worker.memory.pause": False,
        # "distributed.worker.memory.terminate": False,
        "temporary-directory": CACHE_DIR,
        "array.rechunk.method": "p2p",
        # Prevent task fusion (which can cause large memory blowouts)
        "optimization.fuse.active": False,
        "distributed.worker.memory.target": False,
        "distributed.worker.memory.spill": False,
        "distributed.worker.memory.pause": 0.80,
        "distributed.worker.memory.terminate": 0.95,
    }
)
