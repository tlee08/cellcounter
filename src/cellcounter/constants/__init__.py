"""Constants package with backward-compatible re-exports.

For targeted imports, use:
    from cellcounter.constants.coords import Coords
    from cellcounter.constants.cells import CellColumns
    etc.
"""

from cellcounter.constants.annotations import (
    ANNOT_COLUMNS_FINAL,
    ANNOT_COLUMNS_TYPES,
    AnnotColumns,
    AnnotExtraColumns,
    SpecialRegions,
)
from cellcounter.constants.cells import CELL_AGG_MAPPINGS, CELL_IDX_NAME, CellColumns
from cellcounter.constants.config import CUPY_ENABLED, DASK_CUDA_ENABLED
from cellcounter.constants.coords import Coords
from cellcounter.constants.masks import MASK_VOLUME, MaskColumns
from cellcounter.constants.paths import ATLAS_DIR, CACHE_DIR
from cellcounter.constants.processing import PROC_CHUNKS, ROWS_PARTITION, TRFM

__all__ = [
    "ANNOT_COLUMNS_FINAL",
    "ANNOT_COLUMNS_TYPES",
    "ATLAS_DIR",
    "CACHE_DIR",
    "CELL_AGG_MAPPINGS",
    "CELL_IDX_NAME",
    "CUPY_ENABLED",
    "DASK_CUDA_ENABLED",
    "GPU_ENABLED",
    "MASK_VOLUME",
    "PROC_CHUNKS",
    "ROWS_PARTITION",
    "TRFM",
    "AnnotColumns",
    "AnnotExtraColumns",
    "CellColumns",
    "Coords",
    "MaskColumns",
    "SpecialRegions",
]
