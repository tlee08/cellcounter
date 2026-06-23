"""Core functions for cell counting, registration, and I/O operations.

Modules:
- cpu_cellc_funcs: CPU-based cell counting operations (numpy backend)
- gpu_cellc_funcs: GPU-accelerated cell counting (cupy backend)
- reg_funcs: Image registration preprocessing (downsample, reorient)
- map_funcs: Atlas region mapping and annotation utilities
- elastix_funcs: ITK-Elastix registration wrappers
- io_funcs: File I/O (TIFF, Zarr, Parquet)
- batch_combine_funcs: Multi-experiment aggregation
- visual_check_funcs_*: Coordinate-to-image conversion for QC
"""

from .batch_combine_funcs import combine_root
from .cpu_cellc_funcs import CpuCellcFuncs
from .elastix_funcs import registration, transformation_coords, transformation_img
from .gpu_cellc_funcs import GpuCellcFuncs
from .io_funcs import (
    async_read,
    async_read_files,
    async_read_files_run,
    btiff2niftygz,
    btiff2zarr,
    combine_arrs,
    silent_remove,
    tiffs2zarr,
    write_parquet,
    write_tiff,
    zarr2tiff,
)
from .map_funcs import annot_fp2df, combine_nested_regions, df_map_ids, get_cells

__all__ = [
    "CpuCellcFuncs",
    "GpuCellcFuncs",
    "annot_fp2df",
    "async_read",
    "async_read_files",
    "async_read_files_run",
    "btiff2niftygz",
    "btiff2zarr",
    "combine_arrs",
    "combine_nested_regions",
    "combine_root",
    "df_map_ids",
    "get_cells",
    "get_cells",
    "registration",
    "silent_remove",
    "tiffs2zarr",
    "transformation_coords",
    "transformation_img",
    "write_parquet",
    "write_tiff",
    "zarr2tiff",
]
