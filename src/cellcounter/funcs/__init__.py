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
