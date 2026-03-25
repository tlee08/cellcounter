"""File I/O utilities for microscopy images and data tables.

Supports:
- TIFF reading/writing (including bigtiff)
- Zarr conversion for chunked array storage
- Parquet for DataFrame persistence
- Async file reading for parallel I/O
- Format conversions (TIFF↔Zarr↔NIfTI)
"""

import asyncio
import contextlib
import shutil
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import dask
import dask.array as da
import nibabel as nib
import numpy as np
import numpy.typing as npt
import pandas as pd
import tifffile
import zarr

#############################################
# Regular IO
#############################################


def silent_remove(fp: Path | str) -> None:
    """Remove file or directory, suppressing errors if not found.

    Args:
        fp: Path to file or directory to remove.
    """
    fp = Path(fp)
    if fp.is_file():
        with contextlib.suppress(OSError):
            fp.unlink()
    elif fp.is_dir():
        with contextlib.suppress(OSError):
            shutil.rmtree(fp)


#####################################################################
# Parquet IO
#####################################################################


def write_parquet(df: pd.DataFrame, fp: Path | str) -> None:
    """Write DataFrame to Parquet format, creating parent directories.

    Args:
        df: DataFrame to save.
        fp: Output file path.
    """
    fp = Path(fp)
    fp.parent.mkdir(exist_ok=True)
    df.to_parquet(fp)


#############################################
# TIFF Files
#############################################


def read_tiff(src_fp: Path | str) -> npt.NDArray:
    """Read TIFF file, squeezing singleton dimensions.

    Args:
        src_fp: Path to TIFF file.

    Returns:
        Numpy array with squeezed dimensions.
    """
    arr = tifffile.imread(src_fp)
    for _ in np.arange(len(arr.shape)):
        arr = np.squeeze(arr)
    return arr


def write_tiff(arr: npt.NDArray, dst_fp: Path | str) -> None:
    """Write array to TIFF file, creating parent directories.

    Args:
        arr: Numpy array to save.
        dst_fp: Output file path.
    """
    dst_fp = Path(dst_fp)
    dst_fp.parent.mkdir(exist_ok=True)
    tifffile.imwrite(dst_fp, arr)


#############################################
# Async reading
#############################################


async def async_read(
    fp: Path | str, executor: ThreadPoolExecutor, read_func: Callable
) -> list:
    """Asynchronously read a single file in thread pool.

    Args:
        fp: File path to read.
        executor: Thread pool executor.
        read_func: Function to read the file.

    Returns:
        Result of read_func.
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(executor, read_func, fp)


async def async_read_files(fp_ls: list[Path | str], read_func: Callable) -> list:
    """Asynchronously read multiple files in parallel.

    Args:
        fp_ls: List of file paths.
        read_func: Function to read each file.

    Returns:
        List of results from read_func.
    """
    with ThreadPoolExecutor() as executor:
        tasks = [async_read(fp, executor, read_func) for fp in fp_ls]
        return await asyncio.gather(*tasks)


def async_read_files_run(fp_ls: list[Path | str], read_func: Callable) -> list:
    """Synchronously run async file reading.

    Args:
        fp_ls: List of file paths.
        read_func: Function to read each file.

    Returns:
        List of results from read_func.
    """
    return asyncio.run(async_read_files(fp_ls, read_func))


#############################################
# Conversion between array file formats
#############################################


def btiff2zarr(
    src_fp: Path | str,
    dst_fp: Path | str,
    chunks: tuple[int, ...],
) -> None:
    """Convert big TIFF to Zarr format with chunked storage.

    Uses memory-mapped reading to handle large files.

    Args:
        src_fp: Input big TIFF file path.
        dst_fp: Output Zarr directory path.
        chunks: Chunk size for Zarr array.
    """
    # To intermediate tiff
    mmap_arr = tifffile.memmap(src_fp)
    zarr_arr = zarr.open(
        f"{dst_fp}_tmp.zarr",
        mode="w",
        shape=mmap_arr.shape,
        dtype=mmap_arr.dtype,
        chunks=chunks,
    )
    zarr_arr[:] = mmap_arr
    # To final dask tiff
    zarr_arr = da.from_zarr(f"{dst_fp}_tmp.zarr")
    silent_remove(dst_fp)
    zarr_arr.to_zarr(dst_fp, mode="w")
    # Remove intermediate
    silent_remove(f"{dst_fp}_tmp.zarr")


def tiffs2zarr(
    src_fp_ls: tuple[Path | str, ...],
    dst_fp: Path | str,
    chunks: tuple[int, ...],
) -> None:
    """Convert stack of TIFF files to single Zarr.

    Stacks individual TIFFs along new first axis.

    Args:
        src_fp_ls: Tuple of input TIFF file paths.
        dst_fp: Output Zarr directory path.
        chunks: Chunk size for Zarr array.
    """
    # Getting shape and dtype
    arr0 = read_tiff(src_fp_ls[0])
    shape = (len(src_fp_ls), *arr0.shape)
    dtype = arr0.dtype
    # Getting list of dask delayed tiffs
    tiffs_ls = [dask.delayed(read_tiff)(i) for i in src_fp_ls]
    # Getting list of dask array tiffs and rechunking each (in prep later rechunking)
    tiffs_ls = [
        da.from_delayed(i, dtype=dtype, shape=shape[1:]).rechunk(chunks[1:])
        for i in tiffs_ls
    ]
    # Stacking tiffs and rechunking
    arr = da.stack(tiffs_ls, axis=0).rechunk(chunks)
    # Saving to zarr
    silent_remove(dst_fp)
    arr.to_zarr(dst_fp, mode="w")


def zarr2tiff(src_fp: str, dst_fp: str) -> None:
    """Convert Zarr to TIFF format.

    Args:
        src_fp: Input Zarr directory path.
        dst_fp: Output TIFF file path.
    """
    arr = da.from_zarr(src_fp)
    write_tiff(arr, dst_fp)


def btiff2niftygz(src_fp: str, dst_fp: str) -> None:
    """Convert big TIFF to NIfTI-GZ format.

    Args:
        src_fp: Input TIFF file path.
        dst_fp: Output NIfTI-GZ file path.
    """
    arr = tifffile.imread(src_fp)
    nib.Nifti1Image(arr, None).to_filename(dst_fp)


def read_niftygz(fp: str) -> np.typing.NDArray:
    """Read NIfTI-GZ file into numpy array.

    Args:
        fp: Input NIfTI-GZ file path.

    Returns:
        Numpy array of image data.
    """
    img = nib.load(fp)
    return np.array(img.dataobj)
