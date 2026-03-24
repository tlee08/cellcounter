import contextlib
import shutil
from pathlib import Path

import dask
import dask.array as da
import nibabel as nib
import numpy as np
import numpy.typing as npt
import pandas as pd
import tifffile
import zarr

from cellcounter.constants import PROC_CHUNKS

#############################################
# Regular IO
#############################################


def silent_remove(fp: Path | str) -> None:
    """Remove file or dir without throwing errors."""
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
    """Write parquet."""
    fp = Path(fp)
    fp.parent.mkdir(exist_ok=True)
    df.to_parquet(fp)


#############################################
# TIFF Files
#############################################


def read_tiff(src_fp: Path | str) -> npt.NDArray:
    """Read tiff file."""
    arr = tifffile.imread(src_fp)
    for _ in np.arange(len(arr.shape)):
        arr = np.squeeze(arr)
    return arr


def write_tiff(arr: npt.NDArray, dst_fp: Path | str) -> None:
    """Write to tiff file."""
    dst_fp = Path(dst_fp)
    dst_fp.parent.mkdir(exist_ok=True)
    tifffile.imwrite(dst_fp, arr)


#############################################
# Conversion between array file formats
#############################################


def btiff2zarr(
    src_fp: Path | str,
    dst_fp: Path | str,
    chunks: tuple[int, ...] = PROC_CHUNKS,
) -> None:
    """Convert bigtiff to zarr."""
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
    chunks: tuple[int, ...] = PROC_CHUNKS,
) -> None:
    """Convert folder of tiffs to zarr."""
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
    """Convert zarr to tiff."""
    arr = da.from_zarr(src_fp)
    write_tiff(arr, dst_fp)


def btiff2niftygz(src_fp: str, dst_fp: str) -> None:
    """Convert bigtiff to nifty-gz."""
    arr = tifffile.imread(src_fp)
    nib.Nifti1Image(arr, None).to_filename(dst_fp)


def read_niftygz(fp: str) -> np.typing.NDArray:
    """Read nifty-gz."""
    img = nib.load(fp)
    return np.array(img.dataobj)
