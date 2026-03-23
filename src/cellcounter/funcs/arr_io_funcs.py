from pathlib import Path

import dask
import dask.array as da
import nibabel as nib
import numpy as np
import tifffile
import zarr

from cellcounter.constants import PROC_CHUNKS
from cellcounter.utils.io_utils import silent_remove


class ArrIOFuncs:
    """ArrIOFuncs."""

    #############################################
    # REGULAR IO
    #############################################

    @classmethod
    def read_tiff(cls, src_fp: Path | str) -> np.ndarray:
        """Read tiff file."""
        arr = tifffile.imread(src_fp)
        for _ in np.arange(len(arr.shape)):
            arr = np.squeeze(arr)
        return arr

    @classmethod
    def write_tiff(cls, arr: np.ndarray, dst_fp: Path | str) -> None:
        """Write to tiff file."""
        dst_fp = Path(dst_fp)
        dst_fp.parent.mkdir(exist_ok=True)
        tifffile.imwrite(dst_fp, arr)

    #############################################
    # CONVERSIONS
    #############################################

    @classmethod
    def btiff2zarr(
        cls,
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

    @classmethod
    def tiffs2zarr(
        cls,
        src_fp_ls: tuple[Path | str, ...],
        dst_fp: Path | str,
        chunks: tuple[int, ...] = PROC_CHUNKS,
    ) -> None:
        """Convert folder of tiffs to zarr."""
        # Getting shape and dtype
        arr0 = cls.read_tiff(src_fp_ls[0])
        shape = (len(src_fp_ls), *arr0.shape)
        dtype = arr0.dtype
        # Getting list of dask delayed tiffs
        tiffs_ls = [dask.delayed(cls.read_tiff)(i) for i in src_fp_ls]
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

    @classmethod
    def zarr2tiff(cls, src_fp: str, dst_fp: str) -> None:
        """Convert zarr to tiff."""
        arr = da.from_zarr(src_fp)
        cls.write_tiff(arr, dst_fp)

    @classmethod
    def btiff2niftygz(cls, src_fp: str, dst_fp: str) -> None:
        """Convert bigtiff to nifty-gz."""
        arr = tifffile.imread(src_fp)
        nib.Nifti1Image(arr, None).to_filename(dst_fp)

    @classmethod
    def read_niftygz(cls, fp: str) -> np.typing.NDArray:
        """Read nifty-gz."""
        img = nib.load(fp)
        return np.array(img.dataobj)
