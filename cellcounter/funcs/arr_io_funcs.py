import os

import dask
import dask.array as da
import nibabel as nib
import numpy as np
import tifffile
import zarr

# from prefect import task
from cellcounter.constants import PROC_CHUNKS
from cellcounter.utils.io_utils import silent_remove


class ArrIOFuncs:
    #####################################################################
    # REGULAR IO
    #####################################################################

    @classmethod
    def read_tiff(cls, fp: str) -> np.ndarray:
        arr = tifffile.imread(fp)
        for i in np.arange(len(arr.shape)):
            arr = np.squeeze(arr)
        return arr

    @classmethod
    def write_tiff(cls, arr: np.ndarray, fp: str) -> None:
        os.makedirs(os.path.dirname(fp), exist_ok=True)
        tifffile.imwrite(fp, arr)

    #####################################################################
    # CONVERSIONS
    #####################################################################

    @classmethod
    def btiff2zarr(cls, in_fp: str, out_fp: str, chunks: tuple[int, ...] = PROC_CHUNKS) -> None:
        # To intermediate tiff
        mmap_arr = tifffile.memmap(in_fp)
        zarr_arr = zarr.open(
            f"{out_fp}_tmp.zarr",
            mode="w",
            shape=mmap_arr.shape,
            dtype=mmap_arr.dtype,
            chunks=chunks,
        )
        zarr_arr[:] = mmap_arr
        # To final dask tiff
        zarr_arr = da.from_zarr(f"{out_fp}_tmp.zarr")
        zarr_arr.to_zarr(out_fp, overwrite=True)
        # Remove intermediate
        silent_remove(f"{out_fp}_tmp.zarr")

    @classmethod
    def tiff2zarr(
        cls,
        in_fp_ls: tuple[str, ...],
        out_fp: str,
        chunks: tuple[int, ...] = PROC_CHUNKS,
    ) -> None:
        # Getting shape and dtype
        arr0 = cls.read_tiff(in_fp_ls[0])
        shape = (len(in_fp_ls), *arr0.shape)
        dtype = arr0.dtype
        # Getting list of dask delayed tiffs
        tiffs_ls = [dask.delayed(cls.read_tiff)(i) for i in in_fp_ls]
        # Getting list of dask array tiffs and rechunking each (in prep later rechunking)
        tiffs_ls = [da.from_delayed(i, dtype=dtype, shape=shape[1:]).rechunk(chunks[1:]) for i in tiffs_ls]
        # Stacking tiffs and rechunking
        arr = da.stack(tiffs_ls, axis=0).rechunk(chunks)
        # Saving to zarr
        arr.to_zarr(out_fp, overwrite=True)

    @classmethod
    def zarr2tiff(cls, in_fp: str, out_fp: str) -> None:
        arr = da.from_zarr(in_fp)
        cls.write_tiff(out_fp, arr)

    @classmethod
    def btiff2niftygz(cls, in_fp: str, out_fp: str):
        arr = tifffile.imread(in_fp)
        nib.Nifti1Image(arr, None).to_filename(out_fp)

    @classmethod
    def read_niftygz(cls, fp):
        img = nib.load(fp)
        return np.array(img.dataobj)
