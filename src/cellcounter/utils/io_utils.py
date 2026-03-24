from __future__ import annotations

import asyncio
import contextlib
import json
import os
import re
import shutil
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from natsort import natsorted

if TYPE_CHECKING:
    import pandas as pd

# TODO: add request functionality to download Allen Mouse Atlas image:
# Atlas from https://download.alleninstitute.org/informatics-archive/current-release/mouse_ccf/

#####################################################################
#                     Getting filepaths in order
#####################################################################


def get_filepaths(my_dir: Path | str, pattern: str) -> list[Path]:
    r"""Looks in dir for filepaths with the pattern.

    The pattern follows the regex search syntax:
        - Write the pattern identical to the filenames, except for the Z0000 number.
        - Use `*` to indicate "any character any number of times" (i.e. for Z number)
    An example pattern is:
    ```
    `r"Sample_11_zoom0.52_2.5x_dual_side_fusion_2x4 vertical-stitched_T001_Z(\d+?)_C01.tif"`
    ```
    """  # noqa: E501
    my_dir = Path(my_dir)
    fps_all = natsorted(my_dir.iterdir())
    fps = [(my_dir / i) for i in fps_all if re.search(pattern, i)]
    return fps


def rename_slices_filepaths(my_dir: Path | str, pattern: str) -> None:
    """Rename slices.

    TODO: make pattern modular, instead of current (pref)(to_change)(suffix) setup
    Currently just converts slices filenames to <Z,4>.
    """
    my_dir = Path(my_dir)
    for fp in my_dir.iterdir():
        print(fp)
        fp_new = re.sub(
            pattern,
            lambda x: x.group(0).zfill(4),
            str(fp),
        )
        print(fp_new)
        print()


#####################################################################
#                     Make npy headers for ImageJ
#####################################################################


def get_npy_header_size(fp: Path | str) -> int:
    """Get size of numpy header."""
    fp = Path(fp)
    with fp.open(mode="rb") as f:
        h_size = 0
        while True:
            char = f.read(1)
            h_size += 1
            if char == b"\n":
                break
    return h_size


def make_npy_header(fp: Path | str) -> None:
    """Makes a npy mhd header file so ImageJ can read .npy spatial arrays.

    If `fp` is does not have the `.npy` file extension, then adds it.
    """
    fp = Path(fp)
    # Adding ".npy" extension if missing
    if not re.search(r"\.npy$", str(fp)):
        fp = f"{fp}.npy"
    # Making datatype name mapper
    dtype_mapper = {
        "int8": "MET_CHAR",
        "uint8": "MET_UCHAR",
        "int16": "MET_SHORT",
        "uint16": "MET_USHORT",
        "int32": "MET_INT",
        "uint32": "MET_UINT",
        "int64": "MET_LONG",
        "uint64": "MET_ULONG",
        "float32": "MET_FLOAT",
        "float64": "MET_DOUBLE",
    }

    # Loading array
    ar = np.load(fp, mmap_mode="r")
    # Making header contents
    header_content = f"""ObjectType = Image
NDims = 3
BinaryData = True
BinaryDataByteOrderMSB = False
DimSize = {ar.shape[2]} {ar.shape[1]} {ar.shape[0]}
HeaderSize = {get_npy_header_size(fp)}
ElementType = {dtype_mapper[str(ar.dtype)]}
ElementDataFile = {os.path.split(fp)[1]}
"""
    # Saving header file
    header_fp = Path(f"{fp}.mhd")
    with header_fp.open(mode="w") as f:
        f.write(header_content)


def read_json(fp: Path | str) -> dict:
    """Read json from file."""
    fp = Path(fp)
    with fp.open(mode="r") as f:
        return json.load(f)


def write_file(fp: Path | str, content: str) -> None:
    """Write json to file."""
    fp = Path(fp)
    fp.parent.mkdir(exist_ok=True)
    with fp.open(mode="w") as f:
        f.write(content)


def silent_remove(fp: Path | str) -> None:
    """Remove file or dir without throwing errors."""
    fp = Path(fp)
    if fp.is_file():
        with contextlib.suppress(OSError):
            fp.unlink()
    elif fp.is_dir():
        with contextlib.suppress(OSError):
            shutil.rmtree(fp)


def sanitise_smb_df(df: pd.DataFrame) -> pd.DataFrame:
    """Sanitizes the SMB share dataframe.

    Removes any column called "smb-share:server".
    """
    if "smb-share:server" in df.columns:
        df = df.drop(columns="smb-share:server")
    return df


#####################################################################
# DF IO
#####################################################################


def write_parquet(df: pd.DataFrame, fp: Path | str) -> None:
    """Write parquet."""
    fp = Path(fp)
    fp.parent.mkdir(exist_ok=True)
    df.to_parquet(fp)


#####################################################################
# ASYNC READ MULTIPLE FILES
#####################################################################


async def async_read(
    fp: str, executor: ThreadPoolExecutor, read_func: Callable
) -> list:
    """Asynchronously read a single file."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(executor, read_func, fp)


async def async_read_files(fp_ls: list[Path | str], read_func: Callable) -> list:
    """Asynchronously read a list of files and return a list of numpy arrays."""
    with ThreadPoolExecutor() as executor:
        tasks = [async_read(fp, executor, read_func) for fp in fp_ls]
        return await asyncio.gather(*tasks)


def async_read_files_run(fp_ls: list[Path | str], read_func: Callable) -> list:
    """Asynchronously read a list of files and return a list of numpy arrays."""
    return asyncio.run(async_read_files(fp_ls, read_func))
