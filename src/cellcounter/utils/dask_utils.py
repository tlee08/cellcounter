import contextlib
from collections.abc import Callable
from pathlib import Path

import dask
import dask.array
import dask.array as da
import dask.dataframe as dd
import numpy as np
import pandas as pd
from dask import compute
from dask.distributed import Client, SpecCluster

from cellcounter.constants import Coords
from cellcounter.utils.io_utils import silent_remove
from cellcounter.utils.misc_utils import const2iter

#############################################
# Spatially aware array -> coords
#############################################


def block2coords(func: Callable, *args) -> dd.DataFrame:
    """Applies the `func` to `arr`.

    Expects `func` to convert `arr` to coords df (of sorts).

    Importantly, this offsets the coords in each block using ONLY
    the chunks of the first da.Array object in `args`.

    All da.Arrays must have the same shape of chunks but
    can have different chunk sizes.

    Process is:
        - Convert dask arrays to delayed object blocks
        - Perform `func([arr1_blocki, arr2_blocki, ...], *args)` for each block
        - At each block, offset the coords by the block's location in the entire array.
    """
    # Assertions
    arr_blocks_ls = [i for i in args if isinstance(i, da.Array)]
    # Asserting that there is at least one da.Array (i.e curr_chunks not None)
    assert len(arr_blocks_ls) > 0, "At least one da.Array must be passed."
    # Getting first da.Array
    arr0 = arr_blocks_ls[0]
    # Asserting that all da.Arrays have the same shape of chunks
    assert all(i.to_delayed().shape == arr0.to_delayed().shape for i in arr_blocks_ls)
    # Converting chunks tuple[tuple] from chunk sizes to block offsets
    offsets = [np.cumsum([0, *i[:-1]]) for i in arr0.chunks]
    # Creating the meshgrid of offsets to get offsets for each block in order
    z_off, y_off, x_off = np.meshgrid(*offsets, indexing="ij")
    z_off, y_off, x_off = map(np.ravel, (z_off, y_off, x_off))
    # Getting number of blocks
    nblocks = np.prod(arr0.numblocks)
    # Converting dask arrays to list of delayed blocks in args list
    blocks_args = [
        i.to_delayed().ravel()
        if isinstance(i, da.Array)
        else list(const2iter(i, nblocks))
        for i in args
    ]
    # Transposing from (arg, blocks) to (block, arg) dimensions
    args_blocks = list(zip(*blocks_args, strict=True))
    # Applying the function to each block
    results_ls = [
        func_offsetted_delayed(func, args_block, z_offset, y_offset, x_offset)
        for args_block, z_offset, y_offset, x_offset in zip(
            args_blocks, z_off, y_off, x_off, strict=True
        )
    ]
    # NOTE: current workaround for memory/taking-on-too-many-blocks issue
    # is compute each block separately
    return dd.concat([compute(i)[0] for i in results_ls], axis=0, ignore_index=True)


@dask.delayed
def func_offsetted_delayed(
    func: Callable, args: tuple, z_off: int, y_off: int, x_off: int
) -> dd.DataFrame:
    """Offsets the coords in each block.

    Given the block args and offsets, applies the function to each block
    and offsets the outputted coords for the block.

    Intended to be used with `block2coords`.
    """
    # Running the function with the args
    df = func(*args)
    # Offsetting the coords in the df
    df[Coords.Z.value] = (
        df[Coords.Z.value] + z_off if Coords.Z.value in df.columns else z_off
    )
    df[Coords.Y.value] = (
        df[Coords.Y.value] + y_off if Coords.Y.value in df.columns else y_off
    )
    df[Coords.X.value] = (
        df[Coords.X.value] + x_off if Coords.X.value in df.columns else x_off
    )
    # Returning the df
    return df


def coords2block(df: pd.DataFrame, block_info: dict) -> pd.DataFrame:
    """Converts the coords to a block, given the block info.

    The block info is from Dask, in the map_blocks function
    (and other relevant functions like map_block).
    The block info is required so relevant block offsets are used.
    """
    # Getting block info
    z, y, x = block_info[0]["array-location"]
    # Copying df
    df = df.copy()
    # Offsetting
    df[Coords.Z.value] = df[Coords.Z.value] - z[0]
    df[Coords.Y.value] = df[Coords.Y.value] - y[0]
    df[Coords.X.value] = df[Coords.X.value] - x[0]
    return df


#############################################
# Other helpful Dask functions
#############################################


def disk_cache(arr: da.Array, fp: Path | str) -> da.Array:
    """Save array to disk and return the array.

    This is a good way to cache results.
    """
    fp = Path(fp)
    # Make parent dir
    fp.parent.mkdir(exist_ok=True)
    # Remove existing file (if there) - otherwise error thrown by dask
    silent_remove(fp)
    # Write arr to disk as zarr
    arr.to_zarr(fp, mode="w")
    # Return arr
    return da.from_zarr(fp)


@contextlib.contextmanager
def cluster_process(cluster: SpecCluster):
    """Context manager that runs the given Dask cluster and a client.

    Makes a Dask cluster and client, runs the body in the context manager,
    then closes the client and cluster.
    """
    client = Client(cluster)
    print(f"Dask Dashboard is accessible at {client.dashboard_link}")
    try:
        yield
    finally:
        client.close()
        cluster.close()
