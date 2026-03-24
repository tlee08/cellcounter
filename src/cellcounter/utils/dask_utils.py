import contextlib
from collections.abc import Generator
from pathlib import Path

import dask.array as da
import pandas as pd
from dask.distributed import Client, SpecCluster

from cellcounter.constants import Coords
from cellcounter.funcs.io_funcs import silent_remove


def coords2block(df: pd.DataFrame, block_info: dict) -> pd.DataFrame:
    """Convert global coords to block-local coords using dask block_info."""
    z, y, x = block_info[0]["array-location"]
    df = df.copy()
    df[Coords.Z.value] = df[Coords.Z.value] - z[0]
    df[Coords.Y.value] = df[Coords.Y.value] - y[0]
    df[Coords.X.value] = df[Coords.X.value] - x[0]
    return df


def disk_cache(arr: da.Array, fp: Path | str) -> da.Array:
    """Save array to disk as zarr and return lazy array."""
    fp = Path(fp)
    fp.parent.mkdir(exist_ok=True)
    silent_remove(fp)
    arr.to_zarr(fp, mode="w")
    return da.from_zarr(fp)


@contextlib.contextmanager
def cluster_process(cluster: SpecCluster) -> Generator[None, None, None]:
    """Context manager for Dask cluster with automatic cleanup."""
    client = Client(cluster)
    print(f"Dask Dashboard is accessible at {client.dashboard_link}")
    try:
        yield
    finally:
        client.close()
        cluster.close()
