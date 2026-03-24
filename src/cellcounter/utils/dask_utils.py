import contextlib
from collections.abc import Generator
from pathlib import Path

import dask
import dask.array as da
import pandas as pd
from dask.distributed import Client, SpecCluster

from cellcounter.constants import Coords
from cellcounter.constants.paths import CACHE_DIR
from cellcounter.funcs.io_funcs import silent_remove


def setup_dask_configs() -> None:
    """Sets up dask with good configs.

    Run upon importing this library.
    """
    # Setting Dask configuration
    dask.config.set(
        {
            # "distributed.scheduler.active-memory-manager.measure": "managed",
            # "distributed.worker.memory.rebalance.measure": "managed",
            # "distributed.worker.memory.spill": False,
            # "distributed.worker.memory.pause": False,
            # "distributed.worker.memory.terminate": False,
            "temporary-directory": str(CACHE_DIR),
            "array.rechunk.method": "p2p",
            # Prevent task fusion (which can cause large memory blowouts)
            "optimization.fuse.active": False,
            "distributed.worker.memory.target": False,
            "distributed.worker.memory.spill": False,
            "distributed.worker.memory.pause": 0.80,
            "distributed.worker.memory.terminate": 0.95,
        }
    )


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
