from pydantic import BaseModel, PositiveInt


class ClusterConfig(BaseModel):
    """Dask cluster settings."""

    heavy_n_workers: PositiveInt = 2
    heavy_threads_per_worker: PositiveInt = 1
    busy_n_workers: PositiveInt = 6
    busy_threads_per_worker: PositiveInt = 2
