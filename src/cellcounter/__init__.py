"""Cellcounter package."""

from cellcounter.pipeline import Pipeline, VisualCheck
from cellcounter.utils import configure_logger, setup_dask_configs

configure_logger()
setup_dask_configs()

__all__ = [
    "Pipeline",
    "VisualCheck",
]
