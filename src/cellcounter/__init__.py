"""Cellcounter package."""

from cellcounter.pipeline.pipeline import Pipeline
from cellcounter.pipeline.visual_check import VisualCheck
from cellcounter.utils.dask_utils import setup_dask_configs
from cellcounter.utils.logger_utils import configure_logger

configure_logger()
setup_dask_configs()

__all__ = [
    "Pipeline",
    "VisualCheck",
]
