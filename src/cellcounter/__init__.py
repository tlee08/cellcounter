import warnings

from cellcounter.pipeline.pipeline import Pipeline
from cellcounter.pipeline.visual_check import VisualCheck
from cellcounter.utils.dask_utils import setup_dask_configs
from cellcounter.utils.logger import setup_logging

warnings.filterwarnings("ignore")

setup_logging()
setup_dask_configs()

__all__ = [
    "Pipeline",
    "VisualCheck",
]
