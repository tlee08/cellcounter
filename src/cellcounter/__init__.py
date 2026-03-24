import warnings

from cellcounter.funcs.batch_combine_funcs import BatchCombineFuncs as BatchCombineFuncs
from cellcounter.pipeline.pipeline import Pipeline as Pipeline
from cellcounter.pipeline.visual_check import VisualCheck as VisualCheck
from cellcounter.utils.logger import setup_logging

warnings.filterwarnings("ignore")

setup_logging()
