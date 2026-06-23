"""Pipeline orchestration for cell counting and region mapping.

Provides the main Pipeline class for running registration, cell counting,
and mapping workflows, plus visual quality control tools.
"""

from .pipeline import Pipeline
from .visual_check import VisualCheck

__all__ = [
    "Pipeline",
    "VisualCheck",
]
