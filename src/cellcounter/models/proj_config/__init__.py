from __future__ import annotations

"""ProjConfig pydantic model."""

import json
from pathlib import Path
from typing import Self

import yaml
from loguru import logger
from pydantic import BaseModel, ConfigDict, PositiveInt

from cellcounter.models.proj_config.cell_counting_config import CellCountingConfig
from cellcounter.models.proj_config.cluster_config import ClusterConfig
from cellcounter.models.proj_config.dims_config import (
    DimsConfig,
    DimsSliceConfig,
    SliceConfig,
)
from cellcounter.models.proj_config.reference_config import ReferenceConfig
from cellcounter.models.proj_config.registration_config import RegistrationConfig
from cellcounter.models.proj_config.visual_check_config import VisualCheckConfig

# ═══════════════════════════════════════════════════════════════════════════════
# Main Config Model
# ═══════════════════════════════════════════════════════════════════════════════


class ProjConfig(BaseModel):
    """Pydantic model for pipeline configuration."""

    model_config = ConfigDict(
        extra="ignore",
        validate_default=True,
        frozen=True,
    )

    # Zarr storage
    chunks: DimsConfig[PositiveInt] = DimsConfig.from_tuple(500, 500, 500)

    # Tuning crop (subset for parameter exploration)
    tuning_trim: DimsSliceConfig = DimsSliceConfig(
        z=SliceConfig(start=0, stop=100, step=None),
        y=SliceConfig(start=0, stop=2000, step=None),
        x=SliceConfig(start=0, stop=2000, step=None),
    )

    # Nested configs
    cluster: ClusterConfig = ClusterConfig()
    reference: ReferenceConfig = ReferenceConfig()
    registration: RegistrationConfig = RegistrationConfig()
    cell_counting: CellCountingConfig = CellCountingConfig()
    visual_check: VisualCheckConfig = VisualCheckConfig()

    @classmethod
    def read_yaml(cls, fp: Path) -> ProjConfig:
        """Read the config from a yaml file."""
        return cls.model_validate(yaml.safe_load(fp.read_text()))

    def write_yaml(self, fp: Path) -> None:
        """Write the config to a yaml file."""
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(yaml.dump(self.model_dump(), default_flow_style=False))


__all__ = [
    "ProjConfig",
]
