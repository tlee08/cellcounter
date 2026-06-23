"""ProjConfig pydantic model."""

import json
from pathlib import Path
from typing import Self

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


def _deep_merge(target: dict, source: dict) -> dict:
    """Recursively merge source dict into target dict.

    Source values will overwrite target values,
    except when both values are dicts,
    in which case the merge is applied recursively.
    """
    for key, value in source.items():
        if isinstance(value, dict) and key in target and isinstance(target[key], dict):
            target[key] = _deep_merge(target[key], value)
        else:
            target[key] = value
    return target


class ProjConfig(BaseModel):
    """Pydantic model for pipeline configuration."""

    model_config = ConfigDict(
        extra="ignore",
        validate_default=True,
        use_enum_values=True,
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

    def update(self, **kwargs) -> Self:
        """Update configs and return new instance."""
        merged = _deep_merge(self.model_dump(), kwargs)
        return self.model_validate(merged)

    @classmethod
    def read_file(cls, fp: Path | str) -> Self:
        """Read configs from JSON file."""
        fp = Path(fp)
        with fp.open(mode="r", encoding="utf-8") as f:
            content = json.load(f)
        return cls.model_validate(content)

    def write_file(self, fp: Path | str) -> None:
        """Write configs to JSON file."""
        fp = Path(fp)
        fp.parent.mkdir(parents=True, exist_ok=True)
        with fp.open(mode="w", encoding="utf-8") as f:
            f.write(self.model_dump_json(indent=2))

    @classmethod
    def ensure(cls, config_fp: Path | str, updates: dict) -> Self:
        """Load config from file, creating default if needed, and apply updates.

        Args:
            config_fp: Path to config file.
            updates: Fields to update in the config.

        Returns:
            The loaded or created config.
        """
        should_write = False
        config_fp = Path(config_fp)
        # Load existing config or create default if file doesn't exist
        try:
            config = cls.read_file(config_fp)
        except FileNotFoundError:
            logger.info("Config file not found at {} — creating default", config_fp)
            config = cls()
            should_write = True
        # Update the config with any provided updates and pydantic validate
        # Write back to file if updates to ensure the file is always in sync
        if updates:
            config = config.update(**updates)
            should_write = True
        # Write back to file if we modified config
        if should_write:
            config.write_file(config_fp)
        return config


__all__ = [
    "ProjConfig",
]
