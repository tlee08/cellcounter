import json
from pathlib import Path
from typing import Self

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


class ProjConfig(BaseModel):
    """Pydantic model for pipeline configuration."""

    model_config = ConfigDict(
        extra="ignore",
        validate_default=True,
        use_enum_values=True,
    )

    # Zarr storage
    chunks: DimsConfig[PositiveInt] = DimsConfig.from_ls(500, 500, 500)

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
        return self.model_validate(self.model_copy(update=kwargs))

    @classmethod
    def read_file(cls, fp: Path | str) -> Self:
        """Read configs from JSON file."""
        fp = Path(fp)
        with fp.open(mode="r") as f:
            content = json.load(f)
        return cls.model_validate(content)

    def write_file(self, fp: Path | str) -> None:
        """Write configs to JSON file."""
        fp = Path(fp)
        fp.parent.mkdir(exist_ok=True)
        with fp.open(mode="w") as f:
            f.write(self.model_dump_json(indent=2))

    @classmethod
    def ensure(cls, config_fp: Path | str, **updates) -> Self:
        """Load config from file, creating default if needed, and apply updates.

        Args:
            config_fp: Path to config file.
            **updates: Fields to update in the config.

        Returns:
            The loaded or created config.
        """
        config_fp = Path(config_fp)
        try:
            config = cls.read_file(config_fp)
        except FileNotFoundError:
            config = cls()
            config.write_file(config_fp)
        if updates:
            config = config.model_validate(config.model_copy(update=updates))
            config.write_file(config_fp)
        return config
