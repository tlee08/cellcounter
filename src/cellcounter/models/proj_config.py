import json
from enum import Enum
from pathlib import Path
from typing import Self

import numpy as np
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeInt,
    PositiveInt,
    model_validator,
)

from cellcounter.constants import ATLAS_DIR, PROC_CHUNKS

UnitFloat = Field(gt=0, le=1.0)


class RefVersions(Enum):
    """RefVersions."""

    AVERAGE_TEMPLATE_25 = "average_template_25"
    ARA_NISSL_25 = "ara_nissl_25"


class AnnotVersions(Enum):
    """AnnotVersions."""

    CCF_2017_25 = "ccf_2017_25"
    CCF_2016_25 = "ccf_2016_25"
    CCF_2015_25 = "ccf_2015_25"


class MapVersions(Enum):
    """MapVersions."""

    ABA_ANNOTATIONS = "ABA_annotations"
    CM_ANNOTATIONS = "CM_annotations"


class DimsConfig[T](BaseModel):
    """Dimensions configs."""

    z: T
    y: T
    x: T

    @classmethod
    def load_from_ls(cls, z: T, y: T, x: T) -> Self:
        """Helper function to load into struct from list."""
        return cls(z=z, y=y, x=x)


class SliceConfig(BaseModel):
    """Slice configs that map to slice(start, stop, step)`."""

    start: int | None = None
    stop: int | None = None
    step: int | None = None


class DimsSliceConfig(DimsConfig):
    """Slice configs across dimensions z, y, x."""

    z: SliceConfig = SliceConfig()
    y: SliceConfig = SliceConfig()
    x: SliceConfig = SliceConfig()


class ProjConfig(BaseModel):
    """Pydantic model for registration parameters."""

    # NOTE: can set extra as "forbid" to prevent extra keys
    model_config = ConfigDict(
        extra="ignore",
        validate_default=True,
        use_enum_values=True,
    )

    # CLUSTER SETTINGS
    heavy_n_workers: int = 2
    heavy_threads_per_worker: int = 1
    busy_n_workers: int = 6
    busy_threads_per_worker: int = 2
    # REFERENCE
    atlas_dir: Path = ATLAS_DIR
    ref_version: str = RefVersions.AVERAGE_TEMPLATE_25.value
    annot_version: str = AnnotVersions.CCF_2016_25.value
    map_version: str = MapVersions.ABA_ANNOTATIONS.value
    # RAW
    chunks: DimsConfig[PositiveInt] = DimsConfig.load_from_ls(*PROC_CHUNKS)
    # PREPARING IMAGE FOR REGISTRATION
    ref_orient_ls: DimsConfig[int] = DimsConfig.load_from_ls(1, 2, 3)
    ref_trim: DimsSliceConfig = DimsSliceConfig()
    downsample_rough: DimsConfig[PositiveInt] = DimsConfig.load_from_ls(3, 6, 6)
    downsample_fine: DimsConfig[UnitFloat] = DimsConfig.load_from_ls(1.0, 0.6, 0.6)
    reg_trim: DimsSliceConfig = DimsSliceConfig()
    lower_bound: NonNegativeInt = 100
    lower_bound_mapto: NonNegativeInt = 0
    upper_bound: NonNegativeInt = 5000
    upper_bound_mapto: NonNegativeInt = 5000
    # CELL COUNT TUNING CROP
    tuning_trim: DimsSliceConfig = DimsSliceConfig(
        z=SliceConfig(0, 100, None),
        y=SliceConfig(0, 2000, None),
        x=SliceConfig(0, 2000, None),
    )
    # CELL COUNTING
    tophat_radius: PositiveInt = 10
    dog_sigma1: PositiveInt = 1
    dog_sigma2: PositiveInt = 4
    large_gauss_sigma: PositiveInt = 101
    threshd_value: PositiveInt = 60
    min_threshd_size: PositiveInt = 100
    max_threshd_size: PositiveInt = 10000
    maxima_radius: PositiveInt = 10
    min_wshed_size: PositiveInt = 1
    max_wshed_size: PositiveInt = 1000
    # VISUAL CHECK
    heatmap_raw_radius: PositiveInt = 5
    heatmap_trfm_radius: PositiveInt = 3
    # COMBINE ARRAYS
    combine_cellcount_trim: DimsSliceConfig = DimsSliceConfig(
        z=SliceConfig(0, 10, None),
        y=SliceConfig(),
        x=SliceConfig(),
    )

    @model_validator(mode="after")
    def _validate_trims(self) -> Self:
        """Orient validation."""
        vect = np.array(self.ref_orient_ls)
        vect_abs = np.abs(vect)
        vect_abs_sorted = np.sort(vect_abs)
        assert np.all(vect_abs_sorted == np.array([1, 2, 3]))
        return self

    @model_validator(mode="after")
    def _validate_bounds(self) -> Self:
        """Upper and lower bounds validation."""
        assert self.lower_bound_mapto < self.lower_bound
        assert self.lower_bound < self.upper_bound
        assert self.upper_bound < self.upper_bound_mapto
        return self

    def update(self, **kwargs) -> Self:
        """Update configs."""
        return self.model_validate(self.model_copy(update=kwargs))

    @classmethod
    def read_file(cls, fp: Path | str) -> Self:
        """Read configs from file."""
        fp = Path(fp)
        with fp.open(mode="r") as f:
            content = json.load(f)
        model = cls.model_validate(content)
        return model

    def write_file(self, fp: Path | str) -> None:
        """Write file."""
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
        if updates:
            config = config.model_validate(config.model_copy(update=updates))
            config.write_file(config_fp)
        return config
