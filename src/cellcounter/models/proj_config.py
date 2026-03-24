import json
from enum import Enum
from pathlib import Path
from typing import Self

import numpy as np
from pydantic import BaseModel, ConfigDict, model_validator

from cellcounter.constants import ATLAS_DIR, PROC_CHUNKS


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
    zarr_chunksize: tuple[int, int, int] = PROC_CHUNKS
    # REGISTRATION
    ref_orient_ls: tuple[int, int, int] = (1, 2, 3)
    ref_z_trim: tuple[int | None, int | None, int | None] = (None, None, None)
    ref_y_trim: tuple[int | None, int | None, int | None] = (None, None, None)
    ref_x_trim: tuple[int | None, int | None, int | None] = (None, None, None)
    z_rough: int = 3
    y_rough: int = 6
    x_rough: int = 6
    z_fine: float = 1.0
    y_fine: float = 0.6
    x_fine: float = 0.6
    z_trim: tuple[int | None, int | None, int | None] = (None, None, None)
    y_trim: tuple[int | None, int | None, int | None] = (None, None, None)
    x_trim: tuple[int | None, int | None, int | None] = (None, None, None)
    lower_bound: tuple[int, int] = (100, 0)
    upper_bound: tuple[int, int] = (5000, 5000)
    # MASK
    mask_gaus_blur: int = 1
    mask_thresh: int = 300
    # CELL COUNT TUNING CROP
    tuning_z_trim: tuple[int | None, int | None, int | None] = (0, 100, None)
    tuning_y_trim: tuple[int | None, int | None, int | None] = (0, 2000, None)
    tuning_x_trim: tuple[int | None, int | None, int | None] = (0, 2000, None)
    # CELL COUNTING
    tophat_radius: int = 10
    dog_sigma1: int = 1
    dog_sigma2: int = 4
    large_gauss_sigma: int = 101
    threshd_value: int = 60
    min_threshd_size: int = 100
    max_threshd_size: int = 10000
    maxima_radius: int = 10
    min_wshed_size: int = 1
    max_wshed_size: int = 1000
    # VISUAL CHECK
    heatmap_raw_radius: int = 5
    heatmap_trfm_radius: int = 3
    # COMBINE ARRAYS
    combine_cellc_z_trim: tuple[int | None, int | None, int | None] = (0, 10, None)
    combine_cellc_y_trim: tuple[int | None, int | None, int | None] = (None, None, None)
    combine_cellc_x_trim: tuple[int | None, int | None, int | None] = (None, None, None)

    @model_validator(mode="after")
    def _validate_trims(self) -> Self:
        # Orient validation
        vect = np.array(self.ref_orient_ls)
        vect_abs = np.abs(vect)
        vect_abs_sorted = np.sort(vect_abs)
        assert np.all(vect_abs_sorted == np.array([1, 2, 3]))
        # TODO: Size validation
        # TODO: Trim validation
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
