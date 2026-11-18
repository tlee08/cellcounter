from typing import Self

from pydantic import (
    BaseModel,
    NonNegativeFloat,
    PositiveFloat,
    PositiveInt,
    model_validator,
)

from cellcounter.models.proj_config.dims_config import DimsConfig


class CellCountingConfig(BaseModel):
    """Cell detection and filtering parameters.

    Every size-based filter is derived from ``cell_radius_px``, the
    physical cell radius expressed in voxels along each axis. Because
    acquisition voxels are anisotropic (z is usually thicker than x/y),
    each size is per-axis and obtained by scaling the radius by a
    dimensionless factor, so anisotropy is inherited automatically.
    """

    # Cell size - anchors every derived filter size.
    cell_radius_px: DimsConfig[PositiveFloat] = DimsConfig.from_tuple(1.5, 5.0, 5.0)

    # Dimensionless scale factors applied to cell_radius_px.
    tophat_radius_scale: PositiveFloat = 3.6
    dog_sigma1_scale: PositiveFloat = 1.2
    dog_sigma2_scale: PositiveFloat = 2.8
    maxima_radius_scale: PositiveFloat = 2.0

    # Absolute per-axis sigma for large-scale background estimation.
    # None skips the adaptive thresholding step.
    large_gauss_sigma_scale: PositiveFloat | None = None

    # Thresholding — three interchangeable methods (picked by threshold_foreground):
    #   threshold_percentile — keep the top (100-p)% of foreground (DoG > 0) voxels.
    #   threshold_offset_sd  — threshold at mean + offset*sd of foreground voxels.
    #   threshold_value      — fixed absolute intensity threshold (legacy).
    threshold_percentile: PositiveFloat = 98.0
    threshold_offset_sd: NonNegativeFloat = 1.0
    threshold_value: PositiveInt = 100

    # Threshold volume filtering (in voxels)
    min_threshold_volume: PositiveInt = 1
    max_threshold_volume: PositiveInt = 15000

    # Watershed
    min_watershed_volume: PositiveInt = 1
    max_watershed_volume: PositiveInt = 1000

    @property
    def tophat_radius(self) -> DimsConfig[float]:
        """Top-hat footprint radius, per axis."""
        return self.cell_radius_px.scaled(self.tophat_radius_scale)

    @property
    def dog_sigma1(self) -> DimsConfig[float]:
        """Inner DoG Gaussian sigma, per axis."""
        return self.cell_radius_px.scaled(self.dog_sigma1_scale)

    @property
    def dog_sigma2(self) -> DimsConfig[float]:
        """Outer DoG Gaussian sigma, per axis."""
        return self.cell_radius_px.scaled(self.dog_sigma2_scale)

    @property
    def large_gauss_sigma(self) -> DimsConfig[float] | None:
        """Large-scale Gaussian sigma, per axis."""
        if self.large_gauss_sigma_scale is None:
            return None
        return self.cell_radius_px.scaled(self.large_gauss_sigma_scale)

    @property
    def maxima_radius(self) -> DimsConfig[float]:
        """Local-maxima detection radius, per axis."""
        return self.cell_radius_px.scaled(self.maxima_radius_scale)

    @model_validator(mode="after")
    def _validate_sigma(self) -> Self:
        """Validate dog sigma ordering."""
        if not (self.dog_sigma1_scale < self.dog_sigma2_scale):
            err_msg = (
                f"dog_sigma1_scale ({self.dog_sigma1_scale}) must be < "
                f"dog_sigma2_scale ({self.dog_sigma2_scale})"
            )
            raise ValueError(err_msg)
        return self

    @model_validator(mode="after")
    def _validate_threshold_volumes(self) -> Self:
        """Validate threshold size ordering."""
        if not (self.min_threshold_volume < self.max_threshold_volume):
            err_msg = (
                f"min_threshold_volume ({self.min_threshold_volume}) must be < "
                f"max_threshold_volume ({self.max_threshold_volume})"
            )
            raise ValueError(err_msg)
        return self

    @model_validator(mode="after")
    def _validate_watershed_volumes(self) -> Self:
        """Validate watershed size ordering."""
        if not (self.min_watershed_volume < self.max_watershed_volume):
            err_msg = (
                f"min_watershed_volume ({self.min_watershed_volume}) must be < "
                f"max_watershed_volume ({self.max_watershed_volume})"
            )
            raise ValueError(err_msg)
        return self
