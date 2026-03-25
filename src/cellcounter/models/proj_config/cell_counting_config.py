from typing import Self

from pydantic import BaseModel, PositiveInt, model_validator


class CellCountingConfig(BaseModel):
    """Cell detection and filtering parameters."""

    # Filtering
    tophat_radius: PositiveInt = 10
    dog_sigma1: PositiveInt = 1
    dog_sigma2: PositiveInt = 4
    large_gauss_radius: PositiveInt = 101

    # Thresholding
    threshd_value: PositiveInt = 60
    min_threshd_size: PositiveInt = 100
    max_threshd_size: PositiveInt = 10000

    # Watershed
    maxima_radius: PositiveInt = 10
    min_wshed_size: PositiveInt = 1
    max_wshed_size: PositiveInt = 1000

    @model_validator(mode="after")
    def _validate_sigma(self) -> Self:
        """Validate dog sigma ordering."""
        if not (self.dog_sigma1 < self.dog_sigma2):
            err_msg = (
                f"dog_sigma1 ({self.dog_sigma1}) must be < "
                f"dog_sigma2 ({self.dog_sigma2})"
            )
            raise ValueError(err_msg)
        return self

    @model_validator(mode="after")
    def _validate_threshd_sizes(self) -> Self:
        """Validate threshold size ordering."""
        if not (self.min_threshd_size < self.max_threshd_size):
            err_msg = (
                f"min_threshd_size ({self.min_threshd_size}) must be < "
                f"max_threshd_size ({self.max_threshd_size})"
            )
            raise ValueError(err_msg)
        return self

    @model_validator(mode="after")
    def _validate_wshed_sizes(self) -> Self:
        """Validate watershed size ordering."""
        if not (self.min_wshed_size < self.max_wshed_size):
            err_msg = (
                f"min_wshed_size ({self.min_wshed_size}) must be < "
                f"max_wshed_size ({self.max_wshed_size})"
            )
            raise ValueError(err_msg)
        return self
