from typing import Self

from pydantic import (
    BaseModel,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    model_validator,
)

from cellcounter.models.proj_config.dims_config import DimsConfig, DimsSliceConfig


class RegistrationConfig(BaseModel):
    """Image registration parameters."""

    ref_orientation: DimsConfig[int] = DimsConfig.from_tuple(1, 2, 3)
    ref_trim: DimsSliceConfig = DimsSliceConfig()
    downsample_rough: DimsConfig[PositiveInt] = DimsConfig.from_tuple(3, 6, 6)
    downsample_fine: DimsConfig[PositiveFloat] = DimsConfig.from_tuple(1.0, 0.6, 0.6)
    reg_trim: DimsSliceConfig = DimsSliceConfig()
    lower_bound: NonNegativeInt = 100
    lower_bound_mapto: NonNegativeInt = 0
    upper_bound: NonNegativeInt = 5000
    upper_bound_mapto: NonNegativeInt = 5000

    @model_validator(mode="after")
    def _validate_orientation(self) -> Self:
        """Validate orientation vector contains -3 to 3, no repeats."""
        vect = [self.ref_orientation.z, self.ref_orientation.y, self.ref_orientation.x]
        vect_abs = sorted(abs(v) for v in vect)
        if vect_abs != [1, 2, 3]:
            err_msg = (
                f"ref_orientation must contain values 1,2,3 (negated ok), got {vect}"
            )
            raise ValueError(err_msg)
        return self

    @model_validator(mode="after")
    def _validate_downsample(self) -> Self:
        """Validate fine downsample is between 0 and 1."""
        for i in [
            self.downsample_fine.z,
            self.downsample_fine.y,
            self.downsample_fine.x,
        ]:
            if not (0.0 < i <= 1.0):
                err_msg = f"Must have fine downsample values between 0 and 1. Got: {i}"
                raise ValueError(err_msg)
        return self

    @model_validator(mode="after")
    def _validate_bounds(self) -> Self:
        """Validate upper and lower bounds ordering."""
        if not (self.lower_bound_mapto <= self.lower_bound):
            err_msg = (
                f"lower_bound_mapto ({self.lower_bound_mapto}) must be < "
                f"lower_bound ({self.lower_bound})"
            )
            raise ValueError(err_msg)
        if not (self.lower_bound <= self.upper_bound):
            err_msg = (
                f"lower_bound ({self.lower_bound}) must be < "
                f"upper_bound ({self.upper_bound})"
            )
            raise ValueError(err_msg)
        if not (self.upper_bound <= self.upper_bound_mapto):
            err_msg = (
                f"upper_bound ({self.upper_bound}) must be < "
                f"upper_bound_mapto ({self.upper_bound_mapto})"
            )
            raise ValueError(err_msg)
        return self
