from typing import Self

from pydantic import BaseModel


class DimsConfig[T](BaseModel):
    """Dimensions configs for z, y, x axes."""

    z: T
    y: T
    x: T

    @classmethod
    def from_tuple(cls, z: T, y: T, x: T) -> Self:
        """Load into struct from list."""
        return cls(z=z, y=y, x=x)

    def to_tuple(self) -> tuple[T, T, T]:
        """Return tuple in (z, y, x) form."""
        return (self.z, self.y, self.x)


class SliceConfig(BaseModel):
    """Slice configs that map to slice(start, stop, step)."""

    start: int | None = None
    stop: int | None = None
    step: int | None = None

    def to_slice(self) -> slice:
        """Return slice object."""
        return slice(self.start, self.stop, self.step)


class DimsSliceConfig(DimsConfig):
    """Slice configs across dimensions z, y, x."""

    z: SliceConfig = SliceConfig()
    y: SliceConfig = SliceConfig()
    x: SliceConfig = SliceConfig()
