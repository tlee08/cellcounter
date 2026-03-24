from enum import Enum


class Coords(Enum):
    """The de facto order of the 3d dimensions (for tiff and zarr)."""

    Z = "z"
    Y = "y"
    X = "x"
