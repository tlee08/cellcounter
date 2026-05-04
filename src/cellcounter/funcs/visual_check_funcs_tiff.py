"""Memory-mapped coordinate-to-image conversion for TIFF files.

Converts cell coordinates to visualization arrays using numpy memmap
for memory-efficient processing. Use for outputs that fit in disk
but not RAM:
- coords2points: Single-voxel markers at each coordinate
- coords2heatmap: Spherical markers with radius for density visualization
- coords2regions: Region ID labels at each coordinate
"""

import os
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd

from cellcounter.constants import CACHE_DIR, AnnotColumns, Coords
from cellcounter.funcs.io_funcs import silent_remove, write_tiff

#####################################################################
#             Converting coordinates to spatial
#####################################################################


def coords2points_workers(arr: npt.NDArray, coords: pd.DataFrame) -> npt.NDArray:
    """Formatting coord values as (z, y, x), rounding to integers, and filtering."""
    s = arr.shape
    coords = (
        coords[[Coords.Z.value, Coords.Y.value, Coords.X.value]]
        .query(
            f"({Coords.Z.value} >= 0) & ({Coords.Z.value} < {s[0]}) & "
            f"({Coords.Y.value} >= 0) & ({Coords.Y.value} < {s[1]}) & "
            f"({Coords.X.value} >= 0) & ({Coords.X.value} < {s[2]})"
        )
        .round(0)
        .clip(0, 2**16 - 1)
        .astype(np.uint16)
        .to_numpy()
    )
    # Incrementing the coords in the array
    if coords.shape[0] > 0:
        arr[coords[:, 0], coords[:, 1], coords[:, 2]] += 1
    # Return arr
    return arr


def make_mmap(shape: tuple[int, ...], out_fp: Path | str) -> npt.NDArray:
    """Make numpy memory map."""
    # Initialising spatial array
    arr = np.memmap(
        out_fp,
        mode="w+",
        shape=shape,
        dtype=np.uint8,
    )
    return arr


#####################################################################
#             Converting coordinates to spatial
#####################################################################


def coords2points(
    coords: pd.DataFrame,
    shape: tuple[int, ...],
    out_fp: Path | str,
) -> None:
    """Converts list of coordinates to spatial array single points.

    Params:
        coords: A pd.DataFrame of points, with the columns, `x`, `y`, and `z`.
        shape: The dimensions of the output array. Assumes that shape is in format
            `(z, y, x)` (regular for npy and tif file).
        out_fp: The output filename.

    Returns:
        The output image array
    """
    # Initialising spatial array
    temp_fp = CACHE_DIR / f"temp_{os.getpid()}.dat"
    arr = make_mmap(shape, temp_fp)
    # Adding coords to image
    coords2points_workers(arr, coords)
    # Saving the subsampled array
    write_tiff(arr, out_fp)
    # Removing temporary memmap
    silent_remove(temp_fp)


def coords2heatmap(
    coords: pd.DataFrame,
    shape: tuple[int, ...],
    out_fp: Path | str,
    radius: int,
) -> None:
    """Converts list of coordinates to spatial array as voxels.

    Overlaying areas accumulate in intensity.

    Params:
        coords: A pd.DataFrame of points, with the columns, `x`, `y`, and `z`.
        r: radius of the voxels.
        shape: The dimensions of the output array. Assumes that shape is in format
            `(z, y, x)` (regular for npy and tif file).
        out_fp: The output filename.

    Returns:
        The output image array
    """
    # Initialising spatial array
    temp_fp = CACHE_DIR / f"temp_{os.getpid()}.dat"
    arr = make_mmap(shape, temp_fp)

    # Constructing sphere array mask
    zz, yy, xx = np.ogrid[1 : radius * 2, 1 : radius * 2, 1 : radius * 2]
    circ = (
        np.sqrt((xx - radius) ** 2 + (yy - radius) ** 2 + (zz - radius) ** 2) < radius
    )
    # Constructing offset indices
    i = np.arange(-radius + 1, radius)
    z_ind, y_ind, x_ind = np.meshgrid(i, i, i, indexing="ij")
    # Adding coords to image
    for z, y, x, t in zip(
        z_ind.ravel(), y_ind.ravel(), x_ind.ravel(), circ.ravel(), strict=False
    ):
        if t:
            coords_i = coords.copy()
            coords_i[Coords.Z.value] += z
            coords_i[Coords.Y.value] += y
            coords_i[Coords.X.value] += x
            coords2points_workers(arr, coords_i)

    # Saving the subsampled array
    write_tiff(arr, out_fp)
    # Removing temporary memmap
    silent_remove(temp_fp)


def coords2regions(
    coords: pd.DataFrame, shape: tuple[int, ...], out_fp: Path | str
) -> None:
    """Converts list of coordinates to spatial array.

    Params:
        coords: A pd.DataFrame of points, with the columns, `x`, `y`, `z`, and `id`.
        shape: The dimensions of the output array. Assumes that shape is in format
            `(z, y, x)` (regular for npy and tif file).
        out_fp: The output filename.

    Returns:
        The output image array
    """
    # Initialising spatial array
    temp_fp = CACHE_DIR / f"temp_{os.getpid()}.dat"
    arr = make_mmap(shape, temp_fp)

    # Adding coords to image with np.apply_along_axis
    def f(coord: npt.NDArray) -> None:
        # Plotting coord to image. Including only coords within the image's bounds
        if np.all((coord >= 0) & (coord < shape)):
            z, y, x, _id = coord
            arr[z, y, x] = _id

    # Formatting coord values as (z, y, x) and rounding to integers
    coords = (
        coords[[Coords.Z.value, Coords.Y.value, Coords.X.value, AnnotColumns.ID.value]]
        .round(0)
        .clip(0, 2**16 - 1)
        .astype(np.uint16)
    )
    if coords.shape[0] > 0:
        np.apply_along_axis(f, 1, coords)

    # Saving the subsampled array
    write_tiff(arr, out_fp)
    # Removing temporary memmap
    silent_remove(temp_fp)
