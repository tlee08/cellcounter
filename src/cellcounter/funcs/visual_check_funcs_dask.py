"""Dask-based coordinate-to-image conversion for large arrays.

Converts cell coordinates to visualization arrays using Dask for
out-of-core processing of images too large for memory:
- coords2points: Single-voxel markers at each coordinate
- coords2heatmap: Spherical markers with radius for density visualization
- coords2regions: Region ID labels at each coordinate
"""

from pathlib import Path

import dask.array as da
import numpy as np
import numpy.typing as npt
import pandas as pd

from cellcounter.constants import ID, X, Y, Z
from cellcounter.utils.dask_utils import coords2block, disk_cache

#####################################################################
#             Converting coordinates to spatial
#####################################################################


def coords2points_workers(
    arr: npt.NDArray, coords: pd.DataFrame, block_info: dict | None = None
) -> npt.NDArray:
    """Coords to points worker."""
    arr = arr.copy()
    # Offsetting coords with chunk space
    if block_info is not None:
        coords = coords2block(coords, block_info)
    # Formatting coord values as (z, y, x),
    # rounding to integers, and
    # Filtering
    s = arr.shape
    coords = (
        coords[[Z, Y, X]]
        .round(0)
        .query(
            f"({Z} >= 0) & ({Z} < {s[0]}) & "
            f"({Y} >= 0) & ({Y} < {s[1]}) & "
            f"({X} >= 0) & ({X} < {s[2]})"
        )
        .clip(0, 2**16 - 1)
        .astype(np.uint16)
    )
    # Groupby and counts, so we don't drop duplicates
    coords = coords.groupby([Z, Y, X]).size().reset_index(name="counts")
    # Incrementing the coords in array
    if coords.shape[0] > 0:
        arr[
            coords[Z],
            coords[Y],
            coords[X],
        ] += coords["counts"]
    # Return arr
    return arr


def coords2sphere_workers(
    arr: npt.NDArray,
    coords: pd.DataFrame,
    r: int,
    block_info: dict | None = None,
) -> npt.NDArray:
    """Sphere worker."""
    # Offsetting coords with chunk space
    if block_info:
        coords = coords2block(coords, block_info)
    # Formatting coord values as (z, y, x),
    # rounding to integers, and
    # Filtering for pts within the image + radius padding bounds
    s = arr.shape
    coords = (
        coords[[Z, Y, X]]
        .round(0)
        .query(
            f"({Z} > {-1 * r}) & ({Z} < {s[0] + r}) & "
            f"({Y} > {-1 * r}) & ({Y} < {s[1] + r}) & "
            f"({X} > {-1 * r}) & ({X} < {s[2] + r})"
        )
        .clip(0, 2**16 - 1)
        .astype(np.uint16)
    )
    # Constructing index and sphere mask arrays
    i = np.arange(-r, r + 1)
    z_ind, y_ind, x_ind = np.meshgrid(i, i, i, indexing="ij")
    circ = np.square(z_ind) + np.square(y_ind) + np.square(x_ind) <= np.square(r)
    # Adding coords to image
    for z, y, x, t in zip(
        z_ind.ravel(), y_ind.ravel(), x_ind.ravel(), circ.ravel(), strict=True
    ):
        if t:
            coords_i = coords.copy()
            coords_i[Z] += z
            coords_i[Y] += y
            coords_i[X] += x
            arr = coords2points_workers(arr, coords_i)
    # Return arr
    return arr


#####################################################################
#             Converting coordinates to spatial
#####################################################################


def coords2points(
    coords: pd.DataFrame,
    shape: tuple[int, ...],
    out_fp: Path | str,
    chunks: tuple[int, ...],
) -> da.Array:
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
    arr = da.zeros(shape, chunks=chunks, dtype=np.uint8)
    # Adding coords to image
    arr = da.map_blocks(coords2points_workers, arr, coords)
    # Computing and saving
    return disk_cache(arr, out_fp)


def coords2heatmap(
    coords: pd.DataFrame,
    shape: tuple[int, ...],
    out_fp: Path | str,
    radius: int,
    chunks: tuple[int, ...],
) -> da.Array:
    """Converts list of coordinates to spatial array as voxels.

    overlaying areas accumulate in intensity.

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
    arr = da.zeros(shape, chunks=chunks, dtype=np.uint8)
    # Adding coords to image
    arr = arr.map_blocks(
        lambda i, block_info=None: coords2sphere_workers(i, coords, radius, block_info)
    )
    # Computing and saving
    return disk_cache(arr, out_fp)


def coords2regions(
    coords: pd.DataFrame,
    shape: tuple[int, ...],
    out_fp: str,
    chunks: tuple[int, ...],
) -> da.Array:
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
    arr = da.zeros(shape, chunks=chunks, dtype=np.uint8)

    # Adding coords to image with np.apply_along_axis
    def f(coord: npt.NDArray) -> None:
        # Plotting coord to image. Including only coords within the image's bounds
        if np.all((coord >= 0) & (coord < shape)):
            z, y, x, _id = coord
            arr[z, y, x] = _id

    # Formatting coord values as (z, y, x) and rounding to integers
    coords = coords[[Z, Y, X, ID]].round(0).clip(0, 2**16 - 1).astype(np.uint16)
    if coords.shape[0] > 0:
        np.apply_along_axis(f, 1, coords)
    # Computing and saving
    return disk_cache(arr, out_fp)
