"""Dask-based coordinate-to-image conversion for large arrays.

Converts cell coordinates to visualization arrays using Dask for
out-of-core processing of images too large for memory:
- coords2points: Single-voxel markers at each coordinate
- coords2heatmap: Spherical markers with radius for density visualization
- coords2regions: Region ID labels at each coordinate
"""

import dask.array as da
import numpy as np
import numpy.typing as npt
import pandas as pd

from cellcounter.constants import PROC_CHUNKS, AnnotColumns, Coords
from cellcounter.utils.dask_utils import coords2block, disk_cache

#####################################################################
#             Converting coordinates to spatial
#####################################################################


def coords2points_workers(
    arr: npt.NDArray, coords: pd.DataFrame, block_info: dict | None = None
) -> npt.NDArray:
    arr = arr.copy()
    # Offsetting coords with chunk space
    if block_info is not None:
        coords = coords2block(coords, block_info)
    # Formatting coord values as (z, y, x),
    # rounding to integers, and
    # Filtering
    s = arr.shape
    coords = (
        coords[[Coords.Z.value, Coords.Y.value, Coords.X.value]]
        .round(0)
        .astype(np.uint16)
        .query(
            f"({Coords.Z.value} >= 0) & ({Coords.Z.value} < {s[0]}) & "
            f"({Coords.Y.value} >= 0) & ({Coords.Y.value} < {s[1]}) & "
            f"({Coords.X.value} >= 0) & ({Coords.X.value} < {s[2]})"
        )
    )
    # Groupby and counts, so we don't drop duplicates
    coords = (
        coords.groupby([Coords.Z.value, Coords.Y.value, Coords.X.value])
        .size()
        .reset_index(name="counts")
    )  # type: ignore
    # Incrementing the coords inCoords.Y.valuee array
    if coords.shape[0] > 0:
        arr[
            coords[Coords.Z.value],
            coords[Coords.Y.value],
            coords[Coords.X.value],
        ] += coords["counts"]
    # Return arr
    return arr


def coords2sphere_workers(
    arr: npt.NDArray,
    coords: pd.DataFrame,
    r: int,
    block_info: dict | None = None,
) -> npt.NDArray:
    # Offsetting coords with chunk space
    if block_info:
        coords = coords2block(coords, block_info)
    # Formatting coord values as (z, y, x),
    # rounding to integers, and
    # Filtering for pCoords.Y.valuets within the image + radius padding bounds
    s = arr.shape
    coords = (
        coords[[Coords.Z.value, Coords.Y.value, Coords.X.value]]
        .round(0)
        .astype(np.uint16)
        .query(
            f"({Coords.Z.value} > {-1 * r}) & ({Coords.Z.value} < {s[0] + r}) & "
            f"({Coords.Y.value} > {-1 * r}) & ({Coords.Y.value} < {s[1] + r}) & "
            f"({Coords.X.value} > {-1 * r}) & ({Coords.X.value} < {s[2] + r})"
        )
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
            coords_i[Coords.Z.value] += z
            coords_i[Coords.Y.value] += y
            coords_i[Coords.X.value] += x
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
) -> da.Array:
    """Converts list of coordinates to spatial array single points.

    Params:
        coords: A pd.DataFrame of points, with the columns, `x`, `y`, and `z`.
        shape: The dimensions of the output array. Assumes that shape is in format `(z, y, x)` (regular for npy and tif file).
        out_fp: The output filename.

    Returns:
        The output image array
    """
    # Initialising spatial array
    arr = da.zeros(shape, chunks=PROC_CHUNKS, dtype=np.uint8)
    # Adding coords to image
    # arr = arr.map_blocks(
    #     lambda i, block_info=None: coords2points_workers(i, coords, block_info)
    # )
    arr = da.map_blocks(coords2points_workers, arr, coords)
    # Computing and saving
    return disk_cache(arr, out_fp)


def coords2heatmap(
    coords: pd.DataFrame,
    shape: tuple[int, ...],
    out_fp: Path | str,
    radius: int,
) -> da.Array:
    """Converts list of coordinates to spatial array as voxels.

    overlaying areas accumulate in intensity.

    Params:
        coords: A pd.DataFrame of points, with the columns, `x`, `y`, and `z`.
        r: radius of the voxels.
        shape: The dimensions of the output array. Assumes that shape is in format `(z, y, x)` (regular for npy and tif file).
        out_fp: The output filename.

    Returns:
        The output image array
    """
    # Initialising spatial array
    arr = da.zeros(shape, chunks=PROC_CHUNKS, dtype=np.uint8)
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
) -> da.Array:
    """Converts list of coordinates to spatial array.

    Params:
        coords: A pd.DataFrame of points, with the columns, `x`, `y`, `z`, and `id`.
        shape: The dimensions of the output array. Assumes that shape is in format `(z, y, x)` (regular for npy and tif file).
        out_fp: The output filename.

    Returns:
        The output image array
    """
    # Initialising spatial array
    arr = da.zeros(shape, chunks=PROC_CHUNKS, dtype=np.uint8)

    # Adding coords to image with np.apply_along_axis
    def f(coord):
        # Plotting coord to image. Including only coords within the image's bounds
        if np.all((coord >= 0) & (coord < shape)):
            z, y, x, _id = coord
            arr[z, y, x] = _id

    # Formatting coord values as (z, y, x) and rounding to integers
    coords = (
        coords[[Coords.Z.value, Coords.Y.value, Coords.X.value, AnnotColumns.ID.value]]
        .round(0)
        .astype(np.uint16)
    )
    if coords.shape[0] > 0:
        np.apply_along_axis(f, 1, coords)
    # Computing and saving
    return disk_cache(arr, out_fp)
