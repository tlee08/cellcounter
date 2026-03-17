import os

import numpy as np
import pandas as pd

from cellcounter.constants import CACHE_DIR, AnnotColumns, Coords
from cellcounter.funcs.arr_io_funcs import ArrIOFuncs
from cellcounter.utils.io_utils import silent_remove

#####################################################################
#             Converting coordinates to spatial
#####################################################################


class VisualCheckFuncsTiff:
    @classmethod
    def coords2points_workers(cls, arr: np.ndarray, coords: pd.DataFrame):
        # Formatting coord values as (z, y, x),
        # rounding to integers, and
        # Filtering
        s = arr.shape
        coords = (
            coords[[Coords.Z.value, Coords.Y.value, Coords.X.value]]
            .round(0)
            .astype(np.int16)
            .query(
                f"({Coords.Z.value} >= 0) & ({Coords.Z.value} < {s[0]}) & "
                f"({Coords.Y.value} >= 0) & ({Coords.Y.value} < {s[1]}) & "
                f"({Coords.X.value} >= 0) & ({Coords.X.value} < {s[2]})"
            )
            .values
        )  # type: ignore
        # Incrementing the coords in the array
        if coords.shape[0] > 0:
            arr[coords[:, 0], coords[:, 1], coords[:, 2]] += 1
        # Return arr
        return arr

    @classmethod
    def make_mmap(cls, shape: tuple, out_fp: str) -> np.ndarray:
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

    @classmethod
    def coords2points(
        cls,
        coords: pd.DataFrame,
        shape: tuple[int, ...],
        out_fp: str,
    ):
        """
        Converts list of coordinates to spatial array single points.

        Params:
            coords: A pd.DataFrame of points, with the columns, `x`, `y`, and `z`.
            shape: The dimensions of the output array. Assumes that shape is in format `(z, y, x)` (regular for npy and tif file).
            out_fp: The output filename.

        Returns:
            The output image array
        """
        # Initialising spatial array
        temp_fp = os.path.join(CACHE_DIR, f"temp_{os.getpid()}.dat")
        arr = cls.make_mmap(shape, temp_fp)
        # Adding coords to image
        cls.coords2points_workers(arr, coords)
        # Saving the subsampled array
        ArrIOFuncs.write_tiff(arr, out_fp)
        # Removing temporary memmap
        silent_remove(temp_fp)

    @classmethod
    def coords2heatmap(
        cls,
        coords: pd.DataFrame,
        shape: tuple[int, ...],
        out_fp: str,
        radius: int,
    ):
        """
        Converts list of coordinates to spatial array as voxels.
        Overlapping areas accumulate in intensity.

        Params:
            coords: A pd.DataFrame of points, with the columns, `x`, `y`, and `z`.
            r: radius of the voxels.
            shape: The dimensions of the output array. Assumes that shape is in format `(z, y, x)` (regular for npy and tif file).
            out_fp: The output filename.

        Returns:
            The output image array
        """
        # Initialising spatial array
        temp_fp = os.path.join(CACHE_DIR, f"temp_{os.getpid()}.dat")
        arr = cls.make_mmap(shape, temp_fp)

        # Constructing sphere array mask
        zz, yy, xx = np.ogrid[1 : radius * 2, 1 : radius * 2, 1 : radius * 2]
        circ = np.sqrt((xx - radius) ** 2 + (yy - radius) ** 2 + (zz - radius) ** 2) < radius
        # Constructing offset indices
        i = np.arange(-radius + 1, radius)
        z_ind, y_ind, x_ind = np.meshgrid(i, i, i, indexing="ij")
        # Adding coords to image
        for z, y, x, t in zip(z_ind.ravel(), y_ind.ravel(), x_ind.ravel(), circ.ravel()):
            if t:
                coords_i = coords.copy()
                coords_i[Coords.Z.value] += z
                coords_i[Coords.Y.value] += y
                coords_i[Coords.X.value] += x
                cls.coords2points_workers(arr, coords_i)

        # Saving the subsampled array
        ArrIOFuncs.write_tiff(arr, out_fp)
        # Removing temporary memmap
        silent_remove(temp_fp)

    @classmethod
    def coords2regions(cls, coords, shape, out_fp):
        """
        Converts list of coordinates to spatial array.

        Params:
            coords: A pd.DataFrame of points, with the columns, `x`, `y`, `z`, and `id`.
            shape: The dimensions of the output array. Assumes that shape is in format `(z, y, x)` (regular for npy and tif file).
            out_fp: The output filename.

        Returns:
            The output image array
        """
        # Initialising spatial array
        temp_fp = os.path.join(CACHE_DIR, f"temp_{os.getpid()}.dat")
        arr = cls.make_mmap(shape, temp_fp)

        # Adding coords to image with np.apply_along_axis
        def f(coord):
            # Plotting coord to image. Including only coords within the image's bounds
            if np.all((coord >= 0) & (coord < shape)):
                z, y, x, _id = coord
                arr[z, y, x] = _id

        # Formatting coord values as (z, y, x) and rounding to integers
        coords = (
            coords[[Coords.Z.value, Coords.Y.value, Coords.X.value, AnnotColumns.ID.value]].round(0).astype(np.int16)
        )
        if coords.shape[0] > 0:
            np.apply_along_axis(f, 1, coords)

        # Saving the subsampled array
        ArrIOFuncs.write_tiff(arr, out_fp)
        # Removing temporary memmap
        silent_remove(temp_fp)
