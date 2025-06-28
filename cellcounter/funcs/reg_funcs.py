import numpy as np
from scipy import ndimage


class RegFuncs:
    @classmethod
    def downsmpl_rough(cls, arr: np.ndarray, z_scale: int, y_scale: int, x_scale: int) -> np.ndarray:
        """
        Expects scales to be ints
        """
        res = arr[::z_scale, ::y_scale, ::x_scale]
        return res

    @classmethod
    def downsmpl_fine(cls, arr: np.ndarray, z_scale: float, y_scale: float, x_scale: float) -> np.ndarray:
        """
        Expects scales to be floats
        """
        res = ndimage.zoom(arr, (z_scale, y_scale, x_scale))
        return res

    @classmethod
    def reorient(cls, arr: np.ndarray, orient_ls: list[int] | tuple[int, ...]):
        """
        Order of orient_ls is the axis order.
        Negative of an element in orient_ls means that axis is flipped
        Axis order starts from 1, 2, 3, ...

        Example:
            `orient_ls=(-2, 1, 3)` flips the second axis and swaps the first and second axes.
        """
        orient_ls = list(orient_ls)
        # Flipping axes and formatting orient_ls for transposing
        for i in np.arange(len(orient_ls)):
            # ax is the given axes dimension
            ax = orient_ls[i]
            # ax_new is the formatted dimension for transposing (positive and starts from 0)
            ax_new = np.abs(ax) - 1
            orient_ls[i] = ax_new
            # If the given axes dimension is negative, then flip the axis
            if ax < 0:
                arr = np.flip(arr, ax_new)
        # Reordering axes
        arr = arr.transpose(orient_ls)
        return arr
