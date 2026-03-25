"""CPU-based cell counting operations with injectable backend.

Provides filtering, thresholding, labeling, and watershed operations
for 3D microscopy images. Backend (numpy/cupy) is injected at runtime,
enabling GPU acceleration without code duplication.
"""

import logging

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import ndimage as sc_ndimage
from skimage.segmentation import watershed

from cellcounter.constants import CELL_IDX_NAME, CellColumns, Coords

logger = logging.getLogger(__name__)


class CpuCellcFuncs:
    """CPU cell counting funcs with injectable backend."""

    def __init__(
        self,
        xp: type[np] = np,
        xdimage: type[sc_ndimage] = sc_ndimage,
    ) -> None:
        """Initialize CpuCellcFuncs.

        Args:
            xp: Array backend (numpy for CPU, cupy for GPU).
            xdimage: ndimage backend
                (scipy.ndimage for CPU, cupyx.scipy.ndimage for GPU).
        """
        self.xp = xp
        self.xdimage = xdimage

    #############################################
    # Intended to be used as entire np array
    #############################################

    def downsample(
        self, arr: npt.NDArray, z_scale: float, y_scale: float, x_scale: float
    ) -> npt.NDArray:
        """Downsample with float scale factors using zoom.

        GPU-accelerated via cupyx.scipy.ndimage.zoom when in GPU mode.

        Args:
            arr: Input array (numpy or cupy).
            z_scale: Float scale factors.
            y_scale: Float scale factors.
            x_scale: Float scale factors.

        Returns:
            Downsampled array.
        """
        arr = self.xp.asarray(arr)
        return self.xdimage.zoom(arr, (z_scale, y_scale, x_scale))

    def reorient(
        self, arr: npt.NDArray, orient_ls: list[int] | tuple[int, ...]
    ) -> npt.NDArray:
        """Reorder and flip axes.

        Order of orient_ls is the axis order.
        Negative element means that axis is flipped.
        Axis order starts from 1, 2, 3, ...

        Example:
            `orient_ls=(-2, 1, 3)` flips the second axis
            and swaps the first and second axes.

        Args:
            arr: Input array (numpy or cupy).
            orient_ls: Axis ordering with optional flips (negative values).

        Returns:
            Reoriented array.
        """
        arr = self.xp.asarray(arr)
        orient_ls = list(orient_ls)
        for i in range(len(orient_ls)):
            ax = orient_ls[i]
            ax_new = abs(ax) - 1
            orient_ls[i] = ax_new
            if ax < 0:
                arr = self.xp.flip(arr, ax_new)
        return arr.transpose(orient_ls)

    #############################################
    # Intended to be used with dask.array.map_blocks
    #############################################

    def _spherical_footprint(self, radius: int) -> npt.NDArray:
        """Create spherical structuring element.

        Args:
            radius: Radius of sphere.

        Returns:
            Boolean array True within sphere of given radius.
        """
        coords = self.xp.ogrid[
            -radius : radius + 1, -radius : radius + 1, -radius : radius + 1
        ]
        return coords[0] ** 2 + coords[1] ** 2 + coords[2] ** 2 <= radius**2

    def tophat_filt(self, block: npt.NDArray, radius: int = 10) -> npt.NDArray:
        """Calculate white top-hat filter with spherical structuring element.

        White top-hat removes background by subtracting the morphological
        opening (erosion followed by dilation) from the original image.

        Args:
            block: Input array.
            radius: Radius of spherical structuring element.

        Returns:
            Background-subtracted array (original - opening).
        """
        block = self.xp.asarray(block).astype(self.xp.float32)
        logger.debug("Perform white top-hat filter with spherical element")
        footprint = self._spherical_footprint(radius)
        res_block = self.xdimage.white_tophat(block, footprint=footprint)
        logger.debug("ReLu")
        res_block = self.xp.maximum(res_block, 0)
        return res_block.astype(self.xp.uint16)

    def dog_filt(
        self, block: npt.NDArray, sigma1: int = 1, sigma2: int = 2
    ) -> npt.NDArray:
        block = self.xp.asarray(block).astype(self.xp.float32)
        logger.debug("Making gaussian blur 1")
        gaus1 = self.xdimage.gaussian_filter(block, sigma=sigma1)
        logger.debug("Making gaussian blur 2")
        gaus2 = self.xdimage.gaussian_filter(block, sigma=sigma2)
        logger.debug("Subtracting gaussian blurs")
        res_block = gaus1 - gaus2
        logger.debug("ReLu")
        res_block = self.xp.maximum(res_block, 0)
        return res_block.astype(self.xp.uint16)

    def gauss_blur_filt(self, block: npt.NDArray, sigma: int = 10) -> npt.NDArray:
        block = self.xp.asarray(block).astype(self.xp.float32)
        logger.debug("Calculate Gaussian blur")
        res_block = self.xdimage.gaussian_filter(block, sigma=sigma)
        return res_block.astype(self.xp.uint16)

    def gauss_subt_filt(self, block: npt.NDArray, sigma: int = 10) -> npt.NDArray:
        block = self.xp.asarray(block).astype(self.xp.float32)
        logger.debug("Calculate local Gaussian blur")
        gaus = self.xdimage.gaussian_filter(block, sigma=sigma)
        logger.debug("Apply the adaptive filter")
        res_block = block - gaus
        logger.debug("ReLu")
        res_block = self.xp.maximum(res_block, 0)
        return res_block.astype(self.xp.uint16)

    def intensity_cutoff(
        self, block: npt.NDArray, min_: None | float = None, max_: None | float = None
    ) -> npt.NDArray:
        """Performing cutoffs on a 3D tensor."""
        block = self.xp.asarray(block)
        logger.debug("Making cutoffs")
        res_block = block
        if min_ is not None:
            res_block = self.xp.maximum(res_block, min_)
        if max_ is not None:
            res_block = self.xp.minimum(res_block, max_)
        return res_block

    def otsu_thresh(self, block: npt.NDArray) -> npt.NDArray:
        """Perform Otsu's thresholding on a 3D tensor."""
        block = self.xp.asarray(block)
        logger.debug("Calculate histogram")
        hist, _bin_edges = self.xp.histogram(block, bins=256)
        logger.debug("Normalize histogram")
        prob_hist = hist / hist.sum()
        logger.debug("Compute cumulative sum and cumulative mean")
        cum_sum = self.xp.cumsum(prob_hist)
        cum_mean = self.xp.cumsum(prob_hist * self.xp.arange(256))
        logger.debug("Compute global mean")
        global_mean = cum_mean[-1]
        logger.debug(
            "Compute between class variance for all thresholds "
            "and find the threshold that maximizes it"
        )
        numerator = (global_mean * cum_sum - cum_mean) ** 2
        denominator = cum_sum * (1.0 - cum_sum)
        logger.debug("Avoid division by zero")
        denominator = self.xp.where(denominator == 0, float("inf"), denominator)
        between_class_variance = numerator / denominator
        logger.debug("Find the threshold that maximizes the between class variance")
        optimal_threshold = self.xp.argmax(between_class_variance)
        logger.debug("Apply threshold")
        res_block = block > optimal_threshold
        return res_block.astype(self.xp.uint8)

    def mean_thresh(self, block: npt.NDArray, offset_sd: float = 0.0) -> npt.NDArray:
        """Perform adaptive thresholding on a 3D tensor."""
        block = self.xp.asarray(block)
        logger.debug("Get mean and std of ONLY non-zero values")
        arr0 = block[block > 0]
        mu = arr0.mean()
        sd = arr0.std()
        logger.debug("Apply the threshold")
        res_block = block > mu + offset_sd * sd
        return res_block.astype(self.xp.uint8)

    def manual_thresh(self, block: npt.NDArray, val: int) -> npt.NDArray:
        """Perform manual thresholding on a tensor."""
        block = self.xp.asarray(block)
        logger.debug("Applying the threshold")
        res_block = block >= val
        return res_block.astype(self.xp.uint8)

    def mask2label(
        self,
        block: npt.NDArray,
        block_info: dict | None = None,
        max_labels_per_chunk: int | None = None,
    ) -> npt.NDArray:
        """Convert array of mask (usually binary) to contiguous label values.

        If block_info and max_labels_per_chunk are provided, adds a globally
        unique offset to labels based on chunk location. This ensures labels
        from different chunks don't collide when processing arrays in parallel.

        Parameters
        ----------
        arr : npt.NDArray
            Input mask array (usually binary)
        block_info : dict, optional
            Block info from dask map_blocks containing chunk-location and num-chunks
        max_labels_per_chunk : int, optional
            Maximum possible labels per chunk, used to compute unique offsets

        Returns:
        --------
        npt.NDArray
            Labeled array with uint32 dtype (or int64 if using global offsets)
        """
        block = self.xp.asarray(block).astype(self.xp.uint8)
        logger.debug("Labelling contiguous objects uniquely")
        res_block, _ = self.xdimage.label(block)
        res_block = res_block.astype(self.xp.uint64)
        # Add globally unique offset if parameters provided
        if (
            block_info is not None
            and block_info[0]
            and max_labels_per_chunk is not None
        ):
            loc = self.xp.asarray(block_info[0]["chunk-location"])
            grid_shape = block_info[0]["num-chunks"]
            flat_idx = self.xp.ravel_multi_index(loc, grid_shape)
            offset = flat_idx * max_labels_per_chunk
            block[block > 0] += offset.astype(block.dtype)
            logger.debug("Applied label offset: %s", offset)
        return res_block

    def get_boundary_pairs(self, block: npt.NDArray, depth: int = 1) -> npt.NDArray:
        """Find adjacent label pairs at chunk boundaries.

        Only checks the halo-interior interfaces, not the entire chunk interior.
        This is valid because labeling already merged all intra-chunk adjacencies,
        so different labels can only meet at chunk boundaries.

        Parameters:
        -----------
        block : npt.NDArray
            Overlabeled array chunk with halo of depth=1
        depth : int
            Overlap depth (default 1)

        Returns:
        --------
        npt.NDArray
            Array of shape (N, 2) with adjacent label pairs
        """
        block = self.xp.asarray(block)
        pairs = set()
        for axis in range(block.ndim):
            # Check both faces of this axis
            for face in [(depth - 1, depth), (-depth - 1, -depth)]:
                idx_a, idx_b = face
                sl_a = [slice(None)] * block.ndim
                sl_b = [slice(None)] * block.ndim
                sl_a[axis] = idx_a
                sl_b[axis] = idx_b
                a = block[tuple(sl_a)].ravel()
                b = block[tuple(sl_b)].ravel()
                mask = (a > 0) & (b > 0) & (a != b)
                if mask.any():
                    lo = self.xp.minimum(a[mask], b[mask])
                    hi = self.xp.maximum(a[mask], b[mask])
                    pairs.update(zip(lo.tolist(), hi.tolist(), strict=True))
        if pairs:
            # Each row is [label_a, label_b]
            return np.array(list(pairs))
        return np.empty((0, 2))

    def get_label_sizemap(self, block: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray]:
        """Get a dict of label_val : contiguous_size.

        Parameters:
        -----------
        block : npt.NDArray
            Overlabeled array chunk with halo of depth=1

        Returns:
        --------
        npt.NDArray
            Array of shape (N, 2) with ids and counts pairs
        """
        block = self.xp.asarray(block)
        logger.debug("Getting vector of ids and volumes (not incl. 0)")
        mask = block > 0
        labels_fg = block[mask]
        ids, counts = self.xp.unique(labels_fg, return_counts=True)
        # Each row is [id_i, count_i]
        return self.xp.column_stack((ids, counts))

    def map_values_to_arr(
        self,
        block: npt.NDArray,
        ids: npt.NDArray,
        values: npt.NDArray,
    ) -> npt.NDArray:
        """Map label values to their component values.

        Parameters:
        -----------
        block : npt.NDArray
            Label array chunk
        ids : npt.NDArray
            Sorted array of label IDs for lookup
        values : npt.NDArray
            Corresponding component values for each label

        Returns:
        --------
        npt.NDArray
            Array where each voxel contains its component size (0 for background)
        """
        block = self.xp.asarray(block)
        ids = self.xp.asarray(ids)
        values = self.xp.asarray(values)
        res_block = self.xp.zeros(block.shape, dtype=self.xp.uint64)
        mask = block > 0
        if not mask.any():
            return res_block
        labels_vect = block[mask]
        # Find each label's position in the sorted lookup table
        pos = self.xp.searchsorted(ids, labels_vect)
        pos = self.xp.clip(pos, 0, len(ids) - 1)
        valid = ids[pos] == labels_vect  # guard against missing keys
        res_block[mask] = self.xp.where(valid, values[pos], 0)
        return res_block

    def label2volume(self, block: npt.NDArray) -> npt.NDArray:
        """Convert array of label values to contiguous volume (i.e. count) values."""
        ids, counts = self.get_label_sizemap(block)
        res_block = self.map_values_to_arr(block, ids, counts)
        return res_block

    def volume_filter(
        self, block: npt.NDArray, smin: int | None = None, smax: int | None = None
    ) -> npt.NDArray:
        """Assumes `arr` is array of objects labelled with their volumes."""
        block = self.xp.asarray(block)
        logger.debug("Getting filter of small and large object to filter out")
        smin = smin or 0
        smax = smax or block.max()
        filt_objs = (block < smin) | (block > smax)
        logger.debug("Filter out objects (by setting them to 0)")
        block[filt_objs] = 0
        return block

    def get_local_maxima(
        self,
        block: npt.NDArray,
        mask_block: npt.NDArray | None = None,
        radius: int = 5,
    ) -> npt.NDArray:
        """Find local maxima using spherical neighborhood.

        A point is a local maximum if it equals the maximum within a sphere
        of given radius. Note: plateau regions (flat maxima) return all points
        in the plateau, not just the center.

        Args:
            block: Input 3D array.
            radius: Radius of spherical structuring element.
            mask_block: Optional mask; maxima only kept where mask > 0.

        Returns:
            Binary array where True indicates local maxima.
        """
        block = self.xp.asarray(block)
        logger.debug("Finding local maxima with spherical radius %d", radius)
        footprint = self._spherical_footprint(radius)
        max_arr = self.xdimage.maximum_filter(block, footprint=footprint)
        logger.debug("Identifying points where arr == max_arr")
        res_block = block == max_arr
        if mask_block is not None:
            logger.debug("Mask provided. Maxima only in mask regions considered.")
            mask_block = (self.xp.asarray(mask_block) > 0).astype(self.xp.uint8)
            res_block = (res_block * mask_block).astype(self.xp.uint8)
        return res_block

    def mask(self, block: npt.NDArray, mask_block: npt.NDArray) -> npt.NDArray:
        block = self.xp.asarray(block)
        mask_block = self.xp.asarray(mask_block).astype(self.xp.uint8)
        logger.debug("Masking for only maxima within mask")
        res_block = block * (mask_block > 0)
        return res_block

    def wshed_segm(
        self, raw_block: npt.NDArray, maxima_block: npt.NDArray, mask_block: npt.NDArray
    ) -> npt.NDArray:
        """Do watershed segmentation.

        NOTE: NOT GPU accelerated

        Expects `maxima_arr` to have unique labels for each maxima.
        """
        logger.debug("Watershed segmentation")
        res_block = watershed(
            image=-raw_block,
            markers=maxima_block,
            mask=mask_block > 0,
        )
        return res_block

    def get_coords(self, block: npt.NDArray) -> pd.DataFrame:
        """Get coordinates of regions in 3D tensor.

        TODO: Keep only the first row (i.e cell) for each label (groupby).
        """
        logger.debug("Getting coordinates of regions")
        z, y, x = np.where(block)
        logger.debug("Getting IDs of regions (from coords)")
        ids = block[z, y, x]
        logger.debug("Making dataframe")
        df = pd.DataFrame(
            {
                Coords.Z.value: z,
                Coords.Y.value: y,
                Coords.X.value: x,
            },
            index=pd.Index(ids.astype(np.uint32), name=CELL_IDX_NAME),
        ).astype(np.uint16)
        df[CellColumns.VOLUME.value] = -1  # TODO: placeholder
        df[CellColumns.SUM_INTENSITY.value] = -1  # TODO: placeholder
        # df[CellColumns.MAX_INTENSITY.value] = -1  # TODO: placeholder
        return df
