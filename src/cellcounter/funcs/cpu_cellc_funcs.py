import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import ndimage as sc_ndimage
from skimage.segmentation import watershed

from cellcounter.constants import CELL_IDX_NAME, DEPTH, CellColumns, Coords
from cellcounter.utils.logging_utils import init_logger_file

logger = init_logger_file(__name__)


class CpuCellcFuncs:
    xp = np
    xdimage = sc_ndimage

    @classmethod
    def tophat_filt(cls, block: npt.NDArray, sigma: int = 10) -> npt.NDArray:
        """Calculate top hat filter.

        ```
        res = arr - max_filter(min_filter(arr, sigma), sigma)
        ```
        """
        block = cls.xp.asarray(block).astype(cls.xp.float32)
        logger.debug("Perform white top-hat filter")
        res_block = cls.xdimage.white_tophat(block, sigma)
        logger.debug("ReLu")
        res_block = cls.xp.maximum(res_block, 0)
        return res_block.astype(cls.xp.uint16)

    @classmethod
    def dog_filt(
        cls, block: npt.NDArray, sigma1: int = 1, sigma2: int = 2
    ) -> npt.NDArray:
        block = cls.xp.asarray(block).astype(cls.xp.float32)
        logger.debug("Making gaussian blur 1")
        gaus1 = cls.xdimage.gaussian_filter(block, sigma=sigma1)
        logger.debug("Making gaussian blur 2")
        gaus2 = cls.xdimage.gaussian_filter(block, sigma=sigma2)
        logger.debug("Subtracting gaussian blurs")
        res_block = gaus1 - gaus2
        logger.debug("ReLu")
        res_block = cls.xp.maximum(res_block, 0)
        return res_block.astype(cls.xp.uint16)

    @classmethod
    def gauss_blur_filt(cls, block: npt.NDArray, sigma: int = 10) -> npt.NDArray:
        block = cls.xp.asarray(block).astype(cls.xp.float32)
        logger.debug("Calculate Gaussian blur")
        res_block = cls.xdimage.gaussian_filter(block, sigma=sigma)
        return res_block.astype(cls.xp.uint16)

    @classmethod
    def gauss_subt_filt(cls, block: npt.NDArray, sigma: int = 10) -> npt.NDArray:
        block = cls.xp.asarray(block).astype(cls.xp.float32)
        logger.debug("Calculate local Gaussian blur")
        gaus = cls.xdimage.gaussian_filter(block, sigma=sigma)
        logger.debug("Apply the adaptive filter")
        res_block = block - gaus
        logger.debug("ReLu")
        res_block = cls.xp.maximum(res_block, 0)
        return res_block.astype(cls.xp.uint16)

    @classmethod
    def intensity_cutoff(
        cls, block: npt.NDArray, min_: None | float = None, max_: None | float = None
    ) -> npt.NDArray:
        """Performing cutoffs on a 3D tensor."""
        block = cls.xp.asarray(block)
        logger.debug("Making cutoffs")
        res_block = block
        if min_ is not None:
            res_block = cls.xp.maximum(res_block, min_)
        if max_ is not None:
            res_block = cls.xp.minimum(res_block, max_)
        return res_block

    @classmethod
    def otsu_thresh(cls, block: npt.NDArray) -> npt.NDArray:
        """Perform Otsu's thresholding on a 3D tensor."""
        block = cls.xp.asarray(block)
        logger.debug("Calculate histogram")
        hist, bin_edges = cls.xp.histogram(block, bins=256)
        logger.debug("Normalize histogram")
        prob_hist = hist / hist.sum()
        logger.debug("Compute cumulative sum and cumulative mean")
        cum_sum = cls.xp.cumsum(prob_hist)
        cum_mean = cls.xp.cumsum(prob_hist * cls.xp.arange(256))
        logger.debug("Compute global mean")
        global_mean = cum_mean[-1]
        logger.debug(
            "Compute between class variance for all thresholds and find the threshold that maximizes it"
        )
        numerator = (global_mean * cum_sum - cum_mean) ** 2
        denominator = cum_sum * (1.0 - cum_sum)
        logger.debug("Avoid division by zero")
        denominator = cls.xp.where(denominator == 0, float("inf"), denominator)
        between_class_variance = numerator / denominator
        logger.debug("Find the threshold that maximizes the between class variance")
        optimal_threshold = cls.xp.argmax(between_class_variance)
        logger.debug("Apply threshold")
        res_block = block > optimal_threshold
        return res_block.astype(cls.xp.uint8)

    @classmethod
    def mean_thresh(cls, block: npt.NDArray, offset_sd: float = 0.0) -> npt.NDArray:
        """Perform adaptive thresholding on a 3D tensor on GPU."""
        block = cls.xp.asarray(block)
        logger.debug("Get mean and std of ONLY non-zero values")
        arr0 = block[block > 0]
        mu = arr0.mean()
        sd = arr0.std()
        logger.debug("Apply the threshold")
        res_block = block > mu + offset_sd * sd
        return res_block.astype(cls.xp.uint8)

    @classmethod
    def manual_thresh(cls, block: npt.NDArray, val: int) -> npt.NDArray:
        """Perform manual thresholding on a tensor."""
        block = cls.xp.asarray(block)
        logger.debug("Applying the threshold")
        res_block = block >= val
        return res_block.astype(cls.xp.uint8)

    @classmethod
    def offset_labels_by_block(
        cls,
        block: npt.NDArray,
        block_info: dict | None = None,
        max_labels_per_chunk: int | None = None,
    ) -> npt.NDArray:
        """Offset labels by block."""
        block = cls.xp.asarray(block)
        if (
            block_info is not None
            and block_info[0]
            and max_labels_per_chunk is not None
        ):
            loc = cls.xp.asarray(block_info[0]["chunk-location"])
            grid_shape = block_info[0]["num-chunks"]
            flat_idx = cls.xp.ravel_multi_index(loc, grid_shape)
            offset = flat_idx * max_labels_per_chunk
            print(grid_shape)
            print(flat_idx)
            print(offset)
            print(offset.dtype)
            print(block[block > 0])
            print(block[block > 0].dtype)
            block[block > 0] += offset
            logger.debug("Applied label offset: %s", offset)
        return block

    @classmethod
    def mask2label(
        cls,
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
        np.ndarray
            Labeled array with uint32 dtype (or int64 if using global offsets)
        """
        block = cls.xp.asarray(block).astype(cls.xp.uint8)
        logger.debug("Labelling contiguous objects uniquely")
        res_block, _ = cls.xdimage.label(block)
        res_block = res_block.astype(cls.xp.uint64)
        # Add globally unique offset if parameters provided
        res_block = cls.offset_labels_by_block(
            res_block, block_info, max_labels_per_chunk
        )
        print(block.dtype)
        print(res_block.dtype)
        return res_block

    @classmethod
    def get_boundary_pairs(cls, block: npt.NDArray, depth: int = 1) -> npt.NDArray:
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
        np.ndarray
            Array of shape (N, 2) with adjacent label pairs
        """
        block = cls.xp.asarray(block)
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
                    lo = cls.xp.minimum(a[mask], b[mask])
                    hi = cls.xp.maximum(a[mask], b[mask])
                    pairs.update(zip(lo.tolist(), hi.tolist(), strict=True))
        if pairs:
            return cls.cp2np(list(pairs))
        return cls.cp2np(np.empty((0, 2)))

    @classmethod
    def get_label_sizemap(cls, block: npt.NDArray) -> tuple[np.ndarray, np.ndarray]:
        """Get a dict of label_val : contiguous_size."""
        block = cls.xp.asarray(block)
        logger.debug("Getting vector of ids and volumes (not incl. 0)")
        mask = block > 0
        labels_fg = block[mask]
        ids, counts = cls.xp.unique(labels_fg, return_counts=True)
        # Return ids and corresponding counts (as np array)
        return cls.cp2np(ids), cls.cp2np(counts)

    @classmethod
    def map_values_to_arr(
        cls,
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
        np.ndarray
            Array where each voxel contains its component size (0 for background)
        """
        block = cls.xp.asarray(block)
        ids = cls.xp.asarray(ids)
        values = cls.xp.asarray(values)
        res_block = cls.xp.zeros(block.shape, dtype=cls.xp.uint64)
        mask = block > 0
        if not mask.any():
            return res_block
        labels_vect = block[mask]
        # Find each label's position in the sorted lookup table
        pos = cls.xp.searchsorted(ids, labels_vect)
        pos = cls.xp.clip(pos, 0, len(ids) - 1)
        valid = ids[pos] == labels_vect  # guard against missing keys
        res_block[mask] = cls.xp.where(valid, values[pos], 0)
        return res_block

    @classmethod
    def label2volume(cls, block: npt.NDArray) -> npt.NDArray:
        """Convert array of label values to contiguous volume (i.e. count) values."""
        ids, counts = cls.get_label_sizemap(block)
        res_block = cls.map_values_to_arr(block, ids, counts)
        return res_block

    @classmethod
    def volume_filter(
        cls, block: npt.NDArray, smin: int | None = None, smax: int | None = None
    ) -> npt.NDArray:
        """Assumes `arr` is array of objects labelled with their volumes."""
        block = cls.xp.asarray(block)
        logger.debug("Getting filter of small and large object to filter out")
        smin = smin or 0
        smax = smax or block.max()
        filt_objs = (block < smin) | (block > smax)
        logger.debug("Filter out objects (by setting them to 0)")
        block[filt_objs] = 0
        return block

    @classmethod
    def get_local_maxima(
        cls,
        block: npt.NDArray,
        sigma: int = 10,
        mask_block: None | npt.NDArray = None,
        block_info: dict | None = None,
        max_labels_per_chunk: int | None = None,
    ) -> npt.NDArray:
        """Getting local maxima (no connectivity) in a 3D tensor.

        If there is a connected region of maxima, then only the centre point is kept.

        If `mask_arr` is provided, then only maxima within the mask are kept.
        """
        block = cls.xp.asarray(block)
        logger.debug("Making max filter for raw arr (maximum in given area)")
        max_arr = cls.xdimage.maximum_filter(block, sigma)
        logger.debug("Getting local maxima (where arr == max_arr)")
        res_block = block == max_arr
        # If a mask is given, then keep only the maxima within the mask
        if mask_block is not None:
            logger.debug("Mask provided. Maxima only in mask regions considered.")
            mask_block = (cls.xp.asarray(mask_block) > 0).astype(cls.xp.uint8)
            res_block = (res_block * mask_block).astype(cls.xp.uint8)
        # Add globally unique offset if parameters provided
        res_block = cls.offset_labels_by_block(
            res_block, block_info, max_labels_per_chunk
        )
        return res_block

    @classmethod
    def mask(cls, block: npt.NDArray, mask_block: npt.NDArray) -> npt.NDArray:
        block = cls.xp.asarray(block)
        mask_block = cls.xp.asarray(mask_block).astype(cls.xp.uint8)
        logger.debug("Masking for only maxima within mask")
        res_block = block * (mask_block > 0)
        return res_block

    @classmethod
    def wshed_segm(
        cls, raw_block: npt.NDArray, maxima_block: npt.NDArray, mask_block: npt.NDArray
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

    @classmethod
    def wshed_segm_volumes(
        cls, raw_block: npt.NDArray, maxima_block: npt.NDArray, mask_block: npt.NDArray
    ) -> npt.NDArray:
        """Do watershed segmentation with volumes.

        NOTE: NOT GPU accelerated
        """
        # Labelling contiguous maxima with unique labels
        maxima_block = cls.mask2label(maxima_block)
        # Watershed segmentation
        wshed_arr = cls.wshed_segm(raw_block, maxima_block, mask_block)
        # Getting volumes of watershed regions
        res_block = cls.label2volume(wshed_arr)
        return res_block

    @classmethod
    def get_coords(cls, block: npt.NDArray) -> pd.DataFrame:
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

    @classmethod
    def get_cells(
        cls,
        raw_block: npt.NDArray,
        overlap_block: npt.NDArray,
        maxima_labels_block: npt.NDArray,
        wshed_labels_block: npt.NDArray,
        wshed_filt_block: npt.NDArray,
        depth: int = DEPTH,
    ) -> pd.DataFrame:
        """Get the cells from the maxima labels and the watershed segmentation.

        Also get corresponding labels.
        """
        # NOTE: we NEED raw_arr as the first da.Array to get chunking coord offsets correct
        # Asserting arr sizes match between arr_raw, arr_overlap, and depth
        assert raw_block.shape == tuple(i - 2 * depth for i in overlap_block.shape)
        assert overlap_block.shape == maxima_labels_block.shape
        assert overlap_block.shape == wshed_filt_block.shape
        # Converting to xp arrays
        overlap_block = cls.xp.asarray(overlap_block)
        maxima_labels_block = cls.xp.asarray(maxima_labels_block)
        wshed_labels_block = cls.xp.asarray(wshed_labels_block)
        wshed_filt_block = cls.xp.asarray(wshed_filt_block)
        # Trimming maxima labels array to raw array dimensions using `depth`
        slicer = slice(depth, -depth) if depth > 0 else slice(None)
        maxima_labels_trimmed_arr = maxima_labels_block[slicer, slicer, slicer]
        assert raw_block.shape == maxima_labels_trimmed_arr.shape
        # Getting first coord of each unique label in trimmed arr (as some maxima are contiguous)
        # NOTE: np.unique auto flattens arr so reshaping it back with np.unravel_index
        labels_vect, coords_flat = cls.xp.unique(
            maxima_labels_trimmed_arr, return_index=True
        )
        label_max = cls.cp2np(labels_vect.max())
        # Making df of coordinates and measures
        z, y, x = cls.xp.unravel_index(coords_flat, maxima_labels_trimmed_arr.shape)
        cells_df = (
            pd.DataFrame(
                {
                    Coords.Z.value: cls.cp2np(z),
                    Coords.Y.value: cls.cp2np(y),
                    Coords.X.value: cls.cp2np(x),
                },
                index=pd.Index(
                    cls.cp2np(labels_vect).astype(np.uint32), name=CELL_IDX_NAME
                ),
            )
            .drop(index=0)  # Not including the 0 valued row (because it's background)
            .astype(np.uint16)
        )
        cells_df[CellColumns.COUNT.value] = 1
        # Getting wshed_filt_arr (volume) values for each cell (z, y, x). Offsetting by depth.
        cells_df[CellColumns.VOLUME.value] = cls.cp2np(
            wshed_filt_block[
                cells_df[Coords.Z.value] + depth,
                cells_df[Coords.Y.value] + depth,
                cells_df[Coords.X.value] + depth,
            ]
        )
        # Filtering out cells with 0 volume. These cells were evidently filtered out previously in wshed_filt_arr
        cells_df = cells_df.query(f"{CellColumns.VOLUME.value} > 0")
        # Getting summed intensities for each cell
        # For bincount, positional arg is label cat and weights sums is raw arr (helpful for intensity)
        if cls.xp.any(wshed_filt_block > 0) and cells_df.shape[0] > 0:
            sum_intensity = cls.xp.bincount(
                wshed_labels_block[wshed_filt_block > 0].ravel(),
                weights=overlap_block[wshed_filt_block > 0].ravel(),
                minlength=label_max + 1,
            )
            cells_df[CellColumns.SUM_INTENSITY.value] = pd.Series(
                data=cls.cp2np(sum_intensity)
            )
        else:
            cells_df[CellColumns.SUM_INTENSITY.value] = 0.0
        # There should be no na values
        assert np.all(cells_df.notna()), f"{cells_df}\n{cells_df.isna().sum()}"
        return cells_df

    @staticmethod
    def cp2np(arr) -> npt.NDArray:
        """Convert cupy to numpy array."""
        try:
            return arr.get()
        except Exception:
            return arr
