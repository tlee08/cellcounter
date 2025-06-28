import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# from prefect import task
from scipy import ndimage as sc_ndimage
from skimage.segmentation import watershed

from cellcounter.constants import CELL_IDX_NAME, DEPTH, CellColumns, Coords
from cellcounter.utils.logging_utils import init_logger_file


class CpuCellcFuncs:
    xp = np
    xdimage = sc_ndimage

    logger = init_logger_file(__name__)

    @classmethod
    def tophat_filt(cls, arr: np.ndarray, sigma: int = 10) -> np.ndarray:
        """
        Top hat is calculated as:

        ```
        res = arr - max_filter(min_filter(arr, sigma), sigma)
        ```
        """
        arr = cls.xp.asarray(arr).astype(cls.xp.float32)
        cls.logger.debug("Perform white top-hat filter")
        res = cls.xdimage.white_tophat(arr, sigma)
        cls.logger.debug("ReLu")
        res = cls.xp.maximum(res, 0)  # type: ignore
        return res.astype(cls.xp.uint16)

    @classmethod
    def dog_filt(cls, arr: np.ndarray, sigma1=1, sigma2=2) -> np.ndarray:
        arr = cls.xp.asarray(arr).astype(cls.xp.float32)
        cls.logger.debug("Making gaussian blur 1")
        gaus1 = cls.xdimage.gaussian_filter(arr, sigma=sigma1)
        cls.logger.debug("Making gaussian blur 2")
        gaus2 = cls.xdimage.gaussian_filter(arr, sigma=sigma2)
        cls.logger.debug("Subtracting gaussian blurs")
        res = gaus1 - gaus2
        cls.logger.debug("ReLu")
        res = cls.xp.maximum(res, 0)
        return res.astype(cls.xp.uint16)

    @classmethod
    def gauss_blur_filt(cls, arr: np.ndarray, sigma=10) -> np.ndarray:
        arr = cls.xp.asarray(arr).astype(cls.xp.float32)
        cls.logger.debug("Calculate Gaussian blur")
        res = cls.xdimage.gaussian_filter(arr, sigma=sigma)
        return res.astype(cls.xp.uint16)

    @classmethod
    def gauss_subt_filt(cls, arr: np.ndarray, sigma=10) -> np.ndarray:
        arr = cls.xp.asarray(arr).astype(cls.xp.float32)
        cls.logger.debug("Calculate local Gaussian blur")
        gaus = cls.xdimage.gaussian_filter(arr, sigma=sigma)
        cls.logger.debug("Apply the adaptive filter")
        res = arr - gaus
        cls.logger.debug("ReLu")
        res = cls.xp.maximum(res, 0)
        return res.astype(cls.xp.uint16)

    @classmethod
    def intensity_cutoff(cls, arr: np.ndarray, min_: None | float = None, max_: None | float = None) -> np.ndarray:
        """
        Performing cutoffs on a 3D tensor.
        """
        arr = cls.xp.asarray(arr)
        cls.logger.debug("Making cutoffs")
        res = arr
        if min_ is not None:
            res = cls.xp.maximum(res, min_)
        if max_ is not None:
            res = cls.xp.minimum(res, max_)
        return res

    @classmethod
    def otsu_thresh(cls, arr: np.ndarray) -> np.ndarray:
        """
        Perform Otsu's thresholding on a 3D tensor.
        """
        arr = cls.xp.asarray(arr)
        cls.logger.debug("Calculate histogram")
        hist, bin_edges = cls.xp.histogram(arr, bins=256)
        cls.logger.debug("Normalize histogram")
        prob_hist = hist / hist.sum()
        cls.logger.debug("Compute cumulative sum and cumulative mean")
        cum_sum = cls.xp.cumsum(prob_hist)
        cum_mean = cls.xp.cumsum(prob_hist * cls.xp.arange(256))
        cls.logger.debug("Compute global mean")
        global_mean = cum_mean[-1]
        cls.logger.debug("Compute between class variance for all thresholds and find the threshold that maximizes it")
        numerator = (global_mean * cum_sum - cum_mean) ** 2
        denominator = cum_sum * (1.0 - cum_sum)
        cls.logger.debug("Avoid division by zero")
        denominator = cls.xp.where(denominator == 0, float("inf"), denominator)
        between_class_variance = numerator / denominator
        cls.logger.debug("Find the threshold that maximizes the between class variance")
        optimal_threshold = cls.xp.argmax(between_class_variance)
        cls.logger.debug("Apply threshold")
        res = arr > optimal_threshold
        return res.astype(cls.xp.uint8)

    @classmethod
    def mean_thresh(cls, arr: np.ndarray, offset_sd: float = 0.0) -> np.ndarray:
        """
        Perform adaptive thresholding on a 3D tensor on GPU.
        """
        arr = cls.xp.asarray(arr)
        cls.logger.debug("Get mean and std of ONLY non-zero values")
        arr0 = arr[arr > 0]
        mu = arr0.mean()
        sd = arr0.std()
        cls.logger.debug("Apply the threshold")
        res = arr > mu + offset_sd * sd
        return res.astype(cls.xp.uint8)

    @classmethod
    def manual_thresh(cls, arr: np.ndarray, val: int) -> np.ndarray:
        """
        Perform manual thresholding on a tensor.
        """
        arr = cls.xp.asarray(arr)
        cls.logger.debug("Applying the threshold")
        res = arr >= val
        return res.astype(cls.xp.uint8)

    @classmethod
    def mask2label(cls, arr: np.ndarray) -> np.ndarray:
        """
        Convert array of mask (usually binary) to contiguous label values.
        """
        arr = cls.xp.asarray(arr).astype(cls.xp.uint8)
        cls.logger.debug("Labelling contiguous objects uniquely")
        res, _ = cls.xdimage.label(arr)  # type: ignore
        cls.logger.debug("Returning")
        return res.astype(cls.xp.uint32)

    @classmethod
    def label2volume(cls, arr: np.ndarray) -> np.ndarray:
        """
        Convert array of label values to contiguous volume (i.e. count) values.
        """
        arr = cls.xp.asarray(arr)
        cls.logger.debug("Getting vector of ids and volumes (not incl. 0)")
        ids, counts = cls.xp.unique(arr[arr > 0], return_counts=True)
        # NOTE: assumes ids are incrementing from 1
        counts = cls.xp.concatenate([cls.xp.asarray([0]), counts])
        cls.logger.debug("Converting arr intensity to volumes")
        res = counts[arr]
        cls.logger.debug("Returning")
        return res.astype(cls.xp.uint16)

    @classmethod
    def mask2volume(cls, arr: np.ndarray) -> np.ndarray:
        """
        Convert array of mask (usually binary) to contiguous volume (i.e. count) values.
        """
        arr = cls.mask2label(arr)
        res = cls.label2volume(arr)
        return res

    @classmethod
    def visualise_stats(cls, arr: np.ndarray):
        """
        Visualise statistics.

        NOTE: expects arr to be a 3D tensor of a property
        (e.g. volume).
        """
        cls.logger.debug("Converting arr to vector of the ids")
        ids = arr[arr > 0]
        cls.logger.debug("Making histogram")
        fig, ax = plt.subplots()
        sns.histplot(
            x=ids,
            log_scale=True,
            ax=ax,
        )
        return fig

    @classmethod
    def volume_filter(cls, arr: np.ndarray, smin=None, smax=None) -> np.ndarray:
        """
        Assumes `arr` is array of objects labelled with their volumes.
        """
        arr = cls.xp.asarray(arr)
        cls.logger.debug("Getting filter of small and large object to filter out")
        smin = smin if smin else 0
        smax = smax if smax else arr.max()
        filt_objs = (arr < smin) | (arr > smax)
        cls.logger.debug("Filter out objects (by setting them to 0)")
        arr[filt_objs] = 0
        return arr

    @classmethod
    def get_local_maxima(
        cls,
        arr: np.ndarray,
        sigma: int = 10,
        mask_arr: None | np.ndarray = None,
    ) -> np.ndarray:
        """
        Getting local maxima (no connectivity) in a 3D tensor.
        If there is a connected region of maxima, then only the centre point is kept.

        If `mask_arr` is provided, then only maxima within the mask are kept.
        """
        arr = cls.xp.asarray(arr)
        cls.logger.debug("Making max filter for raw arr (maximum in given area)")
        max_arr = cls.xdimage.maximum_filter(arr, sigma)
        cls.logger.debug("Getting local maxima (where arr == max_arr)")
        res = arr == max_arr
        # If a mask is given, then keep only the maxima within the mask
        if mask_arr is not None:
            cls.logger.debug("Mask provided. Maxima only in mask regions considered.")
            mask_arr = (cls.xp.asarray(mask_arr) > 0).astype(cls.xp.uint8)
            res = res * mask_arr
        return res

    @classmethod
    def mask(cls, arr: np.ndarray, mask_arr: np.ndarray) -> np.ndarray:
        arr = cls.xp.asarray(arr)
        mask_arr = cls.xp.asarray(mask_arr).astype(cls.xp.uint8)
        cls.logger.debug("Masking for only maxima within mask")
        res = arr * (mask_arr > 0)
        return res

    @classmethod
    def wshed_segm(cls, raw_arr: np.ndarray, maxima_arr: np.ndarray, mask_arr: np.ndarray) -> np.ndarray:
        """
        NOTE: NOT GPU accelerated

        Expects `maxima_arr` to have unique labels for each maxima.
        """
        cls.logger.debug("Watershed segmentation")
        res = watershed(
            image=-raw_arr,
            markers=maxima_arr,
            mask=mask_arr > 0,
        )
        return res

    @classmethod
    def wshed_segm_volumes(cls, raw_arr: np.ndarray, maxima_arr: np.ndarray, mask_arr: np.ndarray) -> np.ndarray:
        """
        NOTE: NOT GPU accelerated
        """
        # Labelling contiguous maxima with unique labels
        maxima_arr = cls.mask2label(maxima_arr)
        # Watershed segmentation
        wshed_arr = cls.wshed_segm(raw_arr, maxima_arr, mask_arr)
        # Getting volumes of watershed regions
        res = cls.label2volume(wshed_arr)
        return res

    @classmethod
    def get_coords(cls, arr: np.ndarray) -> pd.DataFrame:
        """
        Get coordinates of regions in 3D tensor.

        TODO: Keep only the first row (i.e cell) for each label (groupby).
        """
        cls.logger.debug("Getting coordinates of regions")
        z, y, x = np.where(arr)
        cls.logger.debug("Getting IDs of regions (from coords)")
        ids = arr[z, y, x]
        cls.logger.debug("Making dataframe")
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
    def get_cells_old(
        cls,
        raw_arr: np.ndarray,
        overlap_arr: np.ndarray,
        maxima_arr: np.ndarray,
        mask_arr: np.ndarray,
        depth: int = DEPTH,
    ) -> pd.DataFrame:
        """
        Get the cells from the maxima labels and the watershed segmentation
        (with corresponding labels).
        """
        # Asserting arr sizes match between arr_raw, arr_overlap, and depth
        # NOTE: we NEED raw_arr as the first da.Array to get chunking coord offsets correct
        assert raw_arr.shape == tuple(i - 2 * depth for i in overlap_arr.shape)
        cls.logger.debug("Trimming maxima labels array to raw array dimensions using `d`")
        slicer = slice(depth, -depth) if depth > 0 else slice(None)
        maxima_arr = maxima_arr[slicer, slicer, slicer]
        assert raw_arr.shape == maxima_arr.shape
        cls.logger.debug("Getting unique labels in maxima_arr")
        maxima_l_arr = cls.mask2label(maxima_arr)
        cls.logger.debug("Converting to DataFrame of coordinates and measures")
        # NOTE: getting first coord of each unique label
        # NOTE: np.unique auto flattens arr so reshaping it back with np.unravel_index
        ids_m, ind = cls.xp.unique(maxima_l_arr, return_index=True)
        z, y, x = cls.xp.unravel_index(ind, maxima_l_arr.shape)
        df = (
            pd.DataFrame(
                {
                    Coords.Z.value: cls.cp2np(z),
                    Coords.Y.value: cls.cp2np(y),
                    Coords.X.value: cls.cp2np(x),
                },
                index=pd.Index(cls.cp2np(ids_m).astype(np.uint32), name=CELL_IDX_NAME),
            )
            .drop(index=0)  # Not including the 0 valued row (because there's no cell here)
            .astype(np.uint16)
        )
        cls.logger.debug("Watershed of overlap_arr, seeds maxima_arr, mask mask_arr")
        # NOTE: padding maxima_l_arr because we previously trimmed maxima_arr
        maxima_l_arr = np.pad(cls.cp2np(maxima_l_arr), depth, mode="constant", constant_values=0)
        wshed_arr = cls.wshed_segm(overlap_arr, maxima_l_arr, mask_arr)
        cls.logger.debug("Making vector of watershed region volumes")
        ids_w, counts = cls.xp.unique(wshed_arr[wshed_arr > 0], return_counts=True)
        ids_w = cls.cp2np(ids_w).astype(np.uint32)
        counts = cls.cp2np(counts).astype(np.uint32)
        cls.logger.debug("Getting sum intensity for each cell (wshed)")
        # With bincount, positional arg is the label cat and weights sums is raw arr (helpful for intensity)
        sum_intensity = cls.xp.bincount(
            cls.xp.asarray(wshed_arr[wshed_arr > 0].ravel()),
            weights=cls.xp.asarray(overlap_arr[wshed_arr > 0].ravel()),
            minlength=len(ids_w),
        )
        # NOTE: excluding 0 valued elements means sum_intensity matches with ids_w
        sum_intensity = cls.cp2np(sum_intensity[sum_intensity > 0])
        cls.logger.debug("Adding cell measures to DataFrame")
        idx = pd.Index(ids_w, name=CELL_IDX_NAME)
        df[CellColumns.COUNT.value] = 1
        df[CellColumns.VOLUME.value] = pd.Series(counts, index=idx)
        df[CellColumns.SUM_INTENSITY.value] = pd.Series(sum_intensity, index=idx)
        # df["max_intensity"] = pd.Series(max_intensity, index=idx)
        # Filtering out rows with NaNs in z, y, or x columns (i.e. no na values)
        df = df[df[[Coords.Z.value, Coords.Y.value, Coords.X.value]].isna().sum(axis=1) == 0]
        return df

    @classmethod
    def get_cells(
        cls,
        raw_arr: np.ndarray,
        overlap_arr: np.ndarray,
        maxima_labels_arr: np.ndarray,
        wshed_labels_arr: np.ndarray,
        wshed_filt_arr: np.ndarray,
        depth: int = DEPTH,
    ) -> pd.DataFrame:
        """
        Get the cells from the maxima labels and the watershed segmentation
        (with corresponding labels).
        """
        # NOTE: we NEED raw_arr as the first da.Array to get chunking coord offsets correct
        # Asserting arr sizes match between arr_raw, arr_overlap, and depth
        assert raw_arr.shape == tuple(i - 2 * depth for i in overlap_arr.shape)
        assert overlap_arr.shape == maxima_labels_arr.shape
        assert overlap_arr.shape == wshed_filt_arr.shape
        # Converting to xp arrays
        overlap_arr = cls.xp.asarray(overlap_arr)
        maxima_labels_arr = cls.xp.asarray(maxima_labels_arr)
        wshed_labels_arr = cls.xp.asarray(wshed_labels_arr)
        wshed_filt_arr = cls.xp.asarray(wshed_filt_arr)
        # Trimming maxima labels array to raw array dimensions using `depth`
        slicer = slice(depth, -depth) if depth > 0 else slice(None)
        maxima_labels_trimmed_arr = maxima_labels_arr[slicer, slicer, slicer]
        assert raw_arr.shape == maxima_labels_trimmed_arr.shape
        # Getting first coord of each unique label in trimmed arr (as some maxima are contiguous)
        # NOTE: np.unique auto flattens arr so reshaping it back with np.unravel_index
        labels_vect, coords_flat = cls.xp.unique(maxima_labels_trimmed_arr, return_index=True)
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
                index=pd.Index(cls.cp2np(labels_vect).astype(np.uint32), name=CELL_IDX_NAME),
            )
            .drop(index=0)  # Not including the 0 valued row (because it's background)
            .astype(np.uint16)
        )
        cells_df[CellColumns.COUNT.value] = 1
        # Getting wshed_filt_arr (volume) values for each cell (z, y, x). Offsetting by depth.
        cells_df[CellColumns.VOLUME.value] = cls.cp2np(
            wshed_filt_arr[
                cells_df[Coords.Z.value] + depth,
                cells_df[Coords.Y.value] + depth,
                cells_df[Coords.X.value] + depth,
            ]
        )
        # Filtering out cells with 0 volume. These cells were evidently filtered out previously in wshed_filt_arr
        cells_df = cells_df.query(f"{CellColumns.VOLUME.value} > 0")
        # Getting summed intensities for each cell
        # For bincount, positional arg is label cat and weights sums is raw arr (helpful for intensity)
        if cls.xp.any(wshed_filt_arr > 0):
            sum_intensity = cls.xp.bincount(
                wshed_labels_arr[wshed_filt_arr > 0].ravel(),
                weights=overlap_arr[wshed_filt_arr > 0].ravel(),
                minlength=label_max + 1,
            )
            cells_df.loc[:, CellColumns.SUM_INTENSITY.value] = pd.Series(data=cls.cp2np(sum_intensity))
        else:
            cells_df.loc[:, CellColumns.SUM_INTENSITY.value] = pd.Series(dtype=np.float32)
        # A series with the index used to "auto" filter labels that are not in cells_df
        # Getting the unique values of the untrimmed maxima_labels_arr
        # labels_untrimmed_vect = cls.cp2np(cls.xp.unique(maxima_labels_arr))
        # index = pd.Index(labels_untrimmed_vect, name=CELL_IDX_NAME)
        # sum_intensity = pd.Series(data=cls.cp2np(sum_intensity), index=index) if sum_intensity is not None else pd.Series()
        # There should be no na values
        cls.logger.info("end of function check (get_cells)")
        print("end of function check (get_cells)")
        assert np.all(cells_df.notna())
        return cells_df

    @staticmethod
    def cp2np(arr) -> np.ndarray:
        try:
            return arr.get()
        except Exception:
            return arr
