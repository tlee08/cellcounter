"""Main cell counting pipeline orchestrator.

Coordinates the full workflow:
1. Registration: TIFF→Zarr, downsampling, elastix registration
2. Cell Counting: filtering, thresholding, watershed segmentation
3. Mapping: coordinate transformation, region assignment, aggregation

All pipeline methods use @check_overwrite decorator for file safety.
"""

import functools
import re
import shutil
from collections.abc import Callable
from pathlib import Path

import dask
import dask.array as da
import dask.dataframe as dd
import numpy as np
import pandas as pd
import tifffile
from dask.distributed import LocalCluster, SpecCluster
from loguru import logger
from natsort import natsorted

from cellcounter.constants import (
    ANNOTATED_COLUMNS_FINAL,
    CELL_AGG_MAPPINGS,
    CELL_COLUMNS,
    CUPY_ENABLED,
    DASK_CUDA_ENABLED,
    ID,
    IOV,
    SUM_INTENSITY,
    TRFM,
    VOLUME,
    X,
    Y,
    Z,
)
from cellcounter.funcs import (
    CpuCellcFuncs,
    annot_fp2df,
    btiff2zarr,
    combine_nested_regions,
    combine_root,
    df_map_ids,
    get_cells,
    registration,
    silent_remove,
    tiffs2zarr,
    transformation_coords,
    visual_check_funcs_dask,
    visual_check_funcs_tiff,
    write_parquet,
    write_tiff,
)
from cellcounter.funcs.io_funcs import combine_arrs
from cellcounter.models import ProjConfig, ProjFp, RefFp
from cellcounter.utils import UnionFind, cluster_process, disk_cache, trace

# ===========================================
# Helper Funcs and Decorators
# ===========================================


def _check_overwrite(*fp_attrs: str) -> Callable:
    """Decorator to check if output files exist before running a pipeline step.

    Args:
        *fp_attrs: Names of pfm attributes to check for existence.
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(self, *args, overwrite: bool = False, **kwargs) -> object:  # noqa: ANN001
            if not overwrite:
                for attr in fp_attrs:
                    fp = getattr(self.pfm, attr)
                    if fp.exists():
                        logger.warning(
                            "Output file, {}, already exists - "
                            "not overwriting file. "
                            "To overwrite, specify overwrite=True.",
                            fp,
                        )
                        return None
            return func(self, *args, overwrite=overwrite, **kwargs)

        return wrapper

    return decorator


# ===========================================
# Pipeline Class
# ===========================================
class Pipeline:
    """Cell counting pipeline orchestrator.

    Coordinates registration, cell counting, and mapping workflows.
    All pipeline methods use @check_overwrite decorator for file safety.
    """

    cellc_funcs: CpuCellcFuncs
    _gpu_cluster: Callable[..., SpecCluster]
    _tuning: bool
    _pfm: ProjFp

    def __init__(self, proj_dir: Path | str, *, tuning: bool = False) -> None:
        """Initialize pipeline with project directory.

        Args:
            proj_dir: Path to project directory.
            tuning: If True, use tuning subdirectory for parameters.
        """
        self._tuning = tuning
        self._pfm = ProjFp(proj_dir, tuning=tuning)
        self._set_gpu(enabled=True)

    def get_log_context(self) -> dict:
        """Log context for loguru context."""
        return {"experiment": str(self.pfm.root_dir)}

    @property
    def pfm(self) -> ProjFp:
        """Project filepath model."""
        return self._pfm

    @property
    def tuning(self) -> bool:
        """Is project in tuning version or raw version."""
        return self._tuning

    @property
    def config(self) -> ProjConfig:
        """Project configuration."""
        return ProjConfig.read_yaml(self._pfm.config_fp)

    @staticmethod
    def _cluster(n_workers: int, threads_per_worker: int) -> SpecCluster:
        return LocalCluster(n_workers=n_workers, threads_per_worker=threads_per_worker)

    def heavy_cluster(self) -> SpecCluster:
        """Create cluster with few workers and high memory per worker."""
        return self._cluster(
            self.config.cluster.heavy_n_workers,
            self.config.cluster.heavy_threads_per_worker,
        )

    def busy_cluster(self) -> SpecCluster:
        """Create cluster with many workers and low memory per worker."""
        return self._cluster(
            self.config.cluster.busy_n_workers,
            self.config.cluster.busy_threads_per_worker,
        )

    def gpu_cluster(self) -> SpecCluster:
        """Create GPU-enabled cluster for CUDA operations."""
        return self._gpu_cluster()

    def _set_gpu(self, *, enabled: bool = True) -> None:
        """Switch between GPU and CPU mode at runtime."""
        if enabled:
            if DASK_CUDA_ENABLED:
                from dask_cuda import LocalCUDACluster  # noqa: PLC0415

                self._gpu_cluster = LocalCUDACluster
            else:
                self._gpu_cluster = lambda: LocalCluster(
                    n_workers=1, threads_per_worker=1
                )
            if CUPY_ENABLED:
                from cellcounter.funcs.gpu_cellc_funcs import GpuCellcFuncs  # noqa: PLC0415, I001

                self.cellc_funcs = GpuCellcFuncs()
            else:
                self.cellc_funcs = CpuCellcFuncs()
        else:
            self._gpu_cluster = lambda: LocalCluster(n_workers=2, threads_per_worker=1)
            self.cellc_funcs = CpuCellcFuncs()

    # ===========================================
    # HELPER: GENERIC ZARR PROCESSING STEP
    # ===========================================

    def _spatial_connect_count(self, label_arr: da.array) -> da.array:
        """Connect contiguous foreground components across chunk boundaries.

        Uses Union-Find to merge labels that span chunk boundaries,
        then computes total volume for each connected component.

        Args:
            label_arr: Dask array of labels.

        Returns:
            Dask array where each voxel contains its component's total volume.
        """
        label_overlap = da.overlap.overlap(label_arr, depth=1, boundary=0)
        logger.debug("Finding cross-boundary pairs...")
        # NOTE: for now, computing each delayed chunk sequentially, not in parallel
        # So it works for memory
        delayed_ls = [
            dask.delayed(self.cellc_funcs.get_boundary_pairs)(i)
            for i in label_overlap.to_delayed().ravel()
        ]
        # Computing each chunk sequentially, not parallel, so it works for memory
        pair_arr_ls = [dask.compute(i)[0] for i in delayed_ls]
        pairs_arr = (
            np.concatenate(pair_arr_ls, axis=0) if pair_arr_ls else np.empty((0, 2))
        )
        logger.debug("Cross-boundary pairs found: {}", len(pairs_arr))
        uf = UnionFind()
        for a, b in pairs_arr:
            uf.union(int(a), int(b))
        logger.debug("Aggregating voxels per label...")
        delayed_ls = [
            dask.delayed(self.cellc_funcs.get_label_sizemap)(i)
            for i in label_arr.to_delayed().ravel()
        ]
        # Computing each chunk sequentially, not parallel, so it works for memory
        label_counts_ls = [dask.compute(i)[0] for i in delayed_ls]
        label_counts = (
            np.concatenate(label_counts_ls, axis=0)
            if label_counts_ls
            else np.empty((0, 2))
        )
        labels = label_counts[:, 0]
        counts = label_counts[:, 1]
        logger.debug("Unique labels (foreground): {}", len(labels))
        uf.build_lookup_table(labels, counts)
        logger.debug("Writing output array...")
        return da.map_blocks(
            self.cellc_funcs.map_values_to_arr,
            label_arr,
            ids=uf.sorted_keys,
            values=uf.sorted_sizes,
            dtype=np.uint64,
        )

    # ===========================================
    # STATIC UTILITIES
    # ===========================================

    @staticmethod
    def get_imgs_ls(imgs_dir: Path) -> list:
        """Get sorted list of subdirectories in a directory.

        Args:
            imgs_dir: Directory path to list.

        Returns:
            Naturally sorted list of subdirectory paths.
        """
        return natsorted([fp.name for fp in imgs_dir.iterdir() if fp.is_dir()])

    # ===========================================
    # UPDATE CONFIGS
    # ===========================================

    @trace
    def update_config(
        self,
        default_config_fp: Path,
    ) -> None:
        """Copy the default configs to this project."""
        # Parsing in the new config to see if it is valid
        ProjConfig.read_yaml(default_config_fp)
        # Overwriting the config file with the new config
        shutil.copyfile(default_config_fp, self.pfm.config_fp)

    # ===========================================
    # CONVERT TIFF TO ZARR
    # ===========================================

    @trace
    @_check_overwrite("raw")
    def tiff2zarr(self, in_fp: Path, *, overwrite: bool = False) -> None:
        """Convert TIFF file(s) to Zarr format.

        Args:
            in_fp: Path to TIFF file or directory of TIFF files.
            overwrite: If True, overwrite existing output.
        """
        logger.debug("Making zarr from tiff file(s)")
        with cluster_process(LocalCluster(n_workers=1, threads_per_worker=6)):
            if in_fp.is_dir():
                logger.debug("in_fp ({}) is a directory", in_fp)
                tiffs2zarr(
                    src_fp_ls=tuple(
                        natsorted(
                            in_fp / i
                            for i in in_fp.iterdir()
                            if re.search(r".tif$", str(i))
                        ),
                    ),
                    dst_fp=self.pfm.raw,
                    chunks=self.config.chunks.to_tuple(),
                )
            elif in_fp.is_file():
                logger.debug("in_fp ({}) is a file", in_fp)
                btiff2zarr(
                    src_fp=in_fp,
                    dst_fp=self.pfm.raw,
                    chunks=self.config.chunks.to_tuple(),
                )
            else:
                err_msg = f'Input file path, "{in_fp}" does not exist.'
                raise ValueError(err_msg)

    # ===========================================
    # RECHUNK
    # ===========================================

    @trace
    def rechunk_raw(self, *, overwrite: bool = False) -> None:
        """Rechunk raw zarr to configured chunk size.

        Useful when chunk size needs adjustment after initial conversion.
        """
        with cluster_process(self.busy_cluster()):
            zarr_arr = da.from_zarr(self.pfm.raw)
            desired_chunks = self.config.chunks.to_tuple()
            # Check if already in desired chunks
            if zarr_arr.chunksize == desired_chunks:
                logger.warning("Zarr is already in desired chunks. Not rechunking.")
                return
            # Rechunk
            temp_fp = self.pfm.raw.with_suffix(".rechunk_temp.zarr")
            zarr_rechunked = zarr_arr.rechunk(desired_chunks)
            disk_cache(zarr_rechunked, temp_fp)
            # Remove old zarr and move rechunked zarr to original location
            silent_remove(self.pfm.raw)
            shutil.move(temp_fp, self.pfm.raw)

    # ===========================================
    # REGISTRATION PIPELINE FUNCS
    # ===========================================

    @trace
    @_check_overwrite("ref", "annot", "map", "affine", "bspline")
    def reg_ref_prepare(self, *, overwrite: bool = False) -> None:
        """Prepare reference atlas images for registration.

        Copies and preprocesses reference/annotation images from atlas,
        applying orientation and trimming as configured.
        """
        rfm = RefFp(
            self.config.reference.atlas_dir,
            self.config.reference.ref_version,
            self.config.reference.annot_version,
            self.config.reference.map_version,
        )
        for fp_i, fp_o in [(rfm.ref, self.pfm.ref), (rfm.annot, self.pfm.annot)]:
            arr = tifffile.imread(fp_i)
            arr = self.cellc_funcs.reorient(
                arr, self.config.registration.ref_orientation.to_tuple()
            )
            arr = arr[
                self.config.registration.ref_trim.z.to_slice(),
                self.config.registration.ref_trim.y.to_slice(),
                self.config.registration.ref_trim.x.to_slice(),
            ]
            write_tiff(arr, fp_o)
        shutil.copyfile(rfm.map, self.pfm.map)
        shutil.copyfile(rfm.affine, self.pfm.affine)
        shutil.copyfile(rfm.bspline, self.pfm.bspline)

    @trace
    @_check_overwrite("downsmpl1")
    def reg_img_rough(self, *, overwrite: bool = False) -> None:
        """Rough downsampling of raw image by integer strides.

        First pass downsampling for registration pyramid.
        """
        with cluster_process(self.busy_cluster()):
            raw_arr = da.from_zarr(self.pfm.raw)
            downsmpl1_arr = raw_arr[
                :: self.config.registration.downsample_rough.z,
                :: self.config.registration.downsample_rough.y,
                :: self.config.registration.downsample_rough.x,
            ]
            downsmpl1_arr = downsmpl1_arr.compute()
            write_tiff(downsmpl1_arr, self.pfm.downsmpl1)

    @trace
    @_check_overwrite("downsmpl2")
    def reg_img_fine(self, *, overwrite: bool = False) -> None:
        """Fine downsampling using Gaussian zoom.

        Second pass downsampling for registration pyramid.
        """
        downsmpl1_arr = tifffile.imread(self.pfm.downsmpl1)
        downsmpl2_arr = self.cellc_funcs.downsample(
            downsmpl1_arr,
            self.config.registration.downsample_fine.z,
            self.config.registration.downsample_fine.y,
            self.config.registration.downsample_fine.x,
        )
        write_tiff(downsmpl2_arr, self.pfm.downsmpl2)

    @trace
    @_check_overwrite("trimmed")
    def reg_img_trim(self, *, overwrite: bool = False) -> None:
        """Trim downsampled image to region of interest."""
        downsmpl2_arr = tifffile.imread(self.pfm.downsmpl2)
        trimmed_arr = downsmpl2_arr[
            self.config.registration.reg_trim.z.to_slice(),
            self.config.registration.reg_trim.y.to_slice(),
            self.config.registration.reg_trim.x.to_slice(),
        ]
        write_tiff(trimmed_arr, self.pfm.trimmed)

    @trace
    @_check_overwrite("bounded")
    def reg_img_bound(self, *, overwrite: bool = False) -> None:
        """Apply intensity bounds to trimmed image.

        Clips intensities to configured range for better registration.
        """
        trimmed_arr = tifffile.imread(self.pfm.trimmed)
        bounded_arr = trimmed_arr.copy()
        bounded_arr[bounded_arr < self.config.registration.lower_bound] = (
            self.config.registration.lower_bound_mapto
        )
        bounded_arr[bounded_arr > self.config.registration.upper_bound] = (
            self.config.registration.upper_bound_mapto
        )
        write_tiff(bounded_arr, self.pfm.bounded)

    @trace
    @_check_overwrite("regresult")
    def reg_elastix(self, *, overwrite: bool = False) -> None:
        """Register image with elastix and store transformation components."""
        registration(
            fixed_img_fp=self.pfm.bounded,
            moving_img_fp=self.pfm.ref,
            output_img_fp=self.pfm.regresult,
            affine_fp=self.pfm.affine,
            bspline_fp=self.pfm.bspline,
        )

    # ===========================================
    # CROP RAW ZARR TO MAKE TUNING ZARR
    # ===========================================

    @trace
    def make_tuning_arr(self, *, overwrite: bool = False) -> None:
        """Crop raw zarr to make a smaller zarr for tuning."""
        pfm_prod = ProjFp(self.pfm.root_dir, tuning=False)
        pfm_tuning = ProjFp(self.pfm.root_dir, tuning=True)
        # If tuning zarr already exists and overwrite is False, skip processing
        if pfm_tuning.raw.exists() and not overwrite:
            logger.warning("Tuning raw zarr already exists. Not overwriting.")
            return
        # Crop raw zarr to tuning region and save as new zarr for tuning pipeline steps
        with cluster_process(self.busy_cluster()):
            raw_arr = da.from_zarr(pfm_prod.raw)
            raw_arr = raw_arr[
                self.config.tuning_trim.z.to_slice(),
                self.config.tuning_trim.y.to_slice(),
                self.config.tuning_trim.x.to_slice(),
            ]
            raw_arr = raw_arr.rechunk(self.config.chunks.to_tuple())
            disk_cache(raw_arr, pfm_tuning.raw)

    # ===========================================
    # CELL COUNTING PIPELINE FUNCS
    # ===========================================

    @trace
    @_check_overwrite("bgrm")
    def tophat_filter(self, *, overwrite: bool = False) -> None:
        """Step 1: Top-hat filter (background subtraction)."""
        with cluster_process(self.gpu_cluster()):
            result = da.map_blocks(
                self.cellc_funcs.tophat_filt,
                da.from_zarr(self.pfm.raw),
                radius=self.config.cell_counting.tophat_radius,
            )
            disk_cache(result, self.pfm.bgrm)

    @trace
    @_check_overwrite("dog")
    def dog_filter(self, *, overwrite: bool = False) -> None:
        """Step 2: Difference of Gaussians (edge detection)."""
        with cluster_process(self.gpu_cluster()):
            result = da.map_blocks(
                self.cellc_funcs.dog_filt,
                da.from_zarr(self.pfm.bgrm),
                sigma1=self.config.cell_counting.dog_sigma1,
                sigma2=self.config.cell_counting.dog_sigma2,
            )
            disk_cache(result, self.pfm.dog)

    @trace
    @_check_overwrite("adaptv")
    def adaptive_threshold_prep(self, *, overwrite: bool = False) -> None:
        """Step 3: Gaussian subtraction for adaptive thresholding."""
        with cluster_process(self.gpu_cluster()):
            result = da.map_blocks(
                self.cellc_funcs.gauss_subt_filt,
                da.from_zarr(self.pfm.dog),
                sigma=self.config.cell_counting.large_gauss_radius,
            )
            disk_cache(result, self.pfm.adaptv)

    @trace
    @_check_overwrite("threshd")
    def threshold(self, *, overwrite: bool = False) -> None:
        """Step 4: Manual thresholding."""
        with cluster_process(self.gpu_cluster()):
            result = da.map_blocks(
                self.cellc_funcs.manual_thresh,
                da.from_zarr(self.pfm.adaptv),
                val=self.config.cell_counting.threshd_value,
            )
            disk_cache(result, self.pfm.threshd)

    @trace
    @_check_overwrite("threshd_labels")
    def label_thresholded(self, *, overwrite: bool = False) -> None:
        """Step 5: Label contiguous regions in thresholded image."""
        max_labels = int(np.ceil(np.prod(self.config.chunks.to_tuple())) / 2) + 1
        with cluster_process(self.gpu_cluster()):
            result = da.map_blocks(
                self.cellc_funcs.mask2label,
                da.from_zarr(self.pfm.threshd),
                max_labels_per_chunk=max_labels,
            )
            disk_cache(result, self.pfm.threshd_labels)

    @trace
    @_check_overwrite("threshd_volumes")
    def compute_thresholded_volumes(self, *, overwrite: bool = False) -> None:
        """Step 6: Compute contiguous sizes using union-find."""
        with cluster_process(self.gpu_cluster()):
            sizes_arr = self._spatial_connect_count(
                da.from_zarr(self.pfm.threshd_labels),
            )
            disk_cache(sizes_arr, self.pfm.threshd_volumes)

    @trace
    @_check_overwrite("threshd_filt")
    def filter_thresholded(self, *, overwrite: bool = False) -> None:
        """Step 7: Filter out objects by size."""
        with cluster_process(self.gpu_cluster()):
            result = da.map_blocks(
                self.cellc_funcs.volume_filter,
                da.from_zarr(self.pfm.threshd_volumes),
                smin=self.config.cell_counting.min_threshd_size,
                smax=self.config.cell_counting.max_threshd_size,
            )
            disk_cache(result, self.pfm.threshd_filt)

    @trace
    @_check_overwrite("maxima")
    def detect_maxima(self, *, overwrite: bool = False) -> None:
        """Step 8: Detect local maxima as cell candidates."""
        with cluster_process(self.gpu_cluster()):
            result = da.map_blocks(
                self.cellc_funcs.get_local_maxima,
                da.from_zarr(self.pfm.adaptv),
                da.from_zarr(self.pfm.threshd_filt),
                radius=self.config.cell_counting.maxima_radius,
            )
            disk_cache(result, self.pfm.maxima)

    @trace
    @_check_overwrite("maxima_labels")
    def label_maxima(self, *, overwrite: bool = False) -> None:
        """Step 9: Label maxima points."""
        max_labels = int(np.ceil(np.prod(self.config.chunks.to_tuple())) / 2) + 1
        with cluster_process(self.gpu_cluster()):
            result = da.map_blocks(
                self.cellc_funcs.mask2label,
                da.from_zarr(self.pfm.maxima),
                max_labels_per_chunk=max_labels,
            )
            disk_cache(result, self.pfm.maxima_labels)

    @trace
    @_check_overwrite("wshed_labels")
    def watershed(self, *, overwrite: bool = False) -> None:
        """Step 10: Watershed segmentation."""
        with cluster_process(self.heavy_cluster()):
            result = da.map_blocks(
                self.cellc_funcs.wshed_segm,
                da.from_zarr(self.pfm.adaptv),
                da.from_zarr(self.pfm.maxima_labels),
                da.from_zarr(self.pfm.threshd_filt),
            )
            disk_cache(result, self.pfm.wshed_labels)

    @trace
    @_check_overwrite("wshed_volumes")
    def compute_watershed_volumes(self, *, overwrite: bool = False) -> None:
        """Step 11: Compute watershed volumes using union-find."""
        with cluster_process(self.heavy_cluster()):
            wshed_labels_arr = da.from_zarr(self.pfm.wshed_labels)
            wshed_volumes_arr = self._spatial_connect_count(wshed_labels_arr)
            disk_cache(wshed_volumes_arr, self.pfm.wshed_volumes)

    @trace
    @_check_overwrite("wshed_filt")
    def filter_watershed(self, *, overwrite: bool = False) -> None:
        """Step 12: Filter watershed objects by size."""
        with cluster_process(self.heavy_cluster()):
            result = da.map_blocks(
                self.cellc_funcs.volume_filter,
                da.from_zarr(self.pfm.wshed_volumes),
                smin=self.config.cell_counting.min_wshed_size,
                smax=self.config.cell_counting.max_wshed_size,
            )
            disk_cache(result, self.pfm.wshed_filt)

    @trace
    @_check_overwrite("cells_raw_df")
    def save_cells_table(self, *, overwrite: bool = False) -> None:
        """Step 13: Extract and save cells table with measurements."""
        with cluster_process(self.gpu_cluster()):
            raw_arr = da.from_zarr(self.pfm.raw)
            maxima_labels_arr = da.from_zarr(self.pfm.maxima_labels)
            wshed_labels_arr = da.from_zarr(self.pfm.wshed_labels)
            wshed_filt_arr = da.from_zarr(self.pfm.wshed_filt)

            offsets = [np.cumsum([0, *c[:-1]]) for c in raw_arr.chunks]
            n_blocks = np.prod(raw_arr.numblocks)

            raw_blocks = raw_arr.to_delayed().ravel()
            maxima_blocks = maxima_labels_arr.to_delayed().ravel()
            wshed_blocks = wshed_labels_arr.to_delayed().ravel()
            filt_blocks = wshed_filt_arr.to_delayed().ravel()

            delayed_results = []
            for idx in range(n_blocks):
                block_coords = np.unravel_index(idx, raw_arr.numblocks)
                z_offset, y_offset, x_offset = (
                    offsets[i][block_coords[i]] for i in range(3)
                )
                delayed_results.append(
                    dask.delayed(get_cells)(
                        raw_blocks[idx],
                        maxima_blocks[idx],
                        wshed_blocks[idx],
                        filt_blocks[idx],
                        z_offset,
                        y_offset,
                        x_offset,
                        self.cellc_funcs.xp,
                    ),
                )

            cells_df = dd.from_delayed(delayed_results)

            # If tuning, add offsets back to get coordinates in original raw space
            if self._tuning:
                cells_df[Z] += self.config.tuning_trim.z.start or 0
                cells_df[Y] += self.config.tuning_trim.y.start or 0
                cells_df[X] += self.config.tuning_trim.x.start or 0

            cells_df = cells_df.compute()
            write_parquet(cells_df, self.pfm.cells_raw_df)

    # ===========================================
    # CELL MAPPING FUNCS
    # ===========================================

    @trace
    @_check_overwrite("cells_trfm_df")
    def transform_coords(self, *, overwrite: bool = False) -> None:
        """Transform cell coordinates to reference atlas space."""
        with cluster_process(self.busy_cluster()):
            cells_df = pd.read_parquet(self.pfm.cells_raw_df)
            cells_df = cells_df[[Z, Y, X]]

            cells_df = cells_df / np.array(
                self.config.registration.downsample_rough.to_tuple(),
            )
            cells_df = cells_df * np.array(
                self.config.registration.downsample_fine.to_tuple(),
            )
            cells_df = cells_df - np.array(
                [
                    s.start or 0
                    for s in (
                        self.config.registration.reg_trim.z.to_slice(),
                        self.config.registration.reg_trim.y.to_slice(),
                        self.config.registration.reg_trim.x.to_slice(),
                    )
                ],
            )
            cells_df = pd.DataFrame(cells_df, columns=pd.Series([Z, Y, X]))

            cells_trfm_df = transformation_coords(
                cells_df,
                str(self.pfm.ref),
                str(self.pfm.regresult),
            )
            write_parquet(cells_trfm_df, self.pfm.cells_trfm_df)

    @trace
    @_check_overwrite("cells_df")
    def cell_mapping(self, *, overwrite: bool = False) -> None:
        """Map transformed cell coordinates to region IDs."""
        with cluster_process(self.busy_cluster()):
            cells_df = pd.read_parquet(self.pfm.cells_raw_df)
            coords_trfm = pd.read_parquet(self.pfm.cells_trfm_df)
            cells_df = cells_df.reset_index(drop=True)
            cells_df[f"{Z}_{TRFM}"] = coords_trfm[Z].to_numpy()
            cells_df[f"{Y}_{TRFM}"] = coords_trfm[Y].to_numpy()
            cells_df[f"{X}_{TRFM}"] = coords_trfm[X].to_numpy()

            annot_arr = tifffile.imread(self.pfm.annot)
            s = annot_arr.shape
            trfm_loc = (
                cells_df[[f"{Z}_{TRFM}", f"{Y}_{TRFM}", f"{X}_{TRFM}"]]
                .round(0)
                .query(
                    f"({Z}_{TRFM} >= 0) & "
                    f"({Z}_{TRFM} < {s[0]}) & "
                    f"({Y}_{TRFM} >= 0) & "
                    f"({Y}_{TRFM} < {s[1]}) & "
                    f"({X}_{TRFM} >= 0) & "
                    f"({X}_{TRFM} < {s[2]})",
                )
                .astype(np.uint32)
            )
            cells_df[ID] = pd.Series(
                annot_arr[*trfm_loc.to_numpy().T].astype(np.uint32),
                index=trfm_loc.index,
            ).fillna(-1)

            annot_df = annot_fp2df(self.pfm.map)
            cells_df = df_map_ids(cells_df, annot_df)
            write_parquet(cells_df, self.pfm.cells_df)

    @trace
    @_check_overwrite("cells_agg_df")
    def group_cells(self, *, overwrite: bool = False) -> None:
        """Group cells by region and aggregate."""
        with cluster_process(self.busy_cluster()):
            cells_df = pd.read_parquet(self.pfm.cells_df)
            cells_agg_df = cells_df.groupby(ID).agg(
                CELL_AGG_MAPPINGS,
            )
            cells_agg_df.columns = list(CELL_AGG_MAPPINGS.keys())

            annot_df = annot_fp2df(self.pfm.map)
            cells_agg_df = combine_nested_regions(cells_agg_df, annot_df)
            cells_agg_df[IOV] = cells_agg_df[SUM_INTENSITY] / cells_agg_df[VOLUME]
            cells_agg_df = cells_agg_df[[*ANNOTATED_COLUMNS_FINAL, *CELL_COLUMNS]]
            write_parquet(cells_agg_df, self.pfm.cells_agg_df)

    @trace
    @_check_overwrite("cells_agg_csv")
    def cells2csv(self, *, overwrite: bool = False) -> None:
        """Save aggregated cell data to CSV."""
        cells_agg_df = pd.read_parquet(self.pfm.cells_agg_df)
        cells_agg_df.to_csv(self.pfm.cells_agg_csv)

    # ===========================================
    # HELPER WRAPPER FOR COMBINE IMAGE DATA
    # ===========================================

    @trace
    @staticmethod
    def combine(root_dir: Path, *, overwrite: bool = False) -> None:
        """Combine image data into single table and save."""
        combine_root(root_dir, root_dir.parent, overwrite=overwrite)

    # ===========================================
    # CLEAN
    # ===========================================

    @trace
    def clean_proj(self) -> None:
        """Clean project directory by removing cellcount subdirs.

        Warning: This will delete all intermediate
        outputs of the cell counting pipeline.
        """
        pfm_prod = ProjFp(self.pfm.root_dir, tuning=False)
        silent_remove(pfm_prod.root_dir / pfm_prod.cellcount_sdir)
        pfm_tuning = ProjFp(self.pfm.root_dir, tuning=True)
        silent_remove(pfm_tuning.root_dir / pfm_tuning.cellcount_sdir)
        logger.info("Project {} cleaned.", self.pfm.root_dir)

    # ===========================================
    # VISUAL CHECKS
    # ===========================================

    @_check_overwrite("points_raw")
    def coords2points_raw(self, *, overwrite: bool = False) -> None:
        """Generate single-voxel markers at raw cell coordinates."""
        with cluster_process(self.busy_cluster()):
            visual_check_funcs_dask.coords2points(
                coords=pd.read_parquet(self.pfm.cells_raw_df),
                shape=da.from_zarr(self.pfm.raw).shape,
                out_fp=self.pfm.points_raw,
                chunks=self.config.chunks.to_tuple(),
            )

    @_check_overwrite("heatmap_raw")
    def coords2heatmap_raw(self, *, overwrite: bool = False) -> None:
        """Generate spherical heatmap at raw cell coordinates."""
        with cluster_process(self.busy_cluster()):
            visual_check_funcs_dask.coords2heatmap(
                coords=pd.read_parquet(self.pfm.cells_raw_df),
                shape=da.from_zarr(self.pfm.raw).shape,
                out_fp=self.pfm.heatmap_raw,
                radius=self.config.visual_check.heatmap_raw_radius,
                chunks=self.config.chunks.to_tuple(),
            )

    @_check_overwrite("points_trfm")
    def coords2points_trfm(self, *, overwrite: bool = False) -> None:
        """Generate single-voxel markers at transformed cell coordinates."""
        visual_check_funcs_tiff.coords2points(
            coords=pd.read_parquet(self.pfm.cells_trfm_df),
            shape=tifffile.imread(self.pfm.ref).shape,
            out_fp=self.pfm.points_trfm,
        )

    @_check_overwrite("heatmap_trfm")
    def coords2heatmap_trfm(self, *, overwrite: bool = False) -> None:
        """Generate spherical heatmap at transformed cell coordinates."""
        visual_check_funcs_tiff.coords2heatmap(
            coords=pd.read_parquet(self.pfm.cells_trfm_df),
            shape=tifffile.imread(self.pfm.ref).shape,
            out_fp=self.pfm.heatmap_trfm,
            radius=self.config.visual_check.heatmap_trfm_radius,
        )

    @_check_overwrite("comb_reg")
    def combine_reg(self, *, overwrite: bool = False) -> None:
        """Combine registration images into multi-channel TIFF for viewing."""
        combine_arrs(
            fp_in_ls=(self.pfm.trimmed, self.pfm.bounded, self.pfm.regresult),
            fp_out=self.pfm.comb_reg,
        )

    @_check_overwrite("comb_cellc")
    def combine_cellc(self, *, overwrite: bool = False) -> None:
        """Combine cell counting images into multi-channel TIFF for viewing.

        If not tuning (i.e. is prod), then we have to trim anyway
        - otherwise image is too large.
        """
        z_trim = slice(None)
        y_trim = slice(None)
        x_trim = slice(None)
        if not self.tuning:
            z_trim = self.config.visual_check.cellcount_trim.z.to_slice()
            y_trim = self.config.visual_check.cellcount_trim.y.to_slice()
            x_trim = self.config.visual_check.cellcount_trim.x.to_slice()
        combine_arrs(
            fp_in_ls=(self.pfm.raw, self.pfm.threshd_filt, self.pfm.wshed_filt),
            fp_out=self.pfm.comb_cellc,
            trimmer=(z_trim, y_trim, x_trim),
        )

    @_check_overwrite("comb_heatmap")
    def combine_heatmap_trfm(self, *, overwrite: bool = False) -> None:
        """Combine heatmap image into multi-channel TIFF for viewing."""
        combine_arrs(
            fp_in_ls=(self.pfm.ref, self.pfm.annot, self.pfm.heatmap_trfm),
            fp_out=self.pfm.comb_heatmap,
        )
