"""Main cell counting pipeline orchestrator.

Coordinates the full workflow:
1. Registration: TIFF→Zarr, downsampling, elastix registration
2. Cell Counting: filtering, thresholding, watershed segmentation
3. Mapping: coordinate transformation, region assignment, aggregation

All pipeline methods use @check_overwrite decorator for file safety.
"""

import logging
import re
import shutil
from pathlib import Path

import dask
import dask.array as da
import dask.dataframe as dd
import numpy as np
import pandas as pd
import tifffile
from dask.distributed import LocalCluster
from natsort import natsorted

from cellcounter.constants import (
    ANNOT_COLUMNS_FINAL,
    CELL_AGG_MAPPINGS,
    TRFM,
    AnnotColumns,
    CellColumns,
    Coords,
)
from cellcounter.funcs.elastix_funcs import registration, transformation_coords
from cellcounter.funcs.io_funcs import (
    btiff2zarr,
    silent_remove,
    tiffs2zarr,
    write_parquet,
    write_tiff,
)
from cellcounter.funcs.map_funcs import (
    annot_fp2df,
    combine_nested_regions,
    df_map_ids,
    get_cells,
)
from cellcounter.models.fp_models import get_proj_fp
from cellcounter.models.fp_models.ref_fp import RefFp
from cellcounter.pipeline.abstract_pipeline import AbstractPipeline, _check_overwrite
from cellcounter.utils.dask_utils import cluster_process, disk_cache
from cellcounter.utils.misc_utils import enum2list
from cellcounter.utils.union_find import UnionFind

logger = logging.getLogger(__name__)


class Pipeline(AbstractPipeline):
    """Cell counting pipeline orchestrator.

    Coordinates registration, cell counting, and mapping workflows.
    All pipeline methods use @check_overwrite decorator for file safety.

    Attributes:
        STEPS_REGISTRATION: Tuple of registration step method names.
        STEPS_CELL_COUNTING: Tuple of cell counting step method names.
        STEPS_MAPPING: Tuple of mapping step method names.
    """

    # Pipeline step registry for declarative execution
    STEPS_REGISTRATION = (
        "tiff2zarr",
        "reg_ref_prepare",
        "reg_img_rough",
        "reg_img_fine",
        "reg_img_trim",
        "reg_img_bound",
        "reg_elastix",
    )
    STEPS_CELL_COUNTING = (
        "tophat_filter",
        "dog_filter",
        "adaptive_threshold_prep",
        "threshold",
        "label_thresholded",
        "compute_thresholded_volumes",
        "filter_thresholded",
        "detect_maxima",
        "label_maxima",
        "watershed",
        "compute_watershed_volumes",
        "filter_watershed",
        "save_cells_table",
    )
    STEPS_MAPPING = (
        "transform_coords",
        "cell_mapping",
        "group_cells",
        "cells2csv",
    )

    #############################################
    # HELPER: GENERIC ZARR PROCESSING STEP
    #############################################

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
        logger.debug("Cross-boundary pairs found: %d", len(pairs_arr))
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
        logger.debug("Unique labels (foreground): %d", len(labels))
        uf.build_lookup_table(labels, counts)
        logger.debug("Writing output array...")
        return da.map_blocks(
            self.cellc_funcs.map_values_to_arr,
            label_arr,
            ids=uf.sorted_keys,
            values=uf.sorted_sizes,
            dtype=np.uint64,
        )

    #############################################
    # STATIC UTILITIES
    #############################################

    @staticmethod
    def get_imgs_ls(imgs_dir: Path | str) -> list:
        """Get sorted list of subdirectories in a directory.

        Args:
            imgs_dir: Directory path to list.

        Returns:
            Naturally sorted list of subdirectory paths.
        """
        imgs_dir = Path(imgs_dir)
        return natsorted([fp for fp in imgs_dir.iterdir() if (imgs_dir / fp).is_dir()])

    #############################################
    # CONVERT TIFF TO ZARR
    #############################################

    @_check_overwrite("raw")
    def tiff2zarr(self, in_fp: Path | str, *, overwrite: bool = False) -> None:
        """Convert TIFF file(s) to Zarr format.

        Args:
            in_fp: Path to TIFF file or directory of TIFF files.
            overwrite: If True, overwrite existing output.
        """
        in_fp = Path(in_fp)
        logger.debug("Making zarr from tiff file(s)")
        with cluster_process(LocalCluster(n_workers=1, threads_per_worker=6)):
            if in_fp.is_dir():
                logger.debug("in_fp (%s) is a directory", in_fp)
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
                logger.debug("in_fp (%s) is a file", in_fp)
                btiff2zarr(
                    src_fp=in_fp,
                    dst_fp=self.pfm.raw,
                    chunks=self.config.chunks.to_tuple(),
                )
            else:
                err_msg = f'Input file path, "{in_fp}" does not exist.'
                raise ValueError(err_msg)

    #############################################
    # RECHUNK
    #############################################

    def rechunk_raw(self) -> None:
        """Rechunk raw zarr to configured chunk size.

        Useful when chunk size needs adjustment after initial conversion.
        """
        with cluster_process(self.busy_cluster()):
            zarr_arr = da.from_zarr(self.pfm.raw)
            temp_fp = self.pfm.raw.with_suffix(".rechunk_temp.zarr")
            zarr_rechunked = zarr_arr.rechunk(self.config.chunks.to_tuple())
            disk_cache(zarr_rechunked, temp_fp)
            silent_remove(self.pfm.raw)
            shutil.move(temp_fp, self.pfm.raw)

    #############################################
    # REGISTRATION PIPELINE FUNCS
    #############################################

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

    #############################################
    # CROP RAW ZARR TO MAKE TUNING ZARR
    #############################################

    def make_tuning_arr(self, *, overwrite: bool = False) -> None:
        """Crop raw zarr to make a smaller zarr for tuning."""
        # TODO: make overwrite logic for pfm_tuning
        pfm_prod = get_proj_fp(self.pfm.root_dir, tuning=False)
        pfm_tuning = get_proj_fp(self.pfm.root_dir, tuning=True)
        with cluster_process(self.busy_cluster()):
            raw_arr = da.from_zarr(pfm_prod.raw)
            raw_arr = raw_arr[
                self.config.tuning_trim.z.to_slice(),
                self.config.tuning_trim.y.to_slice(),
                self.config.tuning_trim.x.to_slice(),
            ]
            raw_arr = raw_arr.rechunk(self.config.chunks.to_tuple())
            disk_cache(raw_arr, pfm_tuning.raw)

    #############################################
    # CELL COUNTING PIPELINE FUNCS
    #############################################

    @_check_overwrite("bgrm")
    def tophat_filter(self, *, overwrite: bool = False) -> None:
        """Step 1: Top-hat filter (background subtraction)."""
        with cluster_process(self.gpu_cluster()):
            result = da.map_blocks(
                self.cellc_funcs.tophat_filt,
                da.from_zarr(self.pfm.raw),
                radius=self.pfm.config.cell_counting.tophat_radius,
            )
            disk_cache(result, self.pfm.bgrm)

    @_check_overwrite("dog")
    def dog_filter(self, *, overwrite: bool = False) -> None:
        """Step 2: Difference of Gaussians (edge detection)."""
        with cluster_process(self.gpu_cluster()):
            result = da.map_blocks(
                self.cellc_funcs.dog_filt,
                da.from_zarr(self.pfm.bgrm),
                sigma1=self.pfm.config.cell_counting.dog_sigma1,
                sigma2=self.pfm.config.cell_counting.dog_sigma2,
            )
            disk_cache(result, self.pfm.dog)

    @_check_overwrite("adaptv")
    def adaptive_threshold_prep(self, *, overwrite: bool = False) -> None:
        """Step 3: Gaussian subtraction for adaptive thresholding."""
        with cluster_process(self.gpu_cluster()):
            result = da.map_blocks(
                self.cellc_funcs.gauss_subt_filt,
                da.from_zarr(self.pfm.dog),
                sigma=self.pfm.config.cell_counting.large_gauss_radius,
            )
            disk_cache(result, self.pfm.adaptv)

    @_check_overwrite("threshd")
    def threshold(self, *, overwrite: bool = False) -> None:
        """Step 4: Manual thresholding."""
        with cluster_process(self.gpu_cluster()):
            result = da.map_blocks(
                self.cellc_funcs.manual_thresh,
                da.from_zarr(self.pfm.adaptv),
                val=self.pfm.config.cell_counting.threshd_value,
            )
            disk_cache(result, self.pfm.threshd)

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

    @_check_overwrite("threshd_volumes")
    def compute_thresholded_volumes(self, *, overwrite: bool = False) -> None:
        """Step 6: Compute contiguous sizes using union-find."""
        with cluster_process(self.gpu_cluster()):
            sizes_arr = self._spatial_connect_count(
                da.from_zarr(self.pfm.threshd_labels),
            )
            disk_cache(sizes_arr, self.pfm.threshd_volumes)

    @_check_overwrite("threshd_filt")
    def filter_thresholded(self, *, overwrite: bool = False) -> None:
        """Step 7: Filter out objects by size."""
        with cluster_process(self.gpu_cluster()):
            result = da.map_blocks(
                self.cellc_funcs.volume_filter,
                da.from_zarr(self.pfm.threshd_volumes),
                smin=self.pfm.config.cell_counting.min_threshd_size,
                smax=self.pfm.config.cell_counting.max_threshd_size,
            )
            disk_cache(result, self.pfm.threshd_filt)

    @_check_overwrite("maxima")
    def detect_maxima(self, *, overwrite: bool = False) -> None:
        """Step 8: Detect local maxima as cell candidates."""
        with cluster_process(self.gpu_cluster()):
            result = da.map_blocks(
                self.cellc_funcs.get_local_maxima,
                da.from_zarr(self.pfm.adaptv),
                da.from_zarr(self.pfm.threshd_filt),
                radius=self.pfm.config.cell_counting.maxima_radius,
            )
            disk_cache(result, self.pfm.maxima)

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

    @_check_overwrite("wshed_volumes")
    def compute_watershed_volumes(self, *, overwrite: bool = False) -> None:
        """Step 11: Compute watershed volumes using union-find."""
        with cluster_process(self.heavy_cluster()):
            wshed_labels_arr = da.from_zarr(self.pfm.wshed_labels)
            wshed_volumes_arr = self._spatial_connect_count(wshed_labels_arr)
            disk_cache(wshed_volumes_arr, self.pfm.wshed_volumes)

    @_check_overwrite("wshed_filt")
    def filter_watershed(self, *, overwrite: bool = False) -> None:
        """Step 12: Filter watershed objects by size."""
        with cluster_process(self.heavy_cluster()):
            result = da.map_blocks(
                self.cellc_funcs.volume_filter,
                da.from_zarr(self.pfm.wshed_volumes),
                smin=self.pfm.config.cell_counting.min_wshed_size,
                smax=self.pfm.config.cell_counting.max_wshed_size,
            )
            disk_cache(result, self.pfm.wshed_filt)

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
                cells_df[Coords.Z.value] += self.config.tuning_trim.z.start or 0
                cells_df[Coords.Y.value] += self.config.tuning_trim.y.start or 0
                cells_df[Coords.X.value] += self.config.tuning_trim.x.start or 0

            cells_df = cells_df.compute()
            write_parquet(cells_df, self.pfm.cells_raw_df)

    #############################################
    # CELL MAPPING FUNCS
    #############################################

    @_check_overwrite("cells_trfm_df")
    def transform_coords(self, *, overwrite: bool = False) -> None:
        """Transform cell coordinates to reference atlas space."""
        with cluster_process(self.busy_cluster()):
            cells_df = pd.read_parquet(self.pfm.cells_raw_df)
            cells_df = cells_df[enum2list(Coords)]

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
            cells_df = pd.DataFrame(cells_df, columns=enum2list(Coords))

            cells_trfm_df = transformation_coords(
                cells_df,
                str(self.pfm.ref),
                str(self.pfm.regresult),
            )
            write_parquet(cells_trfm_df, self.pfm.cells_trfm_df)

    @_check_overwrite("cells_df")
    def cell_mapping(self, *, overwrite: bool = False) -> None:
        """Map transformed cell coordinates to region IDs."""
        with cluster_process(self.busy_cluster()):
            cells_df = pd.read_parquet(self.pfm.cells_raw_df)
            coords_trfm = pd.read_parquet(self.pfm.cells_trfm_df)
            cells_df = cells_df.reset_index(drop=True)
            cells_df[f"{Coords.Z.value}_{TRFM}"] = coords_trfm[
                Coords.Z.value
            ].to_numpy()
            cells_df[f"{Coords.Y.value}_{TRFM}"] = coords_trfm[
                Coords.Y.value
            ].to_numpy()
            cells_df[f"{Coords.X.value}_{TRFM}"] = coords_trfm[
                Coords.X.value
            ].to_numpy()

            annot_arr = tifffile.imread(self.pfm.annot)
            s = annot_arr.shape
            trfm_loc = (
                cells_df[
                    [
                        f"{Coords.Z.value}_{TRFM}",
                        f"{Coords.Y.value}_{TRFM}",
                        f"{Coords.X.value}_{TRFM}",
                    ]
                ]
                .round(0)
                .astype(np.int64)
                .astype(np.uint32)
                .query(
                    f"({Coords.Z.value}_{TRFM} >= 0) & "
                    f"({Coords.Z.value}_{TRFM} < {s[0]}) & "
                    f"({Coords.Y.value}_{TRFM} >= 0) & "
                    f"({Coords.Y.value}_{TRFM} < {s[1]}) & "
                    f"({Coords.X.value}_{TRFM} >= 0) & "
                    f"({Coords.X.value}_{TRFM} < {s[2]})",
                )
            )
            cells_df[AnnotColumns.ID.value] = pd.Series(
                annot_arr[*trfm_loc.to_numpy().T].astype(np.uint32),
                index=trfm_loc.index,
            ).fillna(-1)

            annot_df = annot_fp2df(self.pfm.map)
            cells_df = df_map_ids(cells_df, annot_df)
            write_parquet(cells_df, self.pfm.cells_df)

    @_check_overwrite("cells_agg_df")
    def group_cells(self, *, overwrite: bool = False) -> None:
        """Group cells by region and aggregate."""
        with cluster_process(self.busy_cluster()):
            cells_df = pd.read_parquet(self.pfm.cells_df)
            cells_agg_df = cells_df.groupby(AnnotColumns.ID.value).agg(
                CELL_AGG_MAPPINGS,
            )
            cells_agg_df.columns = list(CELL_AGG_MAPPINGS.keys())

            annot_df = annot_fp2df(self.pfm.map)
            cells_agg_df = combine_nested_regions(cells_agg_df, annot_df)
            cells_agg_df[CellColumns.IOV.value] = (
                cells_agg_df[CellColumns.SUM_INTENSITY.value]
                / cells_agg_df[CellColumns.VOLUME.value]
            )
            cells_agg_df = cells_agg_df[[*ANNOT_COLUMNS_FINAL, *enum2list(CellColumns)]]
            write_parquet(cells_agg_df, self.pfm.cells_agg_df)

    @_check_overwrite("cells_agg_csv")
    def cells2csv(self, *, overwrite: bool = False) -> None:
        """Save aggregated cell data to CSV."""
        cells_agg_df = pd.read_parquet(self.pfm.cells_agg_df)
        cells_agg_df.to_csv(self.pfm.cells_agg_csv)

    #############################################
    # CLEAN
    #############################################

    def clean_proj(self) -> None:
        """Clean project directory by removing cellcount subdirs."""
        pfm_prod = get_proj_fp(self.pfm.root_dir, tuning=False)
        silent_remove(pfm_prod.root_dir / pfm_prod.cellcount_sdir)
        pfm_tuning = get_proj_fp(self.pfm.root_dir, tuning=True)
        silent_remove(pfm_tuning.root_dir / pfm_tuning.cellcount_sdir)
        logger.info("Project %s cleaned.", self.pfm.root_dir)

    #############################################
    # RUN PIPELINE
    #############################################

    def run_pipeline_steps(
        self,
        in_fp: str,
        *,
        steps: list[str] | None = None,
        overwrite: bool = False,
    ) -> None:
        """Run pipeline steps in order.

        Args:
            in_fp: Input file path for tiff2zarr.
            steps: Optional list of steps to run. If None, runs full pipeline.
            overwrite: If True, overwrite existing outputs.
        """
        steps = steps or [
            *["tiff2zarr"],
            *list(self.STEPS_REGISTRATION[1:]),
            *["make_tuning_arr"],
            *list(self.STEPS_CELL_COUNTING),
            *list(self.STEPS_MAPPING),
        ]

        for step in steps:
            if step == "tiff2zarr":
                self.tiff2zarr(in_fp, overwrite=overwrite)
            else:
                getattr(self, step)(overwrite=overwrite)
