"""Visual quality control for pipeline outputs.

Generates visualization arrays (points, heatmaps, combined images)
for verifying registration and cell counting results in Napari.
"""

import logging

import dask.array as da
import pandas as pd
import tifffile

from cellcounter.funcs import visual_check_funcs_dask, visual_check_funcs_tiff
from cellcounter.funcs.io_funcs import combine_arrs
from cellcounter.pipeline.abstract_pipeline import AbstractPipeline, _check_overwrite
from cellcounter.utils.dask_utils import cluster_process

logger = logging.getLogger(__name__)


class VisualCheck(AbstractPipeline):
    """Visual quality control for pipeline outputs.

    Generates visualization arrays (points, heatmaps, combined images)
    for verifying registration and cell counting results in Napari.
    """

    @_check_overwrite("points_raw")
    def coords2points_raw(self, *, overwrite: bool = False) -> None:
        """Generate single-voxel markers at raw cell coordinates."""
        with cluster_process(self.busy_cluster()):
            visual_check_funcs_dask.coords2points(
                coords=pd.read_parquet(self.pfm.cells_raw_df),
                shape=da.from_zarr(self.pfm.raw).shape,
                out_fp=self.pfm.points_raw,
                chunks=self.pfm.config.chunks.to_tuple(),
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
                chunks=self.pfm.config.chunks.to_tuple(),
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
            z_trim = self.config.combine.cellcount_trim.z.to_slice()
            y_trim = self.config.combine.cellcount_trim.y.to_slice()
            x_trim = self.config.combine.cellcount_trim.x.to_slice()
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

    @classmethod
    def run_make_visual_checks(cls, proj_dir: str, *, overwrite: bool = False) -> None:
        """Running all visual check pipelines in order."""
        vc = cls(proj_dir, tuning=False)
        vc.combine_reg(overwrite=overwrite)

        for is_tuning in [True, False]:
            vc = cls(proj_dir, tuning=is_tuning)
            vc.coords2points_raw(overwrite=overwrite)
            vc.combine_cellc(overwrite=overwrite)
            vc.coords2points_trfm(overwrite=overwrite)
            vc.coords2heatmap_trfm(overwrite=overwrite)
            vc.combine_heatmap_trfm(overwrite=overwrite)
