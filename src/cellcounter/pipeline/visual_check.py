"""Visual quality control for pipeline outputs.

Generates visualization arrays (points, heatmaps, combined images)
for verifying registration and cell counting results in Napari.
"""

import dask.array as da
import pandas as pd
import tifffile

from cellcounter.funcs import visual_check_funcs_dask, visual_check_funcs_tiff
from cellcounter.funcs.viewer_funcs import combine_arrs
from cellcounter.models.fp_models import check_overwrite
from cellcounter.pipeline.abstract_pipeline import AbstractPipeline
from cellcounter.utils.dask_utils import cluster_process

logger = logging.getLogger(__name__)


class VisualCheck(AbstractPipeline):
    """Visual Check Functions."""

    @check_overwrite("points_raw")
    def coords2points_raw(self, *, overwrite: bool = False) -> None:
        with cluster_process(self.busy_cluster()):
            visual_check_funcs_dask.coords2points(
                coords=pd.read_parquet(self.pfm.cells_raw_df),
                shape=da.from_zarr(self.pfm.raw).shape,
                out_fp=self.pfm.points_raw,
            )

    @check_overwrite("heatmap_raw")
    def coords2heatmap_raw(self, *, overwrite: bool = False) -> None:
        with cluster_process(self.busy_cluster()):
            visual_check_funcs_dask.coords2heatmap(
                coords=pd.read_parquet(self.pfm.cells_raw_df),
                shape=da.from_zarr(self.pfm.raw).shape,
                out_fp=self.pfm.heatmap_raw,
                radius=self.config.heatmap_raw_radius,
            )

    @check_overwrite("points_trfm")
    def coords2points_trfm(self, *, overwrite: bool = False) -> None:
        visual_check_funcs_tiff.coords2points(
            coords=pd.read_parquet(self.pfm.cells_trfm_df),
            shape=tifffile.imread(self.pfm.ref).shape,
            out_fp=self.pfm.points_trfm,
        )

    @check_overwrite("heatmap_trfm")
    def coords2heatmap_trfm(self, *, overwrite: bool = False) -> None:
        visual_check_funcs_tiff.coords2heatmap(
            coords=pd.read_parquet(self.pfm.cells_trfm_df),
            shape=tifffile.imread(self.pfm.ref).shape,
            out_fp=self.pfm.heatmap_trfm,
            radius=self.config.heatmap_trfm_radius,
        )

    @check_overwrite("comb_reg")
    def combine_reg(self, *, overwrite: bool = False) -> None:
        combine_arrs(
            fp_in_ls=(self.pfm.trimmed, self.pfm.bounded, self.pfm.regresult),
            fp_out=self.pfm.comb_reg,
        )

    @check_overwrite("comb_cellc")
    def combine_cellc(self, *, overwrite: bool = False) -> None:
        z_trim = slice(None)
        y_trim = slice(None)
        x_trim = slice(None)
        if not self._tuning:
            z_trim = slice(*self.config.combine_cellc_z_trim)
            y_trim = slice(*self.config.combine_cellc_y_trim)
            x_trim = slice(*self.config.combine_cellc_x_trim)
        combine_arrs(
            fp_in_ls=(self.pfm.raw, self.pfm.threshd_final, self.pfm.wshed_final),
            fp_out=self.pfm.comb_cellc,
            trimmer=(z_trim, y_trim, x_trim),
        )

    @check_overwrite("comb_heatmap")
    def combine_heatmap_trfm(self, *, overwrite: bool = False) -> None:
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
