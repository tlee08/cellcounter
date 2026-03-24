import logging

import dask.array as da
import pandas as pd
import tifffile
from dask.distributed import LocalCluster

from cellcounter.funcs.viewer_funcs import ViewerFuncs
from cellcounter.funcs.visual_check_funcs_dask import VisualCheckFuncsDask
from cellcounter.funcs.visual_check_funcs_tiff import VisualCheckFuncsTiff
from cellcounter.models.fp_models import check_overwrite, get_proj_fm
from cellcounter.models.proj_config import ProjConfig
from cellcounter.utils.dask_utils import cluster_process

logger = logging.getLogger(__name__)


class VisualCheck:
    """Visual Check Functions."""

    # Clusters
    # busy (many workers - carrying low RAM computations)
    n_workers = 6
    threads_per_worker = 2

    @classmethod
    def cluster(cls):
        return LocalCluster(
            n_workers=cls.n_workers, threads_per_worker=cls.threads_per_worker
        )

    ###################################################################################################
    # VISUAL CHECKS FROM DF POINTS
    ###################################################################################################

    @classmethod
    @check_overwrite("points_raw")
    def coords2points_raw(
        cls, proj_dir: str, *, overwrite: bool = False, tuning: bool = False
    ) -> None:
        pfm = get_proj_fm(proj_dir, tuning=tuning)
        with cluster_process(cls.cluster()):
            VisualCheckFuncsDask.coords2points(
                coords=pd.read_parquet(pfm.cells_raw_df),
                shape=da.from_zarr(pfm.raw).shape,
                out_fp=pfm.points_raw,
            )

    @classmethod
    @check_overwrite("heatmap_raw")
    def coords2heatmap_raw(
        cls, proj_dir: str, *, overwrite: bool = False, tuning: bool = False
    ) -> None:
        pfm = get_proj_fm(proj_dir, tuning=tuning)
        with cluster_process(cls.cluster()):
            configs = ProjConfig.read_file(pfm.config_params)
            VisualCheckFuncsDask.coords2heatmap(
                coords=pd.read_parquet(pfm.cells_raw_df),
                shape=da.from_zarr(pfm.raw).shape,
                out_fp=pfm.heatmap_raw,
                radius=configs.heatmap_raw_radius,
            )

    @classmethod
    @check_overwrite("points_trfm")
    def coords2points_trfm(
        cls, proj_dir: str, *, overwrite: bool = False, tuning: bool = False
    ) -> None:
        pfm = get_proj_fm(proj_dir, tuning=tuning)
        VisualCheckFuncsTiff.coords2points(
            coords=pd.read_parquet(pfm.cells_trfm_df),
            shape=tifffile.imread(pfm.ref).shape,
            out_fp=pfm.points_trfm,
        )

    @classmethod
    @check_overwrite("heatmap_trfm")
    def coords2heatmap_trfm(
        cls, proj_dir: str, *, overwrite: bool = False, tuning: bool = False
    ) -> None:
        pfm = get_proj_fm(proj_dir, tuning=tuning)
        configs = ProjConfig.read_file(pfm.config_params)
        VisualCheckFuncsTiff.coords2heatmap(
            coords=pd.read_parquet(pfm.cells_trfm_df),
            shape=tifffile.imread(pfm.ref).shape,
            out_fp=pfm.heatmap_trfm,
            radius=configs.heatmap_trfm_radius,
        )

    ###################################################################################################
    # COMBINING/MERGING ARRAYS IN RGB LAYERS
    ###################################################################################################

    @classmethod
    @check_overwrite("comb_reg")
    def combine_reg(cls, proj_dir: str, *, overwrite: bool = False) -> None:
        pfm = get_proj_fm(proj_dir, tuning=False)
        ViewerFuncs.combine_arrs(
            fp_in_ls=(pfm.trimmed, pfm.bounded, pfm.regresult),
            fp_out=pfm.comb_reg,
        )

    @classmethod
    @check_overwrite("comb_cellc")
    def combine_cellc(
        cls, proj_dir: str, *, overwrite: bool = False, tuning: bool = False
    ) -> None:
        pfm = get_proj_fm(proj_dir, tuning=tuning)
        configs = ProjConfig.read_file(pfm.config_params)
        z_trim = slice(None)
        y_trim = slice(None)
        x_trim = slice(None)
        if not tuning:
            z_trim = slice(*configs.combine_cellc_z_trim)
            y_trim = slice(*configs.combine_cellc_y_trim)
            x_trim = slice(*configs.combine_cellc_x_trim)
        ViewerFuncs.combine_arrs(
            fp_in_ls=(pfm.raw, pfm.threshd_final, pfm.wshed_final),
            fp_out=pfm.comb_cellc,
            trimmer=(z_trim, y_trim, x_trim),
        )

    @classmethod
    @check_overwrite("comb_heatmap")
    def combine_heatmap_trfm(
        cls, proj_dir: str, *, overwrite: bool = False, tuning: bool = False
    ) -> None:
        pfm = get_proj_fm(proj_dir, tuning=tuning)
        ViewerFuncs.combine_arrs(
            fp_in_ls=(pfm.ref, pfm.annot, pfm.heatmap_trfm),
            # 2nd regresult means the combining works in ImageJ
            fp_out=pfm.comb_heatmap,
        )

    ###################################################################################################
    # ALL PIPELINE FUNCTION
    ###################################################################################################

    @classmethod
    def run_make_visual_checks(cls, proj_dir: str, overwrite: bool = False) -> None:
        """Running all visual check pipelines in order."""
        # Registration visual check
        cls.combine_reg(proj_dir, overwrite=overwrite)
        for is_tuning in [
            True,  # Tuning
            False,  # Final
        ]:
            # Cell counting visual checks
            cls.coords2points_raw(proj_dir, overwrite=overwrite, tuning=is_tuning)
            cls.combine_cellc(proj_dir, overwrite=overwrite, tuning=is_tuning)
            # Transformed space visual checks
            cls.coords2points_trfm(proj_dir, overwrite=overwrite, tuning=is_tuning)
            cls.coords2heatmap_trfm(proj_dir, overwrite=overwrite, tuning=is_tuning)
            cls.combine_heatmap_trfm(proj_dir, overwrite=overwrite, tuning=is_tuning)
