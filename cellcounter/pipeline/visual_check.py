import os

import dask.array as da
import pandas as pd
import tifffile
from dask.distributed import LocalCluster

from cellcounter.funcs.viewer_funcs import ViewerFuncs
from cellcounter.funcs.visual_check_funcs_dask import VisualCheckFuncsDask
from cellcounter.funcs.visual_check_funcs_tiff import VisualCheckFuncsTiff
from cellcounter.utils.config_params_model import ConfigParamsModel
from cellcounter.utils.dask_utils import (
    cluster_process,
    da_trim,
    disk_cache,
)
from cellcounter.utils.diagnostics_utils import file_exists_msg
from cellcounter.utils.logging_utils import init_logger_file
from cellcounter.utils.proj_org_utils import ProjFpModel, ProjFpModelTuning


class VisualCheck:
    # Clusters
    # busy (many workers - carrying low RAM computations)
    n_workers = 6
    threads_per_worker = 2

    @classmethod
    def cluster(cls):
        return LocalCluster(n_workers=cls.n_workers, threads_per_worker=cls.threads_per_worker)

    ###################################################################################################
    # VISUAL CHECKS FROM DF POINTS
    ###################################################################################################

    @classmethod
    def cellc_trim_to_final(cls, proj_dir: str, overwrite: bool = False, tuning: bool = False) -> None:
        """
        [DEPRECATED]

        Trimming filtered regions overlaps to make:
        - Trimmed maxima image
        - Trimmed threshold image
        - Trimmed watershed image
        """
        logger = init_logger_file()
        pfm = ProjFpModelTuning(proj_dir) if tuning else ProjFpModel(proj_dir)
        if not overwrite:
            for fp in (pfm.maxima_final, pfm.threshd_final, pfm.wshed_final):
                if os.path.exists(fp):
                    return logger.warning(file_exists_msg(fp))
        with cluster_process(cls.cluster()):
            # Getting configs
            configs = ConfigParamsModel.read_fp(pfm.config_params)
            # Reading input images
            maxima_arr = da.from_zarr(pfm.maxima)
            threshd_filt_arr = da.from_zarr(pfm.threshd_filt)
            wshed_volumes_arr = da.from_zarr(pfm.wshed_volumes)
            # Declaring processing instructions
            maxima_final_arr = da_trim(maxima_arr, d=configs.overlap_depth)
            threshd_final_arr = da_trim(threshd_filt_arr, d=configs.overlap_depth)
            wshed_final_arr = da_trim(wshed_volumes_arr, d=configs.overlap_depth)
            # Computing and saving
            maxima_final_arr = disk_cache(maxima_final_arr, pfm.maxima_final)
            threshd_final_arr = disk_cache(threshd_final_arr, pfm.threshd_final)
            wshed_final_arr = disk_cache(wshed_final_arr, pfm.wshed_final)

    @classmethod
    def coords2points_raw(cls, proj_dir: str, overwrite: bool = False, tuning: bool = False) -> None:
        logger = init_logger_file()
        pfm = ProjFpModelTuning(proj_dir) if tuning else ProjFpModel(proj_dir)
        if not overwrite:
            for fp in (pfm.points_raw,):
                if os.path.exists(fp):
                    return logger.warning(file_exists_msg(fp))
        with cluster_process(cls.cluster()):
            VisualCheckFuncsDask.coords2points(
                coords=pd.read_parquet(pfm.cells_raw_df),
                shape=da.from_zarr(pfm.raw).shape,
                out_fp=pfm.points_raw,
            )

    @classmethod
    def coords2heatmap_raw(cls, proj_dir: str, overwrite: bool = False, tuning: bool = False) -> None:
        logger = init_logger_file()
        pfm = ProjFpModelTuning(proj_dir) if tuning else ProjFpModel(proj_dir)
        if not overwrite:
            for fp in (pfm.heatmap_raw,):
                if os.path.exists(fp):
                    return logger.warning(file_exists_msg(fp))
        with cluster_process(cls.cluster()):
            configs = ConfigParamsModel.read_fp(pfm.config_params)
            VisualCheckFuncsDask.coords2heatmap(
                coords=pd.read_parquet(pfm.cells_raw_df),
                shape=da.from_zarr(pfm.raw).shape,
                out_fp=pfm.heatmap_raw,
                radius=configs.heatmap_raw_radius,
            )

    @classmethod
    def coords2points_trfm(cls, proj_dir: str, overwrite: bool = False, tuning: bool = False) -> None:
        logger = init_logger_file()
        pfm = ProjFpModelTuning(proj_dir) if tuning else ProjFpModel(proj_dir)
        if not overwrite:
            for fp in (pfm.points_trfm,):
                if os.path.exists(fp):
                    return logger.warning(file_exists_msg(fp))
        VisualCheckFuncsTiff.coords2points(
            coords=pd.read_parquet(pfm.cells_trfm_df),
            shape=tifffile.imread(pfm.ref).shape,
            out_fp=pfm.points_trfm,
        )

    @classmethod
    def coords2heatmap_trfm(cls, proj_dir: str, overwrite: bool = False, tuning: bool = False) -> None:
        logger = init_logger_file()
        pfm = ProjFpModelTuning(proj_dir) if tuning else ProjFpModel(proj_dir)
        if not overwrite:
            for fp in (pfm.heatmap_trfm,):
                if os.path.exists(fp):
                    return logger.warning(file_exists_msg(fp))
        configs = ConfigParamsModel.read_fp(pfm.config_params)
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
    def combine_reg(cls, proj_dir: str, overwrite: bool = False) -> None:
        logger = init_logger_file()
        pfm = ProjFpModel(proj_dir)
        if not overwrite:
            for fp in (pfm.comb_reg,):
                if os.path.exists(fp):
                    return logger.warning(file_exists_msg(fp))
        ViewerFuncs.combine_arrs(
            fp_in_ls=(pfm.trimmed, pfm.bounded, pfm.regresult),
            fp_out=pfm.comb_reg,
        )

    @classmethod
    def combine_cellc(cls, proj_dir: str, overwrite: bool = False, tuning: bool = False) -> None:
        logger = init_logger_file()
        pfm = ProjFpModelTuning(proj_dir) if tuning else ProjFpModel(proj_dir)
        if not overwrite:
            for fp in (pfm.comb_cellc,):
                if os.path.exists(fp):
                    return logger.warning(file_exists_msg(fp))
        configs = ConfigParamsModel.read_fp(pfm.config_params)
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
    def combine_heatmap_trfm(cls, proj_dir: str, overwrite: bool = False, tuning: bool = False) -> None:
        logger = init_logger_file()
        pfm = ProjFpModelTuning(proj_dir) if tuning else ProjFpModel(proj_dir)
        if not overwrite:
            for fp in (pfm.comb_heatmap,):
                if os.path.exists(fp):
                    return logger.warning(file_exists_msg(fp))
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
        """
        Running all visual check pipelines in order.
        """
        # Registration visual check
        cls.combine_reg(proj_dir, overwrite=overwrite)
        for is_tuning in [
            True,  # Tuning
            False,  # Final
        ]:
            # Cell counting visual checks
            cls.cellc_trim_to_final(proj_dir, overwrite=overwrite, tuning=is_tuning)
            cls.coords2points_raw(proj_dir, overwrite=overwrite, tuning=is_tuning)
            cls.combine_cellc(proj_dir, overwrite=overwrite, tuning=is_tuning)
            # Transformed space visual checks
            cls.coords2points_trfm(proj_dir, overwrite=overwrite, tuning=is_tuning)
            cls.coords2heatmap_trfm(proj_dir, overwrite=overwrite, tuning=is_tuning)
            cls.combine_heatmap_trfm(proj_dir, overwrite=overwrite, tuning=is_tuning)
