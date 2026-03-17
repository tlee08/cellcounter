import os
from enum import Enum

import pandas as pd
from natsort import natsorted

from cellcounter.constants import (
    ANNOT_COLUMNS_FINAL,
    AnnotColumns,
    CellColumns,
    MaskColumns,
    SpecialRegions,
)
from cellcounter.funcs.map_funcs import (
    MapFuncs,
)
from cellcounter.pipeline.pipeline import Pipeline
from cellcounter.utils.config_params_model import ConfigParamsModel
from cellcounter.utils.io_utils import read_json, sanitise_smb_df
from cellcounter.utils.logging_utils import init_logger
from cellcounter.utils.misc_utils import enum2list
from cellcounter.utils.proj_org_utils import ProjFpModel

# TODO: move to pipeline and refactor

COMBINED_FP = "combined_df"


class CombinedColumns(Enum):
    SPECIMEN = "specimen"
    MEASURE = "measure"


class BatchCombineFuncs:
    logger = init_logger(__name__)

    @classmethod
    def combine_ls_pipeline(
        cls,
        proj_dir_ls: list,
        out_dir: str,
        overwrite: bool = False,
    ):
        # Checking should overwrite
        # If overwrite is False and out_fp_parquet exists, then skip
        out_fp_parquet = os.path.join(out_dir, f"{COMBINED_FP}.parquet")
        out_fp_csv = os.path.join(out_dir, f"{COMBINED_FP}.csv")
        if not overwrite and os.path.exists(out_fp_parquet):
            cls.logger.info(f"Skipping as {out_fp_parquet} already exists.")
            return

        # Asserting that proj_dir_ls is not empty
        assert len(proj_dir_ls) > 0, "proj_dir_ls is empty"
        # Assertions for each project directory
        configs0 = Pipeline.update_configs(proj_dir_ls[0])
        atlas_dir0 = configs0.atlas_dir
        ref_v0 = configs0.ref_version
        annot_v0 = configs0.annot_version
        map_v0 = configs0.map_version
        for proj_dir in proj_dir_ls:
            # Getting project info
            name = os.path.basename(proj_dir)
            pfm = ProjFpModel(proj_dir)
            configs = Pipeline.update_configs(proj_dir)
            atlas_dir = configs.atlas_dir
            ref_v = configs.ref_version
            annot_v = configs.annot_version
            map_v = configs.map_version
            # Asserting that all projects have cells_agg and mask_df files
            assert os.path.exists(pfm.cells_agg_df), f"Missing cells_agg_df for {name}"
            assert os.path.exists(pfm.mask_df), f"Missing mask_df for {name}"
            # Asserting that all projects are using the same origin for reference atlas
            # to verify the same regions are being used
            assert atlas_dir0 == atlas_dir, (
                f"In configs file, there mismatch for {name} and {name}.\n"
                f'atlas_dir values "{atlas_dir0}" and "{atlas_dir}" are not equal'
            )
            assert ref_v0 == ref_v, (
                f"In configs file, there mismatch for {name} and {name}.\n"
                f'ref_v values "{ref_v0}" and "{ref_v}" are not equal'
            )
            assert annot_v0 == annot_v, (
                f"In configs file, there mismatch for {name} and {name}.\n"
                f'annot_v values "{annot_v0}" and "{annot_v}" are not equal'
            )
            assert map_v0 == map_v, (
                f"In configs file, there mismatch for {name} and {name}.\n"
                f'map_v values "{map_v0}" and "{map_v}" are not equal'
            )
        pfm0 = ProjFpModel(proj_dir_ls[0])
        # Making combined_agg_df
        # Starting with annot_df (asserted all the same so using first)
        total_df = MapFuncs.annot_dict2df(read_json(pfm0.map))
        # Adding parent columns to annot_df
        total_df = MapFuncs.annot_df_get_parents(total_df)
        # Adding special rows (e.g. "universe")
        # TODO: is neither clean nor modular
        total_df.loc[-1] = pd.Series({AnnotColumns.NAME.value: SpecialRegions.INVALID.value})
        total_df.loc[0] = pd.Series({AnnotColumns.NAME.value: SpecialRegions.UNIVERSE.value})
        # total_df.loc[np.nan] = pd.Series(
        #     {AnnotColumns.NAME.value: SpecialRegions.NO_LABEL.value}
        # )
        # total_df = df_include_special_ids(total_df)
        # Keeping only the required columns
        total_df = total_df[ANNOT_COLUMNS_FINAL]
        # Making columns a multindex with levels ("annotations", annot columns)
        total_df = pd.concat(
            [total_df],
            keys=["annotations"],
            names=[CombinedColumns.SPECIMEN.value],
            axis=1,
        )
        # Get all experiments
        for proj_dir in proj_dir_ls:
            # Logging which file is being processed
            name = os.path.basename(proj_dir)
            cls.logger.info(f"Running: {name}")
            # Filenames
            pfm = ProjFpModel(proj_dir)
            # CELL_AGG_DF
            # Reading experiment's cells_agg dataframe
            cells_agg_df = pd.read_parquet(pfm.cells_agg_df)
            # Sanitising (removing smb columns)
            cells_agg_df = sanitise_smb_df(cells_agg_df)
            # Keeping only the required columns (not annot columns)
            cells_agg_df = cells_agg_df[enum2list(CellColumns)]
            # MASK_DF
            # Reading experiment's mask_counts dataframe
            mask_df = pd.read_parquet(pfm.mask_df)
            # Keeping only the required columns
            mask_df = mask_df[enum2list(MaskColumns)]
            # Merging cells_agg_df with mask_df to combine columns
            exp_df = pd.merge(
                left=cells_agg_df,
                right=mask_df,
                left_index=True,
                right_index=True,
                how="outer",
            )
            # Making columns a multindex with levels (specimen name, cell agg columns)
            exp_df = pd.concat(
                [exp_df],
                keys=[name],
                names=[CombinedColumns.SPECIMEN.value],
                axis=1,
            )
            # Merging with comb_agg_df (ID is index for both dfs)
            total_df = pd.merge(
                left=total_df,
                right=exp_df,
                left_index=True,
                right_index=True,
                how="outer",
            )
        # Setting column MultiIndex's level names
        total_df.columns = total_df.columns.set_names(enum2list(CombinedColumns))
        # Saving to disk (parquet and csv)
        total_df.to_parquet(out_fp_parquet)
        total_df.to_csv(out_fp_csv)

    @classmethod
    def combine_root_pipeline(
        cls,
        root_dir: str,
        out_dir: str,
        overwrite: bool = False,
    ):
        # Get all experiments in root_dir (any dir with a configs file)
        # NOTE: not using the cells_agg and mask file to check for valid projects
        # so we can catch any projects that are missing these files
        proj_dir_ls = []
        for exp in natsorted(os.listdir(root_dir)):
            proj_dir = os.path.join(root_dir, exp)
            pfm = ProjFpModel(proj_dir)
            try:
                # If proj has config_params file, then add to list of projs to combine
                ConfigParamsModel.model_validate(read_json(pfm.config_params))
                proj_dir_ls.append(proj_dir)
            except FileNotFoundError:
                pass
        # Running combine pipeline
        cls.combine_ls_pipeline(proj_dir_ls, out_dir, overwrite)
