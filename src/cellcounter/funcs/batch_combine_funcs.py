"""Multi-experiment aggregation utilities.

Combines cell counting results from multiple specimens into a single
DataFrame with MultiIndex columns (specimen, measure).

Output structure:
- **Index**: Brain region IDs (from atlas annotation)
- **Columns**: MultiIndex with (specimen, measure) levels
    - "annotations" specimen contains region metadata (name, acronym, etc.)
    - Each project specimen contains cell measurements (count, volume, etc.)
"""

import logging
from pathlib import Path

import pandas as pd
from natsort import natsorted

from cellcounter.constants import (
    ANNOT_COLUMNS_FINAL,
    AnnotColumns,
    CellColumns,
    SpecialRegions,
)
from cellcounter.constants.annotations import CombinedColumns
from cellcounter.funcs.map_funcs import annot_df_get_parents, annot_fp2df
from cellcounter.models.fp_models.proj_fp import ProjFp
from cellcounter.utils.misc_utils import enum2list

logger = logging.getLogger(__name__)

COMBINED_FP = "combined_df"


def _build_annotation_base(pfm: ProjFp) -> pd.DataFrame:
    """Build annotation columns with special regions."""
    annot_df = annot_fp2df(pfm.map)
    annot_df = annot_df_get_parents(annot_df)
    annot_df.loc[-1] = pd.Series(
        {AnnotColumns.NAME.value: SpecialRegions.INVALID.value}
    )
    annot_df.loc[0] = pd.Series(
        {AnnotColumns.NAME.value: SpecialRegions.UNIVERSE.value}
    )
    annot_df = annot_df[ANNOT_COLUMNS_FINAL]
    return pd.concat(
        [annot_df],
        keys=["annotations"],
        names=[CombinedColumns.SPECIMEN.value],
        axis=1,
    )


def _get_reference_config(pfm: ProjFp) -> tuple:
    """Extract atlas reference config tuple for comparison."""
    return (
        pfm.config.reference.atlas_dir,
        pfm.config.reference.ref_version,
        pfm.config.reference.annot_version,
        pfm.config.reference.map_version,
    )


def _validate_project(pfm: ProjFp, reference_config: tuple) -> None:
    """Validate project has required files and matches reference atlas."""
    if not pfm.cells_agg_df.exists():
        msg = f"Missing cells_agg_df for {pfm.root_dir}"
        raise FileNotFoundError(msg)
    config = _get_reference_config(pfm)
    if config != reference_config:
        msg = (
            f"Atlas mismatch for {pfm.root_dir}.\n"
            f"Expected: {reference_config}\n"
            f"Got: {config}"
        )
        raise ValueError(msg)


def _load_cells_agg(pfm: ProjFp) -> pd.DataFrame:
    """Load cell aggregation data for a single project."""
    df = pd.read_parquet(pfm.cells_agg_df)
    return df[enum2list(CellColumns)]


def combine_projects(
    proj_dir_ls: list[Path | str],
    out_dir: Path | str,
    *,
    overwrite: bool = False,
) -> None:
    """Combine results from a list of project directories.

    Args:
        proj_dir_ls: List of project directory paths.
        out_dir: Output directory for combined results.
        overwrite: If True, overwrite existing output files.

    Raises:
        ValueError: If projects use different atlas references.
        FileNotFound: If required data files are missing.
    """
    if not proj_dir_ls:
        msg = "proj_dir_ls is empty"
        raise ValueError(msg)
    out_dir = Path(out_dir)
    out_parquet = out_dir / f"{COMBINED_FP}.parquet"
    out_csv = out_dir / f"{COMBINED_FP}.csv"
    if not overwrite and out_parquet.exists():
        logger.info("Skipping as %s already exists.", out_parquet)
        return
    pfm_ref = ProjFp(proj_dir_ls[0])
    reference_config = _get_reference_config(pfm_ref)
    for proj_dir in proj_dir_ls:
        _validate_project(ProjFp(proj_dir), reference_config)
    combined_df = _build_annotation_base(pfm_ref)
    for proj_dir_raw in proj_dir_ls:
        proj_dir = Path(proj_dir_raw)
        logger.info("Processing: %s", proj_dir.name)
        pfm = ProjFp(proj_dir)
        cells_df = _load_cells_agg(pfm)
        cells_df = pd.concat(
            [cells_df],
            keys=[proj_dir.name],
            names=[CombinedColumns.SPECIMEN.value],
            axis=1,
        )
        combined_df = combined_df.merge(
            cells_df,
            left_index=True,
            right_index=True,
            how="outer",
        )
    combined_df.columns = combined_df.columns.set_names(enum2list(CombinedColumns))
    combined_df.to_parquet(out_parquet)
    combined_df.to_csv(out_csv)
    logger.info("Saved combined results to %s", out_dir)


def discover_projects(root_dir: Path | str) -> list[Path]:
    """Discover valid project directories in a root directory.

    A valid project has a config file.

    Args:
        root_dir: Root directory to search for projects.

    Returns:
        List of project directory paths, naturally sorted.
    """
    root_dir = Path(root_dir)
    projects = []
    for entry in natsorted(root_dir.iterdir()):
        if entry.is_dir():
            pfm = ProjFp(entry)
            if pfm.config_fp.exists():
                projects.append(entry)
    return projects


def combine_root(
    root_dir: Path | str,
    out_dir: Path | str | None = None,
    *,
    overwrite: bool = False,
) -> None:
    """Combine results from all projects in a root directory.

    Automatically discovers project directories (those with config files)
    and combines their results.

    Args:
        root_dir: Root directory containing project subdirectories.
        out_dir: Output directory for combined results. Defaults to root_dir.
        overwrite: If True, overwrite existing output files.
    """
    root_dir = Path(root_dir)
    out_dir = Path(out_dir) if out_dir else root_dir
    proj_dir_ls = discover_projects(root_dir)
    if not proj_dir_ls:
        logger.warning("No projects found in %s", root_dir)
        return
    combine_projects(proj_dir_ls, out_dir, overwrite=overwrite)
