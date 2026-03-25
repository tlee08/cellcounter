"""Atlas region mapping and cell extraction utilities.

Handles:
- Parsing Allen Brain Atlas annotation JSON to DataFrame
- Building parent/child region hierarchies
- Recursive region aggregation for nested structures
- Mapping cell coordinates to region IDs
- Extracting cell data from watershed segmentation
"""

import json
import logging
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd

from cellcounter.constants import (
    ANNOT_COLUMNS_TYPES,
    CELL_IDX_NAME,
    AnnotColumns,
    AnnotExtraColumns,
    CellColumns,
    Coords,
    SpecialRegions,
)

logger = logging.getLogger(__name__)


def annot_fp2df(fp: Path | str) -> pd.DataFrame:
    """Read annotation JSON file and convert to DataFrame.

    Args:
        fp: Path to Allen Brain Atlas annotation JSON file.

    Returns:
        DataFrame with region ID as index and region metadata as columns.
    """
    fp = Path(fp)
    with fp.open(mode="r") as f:
        content = json.load(f)
    annot_df = annot_dict2df(content)
    return annot_df


def annot_dict2df(data_dict: dict) -> pd.DataFrame:
    """Recursively parse Allen Brain Atlas JSON to DataFrame.

    Traverses the nested region hierarchy and extracts metadata for each region.

    Args:
        data_dict: Parsed JSON from Allen Brain Atlas annotation file.

    Returns:
        DataFrame with region ID as index and columns: ATLAS_ID, ONTOLOGY_ID,
        ACRONYM, NAME, COLOR_HEX_TRIPLET, GRAPH_ORDER, ST_LEVEL,
        HEMISPHERE_ID, PARENT_STRUCTURE_ID.
    """

    def recursive_gen(annot_dict: dict) -> pd.DataFrame:
        # Initialising current region as a df with a single row
        curr_row = pd.DataFrame(
            {i.value: [annot_dict[i.value]] for i in AnnotColumns},
        )
        # Recursively concatenating all children
        return pd.concat(
            [
                curr_row,
                *[
                    recursive_gen(i)
                    for i in annot_dict[AnnotExtraColumns.CHILDREN.value]
                ],
            ]
        )

    # Recursively get the region information for all children from roots
    # using function defined in this block
    annot_df = pd.concat([recursive_gen(i) for i in data_dict["msg"]])
    # Cast columns to given types
    for k, v in ANNOT_COLUMNS_TYPES.items():
        annot_df[k] = annot_df[k].astype(v)
    # Set region ID as index
    annot_df = annot_df.set_index(AnnotColumns.ID.value)
    return annot_df


def annot_df_get_parents(annot_df: pd.DataFrame) -> pd.DataFrame:
    """Add parent region acronym column to annotation DataFrame.

    Args:
        annot_df: Annotation DataFrame with region information.

    Returns:
        DataFrame with new PARENT_ACRONYM column. Root regions have NaN.
    """
    # For each region (i.e. row), storing the parent region name in a column
    # by merging the annot_df on parent_structure_id
    # with the annot_df (as parent copy) on its own id
    annot_df = annot_df.merge(
        right=annot_df.rename(
            columns={
                AnnotColumns.ACRONYM.value: AnnotExtraColumns.PARENT_ACRONYM.value,
            }
        )[[AnnotExtraColumns.PARENT_ACRONYM.value]],
        left_on=AnnotColumns.PARENT_STRUCTURE_ID.value,
        right_index=True,
        how="left",
    )
    return annot_df


def annot_df_get_children(annot_df: pd.DataFrame) -> pd.DataFrame:
    """Add children region ID list column to annotation DataFrame.

    Args:
        annot_df: Annotation DataFrame with region information.

    Returns:
        DataFrame with new CHILDREN column containing list of child region IDs.
    """
    # Making copy
    annot_df = annot_df.copy()
    # Making a children list column in cells_agg
    annot_df[AnnotExtraColumns.CHILDREN.value] = [[] for i in range(annot_df.shape[0])]
    # For each row (i.e. region), adding the current row ID to the parent's (by ID)
    # children column list
    for i in annot_df.index:
        i_parent = annot_df.loc[i, AnnotColumns.PARENT_STRUCTURE_ID.value]
        if not pd.isna(i_parent):
            annot_df.at[i_parent, AnnotExtraColumns.CHILDREN.value].append(i)
    return annot_df


def combine_nested_regions(
    cells_agg_df: pd.DataFrame, annot_df: pd.DataFrame
) -> pd.DataFrame:
    """Recursively aggregate child region counts into parent regions.

    Args:
        cells_agg_df: DataFrame of cells grouped by region ID.
        annot_df: Annotation DataFrame with region hierarchy.

    Returns:
        DataFrame with aggregated counts propagated up the region hierarchy.
    """
    # Getting the sum column names (i.e. all columns in cells_agg_d)
    sum_cols = cells_agg_df.columns.to_list()
    # Getting annot_df with parent region information for all regions
    annot_df = annot_df_get_parents(annot_df)
    # Merging the cells_agg df with the annot_df
    # NOTE: annot_df and cells_agg_df both have index as ID
    cells_agg_df = annot_df.merge(
        right=cells_agg_df,
        left_index=True,
        right_index=True,
        how="outer",
    )
    # Getting annot_df with list of children for each region
    # NOTE: we do this after the merge, in case there are NaN's
    # in the cells_agg_df regions (e.g. name is NaN, no-label or universal)
    cells_agg_df = annot_df_get_children(cells_agg_df)
    # Imputing all NaN values in the measure columns with 0
    cells_agg_df[sum_cols] = cells_agg_df[sum_cols].fillna(0)

    # Recursively summing the cells_agg_df columns with each child's and current value
    def recursive_sum(row_id: int) -> pd.DataFrame:
        # NOTE: updates the cells_agg_df in place
        # BASE CASE: no children - use current values
        # REC CASE: has children - recursively sum children values + current values
        cells_agg_df.loc[row_id, sum_cols] += np.sum(
            [
                recursive_sum(child_id)
                for child_id in cells_agg_df.at[
                    row_id, AnnotExtraColumns.CHILDREN.value
                ]
            ],
            axis=0,
        )
        return cells_agg_df.loc[row_id, sum_cols]

    # For each root row (i.e. nodes with no parent region),
    # recursively summing (i.e. top-down recursive approach)
    [
        recursive_sum(row_id)
        for row_id in cells_agg_df[
            cells_agg_df[AnnotColumns.PARENT_STRUCTURE_ID.value].isna()
        ].index
    ]
    # Removing unnecessary columns (AnnotExtraColumns.CHILDREN.value column)
    cells_agg_df = cells_agg_df.drop(columns=[AnnotExtraColumns.CHILDREN.value])
    return cells_agg_df


@classmethod
def annot_df2dict(annot_df: pd.DataFrame) -> list:
    """Convert annotation DataFrame to nested dictionary structure.

    Each dictionary represents a region with its metadata and children.

    Args:
        annot_df: Annotation DataFrame with region information.

    Returns:
        List of root region dictionaries (multiple roots possible).
    """
    # Adding children list to each region
    annot_df = annot_df_get_children(annot_df)

    # Converting to dict
    def recursive_gen(i: int) -> dict:
        # Storing info of current region (i.e. row) in dict
        tree = annot_df.loc[i].to_dict()
        # RECURSIVE CASE: has children - recursively get children info
        # NOTE: also covers base case (no children)
        tree[AnnotExtraColumns.CHILDREN.value] = [
            recursive_gen(j) for j in annot_df.at[i, AnnotExtraColumns.CHILDREN.value]
        ]
        return tree

    # For each root (i.e. nodes with no parent region),
    # recursively storing (i.e. top-down recursive approach)
    tree = [
        recursive_gen(i)
        for i in annot_df[annot_df[AnnotColumns.PARENT_STRUCTURE_ID.value].isna()].index
    ]
    return tree


@classmethod
def df_map_ids(cells_df: pd.DataFrame, annot_df: pd.DataFrame) -> pd.DataFrame:
    """Map region IDs to annotation metadata (name, acronym, etc.).

    Args:
        cells_df: DataFrame of cells with region ID column.
        annot_df: Annotation DataFrame with region information.

    Returns:
        DataFrame with annotation columns added (left join on ID).
    """
    # Getting the annotation name for every cell (zyx coord)
    # Left-joining the cells dataframe with the annotation mappings dataframe
    cells_df = cells_df.merge(
        right=annot_df,
        how="left",
        on=AnnotColumns.ID.value,
    )
    # Including special ids
    cells_df = df_include_special_ids(cells_df)
    return cells_df


@classmethod
def df_include_special_ids(cells_df: pd.DataFrame) -> pd.DataFrame:
    """Add special region labels for invalid/unlabeled coordinates.

    Assigns special names for IDs: -1=invalid, 0=universe, NaN=no_label.

    Args:
        cells_df: DataFrame of cells with region ID and name columns.

    Returns:
        DataFrame with special region names filled in.
    """
    cells_df = cells_df.copy()
    id_col = AnnotColumns.ID.value
    name_col = AnnotColumns.NAME.value
    # Setting points with ID == -1 as "invalid" label
    cells_df.loc[cells_df[id_col] == -1, name_col] = SpecialRegions.INVALID.value
    # Setting points with ID == 0 as "universe" label
    cells_df.loc[cells_df[id_col] == 0, name_col] = SpecialRegions.UNIVERSE.value
    # Setting points with no region map name
    # but have a positive ID value as "no label" label
    cells_df.loc[cells_df[name_col].isna(), name_col] = SpecialRegions.NO_LABEL.value
    return cells_df


def _to_numpy(arr: npt.NDArray) -> npt.NDArray:
    """Convert cupy array to numpy, pass through numpy arrays."""
    return arr.get() if hasattr(arr, "get") else arr


def get_cells(
    raw_block: npt.NDArray,
    maxima_labels_block: npt.NDArray,
    wshed_labels_block: npt.NDArray,
    wshed_filt_block: npt.NDArray,
    z_offset: int,
    y_offset: int,
    x_offset: int,
    xp,
) -> pd.DataFrame:
    """Extract cells from watershed segmentation to DataFrame.

    Args:
        raw_block: Raw intensity values.
        maxima_labels_block: Labels for each maxima point.
        wshed_labels_block: Watershed segmentation labels.
        wshed_filt_block: Filtered watershed volumes.
        z_offset: Z-axis offset for global coordinates.
        y_offset: Y-axis offset for global coordinates.
        x_offset: X-axis offset for global coordinates.
        xp: Array backend (numpy or cupy).

    Returns:
        DataFrame with cell coordinates, volumes, and intensities.
    """
    assert raw_block.shape == maxima_labels_block.shape
    assert raw_block.shape == wshed_filt_block.shape

    raw = xp.asarray(raw_block)
    maxima_labels = xp.asarray(maxima_labels_block)
    wshed_labels = xp.asarray(wshed_labels_block)
    wshed_filt = xp.asarray(wshed_filt_block)
    mask = wshed_filt > 0

    # Get maxima positions (first occurrence of each label)
    labels, coords_flat = xp.unique(maxima_labels[mask], return_index=True)
    z, y, x = xp.unravel_index(coords_flat, maxima_labels.shape)

    # Build cells DataFrame with offset coordinates
    cells_df = pd.DataFrame(
        {
            Coords.Z.value: _to_numpy(z) + z_offset,
            Coords.Y.value: _to_numpy(y) + y_offset,
            Coords.X.value: _to_numpy(x) + x_offset,
        },
        index=pd.Index(_to_numpy(labels).astype(np.uint32), name=CELL_IDX_NAME),
    ).astype(np.uint16)

    if cells_df.shape[0] > 0:
        # Remove background label
        cells_df = cells_df.drop(index=0, errors="ignore")

    cells_df[CellColumns.COUNT.value] = 1

    # Get volume at each maxima position
    cells_df[CellColumns.VOLUME.value] = _to_numpy(
        wshed_filt[
            cells_df[Coords.Z.value] - z_offset,
            cells_df[Coords.Y.value] - y_offset,
            cells_df[Coords.X.value] - x_offset,
        ]
    )

    # Filter out cells with 0 volume
    cells_df = cells_df[cells_df[CellColumns.VOLUME.value] > 0]

    # Compute summed intensities per watershed label
    if cells_df.empty:
        cells_df[CellColumns.SUM_INTENSITY.value] = 0.0
        return cells_df

    fg_labels, inverse = xp.unique(wshed_labels[mask].ravel(), return_inverse=True)
    intensities = xp.bincount(inverse, weights=raw[mask].ravel())
    label_to_intensity = dict(
        zip(_to_numpy(fg_labels), _to_numpy(intensities), strict=True)
    )
    cells_df[CellColumns.SUM_INTENSITY.value] = cells_df.index.map(
        label_to_intensity.get
    ).fillna(0.0)

    return cells_df
