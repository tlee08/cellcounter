import numpy as np
import pandas as pd

from cellcounter.constants import (
    ANNOT_COLUMNS_TYPES,
    AnnotColumns,
    AnnotExtraColumns,
    SpecialRegions,
)
from cellcounter.utils.logging_utils import init_logger_file


class MapFuncs:
    logging = init_logger_file(__name__)

    @classmethod
    def annot_dict2df(cls, data_dict: dict) -> pd.DataFrame:
        """
        Recursively find the region information for all nested objects.

        Returns a new dataframe with index as the region ID
        and the columns:
        - ATLAS_ID
        - ONTOLOGY_ID
        - ACRONYM
        - NAME
        - COLOR_HEX_TRIPLET
        - GRAPH_ORDER
        - ST_LEVEL
        - HEMISPHERE_ID
        - PARENT_STRUCTURE_ID
        """

        def recursive_gen(annot_dict):
            # Initialising current region as a df with a single row
            curr_row = pd.DataFrame(
                {i.value: [annot_dict[i.value]] for i in AnnotColumns},
            )
            # Recursively concatenating all children
            return pd.concat(
                [
                    curr_row,
                    *[recursive_gen(i) for i in annot_dict[AnnotExtraColumns.CHILDREN.value]],
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

    @classmethod
    def annot_df_get_parents(cls, annot_df: pd.DataFrame) -> pd.DataFrame:
        """
        Get the parent region information for all regions
        in the annotation mappings dataframe.

        Returns a new dataframe with index as region ID,
        all original columns, and the new columns:
        - PARENT_ACRONYM

        NOTE: any root rows (i.e. no parent region) will have NaN in the PARENT_ACRONYM column.
        """
        # For each region (i.e. row), storing the parent region name in a column
        # by merging the annot_df on parent_structure_id
        # with the annot_df (as parent copy) on its own id
        annot_df = pd.merge(
            left=annot_df,
            right=annot_df.rename(
                columns={
                    AnnotColumns.ACRONYM.value: AnnotExtraColumns.PARENT_ACRONYM.value,
                }
            )[[AnnotExtraColumns.PARENT_ACRONYM.value]],
            left_on=AnnotColumns.PARENT_STRUCTURE_ID.value,
            right_index=True,
            # right_on=AnnotExtraColumns.PARENT_ID.value,
            how="left",
        )
        return annot_df

    @classmethod
    def annot_df_get_children(cls, annot_df: pd.DataFrame) -> pd.DataFrame:
        """
        Get the children region information for all regions
        in the annotation mappings dataframe.

        Returns a new dataframe with index as region ID
        and the columns:
        - NAME
        - ACRONYM
        - COLOR_HEX_TRIPLET
        - PARENT_STRUCTURE_ID
        - PARENT_ACRONYM
        - CHILDREN
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

    @classmethod
    def combine_nested_regions(cls, cells_agg_df: pd.DataFrame, annot_df: pd.DataFrame):
        """
        Combine (sum) children regions in their parent regions
        in the cells_agg dataframe.

        Done recursively.

        Returns a new dataframe with index as region ID,
        the annotation columns, and the
        same columns as the input `cells_agg_df` dataframe.

        Notes
        -----
        - The `annot_df` is the annotation mappings dataframe.
        - The `cells_agg` is the cells dataframe grouped by region ID (so ID is the index).
        """
        # Getting the sum column names (i.e. all columns in cells_agg_d)
        sum_cols = cells_agg_df.columns.to_list()
        # Getting annot_df with parent region information for all regions
        annot_df = cls.annot_df_get_parents(annot_df)
        # Merging the cells_agg df with the annot_df
        # NOTE: annot_df and cells_agg_df both have index as ID
        cells_agg_df = pd.merge(
            left=annot_df,
            right=cells_agg_df,
            left_index=True,
            right_index=True,
            how="outer",
        )
        # Getting annot_df with list of children for each region
        # NOTE: we do this after the merge, in case there are NaN's
        # in the cells_agg_df regions (e.g. name is NaN, no-label or universal)
        cells_agg_df = cls.annot_df_get_children(cells_agg_df)
        # Imputing all NaN values in the measure columns with 0
        cells_agg_df[sum_cols] = cells_agg_df[sum_cols].fillna(0)

        # Recursively summing the cells_agg_df columns with each child's and current value
        def recursive_sum(row_id):
            # NOTE: updates the cells_agg_df in place
            # BASE CASE: no children - use current values
            # REC CASE: has children - recursively sum children values + current values
            cells_agg_df.loc[row_id, sum_cols] += np.sum(
                [recursive_sum(child_id) for child_id in cells_agg_df.at[row_id, AnnotExtraColumns.CHILDREN.value]],
                axis=0,
            )
            return cells_agg_df.loc[row_id, sum_cols]

        # For each root row (i.e. nodes with no parent region),
        # recursively summing (i.e. top-down recursive approach)
        [
            recursive_sum(row_id)
            for row_id in cells_agg_df[cells_agg_df[AnnotColumns.PARENT_STRUCTURE_ID.value].isna()].index
        ]
        # Removing unnecessary columns (AnnotExtraColumns.CHILDREN.value column)
        cells_agg_df = cells_agg_df.drop(columns=[AnnotExtraColumns.CHILDREN.value])
        return cells_agg_df

    @classmethod
    def annot_df2dict(cls, annot_df: pd.DataFrame) -> list:
        """
        Converts an annotation DataFrame into a nested dictionary structure.
        This function takes an annotation DataFrame, adds a list of children to each region,
        and then recursively converts the DataFrame into a nested dictionary format. Each
        dictionary represents a region and contains its information along with its children.

        NOTE: the outputted dict is actually a list as there
        can be multiple roots.
        """
        # Adding children list to each region
        annot_df = cls.annot_df_get_children(annot_df)

        # Converting to dict
        def recursive_gen(i):
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
        tree = [recursive_gen(i) for i in annot_df[annot_df[AnnotColumns.PARENT_STRUCTURE_ID.value].isna()].index]
        return tree

    @classmethod
    def df_map_ids(cls, cells_df: pd.DataFrame, annot_df: pd.DataFrame) -> pd.DataFrame:
        # Getting the annotation name for every cell (zyx coord)
        # Left-joining the cells dataframe with the annotation mappings dataframe
        cells_df = pd.merge(
            left=cells_df,
            right=annot_df,
            how="left",
            on=AnnotColumns.ID.value,
        )
        # Including special ids
        cells_df = cls.df_include_special_ids(cells_df)
        return cells_df

    @classmethod
    def df_include_special_ids(cls, cells_df: pd.DataFrame) -> pd.DataFrame:
        cells_df = cells_df.copy()
        id_col = AnnotColumns.ID.value
        name_col = AnnotColumns.NAME.value
        # Setting points with ID == -1 as "invalid" label
        cells_df.loc[cells_df[id_col] == -1, name_col] = SpecialRegions.INVALID.value
        # Setting points with ID == 0 as "universe" label
        cells_df.loc[cells_df[id_col] == 0, name_col] = SpecialRegions.UNIVERSE.value
        # Setting points with no region map name (but have a positive ID value) as "no label" label
        cells_df.loc[cells_df[name_col].isna(), name_col] = SpecialRegions.NO_LABEL.value
        return cells_df
