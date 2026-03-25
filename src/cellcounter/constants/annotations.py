from enum import Enum

import numpy as np


class AnnotColumns(Enum):
    """Atlas annotation column names."""

    ID = "id"
    ATLAS_ID = "atlas_id"
    ONTOLOGY_ID = "ontology_id"
    ACRONYM = "acronym"
    NAME = "name"
    COLOR_HEX_TRIPLET = "color_hex_triplet"
    GRAPH_ORDER = "graph_order"
    ST_LEVEL = "st_level"
    HEMISPHERE_ID = "hemisphere_id"
    PARENT_STRUCTURE_ID = "parent_structure_id"


class AnnotExtraColumns(Enum):
    """Other annotation columns derived from the Atlas annotations."""

    PARENT_ID = "parent_id"
    PARENT_ACRONYM = "parent_acronym"
    CHILDREN = "children"


ANNOT_COLUMNS_TYPES = {
    AnnotColumns.ID.value: np.float64,
    AnnotColumns.ATLAS_ID.value: np.float64,
    AnnotColumns.ONTOLOGY_ID.value: np.float64,
    AnnotColumns.ACRONYM.value: str,
    AnnotColumns.NAME.value: str,
    AnnotColumns.COLOR_HEX_TRIPLET.value: str,
    AnnotColumns.GRAPH_ORDER.value: np.float64,
    AnnotColumns.ST_LEVEL.value: np.float64,
    AnnotColumns.HEMISPHERE_ID.value: np.float64,
    AnnotColumns.PARENT_STRUCTURE_ID.value: np.float64,
}

ANNOT_COLUMNS_FINAL = [
    AnnotColumns.NAME.value,
    AnnotColumns.ACRONYM.value,
    AnnotColumns.COLOR_HEX_TRIPLET.value,
    AnnotColumns.PARENT_STRUCTURE_ID.value,
    AnnotExtraColumns.PARENT_ACRONYM.value,
]


class SpecialRegions(Enum):
    """Our own special regions (outside brain)."""

    INVALID = "invalid"
    UNIVERSE = "universe"
    NO_LABEL = "no_label"


class CombinedColumns(Enum):
    """Combined columns."""

    SPECIMEN = "specimen"
    MEASURE = "measure"
