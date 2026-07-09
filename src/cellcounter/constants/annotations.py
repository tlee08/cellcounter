import numpy as np

# ===============================================
# Annotation Columns
# ===============================================

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

ANNOTATED_COLUMNS = [
    ID,
    ATLAS_ID,
    ONTOLOGY_ID,
    ACRONYM,
    NAME,
    COLOR_HEX_TRIPLET,
    GRAPH_ORDER,
    ST_LEVEL,
    HEMISPHERE_ID,
    PARENT_STRUCTURE_ID,
]


# Annotation Extra Columns
PARENT_ID = "parent_id"
PARENT_ACRONYM = "parent_acronym"
CHILDREN = "children"


ANNOTATED_COLUMNS_TYPES = {
    ID: np.float64,
    ATLAS_ID: np.float64,
    ONTOLOGY_ID: np.float64,
    ACRONYM: str,
    NAME: str,
    COLOR_HEX_TRIPLET: str,
    GRAPH_ORDER: np.float64,
    ST_LEVEL: np.float64,
    HEMISPHERE_ID: np.float64,
    PARENT_STRUCTURE_ID: np.float64,
}

ANNOTATED_COLUMNS_FINAL = [
    NAME,
    ACRONYM,
    COLOR_HEX_TRIPLET,
    PARENT_STRUCTURE_ID,
    PARENT_ACRONYM,
]

# ===============================================
# Special Regions
# ===============================================


INVALID = "invalid"
UNIVERSE = "universe"
NO_LABEL = "no_label"

SPECIAL_REGIONS = [INVALID, UNIVERSE, NO_LABEL]

# ===============================================
# Combined Columns
# ===============================================


SPECIMEN = "specimen"
MEASURE = "measure"

COMBINED_COLUMNS = [SPECIMEN, MEASURE]
