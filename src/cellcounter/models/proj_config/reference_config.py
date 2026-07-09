from pathlib import Path

from pydantic import BaseModel

from cellcounter.constants import (
    ABA_ANNOTATIONS,
    ATLAS_DIR,
    AVERAGE_TEMPLATE_25,
    CCF_2016_25,
)


class ReferenceConfig(BaseModel):
    """Reference atlas and annotation settings."""

    atlas_dir: Path = ATLAS_DIR
    ref_version: str = AVERAGE_TEMPLATE_25
    annot_version: str = CCF_2016_25
    map_version: str = ABA_ANNOTATIONS
