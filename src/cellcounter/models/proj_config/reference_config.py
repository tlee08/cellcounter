from pathlib import Path

from pydantic import BaseModel

from cellcounter.constants import ATLAS_DIR, AnnotVersions, MapVersions, RefVersions


class ReferenceConfig(BaseModel):
    """Reference atlas and annotation settings."""

    atlas_dir: Path = ATLAS_DIR
    ref_version: str = RefVersions.AVERAGE_TEMPLATE_25.value
    annot_version: str = AnnotVersions.CCF_2016_25.value
    map_version: str = MapVersions.ABA_ANNOTATIONS.value
