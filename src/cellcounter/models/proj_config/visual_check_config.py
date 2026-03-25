from pydantic import BaseModel, PositiveInt

from cellcounter.models.proj_config.dims_config import DimsSliceConfig, SliceConfig


class VisualCheckConfig(BaseModel):
    """Visual QC settings."""

    heatmap_raw_radius: PositiveInt = 5
    heatmap_trfm_radius: PositiveInt = 3
    cellcount_trim: DimsSliceConfig = DimsSliceConfig(
        z=SliceConfig(start=0, stop=10, step=None),
        y=SliceConfig(),
        x=SliceConfig(),
    )
