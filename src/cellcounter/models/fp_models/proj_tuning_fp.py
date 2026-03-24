import logging
from pathlib import Path

from cellcounter.models.fp_models.proj_fp import ProjFp

logger = logging.getLogger(__name__)


class ProjTuningFp(ProjFp):
    """Project filepath TUNING model."""

    def __init__(self, root_dir: Path | str) -> None:
        """Project filepath TUNING model."""
        super().__init__(root_dir)
        self.raw_sdir = f"{self.raw_sdir}_tuning"
        self.cellcount_sdir = f"{self.cellcount_sdir}_tuning"
        self.analysis_sdir = f"{self.analysis_sdir}_tuning"
        self.visual_sdir = f"{self.visual_sdir}_tuning"
