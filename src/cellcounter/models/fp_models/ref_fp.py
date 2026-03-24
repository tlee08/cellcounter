import logging
from pathlib import Path

from cellcounter.models.fp_models.abstract_fp import AbstractFp

logger = logging.getLogger(__name__)


class RefFp(AbstractFp):
    """Refernce filepath model."""

    def __init__(
        self,
        root_dir: Path | str,
        ref_version: str,
        annot_version: str,
        map_version: str,
    ) -> None:
        """Reference filepath model.

        Notes:
        ------
        atlas_rsc_dir = "/home/linux1/Desktop/iDISCO/resources/atlas_resources/".

        Atlas from https://download.alleninstitute.org/informatics-archive/current-release/mouse_ccf/
        """
        # Storing base attributes
        # and wrapping in FpAttr to reset filepath attributes whenever they are changed
        self.root_dir = Path(root_dir)
        self.ref_version = ref_version
        self.annot_version = annot_version
        self.map_version = map_version
        # Constant subdirectory names
        self.reference_sdir = "reference"
        self.annotation_sdir = "annotation"
        self.mapping_sdir = "region_mapping"
        self.elastix_sdir = "elastix_params"

    @property
    def subdirs_ls(self) -> list[str]:
        """Return list of subdirectories."""
        return [
            self.reference_sdir,
            self.annotation_sdir,
            self.mapping_sdir,
            self.elastix_sdir,
        ]

    @property
    def ref(self) -> Path:
        """Ref."""
        return self.root_dir / self.reference_sdir / f"{self.ref_version}.tif"

    @property
    def annot(self) -> Path:
        """Annot."""
        return self.root_dir / self.annotation_sdir / f"{self.annot_version}.tif"

    @property
    def map(self) -> Path:
        """Map."""
        return self.root_dir / self.mapping_sdir / f"{self.map_version}.json"

    @property
    def affine(self) -> Path:
        """Affine."""
        return self.root_dir / self.elastix_sdir / "align_affine.txt"

    @property
    def bspline(self) -> Path:
        """Bspline."""
        return self.root_dir / self.elastix_sdir / "align_bspline.txt"
