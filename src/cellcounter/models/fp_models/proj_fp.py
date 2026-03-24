import functools
import logging
from pathlib import Path

from cellcounter.models.fp_models.abstract_fp import AbstractFp
from cellcounter.models.proj_config import ProjConfig

logger = logging.getLogger(__name__)


class ProjFp(AbstractFp):
    """Project filepath model."""

    def __init__(self, root_dir: Path | str) -> None:
        """Project filepath model."""
        self.root_dir = Path(root_dir)
        self.raw_sdir = "raw"
        self.registration_sdir = "registration"
        self.mask_sdir = "mask"
        self.cellcount_sdir = "cellcount"
        self.analysis_sdir = "analysis"
        self.visual_sdir = "visual"

    @property
    def subdirs_ls(self) -> list[str]:
        """Return list of subdirectories."""
        return [
            self.raw_sdir,
            self.registration_sdir,
            self.mask_sdir,
            self.cellcount_sdir,
            self.analysis_sdir,
            self.visual_sdir,
        ]

    @property
    def config_fp(self) -> Path:
        """config_params."""
        return self.root_dir / "config_params.json"

    @functools.cached_property
    def config(self) -> ProjConfig:
        """Cached project configuration."""
        return ProjConfig.read_file(self.config_fp)

    @property
    def diagnostics(self) -> Path:
        """Diagnostics."""
        return self.root_dir / "diagnostics.csv"

    @property
    def raw(self) -> Path:
        """Raw."""
        return self.root_dir / self.raw_sdir / "raw.zarr"

    @property
    def ref(self) -> Path:
        """Ref."""
        return self.root_dir / self.registration_sdir / "0a_reference.tif"

    @property
    def annot(self) -> Path:
        """Annot."""
        return self.root_dir / self.registration_sdir / "0b_annotation.tif"

    @property
    def map(self) -> Path:
        """Map."""
        return self.root_dir / self.registration_sdir / "0c_mapping.json"

    @property
    def affine(self) -> Path:
        """Affine."""
        return self.root_dir / self.registration_sdir / "0d_align_affine.txt"

    @property
    def bspline(self) -> Path:
        """Bspline."""
        return self.root_dir / self.registration_sdir / "0e_align_bspline.txt"

    @property
    def downsmpl1(self) -> Path:
        """Downsmpl1."""
        return self.root_dir / self.registration_sdir / "1_downsmpl1.tif"

    @property
    def downsmpl2(self) -> Path:
        """Downsmpl2."""
        return self.root_dir / self.registration_sdir / "2_downsmpl2.tif"

    @property
    def trimmed(self) -> Path:
        """Trimmed."""
        return self.root_dir / self.registration_sdir / "3_trimmed.tif"

    @property
    def bounded(self) -> Path:
        """Bounded."""
        return self.root_dir / self.registration_sdir / "4_bounded.tif"

    @property
    def regresult(self) -> Path:
        """Regresult."""
        return self.root_dir / self.registration_sdir / "5_regresult.tif"

    @property
    def premask_blur(self) -> Path:
        """Premask_blur."""
        return self.root_dir / self.mask_sdir / "1_premask_blur.tif"

    @property
    def mask_fill(self) -> Path:
        """Mask_fill."""
        return self.root_dir / self.mask_sdir / "2_mask_trimmed.tif"

    @property
    def mask_outline(self) -> Path:
        """Mask_outline."""
        return self.root_dir / self.mask_sdir / "3_outline_reg.tif"

    @property
    def mask_reg(self) -> Path:
        """Mask_reg."""
        return self.root_dir / self.mask_sdir / "4_mask_reg.tif"

    @property
    def mask_df(self) -> Path:
        """Mask_df."""
        return self.root_dir / self.mask_sdir / "5_mask.parquet"

    @property
    def bgrm(self) -> Path:
        """Bgrm."""
        return self.root_dir / self.cellcount_sdir / "1_bgrm.zarr"

    @property
    def dog(self) -> Path:
        """Dog."""
        return self.root_dir / self.cellcount_sdir / "2_dog.zarr"

    @property
    def adaptv(self) -> Path:
        """Adaptv."""
        return self.root_dir / self.cellcount_sdir / "3_adaptv.zarr"

    @property
    def threshd(self) -> Path:
        """Threshd."""
        return self.root_dir / self.cellcount_sdir / "4_threshd.zarr"

    @property
    def threshd_labels(self) -> Path:
        """Threshd_labels."""
        return self.root_dir / self.cellcount_sdir / "5_threshd_labels.zarr"

    @property
    def threshd_volumes(self) -> Path:
        """Threshd_volumes."""
        return self.root_dir / self.cellcount_sdir / "6_threshd_volumes.zarr"

    @property
    def threshd_filt(self) -> Path:
        """Threshd_filt."""
        return self.root_dir / self.cellcount_sdir / "7_threshd_filt.zarr"

    @property
    def maxima(self) -> Path:
        """Maxima."""
        return self.root_dir / self.cellcount_sdir / "8_maxima.zarr"

    @property
    def maxima_labels(self) -> Path:
        """Maxima_labels."""
        return self.root_dir / self.cellcount_sdir / "9_maxima_labels.zarr"

    @property
    def wshed_labels(self) -> Path:
        """Wshed_labels."""
        return self.root_dir / self.cellcount_sdir / "10_wshed_labels.zarr"

    @property
    def wshed_volumes(self) -> Path:
        """Wshed_volumes."""
        return self.root_dir / self.cellcount_sdir / "11_wshed_volumes.zarr"

    @property
    def wshed_filt(self) -> Path:
        """Wshed_filt."""
        return self.root_dir / self.cellcount_sdir / "12_wshed_filt.zarr"

    @property
    def cells_raw_df(self) -> Path:
        """Cells_raw_df."""
        return self.root_dir / self.analysis_sdir / "1_cells_raw.parquet"

    @property
    def cells_trfm_df(self) -> Path:
        """Cells_trfm_df."""
        return self.root_dir / self.analysis_sdir / "2_cells_trfm.parquet"

    @property
    def cells_df(self) -> Path:
        """Cells_df."""
        return self.root_dir / self.analysis_sdir / "3_cells.parquet"

    @property
    def cells_agg_df(self) -> Path:
        """Cells_agg_df."""
        return self.root_dir / self.analysis_sdir / "4_cells_agg.parquet"

    @property
    def cells_agg_csv(self) -> Path:
        """Cells_agg_csv."""
        return self.root_dir / self.analysis_sdir / "5_cells_agg.csv"

    @property
    def threshd_final(self) -> Path:
        """Threshd_final."""
        return self.root_dir / self.visual_sdir / "threshd.zarr"

    @property
    def maxima_final(self) -> Path:
        """Maxima_final."""
        return self.root_dir / self.visual_sdir / "maxima.zarr"

    @property
    def wshed_final(self) -> Path:
        """Wshed_final."""
        return self.root_dir / self.visual_sdir / "wshed.zarr"

    @property
    def points_raw(self) -> Path:
        """Points_raw."""
        return self.root_dir / self.visual_sdir / "points_raw.zarr"

    @property
    def heatmap_raw(self) -> Path:
        """Heatmap_raw."""
        return self.root_dir / self.visual_sdir / "heatmap_raw.zarr"

    @property
    def points_trfm(self) -> Path:
        """Points_trfm."""
        return self.root_dir / self.visual_sdir / "points_trfm.tif"

    @property
    def heatmap_trfm(self) -> Path:
        """Heatmap_trfm."""
        return self.root_dir / self.visual_sdir / "heatmap_trfm.tif"

    @property
    def comb_reg(self) -> Path:
        """Comb_reg."""
        return self.root_dir / self.visual_sdir / "comb_reg.tif"

    @property
    def comb_cellc(self) -> Path:
        """Comb_cellc."""
        return self.root_dir / self.visual_sdir / "comb_cellc.tif"

    @property
    def comb_heatmap(self) -> Path:
        """Comb_heatmap."""
        return self.root_dir / self.visual_sdir / "comb_heatmap.tif"
