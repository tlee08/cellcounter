"""Filepath models to ensure consistent project file structure."""

from pathlib import Path


class ProjFp:
    """Project filepath model."""

    def __init__(self, root_dir: Path | str, *, tuning: bool = False) -> None:
        """Project filepath model."""
        self.root_dir = Path(root_dir)
        suffix = "_tuning" if tuning else ""
        self.raw_sdir = f"raw{suffix}"
        self.registration_sdir = "registration"
        self.cellcount_sdir = f"cellcount{suffix}"
        self.analysis_sdir = f"analysis{suffix}"
        self.visual_sdir = f"visual{suffix}"

    @property
    def subdirs_ls(self) -> list[str]:
        """Return list of subdirectories."""
        return [
            self.raw_sdir,
            self.registration_sdir,
            self.cellcount_sdir,
            self.analysis_sdir,
            self.visual_sdir,
        ]

    def make_subdirs(self) -> None:
        """Make project directories from all subdirs."""
        for subdir in self.subdirs_ls:
            if subdir is not None:
                (self.root_dir / subdir).mkdir(parents=True, exist_ok=True)

    @property
    def config_fp(self) -> Path:
        """Config filepath."""
        return self.root_dir / "config_params.json"

    @property
    def diagnostics(self) -> Path:
        """Diagnostics filepath."""
        return self.root_dir / "diagnostics.csv"

    @property
    def raw(self) -> Path:
        """Raw zarr filepath."""
        return self.root_dir / self.raw_sdir / "raw.zarr"

    @property
    def ref(self) -> Path:
        """Reference image filepath."""
        return self.root_dir / self.registration_sdir / "0a_reference.tif"

    @property
    def annot(self) -> Path:
        """Annotation image filepath."""
        return self.root_dir / self.registration_sdir / "0b_annotation.tif"

    @property
    def map(self) -> Path:
        """Mapping filepath."""
        return self.root_dir / self.registration_sdir / "0c_mapping.json"

    @property
    def affine(self) -> Path:
        """Affine transform filepath."""
        return self.root_dir / self.registration_sdir / "0d_align_affine.txt"

    @property
    def bspline(self) -> Path:
        """Bspline transform filepath."""
        return self.root_dir / self.registration_sdir / "0e_align_bspline.txt"

    @property
    def downsmpl1(self) -> Path:
        """Downsample 1 filepath."""
        return self.root_dir / self.registration_sdir / "1_downsmpl1.tif"

    @property
    def downsmpl2(self) -> Path:
        """Downsample 2 filepath."""
        return self.root_dir / self.registration_sdir / "2_downsmpl2.tif"

    @property
    def trimmed(self) -> Path:
        """Trimmed image filepath."""
        return self.root_dir / self.registration_sdir / "3_trimmed.tif"

    @property
    def bounded(self) -> Path:
        """Bounded image filepath."""
        return self.root_dir / self.registration_sdir / "4_bounded.tif"

    @property
    def regresult(self) -> Path:
        """Registration result filepath."""
        return self.root_dir / self.registration_sdir / "5_regresult.tif"

    @property
    def bgrm(self) -> Path:
        """Background removed filepath."""
        return self.root_dir / self.cellcount_sdir / "1_bgrm.zarr"

    @property
    def dog(self) -> Path:
        """Difference of Gaussian filepath."""
        return self.root_dir / self.cellcount_sdir / "2_dog.zarr"

    @property
    def adaptv(self) -> Path:
        """Adaptive threshold filepath."""
        return self.root_dir / self.cellcount_sdir / "3_adaptv.zarr"

    @property
    def threshd(self) -> Path:
        """Thresholded filepath."""
        return self.root_dir / self.cellcount_sdir / "4_threshd.zarr"

    @property
    def threshd_labels(self) -> Path:
        """Thresholded labels filepath."""
        return self.root_dir / self.cellcount_sdir / "5_threshd_labels.zarr"

    @property
    def threshd_volumes(self) -> Path:
        """Thresholded volumes filepath."""
        return self.root_dir / self.cellcount_sdir / "6_threshd_volumes.zarr"

    @property
    def threshd_filt(self) -> Path:
        """Thresholded filtered filepath."""
        return self.root_dir / self.cellcount_sdir / "7_threshd_filt.zarr"

    @property
    def maxima(self) -> Path:
        """Maxima filepath."""
        return self.root_dir / self.cellcount_sdir / "8_maxima.zarr"

    @property
    def maxima_labels(self) -> Path:
        """Maxima labels filepath."""
        return self.root_dir / self.cellcount_sdir / "9_maxima_labels.zarr"

    @property
    def wshed_labels(self) -> Path:
        """Watershed labels filepath."""
        return self.root_dir / self.cellcount_sdir / "10_wshed_labels.zarr"

    @property
    def wshed_volumes(self) -> Path:
        """Watershed volumes filepath."""
        return self.root_dir / self.cellcount_sdir / "11_wshed_volumes.zarr"

    @property
    def wshed_filt(self) -> Path:
        """Watershed filtered filepath."""
        return self.root_dir / self.cellcount_sdir / "12_wshed_filt.zarr"

    @property
    def cells_raw_df(self) -> Path:
        """Cells raw dataframe filepath."""
        return self.root_dir / self.analysis_sdir / "1_cells_raw.parquet"

    @property
    def cells_trfm_df(self) -> Path:
        """Cells transformed dataframe filepath."""
        return self.root_dir / self.analysis_sdir / "2_cells_trfm.parquet"

    @property
    def cells_df(self) -> Path:
        """Cells dataframe filepath."""
        return self.root_dir / self.analysis_sdir / "3_cells.parquet"

    @property
    def cells_agg_df(self) -> Path:
        """Cells aggregated dataframe filepath."""
        return self.root_dir / self.analysis_sdir / "4_cells_agg.parquet"

    @property
    def cells_agg_csv(self) -> Path:
        """Cells aggregated CSV filepath."""
        return self.root_dir / self.analysis_sdir / "5_cells_agg.csv"

    @property
    def points_raw(self) -> Path:
        """Points raw filepath."""
        return self.root_dir / self.visual_sdir / "points_raw.zarr"

    @property
    def heatmap_raw(self) -> Path:
        """Heatmap raw filepath."""
        return self.root_dir / self.visual_sdir / "heatmap_raw.zarr"

    @property
    def points_trfm(self) -> Path:
        """Points transformed filepath."""
        return self.root_dir / self.visual_sdir / "points_trfm.tif"

    @property
    def heatmap_trfm(self) -> Path:
        """Heatmap transformed filepath."""
        return self.root_dir / self.visual_sdir / "heatmap_trfm.tif"

    @property
    def comb_reg(self) -> Path:
        """Combined registration filepath."""
        return self.root_dir / self.visual_sdir / "comb_reg.tif"

    @property
    def comb_cellc(self) -> Path:
        """Combined cell count filepath."""
        return self.root_dir / self.visual_sdir / "comb_cellc.tif"

    @property
    def comb_heatmap(self) -> Path:
        """Combined heatmap filepath."""
        return self.root_dir / self.visual_sdir / "comb_heatmap.tif"


class RefFp:
    """Reference filepath model.

    Atlas from https://download.alleninstitute.org/informatics-archive/current-release/mouse_ccf/
    """

    def __init__(
        self,
        root_dir: Path | str,
        ref_version: str,
        annot_version: str,
        map_version: str,
    ) -> None:
        """Reference filepath model."""
        self.root_dir = Path(root_dir)
        self.ref_version = ref_version
        self.annot_version = annot_version
        self.map_version = map_version
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

    def make_subdirs(self) -> None:
        """Make project directories from all subdirs."""
        for subdir in self.subdirs_ls:
            if subdir is not None:
                (self.root_dir / subdir).mkdir(parents=True, exist_ok=True)

    @property
    def ref(self) -> Path:
        """Reference image filepath."""
        return self.root_dir / self.reference_sdir / f"{self.ref_version}.tif"

    @property
    def annot(self) -> Path:
        """Annotation image filepath."""
        return self.root_dir / self.annotation_sdir / f"{self.annot_version}.tif"

    @property
    def map(self) -> Path:
        """Mapping filepath."""
        return self.root_dir / self.mapping_sdir / f"{self.map_version}.json"

    @property
    def affine(self) -> Path:
        """Affine transform filepath."""
        return self.root_dir / self.elastix_sdir / "align_affine.txt"

    @property
    def bspline(self) -> Path:
        """Bspline transform filepath."""
        return self.root_dir / self.elastix_sdir / "align_bspline.txt"
