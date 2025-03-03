import inspect
import os
from abc import ABC

# ALSO, WAY overcomplicated with ObservedAttr
# Just use properties, even if it is a bit more verbose

# TODO: implement diagnostics message for ecah indiv specimen (i.e. PFM)


class FpModel(ABC):
    root_dir: str
    subdirs_ls: list[str]

    def copy(self):
        # Getting the class constructor parameters
        params_ls = list(dict(inspect.signature(self.__init__).parameters).keys())
        # Constructing an identical model with the corresponding parameter attributes
        return self.__init__(**{param: getattr(self, param) for param in params_ls})

    def make_subdirs(self):
        """
        Make project directories from all subdirs in
        """
        for subdir in self.subdirs_ls:
            if subdir is not None:
                os.makedirs(os.path.join(self.root_dir, subdir), exist_ok=True)

    def export2dict(self) -> dict:
        """
        Returns a dict of all the FpModel attributes
        """
        export_dict = {}
        # For each attribute in the model
        for attr in dir(self):
            # Skipping private attributes
            if attr.startswith("_"):
                continue
            # If the attribute is a str, add it to the export dict
            if isinstance(getattr(self, attr), str):
                export_dict[attr] = getattr(self, attr)
        # Returning
        return export_dict

    @staticmethod
    def raise_not_implemented_err(attr_name: str):
        raise NotImplementedError(
            "This filepath is not implemented.\nActivate this by calling 'set_implement' or explicitly edit this model."
        )


class RefFpModel(FpModel):
    def __init__(self, root_dir: str, ref_version: str, annot_version: str, map_version: str):
        """
        atlas_rsc_dir = "/home/linux1/Desktop/iDISCO/resources/atlas_resources/"

        Atlas from https://download.alleninstitute.org/informatics-archive/current-release/mouse_ccf/
        """
        # Storing base attributes
        # and wrapping in a FpAttr to reset filepath attributes whenever they are changed
        self.root_dir = root_dir
        self.ref_version = ref_version
        self.annot_version = annot_version
        self.map_version = map_version
        # Constant subdirectory names
        self.reference_sdir = "reference"
        self.annotation_sdir = "annotation"
        self.mapping_sdir = "region_mapping"
        self.elastix_sdir = "elastix_params"

    @property
    def subdirs_ls(self):
        return [
            self.reference_sdir,
            self.annotation_sdir,
            self.mapping_sdir,
            self.elastix_sdir,
        ]

    @property
    def ref(self) -> str:
        return os.path.join(self.root_dir, self.reference_sdir, f"{self.ref_version}.tif")

    @property
    def annot(self) -> str:
        return os.path.join(self.root_dir, self.annotation_sdir, f"{self.annot_version}.tif")

    @property
    def map(self) -> str:
        return os.path.join(self.root_dir, self.mapping_sdir, f"{self.map_version}.json")

    @property
    def affine(self) -> str:
        return os.path.join(self.root_dir, self.elastix_sdir, "align_affine.txt")

    @property
    def bspline(self) -> str:
        return os.path.join(self.root_dir, self.elastix_sdir, "align_bspline.txt")


class ProjFpModel(FpModel):
    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        self.raw_sdir = "raw"
        self.registration_sdir = "registration"
        self.mask_sdir = "mask"
        self.cellcount_sdir = "cellcount"
        self.analysis_sdir = "analysis"
        self.visual_sdir = "visual"

    @property
    def subdirs_ls(self):
        return [
            self.raw_sdir,
            self.registration_sdir,
            self.mask_sdir,
            self.cellcount_sdir,
            self.analysis_sdir,
            self.visual_sdir,
        ]

    @property
    def config_params(self) -> str:
        return os.path.join(self.root_dir, "config_params.json")

    @property
    def diagnostics(self) -> str:
        return os.path.join(self.root_dir, "diagnostics.csv")

    @property
    def raw(self) -> str:
        return os.path.join(self.root_dir, self.raw_sdir, "raw.zarr")

    @property
    def ref(self) -> str:
        return os.path.join(self.root_dir, self.registration_sdir, "0a_reference.tif")

    @property
    def annot(self) -> str:
        return os.path.join(self.root_dir, self.registration_sdir, "0b_annotation.tif")

    @property
    def map(self) -> str:
        return os.path.join(self.root_dir, self.registration_sdir, "0c_mapping.json")

    @property
    def affine(self) -> str:
        return os.path.join(self.root_dir, self.registration_sdir, "0d_align_affine.txt")

    @property
    def bspline(self) -> str:
        return os.path.join(self.root_dir, self.registration_sdir, "0e_align_bspline.txt")

    @property
    def downsmpl1(self) -> str:
        return os.path.join(self.root_dir, self.registration_sdir, "1_downsmpl1.tif")

    @property
    def downsmpl2(self) -> str:
        return os.path.join(self.root_dir, self.registration_sdir, "2_downsmpl2.tif")

    @property
    def trimmed(self) -> str:
        return os.path.join(self.root_dir, self.registration_sdir, "3_trimmed.tif")

    @property
    def bounded(self) -> str:
        return os.path.join(self.root_dir, self.registration_sdir, "4_bounded.tif")

    @property
    def regresult(self) -> str:
        return os.path.join(self.root_dir, self.registration_sdir, "5_regresult.tif")

    @property
    def premask_blur(self) -> str:
        return os.path.join(self.root_dir, self.mask_sdir, "1_premask_blur.tif")

    @property
    def mask_fill(self) -> str:
        return os.path.join(self.root_dir, self.mask_sdir, "2_mask_trimmed.tif")

    @property
    def mask_outline(self) -> str:
        return os.path.join(self.root_dir, self.mask_sdir, "3_outline_reg.tif")

    @property
    def mask_reg(self) -> str:
        return os.path.join(self.root_dir, self.mask_sdir, "4_mask_reg.tif")

    @property
    def mask_df(self) -> str:
        return os.path.join(self.root_dir, self.mask_sdir, "5_mask.parquet")

    @property
    def overlap(self) -> str:
        return os.path.join(self.root_dir, self.cellcount_sdir, "0_overlap.zarr")

    @property
    def bgrm(self) -> str:
        return os.path.join(self.root_dir, self.cellcount_sdir, "1_bgrm.zarr")

    @property
    def dog(self) -> str:
        return os.path.join(self.root_dir, self.cellcount_sdir, "2_dog.zarr")

    @property
    def adaptv(self) -> str:
        return os.path.join(self.root_dir, self.cellcount_sdir, "3_adaptv.zarr")

    @property
    def threshd(self) -> str:
        return os.path.join(self.root_dir, self.cellcount_sdir, "4_threshd.zarr")

    @property
    def threshd_volumes(self) -> str:
        return os.path.join(self.root_dir, self.cellcount_sdir, "5_threshd_volumes.zarr")

    @property
    def threshd_filt(self) -> str:
        return os.path.join(self.root_dir, self.cellcount_sdir, "6_threshd_filt.zarr")

    @property
    def maxima(self) -> str:
        return os.path.join(self.root_dir, self.cellcount_sdir, "7_maxima.zarr")

    @property
    def maxima_labels(self) -> str:
        return os.path.join(self.root_dir, self.cellcount_sdir, "7_maxima_labels.zarr")  # NOTE: NEW

    @property
    def wshed_labels(self) -> str:
        return os.path.join(self.root_dir, self.cellcount_sdir, "8_wshed_labels.zarr")  # NOTE: NEW

    @property
    def wshed_volumes(self) -> str:
        return os.path.join(self.root_dir, self.cellcount_sdir, "8_wshed_volumes.zarr")

    @property
    def wshed_filt(self) -> str:
        return os.path.join(self.root_dir, self.cellcount_sdir, "9_wshed_filt.zarr")

    @property
    def cells_raw_df(self) -> str:
        return os.path.join(self.root_dir, self.analysis_sdir, "1_cells_raw.parquet")

    @property
    def cells_trfm_df(self) -> str:
        return os.path.join(self.root_dir, self.analysis_sdir, "2_cells_trfm.parquet")

    @property
    def cells_df(self) -> str:
        return os.path.join(self.root_dir, self.analysis_sdir, "3_cells.parquet")

    @property
    def cells_agg_df(self) -> str:
        return os.path.join(self.root_dir, self.analysis_sdir, "4_cells_agg.parquet")

    @property
    def cells_agg_csv(self) -> str:
        return os.path.join(self.root_dir, self.analysis_sdir, "5_cells_agg.csv")

    @property
    def threshd_final(self) -> str:
        return os.path.join(self.root_dir, self.visual_sdir, "threshd.zarr")

    @property
    def maxima_final(self) -> str:
        return os.path.join(self.root_dir, self.visual_sdir, "maxima.zarr")

    @property
    def wshed_final(self) -> str:
        return os.path.join(self.root_dir, self.visual_sdir, "wshed.zarr")

    @property
    def points_raw(self) -> str:
        return os.path.join(self.root_dir, self.visual_sdir, "points_raw.zarr")

    @property
    def heatmap_raw(self) -> str:
        return os.path.join(self.root_dir, self.visual_sdir, "heatmap_raw.zarr")

    @property
    def points_trfm(self) -> str:
        return os.path.join(self.root_dir, self.visual_sdir, "points_trfm.tif")

    @property
    def heatmap_trfm(self) -> str:
        return os.path.join(self.root_dir, self.visual_sdir, "heatmap_trfm.tif")

    @property
    def comb_reg(self) -> str:
        return os.path.join(self.root_dir, self.visual_sdir, "comb_reg.tif")

    @property
    def comb_cellc(self) -> str:
        return os.path.join(self.root_dir, self.visual_sdir, "comb_cellc.tif")

    @property
    def comb_heatmap(self) -> str:
        return os.path.join(self.root_dir, self.visual_sdir, "comb_points.tif")


class ProjFpModelTuning(ProjFpModel):
    def __init__(self, root_dir: str):
        super().__init__(root_dir)
        self.raw_sdir = f"{self.raw_sdir}_tuning"
        self.registration_sdir = None
        self.mask_sdir = None
        self.cellcount_sdir = f"{self.cellcount_sdir}_tuning"
        self.analysis_sdir = f"{self.analysis_sdir}_tuning"
        self.visual_sdir = f"{self.visual_sdir}_tuning"

    @property
    def ref(self) -> str:
        return self.raise_not_implemented_err("ref")

    @property
    def annot(self) -> str:
        return self.raise_not_implemented_err("annot")

    @property
    def map(self) -> str:
        return self.raise_not_implemented_err("map")

    @property
    def affine(self) -> str:
        return self.raise_not_implemented_err("affine")

    @property
    def bspline(self) -> str:
        return self.raise_not_implemented_err("bspline")

    @property
    def downsmpl1(self) -> str:
        return self.raise_not_implemented_err("downsmpl1")

    @property
    def downsmpl2(self) -> str:
        return self.raise_not_implemented_err("downsmpl2")

    @property
    def trimmed(self) -> str:
        return self.raise_not_implemented_err("trimmed")

    @property
    def bounded(self) -> str:
        return self.raise_not_implemented_err("bounded")

    @property
    def regresult(self) -> str:
        return self.raise_not_implemented_err("regresult")

    @property
    def premask_blur(self) -> str:
        return self.raise_not_implemented_err("premask_blur")

    @property
    def mask_fill(self) -> str:
        return self.raise_not_implemented_err("mask_fill")

    @property
    def mask_outline(self) -> str:
        return self.raise_not_implemented_err("mask_outline")

    @property
    def mask_reg(self) -> str:
        return self.raise_not_implemented_err("mask_reg")

    @property
    def mask_df(self) -> str:
        return self.raise_not_implemented_err("mask_df")
