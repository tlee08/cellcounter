import logging
import re
import shutil
from pathlib import Path

import dask
import dask.array as da
import dask.dataframe as dd
import numpy as np
import pandas as pd
import tifffile
from dask.distributed import LocalCluster, SpecCluster
from natsort import natsorted
from scipy import ndimage

from cellcounter.constants import (
    ANNOT_COLUMNS_FINAL,
    CELL_AGG_MAPPINGS,
    DASK_CUDA_ENABLED,
    ELASTIX_ENABLED,
    GPU_ENABLED,
    TRFM,
    AnnotColumns,
    CellColumns,
    Coords,
    MaskColumns,
)
from cellcounter.funcs.arr_io_funcs import ArrIOFuncs
from cellcounter.funcs.cpu_cellc_funcs import CpuCellcFuncs
from cellcounter.funcs.map_funcs import MapFuncs
from cellcounter.funcs.mask_funcs import MaskFuncs
from cellcounter.funcs.reg_funcs import RegFuncs
from cellcounter.funcs.visual_check_funcs_tiff import VisualCheckFuncsTiff
from cellcounter.models.fp_models import check_overwrite, get_proj_fm
from cellcounter.models.fp_models.ref_fp import RefFp
from cellcounter.models.proj_config import ProjConfig
from cellcounter.utils.dask_utils import (
    cluster_process,
    disk_cache,
)
from cellcounter.utils.io_utils import (
    read_json,
    sanitise_smb_df,
    silent_remove,
    write_parquet,
)
from cellcounter.utils.misc_utils import enum2list, import_extra_error_func
from cellcounter.utils.union_find import UnionFind

logger = logging.getLogger(__name__)


# Optional dependency: gpu (with dask-cuda)
if DASK_CUDA_ENABLED:
    from dask_cuda import LocalCUDACluster
else:
    LocalCUDACluster = lambda: LocalCluster(n_workers=1, threads_per_worker=1)  # noqa: E731
    logger.warning(
        "Warning Dask-Cuda functionality not installed.\n"
        "Using single GPU functionality instead (1 worker)\n"
        "Dask-Cuda currently only available on Linux"
    )
# Optional dependency: gpu
if GPU_ENABLED:
    from cellcounter.funcs.gpu_cellc_funcs import GpuCellcFuncs
else:
    # TODO: allow more flexibility in number of workers here
    LocalCUDACluster = lambda: LocalCluster(n_workers=2, threads_per_worker=1)  # noqa: E731
    GpuCellcFuncs = CpuCellcFuncs
    logger.warning(
        "Warning GPU functionality not installed.\n"
        "Using CPU functionality instead (much slower).\n"
        'Can install with `pip install "cellcounter[gpu]"`'
    )
# Optional dependency: elastix
if ELASTIX_ENABLED:
    from cellcounter.funcs.elastix_funcs import ElastixFuncs
else:
    ElastixFuncs = import_extra_error_func("elastix")
    logger.warning(
        "Warning Elastix functionality not installed and unavailable.\n"
        'Can install with `pip install "cellcounter[elastix]"`'
    )


class Pipeline:
    """Pipeline."""

    # Clusters
    # heavy (few workers - carrying high RAM computations)
    heavy_n_workers = 2
    heavy_threads_per_worker = 1
    # busy (many workers - carrying low RAM computations)
    busy_n_workers = 6
    busy_threads_per_worker = 2
    # gpu
    _gpu_cluster = LocalCUDACluster
    # GPU enabled cell funcs
    cellc_funcs: type[CpuCellcFuncs] = GpuCellcFuncs

    #############################################
    # SETTING PROCESSING CONFIGS (NUMBER OF WORKERS, GPU ENABLED, ETC.)
    #############################################

    @classmethod
    def heavy_cluster(cls) -> SpecCluster:
        """Make heavy cluster."""
        return LocalCluster(
            n_workers=cls.heavy_n_workers,
            threads_per_worker=cls.heavy_threads_per_worker,
        )

    @classmethod
    def busy_cluster(cls) -> SpecCluster:
        """Make busy cluster."""
        return LocalCluster(
            n_workers=cls.busy_n_workers, threads_per_worker=cls.busy_threads_per_worker
        )

    @classmethod
    def gpu_cluster(cls) -> SpecCluster:
        """Make GPU cluster."""
        return cls._gpu_cluster()

    @classmethod
    def set_gpu(cls, *, enabled: bool = True) -> None:
        """Set GPU cluster."""
        if enabled:
            cls._gpu_cluster = LocalCUDACluster
            cls.cellc_funcs = GpuCellcFuncs
        else:
            cls._gpu_cluster = lambda: LocalCluster(
                n_workers=cls.heavy_n_workers,
                threads_per_worker=cls.heavy_threads_per_worker,
            )
            cls.cellc_funcs = CpuCellcFuncs

    #############################################
    # GET LIST OF IMAGES
    #############################################

    @classmethod
    def get_imgs_ls(cls, imgs_dir: Path | str) -> list:
        """Get list of images."""
        imgs_dir = Path(imgs_dir)
        return natsorted([fp for fp in imgs_dir.iterdir() if (imgs_dir / fp).is_dir()])

    #############################################
    # UPDATE CONFIGS
    #############################################

    @classmethod
    def update_configs(cls, proj_dir: Path | str, **kwargs) -> ProjConfig:
        """If config_params file does not exist, makes a new one.

        Then updates the config_params file with the kwargs.
        If there are no kwargs, will not update the file
        (other than making it if it did not exist).

        Also creates all the project sub-directories too.

        Finally, returns the ConfigParamsModel object.
        """
        pfm = get_proj_fm(proj_dir, tuning=False)
        logger.debug("Making all the project sub-directories")
        logger.debug("Reading/creating params json")
        try:
            configs = ProjConfig.read_file(pfm.config_params)
            logger.debug("The configs file exists so using this file.")
        except FileNotFoundError:
            logger.debug("The configs file does NOT exists.")
            configs = ProjConfig()
            logger.debug("Saving newly created configs file.")
            configs.write_file(pfm.config_params)
        if kwargs != {}:
            logger.debug("kwargs is not empty. They are: %s", kwargs)
            configs_new = configs.model_validate(configs.model_copy(update=kwargs))
            if configs_new != configs:
                logger.debug(
                    "New configs are different from old configs. Overwriting to file."
                )
                configs.write_file(pfm.config_params)
        logger.debug("Returning the configs file")
        return configs

    #############################################
    # CONVERT TIFF TO ZARR FUNCS
    #############################################

    @classmethod
    @check_overwrite("raw")
    def tiff2zarr(
        cls, proj_dir: Path | str, in_fp: Path | str, *, overwrite: bool = False
    ) -> None:
        """Convert TIFF file(s) to Zarr format."""
        in_fp = Path(in_fp)
        pfm = get_proj_fm(proj_dir, tuning=False)
        logger.debug("Reading config params")
        configs = ProjConfig.read_file(pfm.config_params)
        logger.debug("Making zarr from tiff file(s)")
        with cluster_process(LocalCluster(n_workers=1, threads_per_worker=6)):
            if in_fp.is_dir():
                logger.debug("in_fp (%s) is a directory", in_fp)
                logger.debug("Making zarr from tiff file stack in directory")
                ArrIOFuncs.tiffs2zarr(
                    src_fp_ls=tuple(
                        natsorted(
                            in_fp / i
                            for i in in_fp.iterdir()
                            if re.search(r".tif$", str(i))
                        )
                    ),
                    dst_fp=pfm.raw,
                    chunks=configs.zarr_chunksize,
                )
            elif in_fp.is_file():
                logger.debug("in_fp (%s) is a file", in_fp)
                logger.debug("Making zarr from big-tiff file")
                ArrIOFuncs.btiff2zarr(
                    src_fp=in_fp,
                    dst_fp=pfm.raw,
                    chunks=configs.zarr_chunksize,
                )
            else:
                err_msg = f'Input file path, "{in_fp}" does not exist.'
                raise ValueError(err_msg)

    #############################################
    # REGISTRATION PIPELINE FUNCS
    #############################################

    @classmethod
    @check_overwrite("ref", "annot", "map", "affine", "bspline")
    def reg_ref_prepare(cls, proj_dir: Path | str, *, overwrite: bool = False) -> None:
        """Prepare reference atlas images for registration."""
        pfm = get_proj_fm(proj_dir, tuning=False)
        # Getting configs
        configs = ProjConfig.read_file(pfm.config_params)
        # Making ref_fp_model of original atlas images filepaths
        rfm = RefFp(
            configs.atlas_dir,
            configs.ref_version,
            configs.annot_version,
            configs.map_version,
        )
        # Making atlas images
        for fp_i, fp_o in [
            (rfm.ref, pfm.ref),
            (rfm.annot, pfm.annot),
        ]:
            # Reading
            arr = tifffile.imread(fp_i)
            # Reorienting
            arr = RegFuncs.reorient(arr, configs.ref_orient_ls)
            # Slicing
            arr = arr[
                slice(*configs.ref_z_trim),
                slice(*configs.ref_y_trim),
                slice(*configs.ref_x_trim),
            ]
            # Saving
            ArrIOFuncs.write_tiff(arr, fp_o)
        # Copying region mapping json to project folder
        shutil.copyfile(rfm.map, pfm.map)
        # Copying transformation files
        shutil.copyfile(rfm.affine, pfm.affine)
        shutil.copyfile(rfm.bspline, pfm.bspline)

    @classmethod
    @check_overwrite("downsmpl1")
    def reg_img_rough(cls, proj_dir: Path | str, *, overwrite: bool = False) -> None:
        pfm = get_proj_fm(proj_dir, tuning=False)
        # Getting configs
        configs = ProjConfig.read_file(pfm.config_params)
        with cluster_process(cls.busy_cluster()):
            # Reading
            raw_arr = da.from_zarr(pfm.raw)
            # Rough downsample
            downsmpl1_arr = RegFuncs.downsmpl_rough(
                raw_arr, configs.z_rough, configs.y_rough, configs.x_rough
            )
            # Computing (from dask array)
            downsmpl1_arr = downsmpl1_arr.compute()
            # Saving
            ArrIOFuncs.write_tiff(downsmpl1_arr, pfm.downsmpl1)
            return

    @classmethod
    @check_overwrite("downsmpl2")
    def reg_img_fine(cls, proj_dir: Path | str, *, overwrite: bool = False) -> None:
        pfm = get_proj_fm(proj_dir, tuning=False)
        # Getting configs
        configs = ProjConfig.read_file(pfm.config_params)
        # Reading
        downsmpl1_arr = tifffile.imread(pfm.downsmpl1)
        # Fine downsample
        downsmpl2_arr = RegFuncs.downsmpl_fine(
            downsmpl1_arr, configs.z_fine, configs.y_fine, configs.x_fine
        )
        # Saving
        ArrIOFuncs.write_tiff(downsmpl2_arr, pfm.downsmpl2)

    @classmethod
    @check_overwrite("trimmed")
    def reg_img_trim(cls, proj_dir: Path | str, *, overwrite: bool = False) -> None:
        pfm = get_proj_fm(proj_dir, tuning=False)
        # Getting configs
        configs = ProjConfig.read_file(pfm.config_params)
        # Reading
        downsmpl2_arr = tifffile.imread(pfm.downsmpl2)
        # Trim
        trimmed_arr = downsmpl2_arr[
            slice(*configs.z_trim),
            slice(*configs.y_trim),
            slice(*configs.x_trim),
        ]
        # Saving
        ArrIOFuncs.write_tiff(trimmed_arr, pfm.trimmed)

    @classmethod
    @check_overwrite("bounded")
    def reg_img_bound(cls, proj_dir: Path | str, *, overwrite: bool = False) -> None:
        pfm = get_proj_fm(proj_dir, tuning=False)
        # Getting configs
        configs = ProjConfig.read_file(pfm.config_params)
        # Asserting that lower bound is less than upper bound
        assert configs.lower_bound[0] < configs.upper_bound[0], (
            "Error in config parameters: lower bound condition must be less than upper bound condition."
        )
        # NOTE: removed upper and lower limit assertions to allow for more flexibility (e.g. "reducing" super high values)
        # assert configs.lower_bound[1] <= configs.lower_bound[0], (
        #     "Error in config parameters: "
        #     "lower bound final value must be less than or equal to lower bound condition."
        # )
        # assert configs.upper_bound[1] >= configs.upper_bound[0], (
        #     "Error in config parameters: "
        #     "upper bound final value must be greater than or equal to upper bound condition."
        # )
        # Reading
        trimmed_arr = tifffile.imread(pfm.trimmed)
        bounded_arr = trimmed_arr
        # Bounding lower
        bounded_arr[bounded_arr < configs.lower_bound[0]] = configs.lower_bound[1]
        # Bounding upper
        bounded_arr[bounded_arr > configs.upper_bound[0]] = configs.upper_bound[1]
        # Saving
        ArrIOFuncs.write_tiff(bounded_arr, pfm.bounded)

    @classmethod
    @check_overwrite("regresult")
    def reg_elastix(cls, proj_dir: Path | str, *, overwrite: bool = False) -> None:
        pfm = get_proj_fm(proj_dir, tuning=False)
        # Running Elastix registration
        ElastixFuncs.registration(
            fixed_img_fp=pfm.bounded,
            moving_img_fp=pfm.ref,
            output_img_fp=pfm.regresult,
            affine_fp=pfm.affine,
            bspline_fp=pfm.bspline,
        )

    #############################################
    # MASK PIPELINE FUNCS
    #############################################

    @classmethod
    @check_overwrite("mask_fill", "mask_outline", "mask_reg", "mask_df")
    def make_mask(cls, proj_dir: Path | str, *, overwrite: bool = False) -> None:
        """Makes mask of actual image in reference space.

        Also stores # and proportion of existent voxels
        for each region.
        """
        pfm = get_proj_fm(proj_dir, tuning=False)
        # Getting configs
        configs = ProjConfig.read_file(pfm.config_params)
        # Reading annot img (proj oriented and trimmed) and bounded img
        annot_arr = tifffile.imread(pfm.annot)
        bounded_arr = tifffile.imread(pfm.bounded)
        # Storing annot_arr shape
        s = annot_arr.shape
        # Making mask
        blur_arr = cls.cellc_funcs.gauss_blur_filt(bounded_arr, configs.mask_gaus_blur)
        ArrIOFuncs.write_tiff(blur_arr, pfm.premask_blur)
        mask_arr = cls.cellc_funcs.manual_thresh(blur_arr, configs.mask_thresh)
        ArrIOFuncs.write_tiff(mask_arr, pfm.mask_fill)

        # Make outline
        outline_df = MaskFuncs.make_outline(mask_arr)
        # Transformix on coords
        outline_df[[Coords.Z.value, Coords.Y.value, Coords.X.value]] = (
            ElastixFuncs.transformation_coords(
                outline_df,
                str(pfm.ref),
                str(pfm.regresult),
            )[[Coords.Z.value, Coords.Y.value, Coords.X.value]]
            .round(0)
            .astype(np.uint32)
        )
        # Filtering out of bounds coords
        outline_df = outline_df.query(
            f"({Coords.Z.value} >= 0) & ({Coords.Z.value} < {s[0]}) & "
            f"({Coords.Y.value} >= 0) & ({Coords.Y.value} < {s[1]}) & "
            f"({Coords.X.value} >= 0) & ({Coords.X.value} < {s[2]})"
        )

        # Make outline img (1 for in, 2 for out)
        VisualCheckFuncsTiff.coords2points(
            pd.DataFrame(outline_df[outline_df.is_in == 1]), s, pfm.mask_outline
        )
        in_arr = tifffile.imread(pfm.mask_outline)
        VisualCheckFuncsTiff.coords2points(
            pd.DataFrame(outline_df[outline_df.is_in == 0]), s, pfm.mask_outline
        )
        out_arr = tifffile.imread(pfm.mask_outline)
        ArrIOFuncs.write_tiff(in_arr + out_arr * 2, pfm.mask_outline)

        # Fill in outline to recreate mask (not perfect)
        mask_reg_arr = MaskFuncs.fill_outline(outline_df, s)
        # Opening (removes FP) and closing (fills FN)
        mask_reg_arr = ndimage.binary_closing(mask_reg_arr, iterations=2).astype(
            np.uint8
        )
        mask_reg_arr = ndimage.binary_opening(mask_reg_arr, iterations=2).astype(
            np.uint8
        )
        # Saving
        ArrIOFuncs.write_tiff(mask_reg_arr, pfm.mask_reg)

        # Counting mask voxels in each region
        # Getting original annot fp by making ref_fp_model
        rfm = RefFpModel(
            configs.atlas_dir,
            configs.ref_version,
            configs.annot_version,
            configs.map_version,
        )
        # Reading original annot
        annot_orig_arr = tifffile.imread(rfm.annot)
        # Getting the annotation name for every cell (zyx coord)
        mask_df = pd.merge(
            left=MaskFuncs.mask2region_counts(
                np.full(annot_orig_arr.shape, 1), annot_orig_arr
            ),
            right=MaskFuncs.mask2region_counts(mask_reg_arr, annot_arr),
            how="left",
            left_index=True,
            right_index=True,
            suffixes=("_annot", "_mask"),
        ).fillna(0)
        # Reading annotation mappings json
        annot_df = MapFuncs.annot_dict2df(read_json(pfm.map))
        # Combining (summing) the mask_df volumes for parent regions using the annot_df
        mask_df = MapFuncs.combine_nested_regions(mask_df, annot_df)
        # Calculating proportion of mask volume in each region
        mask_df[MaskColumns.VOLUME_PROP.value] = (
            mask_df[MaskColumns.VOLUME_MASK.value]
            / mask_df[MaskColumns.VOLUME_ANNOT.value]
        )
        # Selecting and ordering relevant columns
        mask_df = pd.DataFrame(mask_df[[*ANNOT_COLUMNS_FINAL, *enum2list(MaskColumns)]])
        # Saving
        write_parquet(mask_df, pfm.mask_df)

    #############################################
    # CROP RAW ZARR TO MAKE TUNING ZARR
    #############################################

    @classmethod
    @check_overwrite("raw")
    def make_tuning_arr(
        cls, proj_dir: Path | str, *, overwrite: bool = False, tuning: bool = True
    ) -> None:
        """Crop raw zarr to make a smaller zarr for tuning the cell counting pipeline."""
        pfm = get_proj_fm(proj_dir, tuning=tuning)
        logger.debug("Reading config params")
        configs = ProjConfig.read_file(pfm.config_params)
        with cluster_process(cls.busy_cluster()):
            logger.debug("Reading raw zarr from production")
            pfm_prod = get_proj_fm(proj_dir, tuning=False)
            raw_arr = da.from_zarr(pfm_prod.raw)
            logger.debug("Cropping raw zarr")
            raw_arr = raw_arr[
                slice(*configs.tuning_z_trim),
                slice(*configs.tuning_y_trim),
                slice(*configs.tuning_x_trim),
            ]
            raw_arr = raw_arr.rechunk(configs.zarr_chunksize)
            logger.debug("Saving cropped raw zarr")
            raw_arr = disk_cache(raw_arr, pfm.raw)

    #############################################
    # CELL COUNTING PIPELINE FUNCS
    #############################################

    @classmethod
    def spatial_connect_count(cls, label_arr: da.array) -> da.array:
        """Connects contiguous foreground components, and aggregate with given func.

        Handles cross-chunk connectivity by:
        1. Overlapping labeled array to expose boundaries
        2. Finding adjacent label pairs across boundaries
        3. Union-Find to merge labels belonging to same physical region
        4. Mapping each label to its component size
        """
        with cluster_process(cls.gpu_cluster()):
            # Step 1: Overlap array to expose boundaries
            label_overlap = da.overlap.overlap(label_arr, depth=1, boundary=0)
            # Step 2: Find cross-boundary pairs
            logger.debug("Finding cross-boundary pairs...")
            pair_arrays = dask.compute(
                *[
                    dask.delayed(cls.cellc_funcs.get_boundary_pairs)(b)
                    for b in label_overlap.to_delayed().ravel()
                ]
            )
            all_pairs = (
                np.concatenate([p for p in pair_arrays if len(p) > 0], axis=0)
                if pair_arrays
                else np.empty((0, 2), dtype=np.uint64)
            )
            logger.debug("Cross-boundary pairs found: %d", len(all_pairs))
            # Step 3: Union-Find
            uf = UnionFind()
            for a, b in all_pairs:
                uf.union(int(a), int(b))
            # Step 4: Count voxels per label
            logger.debug("aggregating voxels per label...")
            label_voxel_counts_ls = dask.compute(
                *[
                    dask.delayed(cls.cellc_funcs.get_label_sizemap)(b)
                    for b in label_arr.to_delayed().ravel()
                ]
            )
            labels = np.concatenate([i[0] for i in label_voxel_counts_ls])
            counts = np.concatenate([i[1] for i in label_voxel_counts_ls])
            logger.debug("Unique labels (foreground): %d", len(labels))
            # Aggregate by component root
            uf.build_lookup_table(labels, counts)
            # Step 5: Map labels to sizes
            logger.debug("Writing output array...")
            sizes_arr = da.map_blocks(
                cls.cellc_funcs.map_values_to_arr,
                label_arr,
                ids=uf.sorted_keys,
                values=uf.sorted_sizes,
                dtype=np.uint64,
            )
            return sizes_arr

    #############################################
    # CELL COUNTING PIPELINE FUNCS
    #############################################

    @classmethod
    @check_overwrite("bgrm")
    def cellc1(
        cls, proj_dir: Path | str, *, overwrite: bool = False, tuning: bool = False
    ) -> None:
        """Cell counting pipeline - Step 1: Top-hat filter (background subtraction).

        Args:
            proj_dir (str): Project directory path.
            overwrite (bool, optional): If True, overwrite existing outputs. Defaults to False.
            tuning (bool, optional): If True, use the tuning project file model. Defaults to False.

        Returns:
            None
        """
        pfm = get_proj_fm(proj_dir, tuning=tuning)
        # Making Dask cluster
        with cluster_process(cls.gpu_cluster()):
            # Getting configs
            configs = ProjConfig.read_file(pfm.config_params)
            # Reading input images
            raw_arr = da.from_zarr(pfm.raw)
            # Declaring processing instructions
            bgrm_arr = da.map_blocks(
                cls.cellc_funcs.tophat_filt,
                raw_arr,
                configs.tophat_sigma,
            )
            # Computing and saving
            bgrm_arr = disk_cache(bgrm_arr, pfm.bgrm)

    @classmethod
    @check_overwrite("dog")
    def cellc2(
        cls, proj_dir: Path | str, *, overwrite: bool = False, tuning: bool = False
    ) -> None:
        """Cell counting pipeline - Step 2: Difference of Gaussians (edge detection).

        Args:
            proj_dir (str): Project directory path.
            overwrite (bool, optional): If True, overwrite existing outputs. Defaults to False.
            tuning (bool, optional): If True, use the tuning project file model. Defaults to False.

        Returns:
            None
        """
        pfm = get_proj_fm(proj_dir, tuning=tuning)
        # Making Dask cluster
        with cluster_process(cls.gpu_cluster()):
            # Getting configs
            configs = ProjConfig.read_file(pfm.config_params)
            # Reading input images
            bgrm_arr = da.from_zarr(pfm.bgrm)
            # Declaring processing instructions
            dog_arr = da.map_blocks(
                cls.cellc_funcs.dog_filt,
                bgrm_arr,
                configs.dog_sigma1,
                configs.dog_sigma2,
            )
            # Computing and saving
            dog_arr = disk_cache(dog_arr, pfm.dog)

    @classmethod
    @check_overwrite("adaptv")
    def cellc3(
        cls, proj_dir: Path | str, *, overwrite: bool = False, tuning: bool = False
    ) -> None:
        """Cell counting pipeline - Step 3: Gaussian subtraction with large sigma for adaptive thresholding.

        Args:
            proj_dir (str): Project directory path.
            overwrite (bool, optional): If True, overwrite existing outputs. Defaults to False.
            tuning (bool, optional): If True, use the tuning project file model. Defaults to False.

        Returns:
            None
        """
        pfm = get_proj_fm(proj_dir, tuning=tuning)
        # Making Dask cluster
        with cluster_process(cls.gpu_cluster()):
            # Getting configs
            configs = ProjConfig.read_file(pfm.config_params)
            # Reading input images
            dog_arr = da.from_zarr(pfm.dog)
            # Declaring processing instructions
            adaptv_arr = da.map_blocks(
                cls.cellc_funcs.gauss_subt_filt,
                dog_arr,
                configs.large_gauss_sigma,
            )
            # Computing and saving
            adaptv_arr = disk_cache(adaptv_arr, pfm.adaptv)

    @classmethod
    @check_overwrite("threshd")
    def cellc4(
        cls, proj_dir: Path | str, *, overwrite: bool = False, tuning: bool = False
    ) -> None:
        """Cell counting pipeline - Step 4: Manual thresholding (or mean thresholding with standard deviation offset).

        Args:
            proj_dir (str): Project directory path.
            overwrite (bool, optional): If True, overwrite existing outputs. Defaults to False.
            tuning (bool, optional): If True, use the tuning project file model. Defaults to False.

        Returns:
            None
        """
        pfm = get_proj_fm(proj_dir, tuning=tuning)
        # Making Dask cluster
        with cluster_process(cls.gpu_cluster()):
            # Getting configs
            configs = ProjConfig.read_file(pfm.config_params)
            # Reading input images
            adaptv_arr = da.from_zarr(pfm.adaptv)
            # Declaring processing instructions
            threshd_arr = da.map_blocks(
                cls.cellc_funcs.manual_thresh,
                adaptv_arr,
                configs.threshd_value,
            )
            # Computing and saving
            threshd_arr = disk_cache(threshd_arr, pfm.threshd)

    @classmethod
    @check_overwrite("threshd_labels")
    def cellc5(
        cls, proj_dir: Path | str, *, overwrite: bool = False, tuning: bool = False
    ) -> None:
        """Cell counting pipeline - Step 5a: Label contiguous regions with globally unique labels.

        Args:
            proj_dir (str): Project directory path.
            overwrite (bool, optional): If True, overwrite existing outputs. Defaults to False.
            tuning (bool, optional): If True, use the tuning project file model. Defaults to False.

        Returns:
            None
        """
        pfm = get_proj_fm(proj_dir, tuning=tuning)
        # Making Dask cluster
        with cluster_process(cls.gpu_cluster()):
            # Getting configs
            configs = ProjConfig.read_file(pfm.config_params)
            max_labels_per_chunk = int(np.ceil(np.prod(configs.zarr_chunksize)) / 2) + 1
            # Reading input images
            threshd_arr = da.from_zarr(pfm.threshd)
            # Declaring processing instructions
            labels_arr = da.map_blocks(
                cls.cellc_funcs.mask2label,
                threshd_arr,
                max_labels_per_chunk=max_labels_per_chunk,
                dtype=np.uint64,
            )
            # Computing and saving
            disk_cache(labels_arr, pfm.threshd_labels)

    @classmethod
    @check_overwrite("threshd_volumes")
    def cellc6(
        cls, proj_dir: Path | str, *, overwrite: bool = False, tuning: bool = False
    ) -> None:
        """Cell counting pipeline - Step 5b: Compute contiguous sizes using union-find.

        Args:
            proj_dir (str): Project directory path.
            overwrite (bool, optional): If True, overwrite existing outputs. Defaults to False.
            tuning (bool, optional): If True, use the tuning project file model. Defaults to False.

        Returns:
            None
        """
        pfm = get_proj_fm(proj_dir, tuning=tuning)
        # Reading input images
        labels_arr = da.from_zarr(pfm.threshd_labels)
        # Declaring processing instructions
        sizes_arr = cls.spatial_connect_count(labels_arr)
        # Saving
        disk_cache(sizes_arr, pfm.threshd_volumes)

    @classmethod
    @check_overwrite("threshd_filt")
    def cellc7(
        cls, proj_dir: Path | str, *, overwrite: bool = False, tuning: bool = False
    ) -> None:
        """Cell counting pipeline - Step 6: Filter out large objects (likely outlines, not cells).

        Args:
            proj_dir (str): Project directory path.
            overwrite (bool, optional): If True, overwrite existing outputs. Defaults to False.
            tuning (bool, optional): If True, use the tuning project file model. Defaults to False.

        Returns:
            None
        """
        pfm = get_proj_fm(proj_dir, tuning=tuning)
        with cluster_process(cls.gpu_cluster()):
            # Getting configs
            configs = ProjConfig.read_file(pfm.config_params)
            # Reading input images
            threshd_volumes_arr = da.from_zarr(pfm.threshd_volumes)
            # Declaring processing instructions
            threshd_filt_arr = da.map_blocks(
                cls.cellc_funcs.volume_filter,
                threshd_volumes_arr,
                configs.min_threshd_size,
                configs.max_threshd_size,
            )
            # Computing and saving
            threshd_filt_arr = disk_cache(threshd_filt_arr, pfm.threshd_filt)

    @classmethod
    @check_overwrite("maxima")
    def cellc8(
        cls, proj_dir: Path | str, *, overwrite: bool = False, tuning: bool = False
    ) -> None:
        """Cell counting pipeline - Step 7: Get maxima mask of raw image with thresholded-filtered mask.

        Args:
            proj_dir (str): Project directory path.
            overwrite (bool, optional): If True, overwrite existing outputs. Defaults to False.
            tuning (bool, optional): If True, use the tuning project file model. Defaults to False.

        Returns:
            None
        """
        pfm = get_proj_fm(proj_dir, tuning=tuning)
        with cluster_process(cls.gpu_cluster()):
            # Getting configs
            configs = ProjConfig.read_file(pfm.config_params)
            # Reading input images
            raw_arr = da.from_zarr(pfm.raw)
            threshd_filt_arr = da.from_zarr(pfm.threshd_filt)
            # Declaring processing instructions
            maxima_arr = da.map_blocks(
                cls.cellc_funcs.get_local_maxima,
                raw_arr,
                configs.maxima_sigma,
                threshd_filt_arr,
            )
            # Computing and saving
            maxima_arr = disk_cache(maxima_arr, pfm.maxima)

    @classmethod
    @check_overwrite("maxima_labels")
    def cellc9(
        cls, proj_dir: Path | str, *, overwrite: bool = False, tuning: bool = False
    ) -> None:
        """Cell counting pipeline - Step 8: Convert maxima mask to uniquely labelled points.

        Args:
            proj_dir (str): Project directory path.
            overwrite (bool, optional): If True, overwrite existing outputs. Defaults to False.
            tuning (bool, optional): If True, use the tuning project file model. Defaults to False.

        Returns:
            None
        """
        # TODO: Check that the results of cellc10 and cellc7b, cellc8a, cellc10a are the same (df)
        pfm = get_proj_fm(proj_dir, tuning=tuning)
        # Getting configs
        configs = ProjConfig.read_file(pfm.config_params)
        max_labels_per_chunk = int(np.ceil(np.prod(configs.zarr_chunksize)) / 2) + 1
        with cluster_process(cls.gpu_cluster()):
            # Reading input images
            maxima_arr = da.from_zarr(pfm.maxima)
            # Declaring processing instructions
            maxima_labels_arr = da.map_blocks(
                cls.cellc_funcs.mask2label,
                maxima_arr,
                max_labels_per_chunk=max_labels_per_chunk,
                dtype=np.uint64,
            )
            maxima_labels_arr = disk_cache(maxima_labels_arr, pfm.maxima_labels)

    @classmethod
    @check_overwrite("wshed_labels")
    def cellc10(
        cls, proj_dir: Path | str, *, overwrite: bool = False, tuning: bool = False
    ) -> None:
        """Cell counting pipeline - Step 9: Watershed segmentation labels with maxima labels and thresholded-filtered mask.

        Args:
            proj_dir (str): Project directory path.
            overwrite (bool, optional): If True, overwrite existing outputs. Defaults to False.
            tuning (bool, optional): If True, use the tuning project file model. Defaults to False.

        Returns:
            None
        """
        pfm = get_proj_fm(proj_dir, tuning=tuning)
        with cluster_process(cls.heavy_cluster()):
            # Reading input images
            raw_arr = da.from_zarr(pfm.raw)
            maxima_labels_arr = da.from_zarr(pfm.maxima_labels)
            threshd_filt_arr = da.from_zarr(pfm.threshd_filt)
            # Declaring processing instructions
            wshed_labels_arr = da.map_blocks(
                cls.cellc_funcs.wshed_segm,
                raw_arr,
                maxima_labels_arr,
                threshd_filt_arr,
            )
            wshed_labels_arr = disk_cache(wshed_labels_arr, pfm.wshed_labels)

    @classmethod
    @check_overwrite("wshed_volumes")
    def cellc11(
        cls, proj_dir: Path | str, *, overwrite: bool = False, tuning: bool = False
    ) -> None:
        """Cell counting pipeline - Step 10.

        Convert watershed labels to watershed segmentation volumes.
        """
        pfm = get_proj_fm(proj_dir, tuning=tuning)
        with cluster_process(cls.heavy_cluster()):
            # Reading input images
            wshed_labels_arr = da.from_zarr(pfm.wshed_labels)
            # Declaring processing instructions
            wshed_volumes_arr = cls.spatial_connect_count(wshed_labels_arr)
            # Saving
            wshed_volumes_arr = disk_cache(wshed_volumes_arr, pfm.wshed_volumes)

    @classmethod
    @check_overwrite("wshed_filt")
    def cellc12(
        cls, proj_dir: Path | str, *, overwrite: bool = False, tuning: bool = False
    ) -> None:
        """Cell counting pipeline - Step 11.

        Filter out large watershed objects
        (sets large volume values to 0, effectively filtering from segmentation image).
        """
        pfm = get_proj_fm(proj_dir, tuning=tuning)
        with cluster_process(cls.gpu_cluster()):
            # Getting configs
            configs = ProjConfig.read_file(pfm.config_params)
            # Reading input images
            wshed_volumes_arr = da.from_zarr(pfm.wshed_volumes)
            # Declaring processing instructions
            wshed_filt_arr = da.map_blocks(
                cls.cellc_funcs.volume_filter,
                wshed_volumes_arr,
                configs.min_wshed_size,
                configs.max_wshed_size,
            )
            # Computing and saving
            wshed_filt_arr = disk_cache(wshed_filt_arr, pfm.wshed_filt)

    @classmethod
    @check_overwrite("cells_raw_df")
    def cellc13(
        cls, proj_dir: Path | str, *, overwrite: bool = False, tuning: bool = False
    ) -> None:
        """Cell counting pipeline - Step 12.

        Saves a table of cells and their measures.
        Uses raw, maxima_labels, wshed_labels, and wshed_filt.
        """
        pfm = get_proj_fm(proj_dir, tuning=tuning)
        with cluster_process(cls.gpu_cluster()):
            configs = ProjConfig.read_file(pfm.config_params)
            # Read arrays
            raw_arr = da.from_zarr(pfm.raw)
            maxima_labels_arr = da.from_zarr(pfm.maxima_labels)
            wshed_labels_arr = da.from_zarr(pfm.wshed_labels)
            wshed_filt_arr = da.from_zarr(pfm.wshed_filt)

            # Compute block offsets from chunk sizes
            offsets = [np.cumsum([0, *c[:-1]]) for c in raw_arr.chunks]
            n_blocks = np.prod(raw_arr.numblocks)

            # Get delayed blocks for each array
            raw_blocks = raw_arr.to_delayed().ravel()
            maxima_blocks = maxima_labels_arr.to_delayed().ravel()
            wshed_blocks = wshed_labels_arr.to_delayed().ravel()
            filt_blocks = wshed_filt_arr.to_delayed().ravel()

            # Process each block with offsets
            @dask.delayed
            def process_block(raw, maxima, wshed, filt, z_off, y_off, x_off):
                return cls.cellc_funcs.get_cells(
                    raw,
                    maxima,
                    wshed,
                    filt,
                    z_offset=z_off,
                    y_offset=y_off,
                    x_offset=x_off,
                )

            delayed_results = []
            for idx in range(n_blocks):
                block_coords = np.unravel_index(idx, raw_arr.numblocks)
                z_off, y_off, x_off = (offsets[i][block_coords[i]] for i in range(3))
                delayed_results.append(
                    process_block(
                        raw_blocks[idx],
                        maxima_blocks[idx],
                        wshed_blocks[idx],
                        filt_blocks[idx],
                        z_off,
                        y_off,
                        x_off,
                    )
                )

            cells_df = dd.from_delayed(delayed_results)

            # Offset for tuning crop if needed
            if tuning:
                cells_df[Coords.Z.value] += configs.tuning_z_trim[0] or 0
                cells_df[Coords.Y.value] += configs.tuning_y_trim[0] or 0
                cells_df[Coords.X.value] += configs.tuning_x_trim[0] or 0

            cells_df = cells_df.compute()
            write_parquet(cells_df, pfm.cells_raw_df)

    #############################################
    # CELL COUNT REALIGNMENT TO REFERENCE AND AGGREGATION PIPELINE FUNCS
    #############################################

    @classmethod
    @check_overwrite("cells_trfm_df")
    def transform_coords(
        cls, proj_dir: Path | str, *, overwrite: bool = False, tuning: bool = False
    ) -> None:
        """Transform cell coordinates to reference atlas space and save as a parquet file.

        Args:
            proj_dir (str): Project directory path.
            overwrite (bool, optional): If True, overwrite existing outputs. Defaults to False.
            tuning (bool, optional): If True, use the tuning project file model. Defaults to False.

        Returns:
            None

        Notes:
            Saves the cells_trfm dataframe as pandas parquet.
        """
        pfm = get_proj_fm(proj_dir, tuning=tuning)
        # Getting configs
        configs = ProjConfig.read_file(pfm.config_params)
        with cluster_process(cls.busy_cluster()):
            # Setting output key (in the form "<maxima/region>_trfm_df")
            # Getting cell coords
            cells_df = pd.read_parquet(pfm.cells_raw_df)
            # Sanitising (removing smb columns)
            cells_df = sanitise_smb_df(cells_df)
            # Taking only Coords.Z.value, Coords.Y.value, Coords.X.value coord columns
            cells_df = cells_df[enum2list(Coords)]
            # Scaling to resampled rough space
            # NOTE: this downsampling uses slicing so must be computed differently
            cells_df = cells_df / np.array(
                (configs.z_rough, configs.y_rough, configs.x_rough)
            )
            # Scaling to resampled space
            cells_df = cells_df * np.array(
                (configs.z_fine, configs.y_fine, configs.x_fine)
            )
            # Trimming/offsetting to sliced space
            cells_df = cells_df - np.array(
                [s[0] or 0 for s in (configs.z_trim, configs.y_trim, configs.x_trim)]
            )
            # Converting back to DataFrame
            cells_df = pd.DataFrame(cells_df, columns=enum2list(Coords))

            cells_trfm_df = ElastixFuncs.transformation_coords(
                cells_df,
                str(pfm.ref),
                str(pfm.regresult),
            )
            # NOTE: Using pandas parquet. does not work with dask yet
            # cells_df = dd.from_pandas(cells_df, npartitions=1)
            # Fitting resampled space to atlas image with Transformix (from Elastix registration step)
            # cells_df = cells_df.repartition(
            #     npartitions=int(np.ceil(cells_df.shape[0].compute() / ROWSPPART))
            # )
            # cells_df = cells_df.map_partitions(
            #     ElastixFuncs.transformation_coords, pfm.ref.val, pfm.regresult.val
            # )
            write_parquet(cells_trfm_df, pfm.cells_trfm_df)

    @classmethod
    @check_overwrite("cells_df")
    def cell_mapping(
        cls, proj_dir: Path | str, *, overwrite: bool = False, tuning: bool = False
    ) -> None:
        """Map transformed cell coordinates to region IDs and names in the reference atlas.

        Args:
            proj_dir (str): Project directory path.
            overwrite (bool, optional): If True, overwrite existing outputs. Defaults to False.
            tuning (bool, optional): If True, use the tuning project file model. Defaults to False.

        Returns:
            None

        Notes:
            Saves the cells dataframe as pandas parquet.
        """
        pfm = get_proj_fm(proj_dir, tuning=tuning)
        # Getting region for each detected cell (i.e. row) in cells_df
        with cluster_process(cls.busy_cluster()):
            # Reading cells_raw and cells_trfm dataframes
            cells_df = pd.read_parquet(pfm.cells_raw_df)
            coords_trfm = pd.read_parquet(pfm.cells_trfm_df)
            # Sanitising (removing smb columns)
            cells_df = sanitise_smb_df(cells_df)
            coords_trfm = sanitise_smb_df(coords_trfm)
            # Making unique incrementing index
            cells_df = cells_df.reset_index(drop=True)
            # Setting the transformed coords
            cells_df[f"{Coords.Z.value}_{TRFM}"] = coords_trfm[Coords.Z.value].values
            cells_df[f"{Coords.Y.value}_{TRFM}"] = coords_trfm[Coords.Y.value].values
            cells_df[f"{Coords.X.value}_{TRFM}"] = coords_trfm[Coords.X.value].values

            # Reading annotation image
            annot_arr = tifffile.imread(pfm.annot)
            # Getting the annotation ID for every cell (zyx coord)
            # Getting transformed coords (that are within tbe bounds_arr, and their corresponding idx)
            s = annot_arr.shape
            trfm_loc = (
                cells_df[
                    [
                        f"{Coords.Z.value}_{TRFM}",
                        f"{Coords.Y.value}_{TRFM}",
                        f"{Coords.X.value}_{TRFM}",
                    ]
                ]
                .round(0)
                .astype(np.uint32)
                .query(
                    f"({Coords.Z.value}_{TRFM} >= 0) & ({Coords.Z.value}_{TRFM} < {s[0]}) & "
                    f"({Coords.Y.value}_{TRFM} >= 0) & ({Coords.Y.value}_{TRFM} < {s[1]}) & "
                    f"({Coords.X.value}_{TRFM} >= 0) & ({Coords.X.value}_{TRFM} < {s[2]})"
                )
            )
            # Getting the pixel values of each valid transformed coord (hence the specified index)
            # By complex array indexing on ar_annot's (z, y, x) dimensions.
            # nulls are imputed with -1
            cells_df[AnnotColumns.ID.value] = pd.Series(
                annot_arr[*trfm_loc.values.T].astype(np.uint32),
                index=trfm_loc.index,
            ).fillna(-1)

            # Reading annotation mappings dataframe
            annot_df = MapFuncs.annot_dict2df(read_json(pfm.map))
            # Getting the annotation name for every cell (zyx coord)
            cells_df = MapFuncs.df_map_ids(cells_df, annot_df)
            # Saving to disk
            # NOTE: Using pandas parquet. does not work with dask yet
            # cells_df = dd.from_pandas(cells_df)
            write_parquet(cells_df, pfm.cells_df)

    @classmethod
    @check_overwrite("cells_agg_df")
    def group_cells(
        cls, proj_dir: Path | str, *, overwrite: bool = False, tuning: bool = False
    ) -> None:
        """Group cells by region name and aggregate total cell volume and cell count for each region.

        Args:
            proj_dir (str): Project directory path.
            overwrite (bool, optional): If True, overwrite existing outputs. Defaults to False.
            tuning (bool, optional): If True, use the tuning project file model. Defaults to False.

        Returns:
            None

        Notes:
            Saves the cells_agg dataframe as pandas parquet.
        """
        pfm = get_proj_fm(proj_dir, tuning=tuning)
        # Making cells_agg_df
        with cluster_process(cls.busy_cluster()):
            # Reading cells dataframe
            cells_df = pd.read_parquet(pfm.cells_df)
            # Sanitising (removing smb columns)
            cells_df = sanitise_smb_df(cells_df)
            # Grouping cells by region name and aggregating on given mappings
            cells_agg_df = cells_df.groupby(AnnotColumns.ID.value).agg(
                CELL_AGG_MAPPINGS
            )
            cells_agg_df.columns = list(CELL_AGG_MAPPINGS.keys())
            # Reading annotation mappings dataframe
            # Making df of region names and their parent region names
            annot_df = MapFuncs.annot_dict2df(read_json(pfm.map))
            # Combining (summing) the cells_agg_df values for parent regions using the annot_df
            cells_agg_df = MapFuncs.combine_nested_regions(cells_agg_df, annot_df)
            # Calculating integrated average intensity (sum_intensity / volume)
            cells_agg_df[CellColumns.IOV.value] = (
                cells_agg_df[CellColumns.SUM_INTENSITY.value]
                / cells_agg_df[CellColumns.VOLUME.value]
            )
            # Selecting and ordering relevant columns
            cells_agg_df = cells_agg_df[[*ANNOT_COLUMNS_FINAL, *enum2list(CellColumns)]]
            # Saving to disk
            # NOTE: Using pandas parquet. does not work with dask yet
            # cells_agg = dd.from_pandas(cells_agg)
            write_parquet(cells_agg_df, pfm.cells_agg_df)

    @classmethod
    @check_overwrite("cells_agg_csv")
    def cells2csv(
        cls, proj_dir: Path | str, *, overwrite: bool = False, tuning: bool = False
    ) -> None:
        """Save the aggregated cell dataframe to a CSV file.

        Args:
            proj_dir (str): Project directory path.
            overwrite (bool, optional): If True, overwrite existing outputs. Defaults to False.
            tuning (bool, optional): If True, use the tuning project file model. Defaults to False.

        Returns:
            None
        """
        pfm = get_proj_fm(proj_dir, tuning=tuning)
        # Reading cells dataframe
        cells_agg_df = pd.read_parquet(pfm.cells_agg_df)
        # Sanitising (removing smb columns)
        cells_agg_df = sanitise_smb_df(cells_agg_df)
        # Saving to csv
        cells_agg_df.to_csv(pfm.cells_agg_csv)

    #############################################
    # CLEAN IMAGE ANALYSIS DIR BY REMOVING LARGE CELLCOUNT SUBDIR
    #############################################

    @classmethod
    def clean_proj(cls, proj_dir: Path | str) -> None:
        """Clean the project directory by removing large files.

        Namely all files in cellcount subdirectory.
        """
        # Removing cellcount subdirectory
        pfm = get_proj_fm(proj_dir, tuning=False)
        silent_remove(pfm.root_dir / pfm.cellcount_sdir)
        # Removing tuning cellcount subdirectory
        pfm_tuning = get_proj_fm(proj_dir, tuning=True)
        silent_remove(pfm_tuning.root_dir / pfm_tuning.cellcount_sdir)
        logger.info("Project %s cleaned.", proj_dir)

    #############################################
    # MISC: REGISTRATION PIPELINE FUNCS
    #############################################

    @classmethod
    def rechunk(cls, proj_dir: Path | str, src_fp: str, dst_fp: str) -> None:
        """Rechunk a zarr file based on the project config's chunksize."""
        pfm = get_proj_fm(proj_dir, tuning=False)
        configs = ProjConfig.read_file(pfm.config_params)
        with cluster_process(cls.busy_cluster()):
            # Read
            zarr_arr = da.from_zarr(src_fp)
            # Rechunk
            zarr_rechunked = zarr_arr.rechunk(configs.zarr_chunksize)
            # Write
            disk_cache(zarr_rechunked, dst_fp)

    #############################################
    # ALL PIPELINE FUNCTION
    #############################################

    @classmethod
    def run_pipeline(
        cls, in_fp: str, proj_dir: Path | str, *, overwrite: bool = False, **kwargs
    ) -> None:
        """Running all pipelines in order."""
        # Updating project configs
        cls.update_configs(proj_dir, **kwargs)
        # tiff to zarr
        cls.tiff2zarr(proj_dir, in_fp, overwrite=overwrite)
        # Registration
        cls.reg_ref_prepare(proj_dir, overwrite=overwrite)
        cls.reg_img_rough(proj_dir, overwrite=overwrite)
        cls.reg_img_fine(proj_dir, overwrite=overwrite)
        cls.reg_img_trim(proj_dir, overwrite=overwrite)
        cls.reg_img_bound(proj_dir, overwrite=overwrite)
        cls.reg_elastix(proj_dir, overwrite=overwrite)
        # Coverage mask
        cls.make_mask(proj_dir, overwrite=overwrite)
        # Making trimmed image for cell count tuning
        cls.make_tuning_arr(proj_dir, overwrite=overwrite)
        # Cell counting (tuning and final)
        for is_tuning in [True, False]:
            cls.cellc1(proj_dir, overwrite=overwrite, tuning=is_tuning)
            cls.cellc2(proj_dir, overwrite=overwrite, tuning=is_tuning)
            cls.cellc3(proj_dir, overwrite=overwrite, tuning=is_tuning)
            cls.cellc4(proj_dir, overwrite=overwrite, tuning=is_tuning)
            cls.cellc5(proj_dir, overwrite=overwrite, tuning=is_tuning)
            cls.cellc6(proj_dir, overwrite=overwrite, tuning=is_tuning)
            cls.cellc7(proj_dir, overwrite=overwrite, tuning=is_tuning)
            cls.cellc8(proj_dir, overwrite=overwrite, tuning=is_tuning)
            cls.cellc9(proj_dir, overwrite=overwrite, tuning=is_tuning)
            cls.cellc10(proj_dir, overwrite=overwrite, tuning=is_tuning)
        # Cell mapping
        cls.transform_coords(proj_dir, overwrite=overwrite)
        cls.cell_mapping(proj_dir, overwrite=overwrite)
        cls.group_cells(proj_dir, overwrite=overwrite)
        cls.cells2csv(proj_dir, overwrite=overwrite)
