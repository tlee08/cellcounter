"""Run pipeline."""

from pathlib import Path

from cellcounter import Pipeline, VisualCheck
from cellcounter.funcs.batch_combine_funcs import combine_root

if __name__ == "__main__":
    # =========================================
    # CHANGE FOLDER PATHS HERE
    stitched_imgs_dir = "/path/to/tiff_imgs_folder"
    analysis_root_dir = "/path/to/analysis_outputs_folder"
    # CHANGE WHETHER TO OVERWRITE EXISTING FILES
    overwrite = True
    # THIS NEXT LINE RUNS ALL IMAGES IN THE INPUT FOLDER
    imgs_ls = Pipeline.get_imgs_ls(stitched_imgs_dir)
    # THIS LINE RUNS ONLY THE IMAGES IN THE LIST BELOW
    # imgs_ls = ["example_img"]
    # =========================================

    # Convert to PosixPath
    stitched_imgs_dir = Path(stitched_imgs_dir)
    analysis_root_dir = Path(analysis_root_dir)

    assert stitched_imgs_dir != analysis_root_dir

    for img_name in imgs_ls:
        print(f"Running: {img_name}")
        try:
            in_fp = stitched_imgs_dir / img_name
            analysis_img_dir = analysis_root_dir / img_name

            # Create pipeline instance for this project
            pipeline = Pipeline(analysis_img_dir)

            # BULK UPDATE CONFIGS HERE
            pipeline.update_config(
                # RAW
                chunks={"z": 500, "y": 500, "x": 500},
                # TUNING CROP
                tuning_trim={
                    "z": {"start": 700, "stop": 800, "step": None},
                    "y": {"start": 1000, "stop": 3000, "step": None},
                    "x": {"start": 1000, "stop": 3000, "step": None},
                },
                # REGISTRATION
                registration={
                    "ref_orientation": {"z": -2, "y": 3, "x": 1},
                    "ref_trim": {
                        "z": {"start": None, "stop": None, "step": None},
                        "y": {"start": None, "stop": None, "step": None},
                        "x": {"start": None, "stop": None, "step": None},
                    },
                    "downsample_rough": {"z": 3, "y": 6, "x": 6},
                    "downsample_fine": {"z": 1.0, "y": 0.6, "x": 0.6},
                    "reg_trim": {
                        "z": {"start": None, "stop": None, "step": None},
                        "y": {"start": None, "stop": None, "step": None},
                        "x": {"start": None, "stop": None, "step": None},
                    },
                    "lower_bound": 100,
                    "lower_bound_mapto": 0,
                    "upper_bound": 5000,
                    "upper_bound_mapto": 5000,
                },
                # CELL COUNTING
                cell_counting={
                    "tophat_radius": 10,
                    "dog_sigma1": 1,
                    "dog_sigma2": 4,
                    "large_gauss_radius": 101,
                    "threshd_value": 60,
                    "min_threshd_size": 100,
                    "max_threshd_size": 9000,
                    "maxima_radius": 10,
                    "min_wshed_size": 1,
                    "max_wshed_size": 700,
                },
                # VISUAL CHECK
                visual_check={
                    "heatmap_raw_radius": 5,
                    "heatmap_trfm_radius": 3,
                },
                # COMBINE ARRAYS
                combine={
                    "cellcount_trim": {
                        "z": {"start": 750, "stop": 760, "step": None},
                        "y": {"start": None, "stop": None, "step": None},
                        "x": {"start": None, "stop": None, "step": None},
                    },
                },
            )

            # Making zarr from tiff file(s)
            # COMMENT OUT AFTER FIRST RUN (ONLY NEEDED INITIALLY AND VERY SLOW)
            pipeline.tiff2zarr(in_fp, overwrite=overwrite)
            # Preparing reference images
            # COMMENT OUT AFTER FIRST RUN (ONLY NEEDED INITIALLY)
            pipeline.reg_ref_prepare(overwrite=overwrite)
            # COMMENT OUT AFTER FIRST RUN (ONLY NEEDED INITIALLY AND VERY SLOW)
            pipeline.reg_img_rough(overwrite=overwrite)
            # COMMENT OUT ONCE YOU'RE HAPPY WITH THE REGISTRATION PARAMETERS
            pipeline.reg_img_fine(overwrite=overwrite)
            # COMMENT OUT ONCE YOU'RE HAPPY WITH THE REGISTRATION PARAMETERS
            pipeline.reg_img_trim(overwrite=overwrite)
            # COMMENT OUT ONCE YOU'RE HAPPY WITH THE REGISTRATION PARAMETERS
            pipeline.reg_img_bound(overwrite=overwrite)
            # COMMENT OUT ONCE YOU'RE HAPPY WITH THE REGISTRATION PARAMETERS
            pipeline.reg_elastix(overwrite=overwrite)
            # Making trimmed image for cell count tuning
            # COMMENT OUT AFTER FIRST RUN (ONLY NEEDED INITIALLY)
            pipeline.make_tuning_arr(overwrite=overwrite)
            # Cell counting (tuning and final)
            for is_tuning in [
                True,  # Tuning
                False,  # Final (COMMENT OUT UNTIL HAPPY WITH THE CELL COUNT PARAMETERS)
            ]:
                pipeline_tuning = Pipeline(analysis_img_dir, tuning=is_tuning)
                pipeline_tuning.tophat_filter(overwrite=overwrite)
                pipeline_tuning.dog_filter(overwrite=overwrite)
                pipeline_tuning.adaptive_threshold_prep(overwrite=overwrite)
                pipeline_tuning.threshold(overwrite=overwrite)
                pipeline_tuning.label_thresholded(overwrite=overwrite)
                pipeline_tuning.compute_thresholded_volumes(overwrite=overwrite)
                pipeline_tuning.filter_thresholded(overwrite=overwrite)
                pipeline_tuning.detect_maxima(overwrite=overwrite)
                pipeline_tuning.label_maxima(overwrite=overwrite)
                pipeline_tuning.watershed(overwrite=overwrite)
                pipeline_tuning.compute_watershed_volumes(overwrite=overwrite)
                pipeline_tuning.filter_watershed(overwrite=overwrite)
                pipeline_tuning.save_cells_table(overwrite=overwrite)
                # Cell mapping
                # COMMENT OUT UNTIL YOU'RE HAPPY WITH THE CELL COUNT PARAMETERS
                pipeline_tuning.transform_coords(overwrite=overwrite)
                # COMMENT OUT UNTIL YOU'RE HAPPY WITH THE CELL COUNT PARAMETERS
                pipeline_tuning.cell_mapping(overwrite=overwrite)
                # COMMENT OUT UNTIL YOU'RE HAPPY WITH THE CELL COUNT PARAMETERS
                pipeline_tuning.group_cells(overwrite=overwrite)
                # COMMENT OUT UNTIL YOU'RE HAPPY WITH THE CELL COUNT PARAMETERS
                pipeline_tuning.cells2csv(overwrite=overwrite)

            # Registration visual check
            vc = VisualCheck(analysis_img_dir, tuning=False)
            vc.combine_reg(overwrite=overwrite)
            # Transformed space visual checks
            for is_tuning in [
                True,  # Tuning
                False,  # Final (COMMENT OUT UNTIL HAPPY WITH THE CELL COUNT PARAMETERS)
            ]:
                vc = VisualCheck(analysis_img_dir, tuning=is_tuning)
                vc.combine_cellc(overwrite=overwrite)
                vc.coords2heatmap_trfm(overwrite=overwrite)
                vc.combine_heatmap_trfm(overwrite=overwrite)
        except Exception as e:
            print(f"Error in {img_name}: {e}")
    # Combining all experiment dataframes
    combine_root(analysis_root_dir, analysis_root_dir.parent, overwrite=True)
