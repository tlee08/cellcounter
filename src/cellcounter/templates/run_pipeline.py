from pathlib import Path

from cellcounter import BatchCombineFuncs, Pipeline, VisualCheck

if __name__ == "__main__":
    # =========================================
    # CHANGE FOLDER PATHS HERE
    stitched_imgs_dir = Path("/path/to/tiff_imgs_folder")
    analysis_root_dir = Path("/path/to/analysis_outputs_folder")
    # CHANGE WHETHER TO OVERWRITE EXISTING FILES
    overwrite = True

    # THIS NEXT LINE RUNS ALL IMAGES IN THE INPUT FOLDER
    imgs_ls = Pipeline.get_imgs_ls(stitched_imgs_dir)
    # THIS LINE RUNS ONLY THE IMAGES IN THE LIST BELOW
    # imgs_ls = ["example_img"]
    # =========================================

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
                # REGISTRATION
                ref_orient_ls=(-2, 3, 1),
                ref_z_trim=(None, None, None),
                ref_y_trim=(None, None, None),
                ref_x_trim=(None, None, None),
                z_rough=3,
                y_rough=6,
                x_rough=6,
                z_fine=1,
                y_fine=0.6,
                x_fine=0.6,
                z_trim=(None, None, None),
                y_trim=(None, None, None),
                x_trim=(None, None, None),
                lower_bound=(100, 0),
                upper_bound=(5000, 5000),
                # # CELL COUNT TUNING CROP
                tuning_z_trim=(700, 800, None),
                tuning_y_trim=(1000, 3000, None),
                tuning_x_trim=(1000, 3000, None),
                # CELL COUNTING
                tophat_sigma=10,
                dog_sigma1=1,
                dog_sigma2=4,
                large_gauss_sigma=101,
                threshd_value=60,
                min_threshd_size=100,
                max_threshd_size=9000,
                maxima_sigma=10,
                min_wshed_size=1,
                max_wshed_size=700,
                # VISUAL CHECK
                heatmap_raw_radius=5,
                heatmap_trfm_radius=3,
                # COMBINE ARRAYS
                combine_cellc_z_trim=(750, 760, None),
                combine_cellc_y_trim=(None, None, None),
                combine_cellc_x_trim=(None, None, None),
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
                False,  # Final (COMMENT OUT UNTIL YOU'RE HAPPY WITH THE CELL COUNT PARAMETERS)
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
                False,  # Final (COMMENT OUT UNTIL YOU'RE HAPPY WITH THE CELL COUNT PARAMETERS)
            ]:
                vc = VisualCheck(analysis_img_dir, tuning=is_tuning)
                vc.combine_cellc(overwrite=overwrite)
                vc.coords2heatmap_trfm(overwrite=overwrite)
                vc.combine_heatmap_trfm(overwrite=overwrite)
        except Exception as e:
            print(f"Error in {img_name}: {e}")
    # Combining all experiment dataframes
    BatchCombineFuncs.combine_root_pipeline(
        analysis_root_dir, analysis_root_dir.parent, overwrite=True
    )
