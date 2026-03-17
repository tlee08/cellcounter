import os

from cellcounter import BatchCombineFuncs, Pipeline, VisualCheck

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

    assert stitched_imgs_dir != analysis_root_dir

    for img_name in imgs_ls:
        print(f"Running: {img_name}")
        try:
            in_fp = os.path.join(stitched_imgs_dir, img_name)
            analysis_img_dir = os.path.join(analysis_root_dir, img_name)

            # BULK UPDATE CONFIGS HERE
            # COMMENT OUT IF YOU DON'T WANT TO UPDATE THE CONFIGS FROM THE SCRIPT
            Pipeline.update_configs(
                analysis_img_dir,
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
                # OVERLAP
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
            Pipeline.tiff2zarr(analysis_img_dir, in_fp, overwrite=overwrite)
            # Preparing reference images
            # COMMENT OUT AFTER FIRST RUN (ONLY NEEDED INITIALLY)
            Pipeline.reg_ref_prepare(analysis_img_dir, overwrite=overwrite)
            # COMMENT OUT AFTER FIRST RUN (ONLY NEEDED INITIALLY AND VERY SLOW)
            Pipeline.reg_img_rough(analysis_img_dir, overwrite=overwrite)
            # COMMENT OUT ONCE YOU'RE HAPPY WITH THE REGISTRATION PARAMETERS
            Pipeline.reg_img_fine(analysis_img_dir, overwrite=overwrite)
            # COMMENT OUT ONCE YOU'RE HAPPY WITH THE REGISTRATION PARAMETERS
            Pipeline.reg_img_trim(analysis_img_dir, overwrite=overwrite)
            # COMMENT OUT ONCE YOU'RE HAPPY WITH THE REGISTRATION PARAMETERS
            Pipeline.reg_img_bound(analysis_img_dir, overwrite=overwrite)
            # COMMENT OUT ONCE YOU'RE HAPPY WITH THE REGISTRATION PARAMETERS
            Pipeline.reg_elastix(analysis_img_dir, overwrite=overwrite)
            # Making trimmed image for cell count tuning
            # COMMENT OUT AFTER FIRST RUN (ONLY NEEDED INITIALLY)
            Pipeline.make_tuning_arr(analysis_img_dir, overwrite=overwrite)
            # Cell counting (tuning and final)
            for is_tuning in [
                True,  # Tuning
                False,  # Final (COMMENT OUT UNTIL YOU'RE HAPPY WITH THE CELL COUNT PARAMETERS)
            ]:
                Pipeline.img_overlap(analysis_img_dir, overwrite=overwrite, tuning=is_tuning)
                Pipeline.cellc1(analysis_img_dir, overwrite=overwrite, tuning=is_tuning)
                Pipeline.cellc2(analysis_img_dir, overwrite=overwrite, tuning=is_tuning)
                Pipeline.cellc3(analysis_img_dir, overwrite=overwrite, tuning=is_tuning)
                Pipeline.cellc4(analysis_img_dir, overwrite=overwrite, tuning=is_tuning)
                Pipeline.cellc5(analysis_img_dir, overwrite=overwrite, tuning=is_tuning)
                Pipeline.cellc6(analysis_img_dir, overwrite=overwrite, tuning=is_tuning)
                Pipeline.cellc7(analysis_img_dir, overwrite=overwrite, tuning=is_tuning)
                Pipeline.cellc8(analysis_img_dir, overwrite=overwrite, tuning=is_tuning)
                Pipeline.cellc9(analysis_img_dir, overwrite=overwrite, tuning=is_tuning)
                Pipeline.cellc10(analysis_img_dir, overwrite=overwrite, tuning=is_tuning)
                Pipeline.cellc11(analysis_img_dir, overwrite=overwrite, tuning=is_tuning)
                Pipeline.cellc12(analysis_img_dir, overwrite=overwrite, tuning=is_tuning)
                # Cell mapping
                # COMMENT OUT UNTIL YOU'RE HAPPY WITH THE CELL COUNT PARAMETERS
                Pipeline.transform_coords(analysis_img_dir, overwrite=overwrite, tuning=is_tuning)
                # COMMENT OUT UNTIL YOU'RE HAPPY WITH THE CELL COUNT PARAMETERS
                Pipeline.cell_mapping(analysis_img_dir, overwrite=overwrite, tuning=is_tuning)
                # COMMENT OUT UNTIL YOU'RE HAPPY WITH THE CELL COUNT PARAMETERS
                Pipeline.group_cells(analysis_img_dir, overwrite=overwrite, tuning=is_tuning)
                # COMMENT OUT UNTIL YOU'RE HAPPY WITH THE CELL COUNT PARAMETERS
                Pipeline.cells2csv(analysis_img_dir, overwrite=overwrite, tuning=is_tuning)

            # Registration visual check
            VisualCheck.combine_reg(analysis_img_dir, overwrite=overwrite)
            # Transformed space visual checks
            for is_tuning in [
                True,  # Tuning
                False,  # Final (COMMENT OUT UNTIL YOU'RE HAPPY WITH THE CELL COUNT PARAMETERS)
            ]:
                VisualCheck.cellc_trim_to_final(analysis_img_dir, overwrite=overwrite, tuning=is_tuning)
                VisualCheck.combine_cellc(analysis_img_dir, overwrite=overwrite, tuning=is_tuning)
                VisualCheck.coords2heatmap_trfm(analysis_img_dir, overwrite=overwrite, tuning=is_tuning)
                VisualCheck.combine_heatmap_trfm(analysis_img_dir, overwrite=overwrite, tuning=is_tuning)
        except Exception as e:
            print(f"Error in {img_name}: {e}")
    # Combining all experiment dataframes
    BatchCombineFuncs.combine_root_pipeline(analysis_root_dir, os.path.dirname(analysis_root_dir), overwrite=True)
