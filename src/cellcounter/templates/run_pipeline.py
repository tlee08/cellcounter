"""Run pipeline - simplified template for users."""

from pathlib import Path

from cellcounter import Pipeline, VisualCheck
from cellcounter.funcs.batch_combine_funcs import combine_root

if __name__ == "__main__":
    # =========================================
    # CONFIGURATION
    # =========================================
    stitched_imgs_dir = Path("/path/to/tiff_imgs_folder")
    analysis_root_dir = Path("/path/to/analysis_outputs_folder")
    overwrite = False  # Set True to re-run steps

    # Which images to process
    imgs_ls = Pipeline.get_imgs_ls(stitched_imgs_dir)
    # imgs_ls = ["example_img"]  # Or specify specific images

    assert stitched_imgs_dir != analysis_root_dir

    # =========================================
    # PIPELINE CONFIG
    # =========================================
    CONFIG = {
        "chunks": {"z": 500, "y": 500, "x": 500},
        "tuning_trim": {
            "z": {"start": 700, "stop": 800, "step": None},
            "y": {"start": 1000, "stop": 3000, "step": None},
            "x": {"start": 1000, "stop": 3000, "step": None},
        },
        "registration": {
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
        "cell_counting": {
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
        "visual_check": {
            "heatmap_raw_radius": 5,
            "heatmap_trfm_radius": 3,
        },
        "combine": {
            "cellcount_trim": {
                "z": {"start": 750, "stop": 760, "step": None},
                "y": {"start": None, "stop": None, "step": None},
                "x": {"start": None, "stop": None, "step": None},
            },
        },
    }

    # =========================================
    # RUN PIPELINE
    # =========================================
    for img_name in imgs_ls:
        print(f"Running: {img_name}")
        try:
            proj_dir = analysis_root_dir / img_name
            in_fp = stitched_imgs_dir / img_name

            # Create and configure pipeline
            pipeline = Pipeline(proj_dir)
            pipeline.update_config(updates=CONFIG)

            # Run once (skip if outputs exist)
            # Registration
            pipeline.tiff2zarr(in_fp, overwrite=False)
            pipeline.reg_ref_prepare(overwrite=False)
            pipeline.reg_img_rough(overwrite=False)

            # Iterative tuning: set overwrite=True for steps you're adjusting
            pipeline.reg_img_fine(overwrite=overwrite)
            pipeline.reg_img_trim(overwrite=overwrite)
            pipeline.reg_img_bound(overwrite=overwrite)
            pipeline.reg_elastix(overwrite=overwrite)

            # Cell counting setup
            pipeline.make_tuning_arr(overwrite=False)

            # Cell counting (run for both tuning and production)
            for is_tuning in [
                True,
                False,
            ]:
                p = Pipeline(proj_dir, tuning=is_tuning)
                p.tophat_filter(overwrite=overwrite)
                p.dog_filter(overwrite=overwrite)
                p.adaptive_threshold_prep(overwrite=overwrite)
                p.threshold(overwrite=overwrite)
                p.label_thresholded(overwrite=overwrite)
                p.compute_thresholded_volumes(overwrite=overwrite)
                p.filter_thresholded(overwrite=overwrite)
                p.detect_maxima(overwrite=overwrite)
                p.label_maxima(overwrite=overwrite)
                p.watershed(overwrite=overwrite)
                p.compute_watershed_volumes(overwrite=overwrite)
                p.filter_watershed(overwrite=overwrite)
                p.save_cells_table(overwrite=overwrite)

                # Mapping
                p.transform_coords(overwrite=overwrite)
                p.cell_mapping(overwrite=overwrite)
                p.group_cells(overwrite=overwrite)
                p.cells2csv(overwrite=overwrite)

            # Visual checks
            vc = VisualCheck(proj_dir, tuning=False)
            vc.combine_reg(overwrite=overwrite)

            for is_tuning in [
                True,
                False,
            ]:
                vc = VisualCheck(proj_dir, tuning=is_tuning)
                vc.combine_cellc(overwrite=overwrite)
                vc.coords2heatmap_trfm(overwrite=overwrite)
                vc.combine_heatmap_trfm(overwrite=overwrite)

        except Exception as e:
            print(f"Error in {img_name}: {e}")

    # Combine all results
    combine_root(analysis_root_dir, analysis_root_dir.parent, overwrite=True)
