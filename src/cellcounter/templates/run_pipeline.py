import marimo

__generated_with = "0.23.13"
app = marimo.App(width="medium")

with app.setup:
    from pathlib import Path

    import marimo as mo
    from loguru import logger

    from cellcounter import Pipeline
    from cellcounter.models import ProjConfig
    from cellcounter.utils import configure_logger, setup_dask_configs

    configure_logger()
    setup_dask_configs()


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # CellCounter — Run Pipeline

    **Setup**
    * Change the paths and config below as needed.
    * Copy `templates/default_config.yaml` as a starting point.
    * Click **Run Pipeline** when ready — it will take a while.
    """)
    return


@app.cell
def _():
    overwrite = False
    stitch_dir = Path("/path/to/tiff_imgs_folder")
    analysis_dir = Path("analysis_images")
    config_path = Path("default_config.yaml")
    imgs_ls = Pipeline.get_imgs_ls(stitch_dir) if stitch_dir.is_dir() else []

    mo.accordion(
        {
            "Images": imgs_ls or "No images found",
            "Config": ProjConfig.read_yaml(config_path).model_dump()
            if config_path.exists()
            else "Config file not found",
        }
    )
    return analysis_dir, config_path, imgs_ls, overwrite, stitch_dir


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## Run Pipeline
    """)
    return


@app.cell
def _():
    run_btn = mo.ui.run_button(label="Run Pipeline")
    run_btn
    return (run_btn,)


@app.cell
def _(analysis_dir, config_path, imgs_ls, overwrite, run_btn, stitch_dir):
    mo.stop(not run_btn.value, mo.md("Click **Run Pipeline** to start."))

    mo.stop(not imgs_ls, mo.md(f"No images found in `{stitch_dir}`"))
    mo.stop(
        stitch_dir == analysis_dir,
        mo.md("Input and output directories must be different"),
    )

    n_imgs = len(imgs_ls)

    for i, img_name in enumerate(imgs_ls):
        try:
            proj_dir = analysis_dir / img_name
            in_fp = stitch_dir / img_name

            pipeline = Pipeline(proj_dir)
            pipeline.update_config(config_path)

            with mo.status.progress_bar(total=) as bar:
                bar.update(subtitle=f"({i + 1}/{n_imgs}) {img_name}: tiff2zarr")
                pipeline.tiff2zarr(in_fp, overwrite=False)

                bar.update(subtitle=f"({i + 1}/{n_imgs}) {img_name}: registration")
                pipeline.reg_ref_prepare(overwrite=False)
                pipeline.reg_img_rough(overwrite=False)
                pipeline.reg_img_fine(overwrite=overwrite)
                pipeline.reg_img_trim(overwrite=overwrite)
                pipeline.reg_img_bound(overwrite=overwrite)
                pipeline.reg_elastix(overwrite=overwrite)

                bar.update(subtitle=f"({i + 1}/{n_imgs}) {img_name}: tuning arr")
                pipeline.make_tuning_arr(overwrite=False)

                for is_tuning in [True, False]:
                    mode = "tuning" if is_tuning else "production"
                    p = Pipeline(proj_dir, tuning=is_tuning)

                    bar.update(
                        subtitle=f"({i + 1}/{n_imgs}) {img_name}: cellc ({mode})"
                    )
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

                    bar.update(
                        subtitle=f"({i + 1}/{n_imgs}) {img_name}: mapping ({mode})"
                    )
                    p.transform_coords(overwrite=overwrite)
                    p.cell_mapping(overwrite=overwrite)
                    p.group_cells(overwrite=overwrite)
                    p.cells2csv(overwrite=overwrite)

                bar.update(subtitle=f"({i + 1}/{n_imgs}) {img_name}: visual checks")
                pipeline.combine_reg(overwrite=overwrite)

                for is_tuning in [True, False]:
                    p = Pipeline(proj_dir, tuning=is_tuning)
                    p.combine_cellc(overwrite=overwrite)
                    p.coords2heatmap_trfm(overwrite=overwrite)
                    p.combine_heatmap_trfm(overwrite=overwrite)

        except Exception:
            logger.exception(f"Error in {img_name}")
    return (analysis_dir,)


@app.cell
def _(analysis_dir):
    Pipeline.combine(analysis_dir, overwrite=True)


if __name__ == "__main__":
    app.run()
