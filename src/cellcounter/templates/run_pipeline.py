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
    ## Configuration

    Load your config YAML (copy `templates/default_config.yaml` and edit).
    """)
    return


@app.cell
def _():
    config_path = mo.ui.text(
        value=str(Path.cwd() / "default_config.yaml"),
        label="Config file path",
        full_width=True,
    )
    config_path

    fp = Path(config_path.value)
    if not fp.exists():
        mo.md(f"Config file not found: `{fp}`")
        mo.stop(True)

    config = ProjConfig.read_yaml(fp)

    mo.md(f"Loaded config from `{fp}`")
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## Inputs
    """)
    return


@app.cell
def _():
    stitched_imgs_dir = mo.ui.text(
        value="/path/to/tiff_imgs_folder",
        label="TIFF images directory",
        full_width=True,
    )
    analysis_root_dir = mo.ui.text(
        value="/path/to/analysis_outputs_folder",
        label="Analysis output directory",
        full_width=True,
    )
    overwrite = mo.ui.checkbox(label="Overwrite existing outputs")
    mo.vstack([stitched_imgs_dir, analysis_root_dir, overwrite])
    return (stitched_imgs_dir,)


@app.cell
def _(stitched_imgs_dir):
    dir_path = Path(stitched_imgs_dir.value)

    if not dir_path.is_dir():
        mo.md(f"Directory not found: `{dir_path}`")
        mo.stop(True)

    imgs_ls = Pipeline.get_imgs_ls(dir_path)
    if not imgs_ls:
        mo.md("No images found in directory")
        mo.stop(True)

    mo.md(f"Found **{len(imgs_ls)}** image(s)")
    return (imgs_ls,)


@app.cell
def _(imgs_ls):
    selected_imgs = mo.ui.multiselect(
        options=imgs_ls,
        value=imgs_ls,
        label="Images to process",
        full_width=True,
    )
    selected_imgs
    return


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
    return


app._unparsable_cell(
    r"""
    mo.stop(not run_btn.value, mo.md("Click **Run Pipeline** to start."))

    stitch_dir = Path(stitched_imgs_dir.value)
    analysis_dir = Path(analysis_root_dir.value)

    if stitch_dir == analysis_dir:
        mo.md("Input and output directories must be different")
        return

    _is_overwrite = overwrite.value
    n_imgs = len(selected_imgs.value)

    for i, img_name in enumerate(selected_imgs.value):
        try:
            proj_dir = analysis_dir / img_name
            in_fp = stitch_dir / img_name

            pipeline = Pipeline(proj_dir)
            pipeline.update_config(config_path)

            with mo.status.progress_bar(total=28) as bar:
                bar.update(
                    subtitle=f"({i + 1}/{n_imgs}) {img_name}: tiff2zarr"
                )
                pipeline.tiff2zarr(in_fp, overwrite=False)

                bar.update(
                    subtitle=f"({i + 1}/{n_imgs}) {img_name}: registration"
                )
                pipeline.reg_ref_prepare(overwrite=False)
                pipeline.reg_img_rough(overwrite=False)
                pipeline.reg_img_fine(overwrite=_is_overwrite)
                pipeline.reg_img_trim(overwrite=_is_overwrite)
                pipeline.reg_img_bound(overwrite=_is_overwrite)
                pipeline.reg_elastix(overwrite=_is_overwrite)

                bar.update(
                    subtitle=f"({i + 1}/{n_imgs}) {img_name}: tuning arr"
                )
                pipeline.make_tuning_arr(overwrite=False)

                for is_tuning in [True, False]:
                    mode = "tuning" if is_tuning else "production"
                    p = Pipeline(proj_dir, tuning=is_tuning)

                    bar.update(
                        subtitle=f"({i + 1}/{n_imgs}) {img_name}: cellc ({mode})"
                    )
                    p.tophat_filter(overwrite=_is_overwrite)
                    p.dog_filter(overwrite=_is_overwrite)
                    p.adaptive_threshold_prep(overwrite=_is_overwrite)
                    p.threshold(overwrite=_is_overwrite)
                    p.label_thresholded(overwrite=_is_overwrite)
                    p.compute_thresholded_volumes(overwrite=_is_overwrite)
                    p.filter_thresholded(overwrite=_is_overwrite)
                    p.detect_maxima(overwrite=_is_overwrite)
                    p.label_maxima(overwrite=_is_overwrite)
                    p.watershed(overwrite=_is_overwrite)
                    p.compute_watershed_volumes(overwrite=_is_overwrite)
                    p.filter_watershed(overwrite=_is_overwrite)
                    p.save_cells_table(overwrite=_is_overwrite)

                    bar.update(
                        subtitle=f"({i + 1}/{n_imgs}) {img_name}: mapping ({mode})"
                    )
                    p.transform_coords(overwrite=_is_overwrite)
                    p.cell_mapping(overwrite=_is_overwrite)
                    p.group_cells(overwrite=_is_overwrite)
                    p.cells2csv(overwrite=_is_overwrite)

                bar.update(
                    subtitle=f"({i + 1}/{n_imgs}) {img_name}: visual checks"
                )
                pipeline.combine_reg(overwrite=_is_overwrite)

                for is_tuning in [True, False]:
                    p = Pipeline(proj_dir, tuning=is_tuning)
                    p.combine_cellc(overwrite=_is_overwrite)
                    p.coords2heatmap_trfm(overwrite=_is_overwrite)
                    p.combine_heatmap_trfm(overwrite=_is_overwrite)

        except Exception:
            logger.exception(f"Error in {img_name}")
    """,
    name="_"
)


@app.cell
def _(analysis_dir):
    Pipeline.combine(analysis_dir, overwrite=True)
    return


if __name__ == "__main__":
    app.run()
