import marimo

__generated_with = "0.23.13"
app = marimo.App(width="medium")

with app.setup:
    from pathlib import Path

    import dask.array as da
    import marimo as mo
    import tifffile

    from cellcounter.models import ProjConfig, ProjFp
    from cellcounter.utils import configure_logger, setup_dask_configs
    from cellcounter.viewer import view_images

    configure_logger()
    setup_dask_configs()


@app.cell
def _():
    img_categories = {
        "Raw": ["raw"],
        "Registration": [
            "ref",
            "annot",
            "downsmpl1",
            "downsmpl2",
            "trimmed",
            "bounded",
            "regresult",
        ],
        "Cell Counting": [
            "bgrm",
            "dog",
            "adaptv",
            "threshd",
            "threshd_labels",
            "threshd_volumes",
            "threshd_filt",
            "maxima",
            "maxima_labels",
            "wshed_labels",
            "wshed_volumes",
            "wshed_filt",
        ],
        "Visual QC": [
            "points_raw",
            "heatmap_raw",
            "points_trfm",
            "heatmap_trfm",
        ],
    }
    return (img_categories,)


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## Project
    """)


@app.cell
def _():
    proj_dir = mo.ui.text(
        value="/path/to/project",
        label="Project directory",
        full_width=True,
    )
    tuning = mo.ui.switch(label="Tuning mode")
    mo.vstack([proj_dir, tuning])
    return proj_dir, tuning


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## Images to view
    """)


@app.cell
def _(img_categories, proj_dir, tuning):
    dir_path = Path(proj_dir.value)
    mo.stop(
        predicate=not dir_path.is_dir(),
        output=mo.md(f"Directory not found: `{dir_path}`"),
    )

    pfm = ProjFp(dir_path, tuning=tuning.value)

    # Build options grouped by category
    available = {}
    for category, keys in img_categories.items():
        members = {}
        for k in keys:
            fp = getattr(pfm, k)
            exists = fp.exists()
            label = f"{'✅' if exists else '❌'} {k}"
            members[label] = k
        available[category] = members

    mo.md(f"Project: `{dir_path}` — {'tuning' if tuning.value else 'production'} mode")
    return (available,)


@app.cell
def _(available):
    flat_options = {}
    for category, members in available.items():
        for label, value in members.items():
            flat_options[label] = value

    selected_imgs = mo.ui.multiselect(
        options=flat_options,
        value=[],
        label="Image types",
        full_width=True,
    )
    selected_imgs


@app.cell(hide_code=True)
def _():
    mo.md("## Trimmer (optional)")
    mo.md("Crop region-of-interest per axis. Leave blank for full range.")


@app.cell(hide_code=True)
def _(pfm):
    mo.stop(
        predicate=not pfm.raw.exists(),
        output=mo.md(f"Raw file not found: `{pfm.raw}`"),
    )

    raw_shape = da.from_zarr(pfm.raw).shape

    mo.stop(
        predicate=not pfm.config_fp.exists(),
        output=mo.md(f"Config file not found: `{pfm.config_fp}`"),
    )

    config = ProjConfig.read_yaml(pfm.config_fp)
    rough = config.registration.downsample_rough
    fine = config.registration.downsample_fine
    ds_factor = (
        rough.z / fine.z,
        rough.y / fine.y,
        rough.x / fine.x,
    )

    preview_available = downsmpl2_arr = downsmpl2_shape = None
    if pfm.downsmpl2.exists():
        try:
            downsmpl2_arr = tifffile.imread(str(pfm.downsmpl2))
            downsmpl2_shape = downsmpl2_arr.shape
            preview_available = True
        except Exception:
            pass

    ds_factor, preview_available, raw_shape, downsmpl2_shape, downsmpl2_arr


@app.cell
def _(raw_shape):
    z_slider = mo.ui.range_slider(
        start=0,
        stop=raw_shape[0],
        step=1,
        value=[0, raw_shape[0]],
        label="Z range",
        full_width=True,
    )
    y_slider = mo.ui.range_slider(
        start=0,
        stop=raw_shape[1],
        step=1,
        value=[0, raw_shape[1]],
        label="Y range",
        full_width=True,
    )
    x_slider = mo.ui.range_slider(
        start=0,
        stop=raw_shape[2],
        step=1,
        value=[0, raw_shape[2]],
        label="X range",
        full_width=True,
    )
    use_trimmer = mo.ui.checkbox(label="Apply trimmer")
    mo.vstack(
        [
            use_trimmer,
            mo.hstack([z_slider, y_slider, x_slider], gap=0.5),
        ]
    )


@app.cell(hide_code=True)
def _(raw_shape):
    z_view_slider = mo.ui.slider(
        start=0,
        stop=raw_shape[0] - 1,
        step=1,
        value=raw_shape[0] // 2,
        label="Preview Z slice (raw coords)",
        full_width=True,
    )
    z_view_slider


@app.cell(hide_code=True)
def _(
    z_slider,
    y_slider,
    x_slider,
    z_view_slider,
    downsmpl2_arr,
    ds_factor,
    preview_available,
):
    mo.stop(
        predicate=not preview_available,
        output=mo.md("No `downsmpl2` image available for preview."),
    )

    z_start, z_stop = z_slider.value
    y_start, y_stop = y_slider.value
    x_start, x_stop = x_slider.value

    if z_stop <= z_start or y_stop <= y_start or x_stop <= x_start:
        return

    ds_z_start = int(round(z_start / ds_factor[0]))
    ds_z_stop = int(round(z_stop / ds_factor[0]))
    ds_y_start = int(round(y_start / ds_factor[1]))
    ds_y_stop = int(round(y_stop / ds_factor[1]))
    ds_x_start = int(round(x_start / ds_factor[2]))
    ds_x_stop = int(round(x_stop / ds_factor[2]))

    ds_shape = downsmpl2_arr.shape
    ds_z_start = max(0, min(ds_z_start, ds_shape[0] - 1))
    ds_z_stop = max(1, min(ds_z_stop, ds_shape[0]))
    ds_y_start = max(0, min(ds_y_start, ds_shape[1] - 1))
    ds_y_stop = max(1, min(ds_y_stop, ds_shape[1]))
    ds_x_start = max(0, min(ds_x_start, ds_shape[2] - 1))
    ds_x_stop = max(1, min(ds_x_stop, ds_shape[2]))

    cropped = downsmpl2_arr[
        ds_z_start:ds_z_stop, ds_y_start:ds_y_stop, ds_x_start:ds_x_stop
    ]

    if cropped.shape[1] == 0 or cropped.shape[2] == 0:
        return

    z_view_raw = z_view_slider.value
    z_view_cropped = int(round((z_view_raw - z_start) / ds_factor[0]))
    z_view_cropped = max(0, min(z_view_cropped, cropped.shape[0] - 1))

    mo.vstack(
        [
            mo.md(
                f"Raw trim: Z[{z_start}:{z_stop}] "
                f"Y[{y_start}:{y_stop}] "
                f"X[{x_start}:{x_stop}]"
            ),
            mo.md(f"Downsampled crop shape: {cropped.shape}"),
            mo.image(
                cropped[z_view_cropped, :, :],
                width="100%",
                caption=f"z≈{z_view_raw} (raw coords, crop z-idx={z_view_cropped})",
            ),
        ]
    )


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## View in Napari

    Napari opens in a **separate window**.
    The notebook will wait until you close Napari.
    """)


@app.cell
def _():
    view_btn = mo.ui.run_button(label="Open in Napari")
    mo.vstack([view_btn])
    return (view_btn,)


@app.cell
def _(
    view_btn,
    selected_imgs,
    use_trimmer,
    z_slider,
    y_slider,
    x_slider,
    pfm,
):
    mo.stop(
        not view_btn.value,
        mo.md("Click **Open in Napari** to view selected images."),
    )

    if not selected_imgs.value:
        mo.md("No images selected.")

    # Resolve file paths from selected image types
    imgs_fp_ls = [getattr(pfm, name) for name in selected_imgs.value]

    # Build trimmer
    if use_trimmer.value:
        z_start, z_stop = z_slider.value
        y_start, y_stop = y_slider.value
        x_start, x_stop = x_slider.value
        trimmer = (
            slice(z_start, z_stop),
            slice(y_start, y_stop),
            slice(x_start, x_stop),
        )
    else:
        trimmer = None

    mo.md(f"Opening {len(imgs_fp_ls)} image(s) in Napari...")
    view_images(imgs_fp_ls=imgs_fp_ls, trimmer=trimmer)
    mo.md("Napari closed. You may re-launch or adjust selections.")


if __name__ == "__main__":
    app.run()
