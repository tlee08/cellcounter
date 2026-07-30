import marimo

__generated_with = "0.23.15"
app = marimo.App(width="medium")

with app.setup:
    from pathlib import Path

    import dask.array as da
    import marimo as mo
    import numpy as np
    import tifffile

    from cellcounter.constants import IMAGE_CATEGORIES
    from cellcounter.models import ProjConfig, ProjFp
    from cellcounter.utils import configure_logger, setup_dask_configs
    from cellcounter.viewer import async_view_images

    configure_logger()
    setup_dask_configs()


@app.cell
def _():
    mo.md("""
    ## Project
    """)


@app.cell
def _():
    img_dir = Path("path/to/image_dir")

    pfm = ProjFp(img_dir, tuning=False)

    # Build options grouped by category
    available = {}
    for category, keys in IMAGE_CATEGORIES.items():
        members = {}
        for k in keys:
            fp = getattr(pfm, k)
            exists = fp.exists()
            label = f"{'✅' if exists else '❌'} {k}"
            members[label] = k
        available[category] = members

    mo.accordion(
        {
            "Image Directory": mo.md(
                f"{img_dir} ({'exists' if img_dir.exists() else "doesn't exist"})"
            ),
            "Available Layers": available,
        },
        multiple=True,
    )
    return (pfm,)


@app.cell
def _():
    selected_imgs = [
        "raw",
        "bgrm",
        "dog",
        "adaptv",
        "threshd",
        # "threshd_labels",
        "threshd_volumes",
        "threshd_filt",
        "maxima",
        # "maxima_labels",
        # "wshed_labels",
        "wshed_volumes",
        "wshed_filt",
    ]
    return (selected_imgs,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Trimmer

    Crop region-of-interest per axis. Leave blank for full range.
    """)


@app.cell
def _(pfm):
    mo.stop(
        predicate=not pfm.raw.exists(),
        output=mo.md(f"Raw file not found: `{pfm.raw}`"),
    )
    mo.stop(
        predicate=not pfm.config_fp.exists(),
        output=mo.md(f"Config file not found: `{pfm.config_fp}`"),
    )

    raw_shape = da.from_zarr(pfm.raw).shape
    config = ProjConfig.read_yaml(pfm.config_fp)
    rough = config.registration.downsample_rough
    fine = config.registration.downsample_fine
    ds_factor = (
        rough.z / fine.z,
        rough.y / fine.y,
        rough.x / fine.x,
    )

    preview_available = downsmpl2_arr = None
    if pfm.downsmpl2.exists():
        downsmpl2_arr = tifffile.imread(str(pfm.downsmpl2))
        preview_available = True
    return downsmpl2_arr, ds_factor, preview_available, raw_shape


@app.cell
def _(raw_shape):
    z_slider = mo.ui.range_slider(
        start=0,
        stop=raw_shape[0],
        label="Z range",
        full_width=True,
    )
    y_slider = mo.ui.range_slider(
        start=0,
        stop=raw_shape[1],
        label="Y range",
        full_width=True,
    )
    x_slider = mo.ui.range_slider(
        start=0,
        stop=raw_shape[2],
        label="X range",
        full_width=True,
    )

    mo.vstack([z_slider, y_slider, x_slider])
    return x_slider, y_slider, z_slider


@app.cell
def _():
    z_view_slider = mo.ui.slider(
        start=0,
        stop=1,
        step=0.01,
        label="Preview Z slice (proportion)",
        full_width=True,
    )
    contrast_slider = mo.ui.range_slider(
        start=0,
        stop=1,
        step=0.01,
        label="Contrast range",
        full_width=True,
    )

    mo.vstack([mo.hstack([z_view_slider, contrast_slider])])
    return contrast_slider, z_view_slider


@app.cell
def _(
    downsmpl2_arr,
    ds_factor,
    preview_available,
    x_slider,
    y_slider,
    z_slider,
):
    # Crop first (and caches)
    mo.stop(
        predicate=not preview_available,
        output=mo.md("No `downsmpl2` image available for preview."),
    )

    z_start, z_stop = z_slider.value
    y_start, y_stop = y_slider.value
    x_start, x_stop = x_slider.value

    ds_z_start = int(round(z_start / ds_factor[0]))
    ds_z_stop = int(round(z_stop / ds_factor[0]))
    ds_y_start = int(round(y_start / ds_factor[1]))
    ds_y_stop = int(round(y_stop / ds_factor[1]))
    ds_x_start = int(round(x_start / ds_factor[2]))
    ds_x_stop = int(round(x_stop / ds_factor[2]))

    cropped_arr = downsmpl2_arr[
        ds_z_start:ds_z_stop, ds_y_start:ds_y_stop, ds_x_start:ds_x_stop
    ]
    return (
        cropped_arr,
        ds_x_start,
        ds_x_stop,
        ds_y_start,
        ds_y_stop,
        ds_z_start,
        ds_z_stop,
        x_start,
        x_stop,
        y_start,
        y_stop,
        z_start,
        z_stop,
    )


@app.cell
def _(
    contrast_slider,
    cropped_arr,
    ds_factor,
    ds_x_start,
    ds_x_stop,
    ds_y_start,
    ds_y_stop,
    ds_z_start,
    ds_z_stop,
    x_start,
    x_stop,
    y_start,
    y_stop,
    z_start,
    z_stop,
    z_view_slider,
):
    # Within crop (cached), get z-slice and contrast
    z_ind = int(round(z_view_slider.value * (z_stop - z_start) + z_start))
    ds_z_ind = int(round(z_ind / ds_factor[0]))
    ds_z_ind_offset = np.clip(ds_z_ind - ds_z_start, 0, cropped_arr.shape[0] - 1)

    vmin_total = cropped_arr.min()
    vmax_total = cropped_arr.max()
    vrange_total = vmax_total - vmin_total
    vmin_adjusted = vmin_total + contrast_slider.value[0] * vrange_total
    vmax_adjusted = vmax_total - (1 - contrast_slider.value[1]) * vrange_total

    cropped_arr_slice = np.clip(
        cropped_arr[ds_z_ind_offset, :, :], vmin_adjusted, vmax_adjusted
    )

    mo.hstack(
        [
            mo.image(
                cropped_arr_slice,
                style={
                    "object-fit": "contain",
                    "width": "100%",
                    "height": "auto",
                },
            ),
            {
                "raw": {
                    "z": [z_start, z_stop],
                    "y": [y_start, y_stop],
                    "x": [x_start, x_stop],
                    "shape": [
                        z_stop - z_start,
                        y_stop - y_start,
                        x_stop - x_start,
                    ],
                    "z_index": z_ind,
                },
                "downsampled": {
                    "z": [ds_z_start, ds_z_stop],
                    "y": [ds_y_start, ds_y_stop],
                    "x": [ds_x_start, ds_x_stop],
                    "shape": cropped_arr.shape,
                    "z_index": ds_z_ind,
                },
                "slice": {
                    "vrange_total": [vmin_total, vmax_total],
                    "vrange_adjusted": [vmin_adjusted, vmax_adjusted],
                },
            },
        ],
    )


@app.cell
def _():
    mo.md("""
    ## View in Napari

    Napari opens in a **separate window**.
    The notebook will wait until you close Napari.
    """)


@app.cell
def _(selected_imgs, x_start, x_stop, y_start, y_stop, z_start, z_stop):
    # Estimate size and give warning

    estimated_size_bytes = (
        (z_stop - z_start)
        * (y_stop - y_start)
        * (x_stop - x_start)
        * 4
        * len(selected_imgs)
    )

    mb_factor = 1024**2
    gb_factor = 1024**3

    recommended_threshold_size = 1e10

    estimated_size_callout = mo.callout(
        value=mo.md(f"""
    Estimated size to view in memory: {estimated_size_bytes / gb_factor:_.4f} GB (or {estimated_size_bytes / mb_factor:_.0f} MB)

    **Highly recommended** to have this size be less than {recommended_threshold_size / gb_factor:_.4f} GB
    to avoid slow load times and OOM.

    Adjust the trimming or the selected layers to adjust the size
    """),
        kind="neutral"
        if estimated_size_bytes < recommended_threshold_size
        else "danger",
    )
    estimated_size_callout


@app.cell
async def _(pfm, selected_imgs, x_start, x_stop, y_start, y_stop, z_start, z_stop):
    # Resolve file paths from selected image types
    imgs_fp_ls = [getattr(pfm, name) for name in selected_imgs]

    mo.md(f"Opening {len(imgs_fp_ls)} image(s) in Napari...")
    await async_view_images(
        imgs_fp_ls=imgs_fp_ls,
        trimmer=(
            slice(z_start, z_stop),
            slice(y_start, y_stop),
            slice(x_start, x_stop),
        ),
    )
    mo.md("Napari closed. You may re-launch or adjust selections.")


@app.cell
def _():

    return


if __name__ == "__main__":
    app.run()
