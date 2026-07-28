import marimo

__generated_with = "0.23.13"
app = marimo.App(width="medium")

with app.setup:
    from pathlib import Path

    import marimo as mo

    from cellcounter.models import ProjFp
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


@app.cell
def _():
    z_start = mo.ui.number(value=0, label="Z start")
    z_stop = mo.ui.number(value=510, label="Z stop")
    y_start = mo.ui.number(value=1500, label="Y start")
    y_stop = mo.ui.number(value=3000, label="Y stop")
    x_start = mo.ui.number(value=1500, label="X start")
    x_stop = mo.ui.number(value=3000, label="X stop")

    use_trimmer = mo.ui.checkbox(label="Apply trimmer")
    mo.vstack(
        [
            use_trimmer,
            mo.hstack([z_start, z_stop, y_start, y_stop, x_start, x_stop], gap=0.5),
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
    z_start,
    y_start,
    x_start,
    z_stop,
    y_stop,
    pfm,
    x_stop,
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
        trimmer = (
            slice(z_start.value, z_stop.value),
            slice(y_start.value, y_stop.value),
            slice(x_start.value, x_stop.value),
        )
    else:
        trimmer = None

    mo.md(f"Opening {len(imgs_fp_ls)} image(s) in Napari...")
    view_images(imgs_fp_ls=imgs_fp_ls, trimmer=trimmer)
    mo.md("Napari closed. You may re-launch or adjust selections.")


if __name__ == "__main__":
    app.run()
