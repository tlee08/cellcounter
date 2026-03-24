import streamlit as st

from cellcounter.pipeline.pipeline import Pipeline

from .gui_funcs import PROJ_DIR, init_var, page_decorator

PIPELINE = "pipeline"
CHECKBOXES = f"{PIPELINE}_checkboxes"
OVERWRITE = f"{PIPELINE}_overwrite"
RUN = f"{PIPELINE}_run"
IS_TUNING = f"{PIPELINE}_is_tuning"

# Pipeline step names for checkboxes
STEP_NAMES = [
    "reg_ref_prepare",
    "reg_img_rough",
    "reg_img_fine",
    "reg_img_trim",
    "reg_img_bound",
    "reg_elastix",
    "tophat_filter",
    "dog_filter",
    "adaptive_threshold_prep",
    "threshold",
    "label_thresholded",
    "compute_thresholded_volumes",
    "filter_thresholded",
    "detect_maxima",
    "label_maxima",
    "watershed",
    "compute_watershed_volumes",
    "filter_watershed",
    "save_cells_table",
    "transform_coords",
    "cell_mapping",
    "group_cells",
    "cells2csv",
]


def is_tuning_func() -> None:
    # Set tuning mode
    pass


@page_decorator()
def page4_pipeline() -> None:
    """Displays the pipeline page in the GUI, allowing users to select and run various pipeline functions."""
    # Initialising session state variables
    init_var(CHECKBOXES, dict.fromkeys(STEP_NAMES, False))
    init_var(OVERWRITE, False)
    init_var(IS_TUNING, False)

    # Recalling session state variables
    proj_dir = st.session_state[PROJ_DIR]
    is_tuning = st.session_state[IS_TUNING]

    st.write("## Pipeline")
    # Making overwrite box
    st.toggle(
        label="Overwrite",
        value=st.session_state[OVERWRITE],
        key=OVERWRITE,
    )
    # Making is_tuning box
    st.toggle(
        label="Switch to tuning mode",
        value=is_tuning,
        on_change=is_tuning_func,
        key=IS_TUNING,
    )
    if is_tuning:
        st.write("Tuning mode ON")

    # Making pipeline checkboxes
    pipeline_checkboxes = st.session_state[CHECKBOXES]
    for step_name in STEP_NAMES:
        st.checkbox(
            label=step_name,
            value=pipeline_checkboxes.get(step_name, False),
            key=f"{PIPELINE}_{step_name}",
        )

    # Button: run pipeline
    st.button(
        label="Run pipeline",
        key=RUN,
    )
    if st.session_state[RUN]:
        # Create pipeline instance
        pipeline = Pipeline(proj_dir, tuning=is_tuning)

        # Showing selected pipeline
        st.write("Running:")
        for step_name in STEP_NAMES:
            if pipeline_checkboxes.get(step_name, False):
                st.write(f"- {step_name}")

        # Run selected steps
        for step_name in STEP_NAMES:
            if pipeline_checkboxes.get(step_name, False):
                getattr(pipeline, step_name)(overwrite=st.session_state[OVERWRITE])
