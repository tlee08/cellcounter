import streamlit as st

from cellcounter.pipeline.pipeline import Pipeline

from .gui_funcs import PROJ_DIR, init_var, page_decorator

PIPELINE = "pipeline"
CHECKBOXES = f"{PIPELINE}_checkboxes"
OVERWRITE = f"{PIPELINE}_overwrite"
RUN = f"{PIPELINE}_run"
IS_TUNING = f"{PIPELINE}_is_tuning"


def is_tuning_func():
    # Set tuning mode
    pass


@page_decorator()
def page4_pipeline():
    """
    Displays the pipeline page in the GUI, allowing users to select and run various pipeline functions.

    This function performs the following tasks:
    1. Retrieves the project directory and project file model from the session state.
    2. Displays a toggle for overwriting existing data.
    3. Creates checkboxes for each pipeline function, allowing users to select which functions to run.
    4. Provides a button to run the selected pipeline functions.
    5. Executes the selected pipeline functions when the button is pressed, respecting the overwrite setting.

    Session State Variables:
    - proj_dir: The directory of the current project.
    - pipeline_overwrite: Boolean indicating whether to overwrite existing data.
    - pipeline_checkboxes: Dictionary mapping pipeline functions to their checkbox states.
    - pipeline_run_btn: Boolean indicating whether the "Run pipeline" button has been pressed.
    """
    # Initialising session state variables (if necessary)
    init_var(
        CHECKBOXES,
        # TODO: dynamically generate this dictionary
        {
            Pipeline.reg_ref_prepare: False,
            Pipeline.reg_img_rough: False,
            Pipeline.reg_img_fine: False,
            Pipeline.reg_img_trim: False,
            Pipeline.reg_elastix: False,
            Pipeline.make_mask: False,
            Pipeline.img_overlap: False,
            Pipeline.cellc1: False,
            Pipeline.cellc2: False,
            Pipeline.cellc3: False,
            Pipeline.cellc4: False,
            Pipeline.cellc5: False,
            Pipeline.cellc6: False,
            Pipeline.cellc7: False,
            Pipeline.cellc8: False,
            Pipeline.cellc9: False,
            Pipeline.cellc10: False,
            Pipeline.cellc11: False,
            Pipeline.cellc_coords_only: False,
            Pipeline.transform_coords: False,
            Pipeline.cell_mapping: False,
            Pipeline.group_cells: False,
            Pipeline.cells2csv: False,
            Pipeline.coords2points_raw: False,
            Pipeline.coords2heatmap_raw: False,
            Pipeline.coords2points_trfm: False,
            Pipeline.coords2heatmap_trfm: False,
        },
    )
    init_var(OVERWRITE, False)
    init_var(IS_TUNING, False)

    # Recalling session state variables
    proj_dir = st.session_state[PROJ_DIR]
    pfm = Pipeline.get_pfm(proj_dir)

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
        value=st.session_state[IS_TUNING],
        on_change=is_tuning_func,
        key=IS_TUNING,
    )
    if st.session_state[IS_TUNING]:
        st.write("Tuning mode ON")
        pfm = pfm.copy().convert_to_tuning()
    # Making pipeline checkboxes
    pipeline_checkboxes = st.session_state[CHECKBOXES]
    for func in pipeline_checkboxes:
        st.checkbox(
            label=func.__name__,
            value=pipeline_checkboxes[func],
            key=f"{PIPELINE}_{func.__name__}",
        )
    # Button: run pipeline
    st.button(
        label="Run pipeline",
        key=RUN,
    )
    if st.session_state[RUN]:
        # Showing selected pipeline
        st.write("Running:")
        for func in pipeline_checkboxes:
            if pipeline_checkboxes[func]:
                st.write(f"- {func.__name__}")
        # TODO: ensure this is blocking
        for func in pipeline_checkboxes:
            if pipeline_checkboxes[func]:
                func(
                    pfm=pfm,
                    overwrite=st.session_state[OVERWRITE],
                )
