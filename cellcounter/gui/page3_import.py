import os

import streamlit as st

from cellcounter.gui.gui_funcs import (
    PROJ_DIR,
    ProjDirStatus,
    init_var,
    page_decorator,
)
from cellcounter.pipeline.pipeline import Pipeline

IMPORT = "import"
OVERWRITE = f"{IMPORT}_overwrite"
INPUT_SRC = f"{IMPORT}_input_src"
INPUT_SRC_STATUS = f"{IMPORT}_input_src_status"
DISABLED = f"{IMPORT}_disabled"
RUN = f"{IMPORT}_run"


def input_src_func():
    # Updating own input variable
    st.session_state[INPUT_SRC] = st.session_state[f"{INPUT_SRC}_w"]
    # Updating session state: INPUT_SRC_STATUS
    if not st.session_state[INPUT_SRC]:
        st.session_state[INPUT_SRC_STATUS] = ProjDirStatus.NOT_SET
    elif not os.path.exists(st.session_state[INPUT_SRC]):
        st.session_state[INPUT_SRC_STATUS] = ProjDirStatus.NOT_EXIST
    else:
        st.session_state[INPUT_SRC_STATUS] = ProjDirStatus.VALID
    # Updating session state: DISABLED
    st.session_state[DISABLED] = st.session_state[INPUT_SRC_STATUS] != ProjDirStatus.VALID


@page_decorator()
def page3_import():
    """ """
    # Initialising session state variables
    init_var(OVERWRITE, False)
    init_var(INPUT_SRC, None)
    init_var(INPUT_SRC_STATUS, ProjDirStatus.NOT_SET)
    init_var(DISABLED, True)

    # Recalling session state variables
    proj_dir = st.session_state[PROJ_DIR]
    pfm = Pipeline.get_pfm(proj_dir)

    st.write("## Import Image")
    st.write("Imports image into project directory as a zarr chunked file.")
    # Overwrite box
    st.toggle(
        label="Overwrite",
        value=st.session_state[OVERWRITE],
        key=OVERWRITE,
    )
    # Input: Source file or directory of image
    st.text_input(
        label="Image source filepath",
        value=st.session_state[INPUT_SRC],
        on_change=input_src_func,
        key=f"{INPUT_SRC}_w",
    )
    # Button: Combine projects
    st.button(
        label="Run",
        disabled=st.session_state[DISABLED],
        key=RUN,
    )
    # Running pipeline
    if st.session_state[RUN]:
        st.write("Running")
        st.write(f"Importing image from {st.session_state[INPUT_SRC]} to {proj_dir}")
        # Running tiff2zarr
        Pipeline.tiff2zarr(
            pfm=pfm,
            in_fp=st.session_state[INPUT_SRC],
            overwrite=st.session_state[OVERWRITE],
        )
