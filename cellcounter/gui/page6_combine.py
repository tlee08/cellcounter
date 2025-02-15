import os

import streamlit as st
from natsort import natsorted

from cellcounter.funcs.batch_combine_funcs import BatchCombineFuncs
from cellcounter.gui.page1_init import INPUT_M as PDIR_INPUT_M
from cellcounter.pipeline.pipeline import Pipeline

from .gui_funcs import ProjDirStatus, init_var, page_decorator

COMBINE = "combine"
USE_PDIR = f"{COMBINE}_use_pdir"
OVERWRITE = f"{COMBINE}_overwrite"
INPUT_ROOT = f"{COMBINE}_input_root"
INPUT_ROOT_STATUS = f"{COMBINE}_input_root_status"
INPUT_OUT = f"{COMBINE}_input_out"
INPUT_OUT_STATUS = f"{COMBINE}_input_out_status"
CHECKBOX_ALL = f"{COMBINE}_checkbox_all"
CHECKBOXES = f"{COMBINE}_checkboxes"
DISABLED = f"{COMBINE}_disabled"
RUN = f"{COMBINE}_run"


def update_disabled():
    # Disabled is True if:
    # - input_root or
    # - input_out is not valid or
    # - no checkboxes are selected
    st.session_state[DISABLED] = (
        st.session_state[INPUT_ROOT_STATUS] != ProjDirStatus.VALID
        or st.session_state[INPUT_OUT_STATUS] != ProjDirStatus.VALID
        or len(st.session_state[CHECKBOXES]) == 0
    )


def use_pdir_func():
    # Updating session_state: INPUT_ROOT from PDIR_INPUT_M (in page1_init)
    st.session_state[f"{INPUT_ROOT}_w"] = st.session_state.get(PDIR_INPUT_M, None)
    # Running input_root_func (for on_change)
    input_root_func()


def input_root_func():
    # Updating own input variable
    st.session_state[INPUT_ROOT] = st.session_state[f"{INPUT_ROOT}_w"]
    # Updating session state: INPUT_ROOT_STATUS
    if not st.session_state[INPUT_ROOT]:
        st.session_state[INPUT_ROOT_STATUS] = ProjDirStatus.NOT_SET
    elif not os.path.isdir(st.session_state[INPUT_ROOT]):
        st.session_state[INPUT_ROOT_STATUS] = ProjDirStatus.NOT_EXIST
    else:
        st.session_state[INPUT_ROOT_STATUS] = ProjDirStatus.VALID
    # Updating session state: CHECKBOXES
    st.session_state[CHECKBOXES] = {}
    # Getting list of valid directories in root directory
    for i in natsorted(os.listdir(st.session_state[INPUT_ROOT])):
        # Checking project has configs_params, cells_agg, and mask df files
        pdir_i = os.path.join(st.session_state[INPUT_ROOT], i)
        pfm_i = Pipeline.get_pfm(pdir_i)
        if (
            os.path.exists(pfm_i.config_params)
            and os.path.exists(pfm_i.cells_agg_df)
            and os.path.exists(pfm_i.mask_df)
        ):
            st.session_state[CHECKBOXES][i] = False
    # Updating session state: DISABLED
    update_disabled()


def input_out_func():
    # Updating own input variable
    st.session_state[INPUT_OUT] = st.session_state[f"{INPUT_OUT}_w"]
    # Updating session state: INPUT_OUT_STATUS
    if not st.session_state[INPUT_OUT]:
        st.session_state[INPUT_OUT_STATUS] = ProjDirStatus.NOT_SET
    elif not os.path.isdir(st.session_state[INPUT_OUT]):
        st.session_state[INPUT_OUT_STATUS] = ProjDirStatus.NOT_EXIST
    else:
        st.session_state[INPUT_OUT_STATUS] = ProjDirStatus.VALID
    # Updating session state: DISABLED
    update_disabled()


def checkbox_all_func():
    # Updating own input variable
    st.session_state[CHECKBOX_ALL] = st.session_state[f"{CHECKBOX_ALL}_w"]
    # Updating all checkboxes
    for i in st.session_state[CHECKBOXES]:
        st.session_state[f"{COMBINE}_{i}"] = st.session_state[CHECKBOX_ALL]
    # Updating session state: DISABLED
    update_disabled()


@page_decorator(check_proj_dir=False)
def page6_combine():
    # Initialising session state variables
    init_var(OVERWRITE, False)
    init_var(INPUT_ROOT, None)
    init_var(INPUT_ROOT_STATUS, ProjDirStatus.NOT_SET)
    init_var(INPUT_OUT, None)
    init_var(INPUT_OUT_STATUS, ProjDirStatus.NOT_SET)
    init_var(CHECKBOX_ALL, False)
    init_var(CHECKBOXES, {})
    init_var(DISABLED, True)
    init_var(PDIR_INPUT_M, None)  # from page1_init

    # Recalling session state variables

    # Title
    st.write("## Combine Projects in a Directory")
    # Button: use root directory from page1_init
    st.button(
        label="Use Root Directory from 'Init Project'",
        on_click=use_pdir_func,
        disabled=st.session_state[PDIR_INPUT_M] is None,
        key=USE_PDIR,
    )
    # Overwrite box
    st.toggle(
        label="Overwrite",
        value=st.session_state[OVERWRITE],
        key=OVERWRITE,
    )
    # Input: Root Projects Directory
    st.text_input(
        label="Root Directory",
        value=st.session_state[INPUT_ROOT],
        on_change=input_root_func,
        key=f"{INPUT_ROOT}_w",
    )
    # Input: Output Directory
    st.text_input(
        label="Output Directory",
        value=st.session_state[INPUT_OUT],
        on_change=input_out_func,
        key=f"{INPUT_OUT}_w",
    )
    # Error messages for input_root
    if st.session_state[INPUT_ROOT_STATUS] == ProjDirStatus.NOT_SET:
        st.warning("Root directory not set")
    elif st.session_state[INPUT_ROOT_STATUS] == ProjDirStatus.NOT_EXIST:
        st.warning("Root directory does not exist")
    elif st.session_state[INPUT_ROOT_STATUS] == ProjDirStatus.VALID:
        # Making checkboxes for projects inside root directory
        st.write("### Select Projects")
        st.write("Only projects with cells_agg and mask df files are shown.")
        st.write("Must select at least one project to run combine pipeline.")
        st.checkbox(
            label="Select All",
            value=st.session_state[CHECKBOX_ALL],
            on_change=checkbox_all_func,
            key=f"{CHECKBOX_ALL}_w",
        )
        # Making checkboxes
        with st.container(height=250):
            for i in st.session_state[CHECKBOXES]:
                st.session_state[CHECKBOXES][i] = st.checkbox(
                    label=i,
                    value=st.session_state[CHECKBOXES][i],
                    key=f"{COMBINE}_{i}",
                )
    # Error messages for input_out
    if st.session_state[INPUT_OUT_STATUS] == ProjDirStatus.NOT_SET:
        st.warning("Output directory not set.")
    elif st.session_state[INPUT_OUT_STATUS] == ProjDirStatus.NOT_EXIST:
        st.warning(
            "Output directory does not exist.\n\n"
            "Please make one or specify a valid directory."
        )
    elif st.session_state[INPUT_OUT_STATUS] == ProjDirStatus.VALID:
        st.success("Output directory exists and ready to save files to.")
    # Button: Combine projects
    st.button(
        label="Combine",
        disabled=st.session_state[DISABLED],
        key=RUN,
    )
    if st.session_state[RUN]:
        # Making list of project directories
        proj_dir_ls = [k for k, v in st.session_state[CHECKBOXES].items() if v]
        st.write("Combining:")
        for i in proj_dir_ls:
            st.write(f"- {i}")
        # Updating project directories to full path
        proj_dir_ls = [
            os.path.join(st.session_state[INPUT_ROOT], i) for i in proj_dir_ls
        ]
        # Running combine func
        BatchCombineFuncs.combine_ls_pipeline(
            proj_dir_ls=proj_dir_ls,
            out_dir=st.session_state[INPUT_OUT],
            overwrite=st.session_state[OVERWRITE],
        )
