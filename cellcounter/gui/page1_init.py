import os

import streamlit as st
from natsort import natsorted

from cellcounter.gui.gui_funcs import (
    PROJ_DIR,
    PROJ_DIR_STATUS,
    ProjDirStatus,
    init_var,
    load_configs,
    page_decorator,
)
from cellcounter.pipeline.pipeline import Pipeline

INIT = "init"
INPUT_S = f"{INIT}_input_s"
INPUT_M = f"{INIT}_input_m"
SELECT_M = f"{INIT}_select_m"
SELECT_M_OPTIONS = f"{INIT}_select_m_options"
SELECT_M_INDEX = f"{INIT}_select_m_index"
INPUT = f"{INIT}_input"
DISABLED = f"{INIT}_disabled"
SET = f"{INIT}_set"
CREATE = f"{INIT}_create"


def input_s_func():
    # Updating own input variable
    st.session_state[INPUT_S] = st.session_state[f"{INPUT_S}_w"]
    # Updating input and disabled variables
    st.session_state[INPUT] = st.session_state[INPUT_S]
    st.session_state[DISABLED] = st.session_state[INPUT] is None


def input_m_func():
    # Updating own input variable
    st.session_state[INPUT_M] = st.session_state[f"{INPUT_M}_w"]
    # Setting selectbox options
    st.session_state[SELECT_M_OPTIONS] = []
    # If input string is not None and is a directory
    if (st.session_state[INPUT_M] is not None) and (os.path.isdir(st.session_state[INPUT_M])):
        # Getting list of directories in the root string
        st.session_state[SELECT_M_OPTIONS] = natsorted(
            [
                i
                for i in os.listdir(st.session_state[INPUT_M])
                if os.path.isdir(os.path.join(st.session_state[INPUT_M], i))
            ]
        )
    # Setting selectbox index
    st.session_state[SELECT_M_INDEX] = None
    # Resetting input and disabled variables
    st.session_state[INPUT] = None
    st.session_state[DISABLED] = True


def select_m_func():
    # Updating own input variable
    st.session_state[SELECT_M] = st.session_state[f"{SELECT_M}_w"]
    # Updatating selectbox index
    st.session_state[SELECT_M_INDEX] = st.session_state[SELECT_M_OPTIONS].index(st.session_state[SELECT_M])
    # Updating input and disabled variables
    if st.session_state[INPUT_M] and st.session_state[SELECT_M]:
        st.session_state[INPUT] = os.path.join(st.session_state[INPUT_M], st.session_state[SELECT_M])
        st.session_state[DISABLED] = False
    else:
        st.session_state[INPUT] = None
        st.session_state[DISABLED] = True


def set_func():
    """
    Changes the project directory in session state.
    Runs relevant checks.
    """
    pdir_input = st.session_state[INPUT]
    # Setting session_state variables based on proj_dir checks
    # Checking if project directory exists
    if pdir_input is None:
        st.session_state[PROJ_DIR_STATUS] = ProjDirStatus.NOT_SET
    elif not os.path.isdir(pdir_input):
        st.session_state[PROJ_DIR_STATUS] = ProjDirStatus.NOT_EXIST
    else:
        # Storing project directory
        st.session_state[PROJ_DIR] = pdir_input
        try:
            # Project directory is initialised (has configs file)
            load_configs()
            st.session_state[PROJ_DIR_STATUS] = ProjDirStatus.VALID
        except (FileNotFoundError, NotADirectoryError):
            # Project directory is not initialised (give option to create)
            st.session_state[PROJ_DIR_STATUS] = ProjDirStatus.NOT_INIT


def create_func():
    """
    Function to make new project.

    Makes project folders and configs file.
    """
    proj_dir = st.session_state[PROJ_DIR]
    pfm = Pipeline.get_pfm(proj_dir)
    Pipeline.update_configs(pfm)
    # Rerunning set func to update session state
    st.session_state[INPUT] = proj_dir
    set_func()


@page_decorator(check_proj_dir=False)
def page1_init():
    """
    Initializes the project page in the GUI.
    This function sets up the user interface for initializing a project directory.
    It includes input fields for the root directory and buttons to confirm the
    project directory or create a new project if the directory is not initialized.

    The function performs the following steps:
    1. Displays the title "Init Project".
    2. Provides an input field for the user to specify the root directory.
    3. Includes a button to confirm the project directory.
    4. Checks if the specified project directory exists:
       - If it does not exist, displays an error message.
       - If it exists, attempts to load project configurations:
         - If configurations are found, displays a success message.
         - If configurations are not found, displays a warning and provides an
           option to create a new project in the specified directory.

    Note
    ----
    - The function uses Streamlit (`st`) for the GUI components.
    - The project directory and its state are managed using `st.session_state`.

    Raises
    -----
    - FileNotFoundError: If the project directory does not contain the required
      configuration files.
    """
    # Initialising session state variables
    init_var(INPUT_S, None)
    init_var(INPUT_M, None)
    init_var(SELECT_M, None)
    init_var(SELECT_M_OPTIONS, [])
    init_var(SELECT_M_INDEX, None)
    init_var(INPUT, None)
    init_var(DISABLED, True)

    # Title
    st.write("## Init Project")
    # tabs: single or multi project
    tabs = st.tabs(["Single Project", "Multiple Projects"])
    with tabs[0]:
        # Input: Project Directory
        st.text_input(
            label="Project Directory",
            value=st.session_state[INPUT_S],
            on_change=input_s_func,
            key=f"{INPUT_S}_w",
        )
    with tabs[1]:
        # Input: Root Projects Directory
        st.text_input(
            label="Root Directory",
            value=st.session_state[INPUT_M],
            on_change=input_m_func,
            key=f"{INPUT_M}_w",
        )
        if len(st.session_state[SELECT_M_OPTIONS]) == 0:
            st.warning("Root directory is either does not exist or has no folders.")
        else:
            # selectbox: folders (i.e. projects) inside root directory
            st.write("Only folders inside the root directory are listed.")
            st.selectbox(
                label="Projects",
                options=st.session_state[SELECT_M_OPTIONS],
                index=st.session_state[SELECT_M_INDEX],
                on_change=select_m_func,
                key=f"{SELECT_M}_w",
            )
    # Button: Set project directory
    st.button(
        label="Set project directory",
        on_click=set_func,
        disabled=st.session_state[DISABLED],
        key=SET,
    )
    # container: outcome of project directory input
    with st.container():
        if st.session_state[PROJ_DIR_STATUS] == ProjDirStatus.NOT_SET:
            st.warning("Project directory not set")
        elif st.session_state[PROJ_DIR_STATUS] == ProjDirStatus.NOT_EXIST:
            st.error(
                "Project directory does not exist (or is a file and not directory).\n\n"
                "Reverting to existing project directory (if one is set)."
            )
        elif st.session_state[PROJ_DIR_STATUS] == ProjDirStatus.NOT_INIT:
            st.warning(
                "Project directory does not contain config_params file.\n\n"
                "You can create a new project in this directory."
            )
            st.button(
                label="Create new project",
                on_click=create_func,
                key=CREATE,
            )
        elif st.session_state[PROJ_DIR_STATUS] == ProjDirStatus.VALID:
            st.success("Project directory loaded")
