import functools
from enum import Enum
from typing import Any

import streamlit as st

from cellcounter.pipeline.pipeline import Pipeline
from cellcounter.utils.config_params_model import ConfigParamsModel
from cellcounter.utils.io_utils import read_json

PROJ_DIR = "proj_dir"
PROJ_DIR_STATUS = "proj_dir_status"

CONFIGS = "configs"


class SliceNames(Enum):
    START = "start"
    STOP = "stop"
    STEP = "step"


class ProjDirStatus(Enum):
    NOT_SET = "not_set"
    NOT_EXIST = "not_exist"
    NOT_INIT = "not_init"
    VALID = "valid"


#####################################################################
# Pipeline Functions (callbacks)
#####################################################################


def init_var(name: str, default: Any):
    """
    Initialising session state variable with default value if not already set.
    """
    st.session_state[name] = st.session_state.get(name, default)


def load_configs():
    """
    Loading in configs to session state from project directory.

    NOTE: does not catch errors
    """
    proj_dir = st.session_state[PROJ_DIR]
    pfm = Pipeline.get_pfm(proj_dir)
    fp = pfm.config_params
    st.session_state[CONFIGS] = ConfigParamsModel.model_validate(read_json(fp))


#####################################################################
# Common Funcs
#####################################################################


def page_decorator(check_proj_dir=True):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Recalling session state variables
            proj_dir = st.session_state[PROJ_DIR]
            proj_dir_status = st.session_state[PROJ_DIR_STATUS]
            # Checking whether project exists
            with st.sidebar:
                st.subheader(f"Root Directory: {proj_dir}")
                # Outputting project directory status
                if proj_dir_status == ProjDirStatus.NOT_SET:
                    st.warning("Project directory is not set")
                elif proj_dir_status == ProjDirStatus.NOT_EXIST:
                    st.warning("Project directory does not exist")
                elif proj_dir_status == ProjDirStatus.NOT_INIT:
                    st.warning("Project directory is not initialised")
                elif proj_dir_status == ProjDirStatus.VALID:
                    load_configs()
                    st.success("Loaded project directory and configs")
            # If project not exists and check_proj_dir is True
            # then don't render rest of page
            if check_proj_dir and proj_dir_status != ProjDirStatus.VALID:
                st.error("Project is not initialised.\n\nPlease set or create a project directory.")
                return
            return func(*args, **kwargs)

        return wrapper

    return decorator
