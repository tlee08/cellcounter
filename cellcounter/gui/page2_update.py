from enum import Enum, EnumType
from types import NoneType, UnionType
from typing import Any, Callable, Optional, Union, get_args, get_origin

import streamlit as st
from pydantic import BaseModel
from streamlit.delta_generator import DeltaGenerator

from cellcounter.constants import Coords
from cellcounter.gui.gui_funcs import (
    CONFIGS,
    PROJ_DIR,
    SliceNames,
    load_configs,
    page_decorator,
)
from cellcounter.pipeline.pipeline import Pipeline
from cellcounter.utils.config_params_model import ConfigParamsModel
from cellcounter.utils.io_utils import write_json
from cellcounter.utils.misc_utils import const2list, dictlists2listdicts, enum2list

UPDATE = "update"
VALUE = f"{UPDATE}_value"
IS_NONE = f"{UPDATE}_is_none"
DEFAULT = f"{UPDATE}_default"
CONFIGS_RESET = f"{UPDATE}_reset"
CONFIGS_SAVE = f"{UPDATE}_save"


class NO_DEFAULT:
    pass


SUBLABEL_NAMES_MAP = {
    "chunksize": tuple(enum2list(Coords)),
    "ref_orient_ls": tuple(enum2list(Coords)),
    "ref_z_trim": tuple(enum2list(SliceNames)),
    "ref_y_trim": tuple(enum2list(SliceNames)),
    "ref_x_trim": tuple(enum2list(SliceNames)),
    "z_trim": tuple(enum2list(SliceNames)),
    "y_trim": tuple(enum2list(SliceNames)),
    "x_trim": tuple(enum2list(SliceNames)),
}


class ConfigsUpdater:
    """
    There's containers and columns, but returns
    the inputted values in the end
    (i.e. containers not returned).
    """

    @classmethod
    def init_inputter(
        cls,
        label: str,
        curr: Any = None,
        nullable: bool = False,
        default: Any = NO_DEFAULT,
        container=None,
    ) -> tuple[DeltaGenerator, Any, bool]:
        # Making container
        container = container or st
        container = container.container(border=True)
        # Header
        container.write(label)
        # If default exists, making a default button
        if default != NO_DEFAULT:
            container.button(
                label=f"Set as default (`{default}`)",
                key=f"{DEFAULT}_{label}",
            )
            # If button clicked, setting curr to default
            # and update widgets accordingly
            if st.session_state[f"{DEFAULT}_{label}"]:
                curr = default
                st.session_state[f"{VALUE}_{label}"] = curr
                st.session_state[f"{IS_NONE}_{label}"] = curr is None
        # If nullable, making nullable checkbox
        is_none = False
        if nullable:
            is_none = container.toggle(
                label="Set to None",
                value=curr is None,
                key=f"{IS_NONE}_{label}",
            )
        return container, curr, is_none

    @classmethod
    def enum_input(
        cls,
        label: str,
        my_enum: EnumType,
        curr: Optional[Any] = None,
        nullable: bool = False,
        default: Any = NO_DEFAULT,
        container=None,
        **kwargs,
    ) -> Optional[Any]:
        # Initialising container, nullable, and default widgets
        container, curr, is_none = cls.init_inputter(
            label, curr, nullable, default, container
        )
        # Selectbox
        output = container.selectbox(
            label=label,
            options=enum2list(my_enum),
            index=enum2list(my_enum).index(curr) if curr else None,
            disabled=is_none,
            key=f"{VALUE}_{label}",
            label_visibility="collapsed",
        )  # type: ignore
        return None if is_none else output

    @classmethod
    def int_input(
        cls,
        label: str,
        curr: Optional[int] = None,
        nullable: bool = False,
        default: Any = NO_DEFAULT,
        container=None,
        **kwargs,
    ) -> Optional[int]:
        # Initialising container, nullable, and default widgets
        container, curr, is_none = cls.init_inputter(
            label, curr, nullable, default, container
        )
        output = container.number_input(
            label=label,
            value=curr,
            step=1,
            disabled=is_none,
            key=f"{VALUE}_{label}",
            label_visibility="collapsed",
        )  # type: ignore
        return None if is_none else output

    @classmethod
    def float_input(
        cls,
        label: str,
        curr: Optional[float] = None,
        nullable: bool = False,
        default: Any = NO_DEFAULT,
        container=None,
        **kwargs,
    ) -> Optional[float]:
        # Initialising container, nullable, and default widgets
        container, curr, is_none = cls.init_inputter(
            label, curr, nullable, default, container
        )
        output = container.number_input(
            label=label,
            value=curr,
            step=0.05,
            disabled=is_none,
            key=f"{VALUE}_{label}",
            label_visibility="collapsed",
        )  # type: ignore
        return None if is_none else output

    @classmethod
    def str_input(
        cls,
        label: str,
        curr: Optional[str] = None,
        nullable: bool = False,
        default: Any = NO_DEFAULT,
        container=None,
        **kwargs,
    ) -> Optional[str]:
        # Initialising container, nullable, and default widgets
        container, curr, is_none = cls.init_inputter(
            label, curr, nullable, default, container
        )
        output = container.text_input(
            label=label,
            value=curr,
            disabled=is_none,
            key=f"{VALUE}_{label}",
            label_visibility="collapsed",
        )
        return None if is_none else output

    @classmethod
    def tuple_inputs(
        cls,
        label: str,
        n: int,
        func: Callable | tuple[Callable, ...],
        nullable: bool | tuple[bool, ...] = False,
        default: Any | tuple[Any, ...] = NO_DEFAULT,
        container=None,
        sublabels: Optional[tuple[str, ...]] = None,
        **kwargs,
    ) -> tuple[Any, ...]:
        # Making container
        container = container or st
        container = container.container(border=True)
        # Header
        container.write(label)
        # For each number input, making into list to input to func
        columns_ls = container.columns(n)
        output_ls = [None for _ in range(n)]
        func_ls = func if isinstance(func, tuple) else const2list(func, n)
        nullable_ls = (
            nullable if isinstance(nullable, tuple) else const2list(nullable, n)
        )
        default_ls = default if isinstance(default, tuple) else const2list(default, n)
        sublabels_ls = sublabels if isinstance(sublabels, tuple) else range(n)
        sublabels_ls = [f"{label}_{sublabel}" for sublabel in sublabels_ls]
        # Making kwargs into kwargs_ls dict of lists
        kwargs_ls = {
            k: v if isinstance(v, tuple) else const2list(v, n)
            for k, v in kwargs.items()
        }
        # Asserting all list lengths are equal to n
        assert len(func_ls) == n
        assert len(nullable_ls) == n
        assert len(default_ls) == n
        assert len(sublabels_ls) == n
        # Asserting all kwargs_ls list lengths are equal to n
        for k, v in kwargs_ls.items():
            assert len(v) == n
        # "Transposing" kwargs_ls so it becomes a list of dicts.
        kwargs_ls = dictlists2listdicts(kwargs_ls)
        # Making inputs
        output_ls = tuple(
            func_ls[i](
                label=sublabels_ls[i],
                nullable=nullable_ls[i],
                default=default_ls[i],
                container=columns_ls[i],
                **kwargs_ls[i],
            )
            for i in range(n)
        )
        return output_ls

    @classmethod
    def type2updater(cls, my_type: type) -> Callable:
        if issubclass(my_type, Enum):
            return lambda **kwargs: cls.enum_input(my_enum=my_type, **kwargs)
        elif my_type is int:
            return cls.int_input
        elif my_type is float:
            return cls.float_input
        elif my_type is str:
            return cls.str_input
        else:
            raise NotImplementedError(f"Type {my_type} not implemented")

    @staticmethod
    def get_type_and_nullable(my_type):
        """
        Returns tuple of:
        - The non-nullable type
        - Whether the type is nullable
        """
        origin = get_origin(my_type) or my_type
        args = list(get_args(my_type))
        nullable = False
        # If a Union, then checking if NoneType is a possible type
        if origin in [Union, UnionType, Optional] and NoneType in args:
            args.remove(NoneType)
            nullable = True
            # TODO: what happens if multiple args?
            origin = args[0]
        # Returns (non-nullable type, is_nullable)
        return nullable, origin

    @classmethod
    def field2updater(
        cls,
        pydantic_instance: BaseModel,
        field_name: str,
        **kwargs,
    ):
        """
        Builds a streamlit value updater widget for a given pydantic field.

        The output is saved to the pydantic_instance object AND the session state.
        Session state values are saved as:
            - f"{DEFAULT}_{label}"
            - f"{VALUE}_{label}" or f"{VALUE}_{label}_{sublabel}" (for tuple)
            - f"{IS_NONE}_{label}" or f"{IS_NONE}_{label}_{sublabel}" (for tuple)
        """
        # Getting field value and info
        curr = getattr(pydantic_instance, field_name)
        field_info = pydantic_instance.model_fields[field_name]
        default = field_info.default
        # Getting type and nullable
        nullable, my_type = cls.get_type_and_nullable(field_info.annotation)
        args = get_args(field_info.annotation)

        # If a tuple, then building tuple inputs
        output = None
        if issubclass(my_type, tuple):
            # Building tuple inputs
            funcs_ls = []
            nullable_ls = []
            for i, arg in enumerate(args):
                nullable, my_type = cls.get_type_and_nullable(arg)
                funcs_ls.append(cls.type2updater(my_type))
                nullable_ls.append(nullable)
            output = cls.tuple_inputs(
                label=field_name,
                n=len(curr),
                func=tuple(funcs_ls),
                nullable=tuple(nullable_ls),
                default=default,
                curr=curr,
                **kwargs,
            )
        else:
            # Otherwise, using type2updated immediately
            output = cls.type2updater(my_type)(
                label=field_name,
                curr=curr,
                nullable=nullable,
                default=default,
                **kwargs,
            )
        # Setting to pydantic_instance
        setattr(pydantic_instance, field_name, output)

    @classmethod
    def field2updater_mapped(
        cls,
        pydantic_instance: BaseModel,
        field_name: str,
        **kwargs,
    ):
        sublabels = SUBLABEL_NAMES_MAP.get(field_name, None)
        cls.field2updater(
            pydantic_instance=pydantic_instance,
            field_name=field_name,
            sublabels=sublabels,
            **kwargs,
        )


@staticmethod
def configs_reset_func():
    """
    For each config parameter, resets the value to the value from disk.

    Also updates the session state variable that are dependent on this value:
    - `{label}`: the value
    - `{label}_is_none`: whether the value is None
    """
    # Loading and getting configs from disk
    load_configs()
    configs: ConfigParamsModel = st.session_state[CONFIGS]
    # For each field, setting the value to the value from disk
    for label, value in configs.model_dump().items():
        if isinstance(value, tuple):
            # If field is a tuple, then setting each value in tuple separately
            # Expects an entry in SUBLABEL_MAP
            for i, sublabel in enumerate(SUBLABEL_NAMES_MAP[label]):
                st.session_state[f"{VALUE}_{label}_{sublabel}"] = value[i]
                st.session_state[f"{IS_NONE}_{label}_{sublabel}"] = value[i] is None
        else:
            # Otherwise, setting the value directly
            st.session_state[f"{VALUE}_{label}"] = value
            st.session_state[f"{IS_NONE}_{label}"] = value is None


def configs_save_func():
    """
    Saving configs from session state to project directory.

    NOTE: does not catch errors
    """
    configs: ConfigParamsModel = st.session_state[CONFIGS]
    proj_dir = st.session_state[PROJ_DIR]
    pfm = Pipeline.get_pfm(proj_dir)
    fp = pfm.config_params
    write_json(fp, configs.model_dump())


@page_decorator()
def page2_configs():
    """
    Displays and allows editing of configuration parameters for the project.
    This function uses Streamlit to create an interactive GUI for editing various
    configuration parameters stored in the session state. The parameters are grouped
    into different categories such as Reference, Raw, Registration, Mask, Overlap,
    and Cell Counting. Each category is expandable and allows the user to update
    specific fields.

    The function performs the following tasks:
    1. Retrieves the current configuration parameters from the session state.
    2. Displays the configuration parameters in expandable sections for editing.
    3. Validates and updates the configuration parameters in the session state.
    4. Provides a button to save the updated configuration parameters.
    The configuration parameters are instances of the `ConfigParamsModel` class.

    Note
    ----
        The function assumes that `ConfigsUpdater` and `ConfigParamsModel` are
        defined elsewhere in the codebase.

    Raises
    ------
        ValidationError: If the configuration parameters do not pass validation.
    """
    # Initialising session state variables

    # Recalling session state variables
    configs: ConfigParamsModel = st.session_state[CONFIGS]

    st.write("# Edit Configs")
    with st.expander("Reference"):
        ConfigsUpdater.field2updater_mapped(configs, "atlas_dir")
        ConfigsUpdater.field2updater_mapped(configs, "ref_version")
        ConfigsUpdater.field2updater_mapped(configs, "annot_version")
        ConfigsUpdater.field2updater_mapped(configs, "map_version")
    with st.expander("Raw"):
        ConfigsUpdater.field2updater_mapped(configs, "zarr_chunksize")
    with st.expander("Registration"):
        # TODO: numerical it is unintuitive for selecting axes in ref_orienf_ls
        ConfigsUpdater.field2updater_mapped(configs, "ref_orient_ls")
        ConfigsUpdater.field2updater_mapped(configs, "ref_z_trim")
        ConfigsUpdater.field2updater_mapped(configs, "ref_y_trim")
        ConfigsUpdater.field2updater_mapped(configs, "ref_x_trim")
        ConfigsUpdater.field2updater_mapped(configs, "z_rough")
        ConfigsUpdater.field2updater_mapped(configs, "y_rough")
        ConfigsUpdater.field2updater_mapped(configs, "x_rough")
        ConfigsUpdater.field2updater_mapped(configs, "z_fine")
        ConfigsUpdater.field2updater_mapped(configs, "y_fine")
        ConfigsUpdater.field2updater_mapped(configs, "x_fine")
        ConfigsUpdater.field2updater_mapped(configs, "z_trim")
        ConfigsUpdater.field2updater_mapped(configs, "y_trim")
        ConfigsUpdater.field2updater_mapped(configs, "x_trim")
    with st.expander("Mask"):
        ConfigsUpdater.field2updater_mapped(configs, "mask_gaus_blur")
        ConfigsUpdater.field2updater_mapped(configs, "mask_thresh")
    with st.expander("Cell Count Tuning"):
        ConfigsUpdater.field2updater_mapped(configs, "tuning_z_trim")
        ConfigsUpdater.field2updater_mapped(configs, "tuning_y_trim")
        ConfigsUpdater.field2updater_mapped(configs, "tuning_x_trim")
    with st.expander("Overlap"):
        ConfigsUpdater.field2updater_mapped(configs, "overlap_depth")
    with st.expander("Cell Counting"):
        ConfigsUpdater.field2updater_mapped(configs, "tophat_sigma")
        ConfigsUpdater.field2updater_mapped(configs, "dog_sigma1")
        ConfigsUpdater.field2updater_mapped(configs, "dog_sigma2")
        ConfigsUpdater.field2updater_mapped(configs, "large_gauss_sigma")
        ConfigsUpdater.field2updater_mapped(configs, "threshd_value")
        ConfigsUpdater.field2updater_mapped(configs, "min_threshd_size")
        ConfigsUpdater.field2updater_mapped(configs, "max_threshd_size")
        ConfigsUpdater.field2updater_mapped(configs, "maxima_sigma")
        ConfigsUpdater.field2updater_mapped(configs, "min_wshed_size")
        ConfigsUpdater.field2updater_mapped(configs, "max_wshed_size")
    with st.expander("Visual Check"):
        ConfigsUpdater.field2updater_mapped(configs, "heatmap_raw_radius")
        ConfigsUpdater.field2updater_mapped(configs, "heatmap_trfm_radius")
    with st.expander("Combine Arrays"):
        ConfigsUpdater.field2updater_mapped(configs, "combine_cellc_z_trim")
        ConfigsUpdater.field2updater_mapped(configs, "combine_cellc_y_trim")
        ConfigsUpdater.field2updater_mapped(configs, "combine_cellc_x_trim")

    # Checking configs and updating in session_state
    # NOTE: an error can occur with the validation here
    configs = ConfigParamsModel.model_validate(configs)
    st.session_state[CONFIGS] = configs

    # Showing updated configs
    st.write("# See Project Configs")
    with st.expander("See Configs"):
        # JSON configs
        st.json(configs.model_dump())

    columns = st.columns(2)
    with columns[0]:
        # Button: Reset to old saved configs
        st.button(
            label="Reset",
            on_click=configs_reset_func,
            key=CONFIGS_RESET,
        )
        if st.session_state[CONFIGS_RESET]:
            st.success("Resetted project directory and configs")
    with columns[1]:
        # Button: Save new configs
        columns[1].button(
            label="Save",
            on_click=configs_save_func,
            key=CONFIGS_SAVE,
        )
        if st.session_state[CONFIGS_SAVE]:
            st.success("New configs saved to project directory")
