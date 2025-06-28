import inspect
from enum import EnumType
from importlib.util import find_spec
from typing import Any, Iterable


def get_module_dir(module_name: str) -> str:
    module_spec = find_spec(module_name)
    if module_spec is None:
        raise ModuleNotFoundError(f"Module '{module_name}' not found.")
    submodules = module_spec.submodule_search_locations
    if not submodules:
        raise ModuleNotFoundError(f"Module '{module_name}' has no submodules.")
    return submodules[0]


def import_extra_error_func(extra_dep_name: str):
    def error_func(*args, **kwargs):
        raise ImportError(
            f'{extra_dep_name} dependency not installed.\nInstall with `pip install "cellcounter[{extra_dep_name}]"`'
        )

    return error_func


def enum2list(my_enum: EnumType) -> list[Any]:
    return [e.value for e in my_enum]  # type: ignore


def const2iter(x: Any, n: int) -> Iterable[Any]:
    """
    Iterates the object, `x`, `n` times.
    """
    for _ in range(n):
        yield x


def const2list(x: Any, n: int) -> list[Any]:
    """
    Iterates the list, `ls`, `n` times.
    """
    return [x for _ in range(n)]


def dictlists2listdicts(my_dict):
    """
    Converts a dict of lists to a list of dicts.
    """
    # Asserting that all values (lists) have same size
    n = len(list(my_dict.values())[0])
    for i in my_dict.values():
        assert len(i) == n
    # Making list of dicts
    return [{k: v[i] for k, v in my_dict.items()} for i in range(n)]


def listdicts2dictlists(my_list):
    """
    Converts a list of dicts to a dict of lists.
    """
    # Asserting that each dict has the same keys
    keys = my_list[0].keys()
    for i in my_list:
        assert i.keys() == keys
    # Making dict of lists
    return {k: [v[k] for v in my_list] for k in keys}


def get_func_name_in_stack(levels_back: int = 1) -> str:
    """
    Returns the name of the function that called this function.
    This is useful for debugging and dynamically changing function behavior
    (e.g. getting attributes according to the functions name).

    Parameters
    ----------
    levels_back : int
        The number of levels back in the stack to get the function name from.
        0 is the function itself ("get_func_name_in_stack"), 1 is the function it's called from, etc.
        Default is 1 (i.e. the function that called this function).

    Returns
    -------
    str
        The name of the function at the given stack level. If the level is out of range, returns an empty string.

    Notes
    -----
    If this function is called from the main script (i.e. no function),
    it will return an empty string.

    Examples
    --------
    Where `levels_back = 0`
    ```
    f_name = get_func_name_in_stack(0)
    # f_name == "get_func_name_in_stack"
    ```
    Where `levels_back = 1`
    ```
    def my_func():
        f_name = get_func_name_in_stack(1)
        # f_name == "my_func"
    ```
    Where `levels_back = 2`
    ```
    def my_func():
        f_name = get_func_name_in_stack(2)
        # f_name == ""
    ```
    """
    # Getting the current frame
    c_frame = inspect.currentframe()
    # Traverse back the specified number of levels
    for _ in range(levels_back):
        if c_frame is None:
            return ""
        c_frame = c_frame.f_back
    # If the frame is None, return an empty string
    if c_frame is None:
        return ""
    # Returning function name
    return c_frame.f_code.co_name
