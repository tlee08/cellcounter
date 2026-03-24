from collections.abc import Iterable
from enum import EnumType
from importlib.util import find_spec
from typing import Any


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
    return [e.value for e in my_enum]  # ty:ignore[unresolved-attribute]


def const2iter(x: Any, n: int) -> Iterable[Any]:
    """Iterates the object, `x`, `n` times."""
    for _ in range(n):
        yield x


def const2list(x: Any, n: int) -> list[Any]:
    """Iterates the list, `ls`, `n` times."""
    return [x for _ in range(n)]


def dictlists2listdicts(my_dict: dict[str, list[Any]]) -> list[dict[str, Any]]:
    """Converts a dict of lists to a list of dicts."""
    # Asserting that all values (lists) have same size
    n = len(list(my_dict.values())[0])
    for i in my_dict.values():
        assert len(i) == n
    # Making list of dicts
    return [{k: v[i] for k, v in my_dict.items()} for i in range(n)]


def listdicts2dictlists(my_list: list[dict[str, Any]]) -> dict[str, list[Any]]:
    """Converts a list of dicts to a dict of lists."""
    # Asserting that each dict has the same keys
    keys = my_list[0].keys()
    for i in my_list:
        assert i.keys() == keys
    # Making dict of lists
    return {k: [v[k] for v in my_list] for k in keys}
