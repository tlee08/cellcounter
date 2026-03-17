"""
Utility functions.
"""

from __future__ import annotations

import os
from typing import Any

from jinja2 import Environment, PackageLoader

from cellcounter.utils.diagnostics_utils import file_exists_msg


def render_template(tmpl_name: str, pkg_name: str, pkg_subdir: str, **kwargs: Any) -> str:
    """
    Renders the given template with the given arguments.
    """
    # Loading the Jinja2 environment
    env = Environment(loader=PackageLoader(pkg_name, pkg_subdir))
    # Getting the template
    template = env.get_template(tmpl_name)
    # Rendering the template
    return template.render(**kwargs)


def save_template(tmpl_name: str, pkg_name: str, pkg_subdir: str, dst_fp: str, **kwargs: Any) -> None:
    """
    Renders the given template with the given arguments and saves it to the out_fp.
    """
    # Rendering the template
    rendered = render_template(tmpl_name, pkg_name, pkg_subdir, **kwargs)
    # Making the directory if it doesn't exist
    os.makedirs(os.path.dirname(dst_fp), exist_ok=True)
    # Saving the rendered template
    with open(dst_fp, "w") as f:
        f.write(rendered)


def import_static_templates_script(
    description: str,
    templates_ls: list[str],
    pkg_name: str,
    pkg_subdir: str,
    root_dir: str = ".",
    overwrite: bool = False,
    dialogue: bool = True,
) -> None:
    """
    A function to import static templates to a folder.
    Useful for calling scripts from the command line.
    """
    if dialogue:
        # Dialogue to check if the user wants to make the files
        to_continue = input(f"Running {description} in current directory. Continue? [y/N]: ").lower() + " "
        if to_continue[0] != "y":
            print("Exiting.")
            return
        # Dialogue to check if the user wants to overwrite the files
        to_overwrite = input("Overwrite existing files? [y/N]: ").lower() + " "
        overwrite = to_overwrite[0] == "y"
    # Making the root folder
    os.makedirs(root_dir, exist_ok=True)
    # Copying the Python files to the project folder
    for template_fp in templates_ls:
        dst_fp = os.path.join(root_dir, template_fp)
        if not overwrite and os.path.exists(dst_fp):
            # Check if we should skip importing (i.e. overwrite is False and file exists)
            print(file_exists_msg(dst_fp))
            continue
        save_template(template_fp, pkg_name, pkg_subdir, dst_fp)
