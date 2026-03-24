"""Utility functions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from jinja2 import Environment, PackageLoader


def render_template(
    tmpl_name: str, pkg_name: str, pkg_subdir: str, **kwargs: Any
) -> str:
    """Renders the given template with the given arguments."""
    # Loading the Jinja2 environment
    env = Environment(loader=PackageLoader(pkg_name, pkg_subdir), autoescape=True)
    # Getting the template
    template = env.get_template(tmpl_name)
    # Rendering the template
    return template.render(**kwargs)


def save_template(
    tmpl_name: str, pkg_name: str, pkg_subdir: str, dst_fp: Path | str, **kwargs: Any
) -> None:
    """Renders the given template with the given arguments and saves it to the out_fp."""
    dst_fp = Path(dst_fp)
    # Rendering the template
    rendered = render_template(tmpl_name, pkg_name, pkg_subdir, **kwargs)
    # Making the directory if it doesn't exist
    dst_fp.parent.mkdir(exist_ok=True)
    # Saving the rendered template
    with dst_fp.open(mode="w") as f:
        f.write(rendered)


def import_static_templates_script(
    description: str,
    templates_ls: list[str],
    pkg_name: str,
    pkg_subdir: str,
    root_dir: Path | str = ".",
    *,
    overwrite: bool = False,
    dialogue: bool = True,
) -> None:
    """A function to import static templates to a folder.

    Useful for calling scripts from the command line.
    """
    root_dir = Path(root_dir)
    if dialogue:
        # Dialogue to check if the user wants to make the files
        to_continue = (
            input(
                f"Running {description} in current directory. Continue? [y/N]: "
            ).lower()
            + " "
        )
        if to_continue[0] != "y":
            print("Exiting.")
            return
        # Dialogue to check if the user wants to overwrite the files
        to_overwrite = input("Overwrite existing files? [y/N]: ").lower() + " "
        overwrite = to_overwrite[0] == "y"
    # Making the root folder
    root_dir.mkdir(exist_ok=True)
    # Copying the Python files to the project folder
    for template_fp in templates_ls:
        dst_fp = root_dir / template_fp
        if not overwrite and dst_fp.exists():
            # Check if we should skip importing (i.e. overwrite is False and file exists)
            print(f"File already exists: {dst_fp}")
            continue
        save_template(template_fp, pkg_name, pkg_subdir, dst_fp)
