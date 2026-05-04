"""Template utilities."""

from pathlib import Path
from typing import Any

from jinja2 import Environment, PackageLoader


def confirm(prompt: str, *, default: bool = False) -> bool:
    """Get yes/no confirmation from user."""
    hint = "[Y/n]" if default else "[y/N]"
    while True:
        response = input(f"{prompt} {hint}: ").strip().lower()
        if not response:
            return default
        if response in ("y", "yes"):
            return True
        if response in ("n", "no"):
            return False
        print("Please enter 'y' or 'n'.")


def save_template(template_name: str, dst: Path, **kwargs: Any) -> None:
    """Render and save a template."""
    env = Environment(loader=PackageLoader("cellcounter", "templates"), autoescape=True)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(env.get_template(template_name).render(**kwargs))
