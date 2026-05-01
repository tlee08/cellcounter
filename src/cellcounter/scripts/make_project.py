"""Create a new cellcounter project with template scripts."""

import importlib.resources
from pathlib import Path


def confirm(prompt: str, default: bool = False) -> bool:
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


def main() -> None:
    """Make a cellcounter pipeline script in the current directory."""
    # Check with user
    confirm_create = confirm("Create cellcounter scripts in current directory?")
    if not confirm_create:
        print("Exiting.")
        return

    overwrite = confirm("Overwrite existing files?", default=False)

    # Copy templates
    templates = importlib.resources.files("cellcounter.templates")
    for template_name in ["run_pipeline.py", "view_img.py"]:
        dst = Path(template_name)
        if dst.exists() and not overwrite:
            print(f"Skipping {template_name} (already exists)")
            continue

        template_content = (templates / template_name).read_text()
        dst.write_text(template_content)
        print(f"Created {template_name}")


if __name__ == "__main__":
    main()
