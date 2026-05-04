"""Create a new cellcounter project with template scripts."""

import importlib.resources
from pathlib import Path

from cellcounter.utils.template_utils import confirm


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
