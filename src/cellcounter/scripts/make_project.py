"""Create a new cellcounter project with template scripts."""

from pathlib import Path

from cellcounter.utils.template_utils import confirm, save_template


def main() -> None:
    """Make a cellcounter pipeline script in the current directory."""
    if not confirm("Create cellcounter scripts in current directory?"):
        return

    overwrite = confirm("Overwrite existing files?")

    if overwrite or not Path("run_pipeline.py").exists():
        save_template("run_pipeline.py", Path("run_pipeline.py"))

    if overwrite or not Path("view_img.py").exists():
        save_template("view_img.py", Path("view_img.py"))


if __name__ == "__main__":
    main()
