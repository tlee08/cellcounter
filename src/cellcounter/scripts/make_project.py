"""Create a new cellcounter project with template files."""

import shutil
import sys
from importlib.resources import files
from pathlib import Path

TEMPLATE_DIR = Path(str(files("cellcounter.templates")))
FILES = ["run_pipeline.py", "view_img.py", "default_config.yaml"]
FOLDERS = ["analysis_images"]


def _confirm(prompt: str) -> bool:
    while True:
        resp = input(f"{prompt} [y/N]: ").strip().lower()
        if resp in ("y", "yes"):
            return True
        if resp in ("", "n", "no"):
            return False


def main() -> None:
    target = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path.cwd()

    if not _confirm(f"Create project in {target}?"):
        return

    target.mkdir(parents=True, exist_ok=True)

    # Make files
    for file in FILES:
        dst_fp = target / file
        if dst_fp.exists() and not _confirm(f"Overwrite {file}?"):
            continue
        shutil.copy(str(TEMPLATE_DIR / file), dst_fp)
        print(f"Copied: {file}")

    # Make folders
    for folder in FOLDERS:
        (target / folder).mkdir(parents=True, exist_ok=True)

    print(f"\nNext:\n  marimo edit {target / 'run_pipeline.py'}")


if __name__ == "__main__":
    main()
