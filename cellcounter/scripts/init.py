# Download Allen Brain Atlas atlas resources
# Atlas from https://download.alleninstitute.org/informatics-archive/current-release/mouse_ccf/
# Also have a mirror at: https://www.dropbox.com/scl/fi/63lavisnxp2osri44029s/atlas_resources.zip?rlkey=qveuyr7awqcf67n36s944wkp6&st=tbe19m64&dl=0
# Downloadable by adding "dl=1" as a query to the URL


import os
import subprocess

from cellcounter.constants import ATLAS_DIR, CACHE_DIR


def main() -> None:
    """
    Sets up atlas resources. in the ~/.cellcounter cache directory.
    """
    # Downloading the Allen Brain Atlas
    atlas_url = "https://www.dropbox.com/scl/fi/63lavisnxp2osri44029s/atlas_resources.zip?rlkey=qveuyr7awqcf67n36s944wkp6&st=sydcwxmy&dl=1"
    output_fp = f"{ATLAS_DIR}.zip"
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Assert whether atlas resources already exist
    assert not os.path.exists(ATLAS_DIR), f"Atlas resources already exist at {ATLAS_DIR}!"

    # Running
    for cmd_str in [
        f'wget "{atlas_url}" -O {output_fp}',
        f"unzip {output_fp} -d {CACHE_DIR}",
    ]:
        try:
            subprocess.run(
                cmd_str,
                shell=True,
                check=True,
            )
        except subprocess.CalledProcessError as e:
            print(f"Error: {e}")
    # Delete the zip file
    os.remove(output_fp)


if __name__ == "__main__":
    main()
