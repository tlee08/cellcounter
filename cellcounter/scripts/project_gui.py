import os
import subprocess

from cellcounter.gui.main import main


def run_script():
    """
    Running the streamlit script.

    Note that it must be run in a subprocess to make the call:
    ```
    streamlit run /path/to/gui.py
    ```
    """
    curr_fp = os.path.abspath(__file__)
    subprocess.run(["streamlit", "run", curr_fp])


if __name__ == "__main__":
    main()
