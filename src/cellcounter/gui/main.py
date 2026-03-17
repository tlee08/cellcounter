import streamlit as st

from cellcounter.gui.gui_funcs import PROJ_DIR, PROJ_DIR_STATUS, ProjDirStatus
from cellcounter.gui.page1_init import page1_init
from cellcounter.gui.page2_update import page2_configs
from cellcounter.gui.page4_pipeline import page4_pipeline
from cellcounter.gui.page5_view import page5_view
from cellcounter.gui.page6_combine import page6_combine

# logging.basicConfig(level=logging.INFO)


#####################################################################
# Streamlit application
#####################################################################


def main():
    # Initialising session state
    if PROJ_DIR not in st.session_state:
        st.session_state[PROJ_DIR] = None
        st.session_state[PROJ_DIR_STATUS] = ProjDirStatus.NOT_SET
    # Title
    st.title("Cellcounter Processing Pipeline")
    # Multi-page navigation
    pg = st.navigation(
        [
            st.Page(page1_init),
            st.Page(page2_configs),
            st.Page(page4_pipeline),
            st.Page(page5_view),
            st.Page(page6_combine),
        ]
    )
    pg.run()


if __name__ == "__main__":
    main()
