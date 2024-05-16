import streamlit as st
from streamlit_option_menu import option_menu

import leaf_gm_architecture
from leaf_gm_architecture.streamlit_utils import load_dataset, load_aggregated_data, PageNames
from leaf_gm_architecture import (
    streamlit_crossval,
    streamlit_crosspred_global_pft,
    streamlit_crosspred_pft_pft)


def main():
    st.set_page_config(
        page_title='leaf_gm_architecture',
        page_icon=':herb:',
        layout="wide",
        menu_items=None)
    
    selected = option_menu(
        menu_title=None,
        options=[page.value for page in PageNames],
        menu_icon="cast",
        default_index=0,  # web app starts on the About page
        orientation="horizontal",
    )

    # initialize session state for raw dataset
    if 'global_df' not in st.session_state:
        st.session_state['global_df'] = load_dataset()

    # initialize session state for aggregated data
    if 'aggregated_df' not in st.session_state:
        st.session_state['aggregated_df'] = load_aggregated_data()

    if selected == PageNames.ABOUT.value:
        st.title('leaf_gm_architecture')
        st.subheader('Early Test Version')
        st.write(f"""
        This is a demo for the paper [Rahimi-Majd et al. (2024). Nonlinear models based on leaf architecture traits explain the variability of mesophyll conductance across plant species](https://doi.org/10.22541/au.171169724.40618321/v1) (PREPRINT).

        The scientific code is available [here](https://github.com/MRahimiMajd/leaf_gm_architecture).
        The code for the web app is [also available](https://github.com/arne-cl/leaf_gm_architecture).
        """)

    if selected == PageNames.CROSSVAL.value:
        streamlit_crossval.app()

    if selected == PageNames.CROSSPRED_GLOBAL_PFT.value:
        streamlit_crosspred_global_pft.app()

    if selected == PageNames.CROSSPRED_PFT_PFT.value:
        streamlit_crosspred_pft_pft.app()


if __name__ == "__main__":
    main()
