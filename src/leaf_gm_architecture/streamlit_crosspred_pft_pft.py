import streamlit as st
import numpy as np
import pandas as pd

import leaf_gm_architecture.functions as gm
from leaf_gm_architecture.streamlit_utils import dict_to_tables, PageNames


@st.cache_data
def get_PFT_PFT_results(df_agg, PFT_train, PFT_test, traits_list, ensemble_size, min_train_rows, min_test_rows):
    return gm.cross_prediction_PFT_PFT(
        df_agg=df_agg,
        PFT_train=PFT_train,
        PFT_test=PFT_test,
        traits_list=traits_list,
        ensemble_size=ensemble_size,
        minimum_train_rows=min_train_rows,
        minimum_test_rows=min_test_rows
    )

@st.cache_data
def get_PFT_PFT_comb_interest_results(df_agg, PFT_train, PFT_test, combination_of_interest, ensemble_size, min_train_rows, min_test_rows):
    return gm.cross_prediction_PFT_PFT_with_combination_of_interest(
        df_agg=df_agg,
        PFT_train=PFT_train,
        PFT_test=PFT_test,
        combination_of_interest=combination_of_interest,
        ensemble_size=ensemble_size,
        minimum_train_rows=min_train_rows,
        minimum_test_rows=min_test_rows
    )


def app():
    st.header(PageNames.CROSSPRED_PFT_PFT.value)
    
    # initialize session state if it doesn't exist
    if 'selected_traits_detailed_analysis' not in st.session_state:
        st.session_state.selected_traits_detailed_analysis = []
    if 'PFT_PFT_results' not in st.session_state:
        st.session_state.PFT_PFT_results = None

    pft_group_options = list(gm.PFTs.keys())
    selected_pft_group_train = st.selectbox(
        'Select PFT Group for Training',
        options=pft_group_options,
        index=pft_group_options.index('ferns'))
    selected_pft_group_test = st.selectbox(
        'Select PFT Group for Testing',
        options=pft_group_options,
        index=pft_group_options.index('evergreen_gymnosperms'))

    selected_traits = st.multiselect(
        'Select traits', 
        ['LMA', 'T_mesophyll', 'fias_mesophyll', 'T_cw', 'T_cyt', 'T_chloroplast', 'Sm', 'Sc', 'T_leaf', 'D_leaf'],
        default=['LMA', 'T_mesophyll', 'fias_mesophyll', 'T_cw', 'T_cyt', 'T_chloroplast', 'Sm', 'Sc', 'T_leaf', 'D_leaf']
    )
    ensemble_size = st.number_input('Ensemble Size', min_value=1, value=5, step=1)
    min_train_rows = st.number_input('Minimum Train Rows', min_value=1, value=40, step=1)
    min_test_rows = st.number_input('Minimum Test Rows', min_value=1, value=10, step=1)

    if st.button('Calculate Cross Prediction: PFT-PFT'):
        PFT_PFT_results = get_PFT_PFT_results(
            df_agg=st.session_state['aggregated_df'],
            PFT_train=gm.PFTs[selected_pft_group_train],
            PFT_test=gm.PFTs[selected_pft_group_test],
            traits_list=selected_traits, 
            ensemble_size=ensemble_size,
            min_train_rows=min_train_rows, 
            min_test_rows=min_test_rows
        )
        
        # store results in session state
        st.session_state.PFT_PFT_results = PFT_PFT_results

    # check if we have results in session state
    if st.session_state.PFT_PFT_results is not None:
        PFT_PFT_results = st.session_state.PFT_PFT_results
        
        if PFT_PFT_results.shape[0] > 0:
            st.write("Trained models:", PFT_PFT_results.shape[0])
            st.write(PFT_PFT_results)
            average_imps_g, average_values_c = gm.total_importances(PFT_PFT_results)
            st.write("The IMP_G of the contributing traits in the trained models:", average_imps_g)
            st.write("The IMP_C of the contributing traits in the trained models:", average_values_c)

            # extract unique traits from PFT_PFT_results
            trait_options = set()
            for traits_list in PFT_PFT_results['Traits']:
                trait_options.update(traits_list)
            trait_options = sorted(trait_options)

            # let user select combination of traits from extracted options
            selected_traits_detailed_analysis = st.multiselect(
                'Select combination of traits for detailed analysis',
                options=trait_options,
                default=st.session_state.selected_traits_detailed_analysis
            )

            # update session state with current selection
            st.session_state.selected_traits_detailed_analysis = selected_traits_detailed_analysis

            if selected_traits_detailed_analysis:
                PFT_PFT_comb_interest_results, model = get_PFT_PFT_comb_interest_results(
                    df_agg=st.session_state['aggregated_df'],
                    PFT_train=gm.PFTs[selected_pft_group_train],
                    PFT_test=gm.PFTs[selected_pft_group_test],
                    combination_of_interest=selected_traits_detailed_analysis, 
                    ensemble_size=ensemble_size,
                    min_train_rows=min_train_rows, 
                    min_test_rows=min_test_rows
                )

                if model is not None:
                    st.header(PageNames.CROSSPRED_PFT_PFT.value + " (with combination of interest)")                    
                    predictability_scores_df, importances_df = dict_to_tables(PFT_PFT_comb_interest_results)

                    st.subheader('Predictability scores')
                    st.dataframe(predictability_scores_df)

                    st.subheader('Gini importances')
                    st.dataframe(importances_df)
                else:
                    st.error("The number of data points is less than the minimum required for the selected combination of traits.")                
            else:
                st.error("Please select at least one trait to perform a detailed analysis.")
        else:
            st.write("There are no trained models because of data availability!")
