from enum import Enum

import numpy as np
import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu


class PageNames(Enum):
    ABOUT = "About"
    CROSSVAL = "Cross Validation"
    CROSSPRED_GLOBAL_PFT = "Cross Prediction: global-PFT"
    CROSSPRED_PFT_PFT = "Cross Prediction: PFT-PFT"

def load_dataset():
    return pd.read_excel('gm_dataset_Knauer_et_al_2022.xlsx', sheet_name='data')

@st.cache_data  # this caches the aggregated input data between runs
def load_aggregated_data():
    return pd.read_parquet('gm_dataset_Knauer_et_al_2022_aggregated.parquet')


def dict_to_tables(results):
    """
    Helper function to convert CV_with_PFT_and_combination_of_interest()
    result dictionary into individual dataframes.

    Parameters
    ----------
    results : dict
        The resulting predictability scores and importance of the given
        traits calculated in CV_with_PFT_and_combination_of_interest().
        
    Returns 
    -------
    predictability_scores_df : pd.DataFrame
        DataFrame with one column per predictability score
    importances_df : pd.DataFrame
        DataFrame with one column per importance value
    """
    # create predictability scores DataFrame by excluding 'importances'
    predictability_scores_df = pd.DataFrame({k: [v] for k, v in results.items() if k != 'importances'})

    if results['importances'] is np.nan:
        importances_df = pd.DataFrame({'Value': results['importances']}, index=[0])
    else:
        importances_df = pd.DataFrame({'Value': results['importances']})

    return predictability_scores_df, importances_df
