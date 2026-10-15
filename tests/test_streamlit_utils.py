import numpy as np
import pandas as pd

from leaf_gm_architecture.streamlit_utils import (
    PageNames,
    dict_to_tables,
    load_aggregated_data,
    load_dataset,
)


def test_page_names():
    # the menu bar of the web app shows these entries in this order
    assert [page.value for page in PageNames] == [
        'About',
        'Cross Validation',
        'Cross Prediction: global-PFT',
        'Cross Prediction: PFT-PFT',
    ]


def test_load_dataset():
    """The raw Excel data (1883 rows) still loads unchanged."""
    df = load_dataset()
    assert 'gm' in df.columns
    assert 'plant_functional_type' in df.columns
    # 1883 rows: the raw data contains several measurements (methods) of
    # the same data point, which data_aggregation() merges into one row
    assert len(df) == 1883
    assert len(df) > len(load_aggregated_data())


def test_load_aggregated_data():
    """
    The aggregated dataset (1625 rows) still loads unchanged, and every
    row is one unique observation (no duplicated data points).
    """
    df = load_aggregated_data()
    assert 'gm' in df.columns
    assert 'plant_functional_type' in df.columns
    assert len(df) == 1625
    grouping_cols = ['refkey', 'species', 'cultivar_variety',
                     'population_year', 'growth_environment']
    assert df.groupby(grouping_cols, dropna=False).ngroups == len(df)


def test_dict_to_tables_with_importances():
    results = {
        'R2': 0.5, 'R2_err': 0.1, 'R2_adj': 0.4, 'R2_adj_err': 0.1,
        'r': 0.7, 'r_err': 0.05,
        'importances': np.array([0.6, 0.4]),
    }
    predictability_scores_df, importances_df = dict_to_tables(results)
    assert list(predictability_scores_df.columns) == [
        'R2', 'R2_err', 'R2_adj', 'R2_adj_err', 'r', 'r_err']
    assert predictability_scores_df.shape == (1, 6)
    assert importances_df.shape == (2, 1)
    assert list(importances_df['Value']) == [0.6, 0.4]


def test_dict_to_tables_with_nan_importances():
    # if a model could not be trained, the importances are NaN and the
    # resulting table has a single NaN value instead of one row per trait
    results = {
        'R2': np.nan, 'R2_err': np.nan, 'R2_adj': np.nan, 'R2_adj_err': np.nan,
        'r': np.nan, 'r_err': np.nan,
        'importances': np.nan,
    }
    predictability_scores_df, importances_df = dict_to_tables(results)
    assert predictability_scores_df.shape == (1, 6)
    assert importances_df.shape == (1, 1)
    assert pd.isna(importances_df['Value'][0])
