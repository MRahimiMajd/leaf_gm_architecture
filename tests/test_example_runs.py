import numpy as np
import pandas as pd

import leaf_gm_architecture.functions as gm


def check_dicts_almost_equal(expected_dict, actual_dict, atol=1e-1):
    for key, expected in expected_dict.items():
        actual = actual_dict[key]
        if isinstance(expected, dict):
            # Recursively check nested dictionaries
            check_dicts_almost_equal(expected, actual, atol=atol)
        elif isinstance(expected, np.ndarray):
            # Compare arrays
            assert np.allclose(actual, expected, atol=atol), f"Mismatch in key {key}"
        else:
            # Compare scalar values
            assert np.isclose(actual, expected, atol=atol), f"Mismatch in key {key}"


def test_example_run1():
    """Check that the first example from the README still works as expected."""
    aggregated_df = pd.read_parquet('gm_dataset_Knauer_et_al_2022_aggregated.parquet')
    traits = ['T_cw','Sc','T_leaf', 'D_leaf']

    expected_results = {
        'R2': 0.6074139550478569,
        'R2_err': 0.023959111708946965,
        'R2_adj': 0.5150407680002939,
        'R2_adj_err': 0.029596549758110963,
        'r': 0.8207050336112371,
        'r_err': 0.012472228165087555,
        'importances': np.array([0.62323259, 0.15238046, 0.13257167, 0.09181528])
    }

    actual_results, _model = gm.CV_with_PFT_and_combination_of_interest(
        aggregated_df,
        gm.PFTs['global_set'],
        traits,
        ensemble_size=50,
        min_rows=50)
    check_dicts_almost_equal(expected_results, actual_results)


def test_example_run2():
    aggregated_df = pd.read_parquet('gm_dataset_Knauer_et_al_2022_aggregated.parquet')
    traits = ['LMA','T_mesophyll','fias_mesophyll','T_cw','T_cyt','T_chloroplast','Sm','Sc','T_leaf', 'D_leaf']
    table_of_results = gm.cross_prediction_global_PFT(
        aggregated_df,
        ['ferns'],
        traits,
        ensemble_size=5,
        minimum_train_rows=40,
        minimum_test_rows=10
    )

    assert table_of_results.shape == (55, 9)  # 55 trained models

    average_imps_g, average_values_c = gm.total_importances(table_of_results) 

    # IMP_G of the contributing traits in the trained models
    expected_average_imps_g = {
        'T_leaf': {0: 0.023833485819961923},
        'LMA': {0: 0.1049070250235918},
        'T_chloroplast': {0: 0.06448465755381468},
        'T_cw': {0: 0.3704522732926987},
        'Sc': {0: 0.4363225583099329}
    }
    actual_average_imps_g = average_imps_g.to_dict()
    check_dicts_almost_equal(expected_average_imps_g, actual_average_imps_g)

    # IMP_C of the contributing traits in the trained models
    expected_average_values_c = {
        'T_leaf': {0: 0.06910974898688242},
        'LMA': {0: 0.2975612843786347},
        'T_chloroplast': {0: 0.19856612330772716},
        'T_cw': {0: 0.23619672001902867},
        'Sc': {0: 0.19856612330772716}
    }
    actual_average_values_c = average_values_c.to_dict()
    check_dicts_almost_equal(expected_average_values_c, actual_average_values_c)
