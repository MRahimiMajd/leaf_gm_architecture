import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor

import leaf_gm_architecture.functions as gm


def make_rf_df(pft_label, n, trait='T_cw', n_complete_for_extra_trait=0):
    """
    Small synthetic dataset with the columns that the RF helpers expect
    (first 'gm', then trait columns). The first n_complete_for_extra_trait
    rows additionally contain the trait 'T_leaf', the rest are NaN there.
    """
    rng = np.random.default_rng(7)
    df = pd.DataFrame({
        'plant_functional_type': pft_label,
        'gm': rng.normal(0.10, 0.01, n),
        trait: rng.normal(0.20, 0.02, n),
        'T_leaf': np.nan,
    })
    if n_complete_for_extra_trait:
        df.loc[df.index[:n_complete_for_extra_trait], 'T_leaf'] = rng.normal(0.10, 0.01, n_complete_for_extra_trait)
    return df


def assert_scores_structure(res, n_traits=1):
    """Check the result dictionary of a successful RF run."""
    for key in ['R2', 'R2_err', 'R2_adj', 'R2_adj_err', 'r', 'r_err']:
        assert np.isfinite(res[key]), f'{key} should be a finite number'
    importances = res['importances']
    assert isinstance(importances, np.ndarray)
    assert importances.shape == (n_traits,)
    assert importances.sum() == pytest.approx(1.0)


@pytest.mark.xfail(
    reason='known bug: in the "not enough data" branch of RF_with_split the '
           'model variable is never assigned, so it crashes instead of '
           'returning NaN scores (RF_with_train_and_test_data sets '
           'model=None correctly). The fix is planned in ticket 0006; '
           'when it lands, this strict xfail turns into an error until '
           'the mark is removed.',
    strict=True)
def test_rf_with_split_not_enough_data():
    """
    Not enough data: RF_with_split should return NaN scores and no model.
    Today it crashes with an UnboundLocalError, so this test is expected
    to fail (see the xfail mark above).
    """
    small_df = make_rf_df('mosses', 5)
    res, model = gm.RF_with_split(small_df, ensemble=2, minimum_data=50)
    assert model is None
    for key in ['R2', 'R2_err', 'R2_adj', 'R2_adj_err', 'r', 'r_err', 'importances']:
        assert pd.isna(res[key]), f'{key} should be NaN'


def test_rf_with_split_trains_on_enough_data():
    df = make_rf_df('mosses', 30)
    # in the app, make_combination_df() builds the model input
    comb_df = gm.make_combination_df(df, ['T_cw'])
    res, model = gm.RF_with_split(comb_df, ensemble=2, minimum_data=10)
    assert isinstance(model, RandomForestRegressor)
    assert_scores_structure(res)


def test_rf_with_train_and_test_data_not_enough_data():
    train_df = make_rf_df('ferns', 5)
    test_df = make_rf_df('mosses', 5)
    res, model = gm.RF_with_train_and_test_data(
        train_df, test_df, ensemble=2, minimum_train_data=50, minimum_test_data=10)
    assert model is None
    for key in ['R2', 'R2_err', 'R2_adj', 'R2_adj_err', 'r', 'r_err', 'importances']:
        assert pd.isna(res[key]), f'{key} should be NaN'


def test_rf_with_train_and_test_data_trains_on_enough_data():
    train_df = gm.make_combination_df(make_rf_df('ferns', 30), ['T_cw'])
    test_df = gm.make_combination_df(make_rf_df('mosses', 20), ['T_cw'])
    res, model = gm.RF_with_train_and_test_data(
        train_df, test_df, ensemble=2, minimum_train_data=10, minimum_test_data=5)
    assert isinstance(model, RandomForestRegressor)
    assert_scores_structure(res)


def test_trait_pairs_correlation():
    df = pd.DataFrame({'a': [1.0, 2.0, 3.0, 4.0, 5.0],
                       'b': [2.0, 4.0, 6.0, 8.0, 10.0]})
    pearson_r, p_value = gm.trait_pairs_correlation(df, 'a', 'b')
    assert pearson_r == pytest.approx(1.0)
    assert p_value == pytest.approx(0.0, abs=1e-10)


def test_trait_pairs_correlation_drops_rows_with_missing_values():
    # only the rows where both traits are present may be used
    df = pd.DataFrame({'a': [1.0, 2.0, 3.0, np.nan],
                       'b': [2.0, 4.0, np.nan, 8.0]})
    pearson_r, _p_value = gm.trait_pairs_correlation(df, 'a', 'b')
    assert pearson_r == pytest.approx(1.0)


def test_cv_with_pft_of_interest():
    df = pd.concat([
        make_rf_df('mosses', 12),
        make_rf_df('ferns', 3),  # other PFT, must be filtered out
    ], ignore_index=True)

    table = gm.CV_with_PFT_of_interest(
        df, ['mosses'], ['T_cw', 'T_leaf'], ensemble_size=2, min_rows=5)
    assert list(table.columns) == ['Traits', 'N', 'R2', 'R2_err', 'R2_adj',
                                   'R2_adj_err', 'r', 'r_err', 'importances']
    # (T_cw,) has 12 complete rows, combinations with T_leaf have none:
    # the too-small combinations are skipped
    assert table.shape[0] == 1
    assert table.loc[0, 'Traits'] == ['T_cw']
    assert table.loc[0, 'N'] == 12


def test_cv_with_pft_of_interest_without_enough_data():
    df = make_rf_df('mosses', 12)
    table = gm.CV_with_PFT_of_interest(
        df, ['mosses'], ['T_cw', 'T_leaf'], ensemble_size=2, min_rows=50)
    assert table.shape[0] == 0


def test_cross_prediction_global_pft_with_combination_of_interest():
    df = pd.concat([
        make_rf_df('evergreen angiosperms', 30),  # train set (non-overlapping)
        make_rf_df('ferns', 20),                  # test set (PFT of interest)
    ], ignore_index=True)
    res, model = gm.cross_prediction_global_PFT_with_combination_of_interest(
        df, ['ferns'], ['T_cw'], ensemble_size=2,
        minimum_train_rows=10, minimum_test_rows=5)
    assert isinstance(model, RandomForestRegressor)
    assert_scores_structure(res)


def test_cross_prediction_global_pft():
    # 30 complete train rows, only 5 of them with T_leaf; 20 test rows.
    df = pd.concat([
        make_rf_df('evergreen angiosperms', 30, n_complete_for_extra_trait=5),
        make_rf_df('ferns', 20, n_complete_for_extra_trait=20),
    ], ignore_index=True)

    table = gm.cross_prediction_global_PFT(
        df, ['ferns'], ['T_cw', 'T_leaf'], ensemble_size=2,
        minimum_train_rows=5, minimum_test_rows=3)
    assert table.shape == (1, 9)
    # (T_cw,) trains on 30 and tests on 20 rows; combinations with
    # T_leaf have only 5 train rows, fewer than the 20 test rows
    assert table.loc[0, 'Traits'] == ['T_cw']
    assert table.loc[0, 'N'] == [30, 20]


def test_cross_prediction_pft_pft_with_combination_of_interest():
    df = pd.concat([
        make_rf_df('ferns', 25),
        make_rf_df('mosses', 15),
    ], ignore_index=True)
    res, model = gm.cross_prediction_PFT_PFT_with_combination_of_interest(
        df, ['ferns'], ['mosses'], ['T_cw'], ensemble_size=2,
        minimum_train_rows=10, minimum_test_rows=5)
    assert isinstance(model, RandomForestRegressor)
    assert_scores_structure(res)


def test_cross_prediction_pft_pft():
    # 25 train rows, only 5 of them with T_leaf; 15 test rows, 10 of them
    # with T_leaf.
    train_df = make_rf_df('ferns', 25, n_complete_for_extra_trait=5)
    test_df = make_rf_df('mosses', 15, n_complete_for_extra_trait=10)
    df = pd.concat([train_df, test_df], ignore_index=True)

    table = gm.cross_prediction_PFT_PFT(
        df, ['ferns'], ['mosses'], ['T_cw', 'T_leaf'], ensemble_size=2,
        minimum_train_rows=5, minimum_test_rows=3)
    assert table.shape == (1, 9)
    # (T_cw,) trains on 25 and tests on 15 rows; combinations with
    # T_leaf have only 5 train rows, fewer than the 10 test rows
    assert table.loc[0, 'Traits'] == ['T_cw']
    assert table.loc[0, 'N'] == [25, 15]


def test_available_pft_pairs():
    pft_dict = {
        'group_a': ['mosses'],
        'group_b': ['ferns'],
        'group_c': ['mosses', 'ferns'],  # overlaps with group_a and group_b
        'group_d': ['CAM plants'],       # too little data
    }
    df = pd.concat([
        make_rf_df('mosses', 20),
        make_rf_df('ferns', 15),
        make_rf_df('CAM plants', 2),
    ], ignore_index=True)

    available_pairs = gm.available_PFT_pairs(
        df, pft_dict, ['T_cw'], minimum_train_rows=5, minimum_test_rows=3)

    # only the pair without overlap and with enough train and test data
    # survives; the reverse pair has fewer train than test rows and is
    # skipped
    assert available_pairs == {('group_a', 'group_b'): 1}
