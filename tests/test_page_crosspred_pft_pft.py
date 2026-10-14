import numpy as np

import leaf_gm_architecture.functions as gm
from leaf_gm_architecture.streamlit_crosspred_pft_pft import (
    get_PFT_PFT_comb_interest_results,
)
from leaf_gm_architecture.streamlit_utils import PageNames
from tests.conftest import (
    make_app,
    make_synthetic_aggregated_df,
    make_synthetic_global_df,
    widget_by_label,
)


def seed_session_state(at):
    """
    The page expects the app entry point to have loaded the datasets
    already. Use synthetic data, so the page runs fast and its skip branches
    are reachable.
    """
    at.session_state['global_df'] = make_synthetic_global_df()
    at.session_state['aggregated_df'] = make_synthetic_aggregated_df()


def configure_widgets(at):
    widget_by_label(at.selectbox, 'Select PFT Group for Training').select('ferns')
    widget_by_label(at.selectbox, 'Select PFT Group for Testing').select('woody_evergreen_angiosperms')
    widget_by_label(at.multiselect, 'Select traits').set_value(
        ['T_cw', 'Sc', 'T_leaf', 'D_leaf'])
    widget_by_label(at.number_input, 'Ensemble Size').set_value(2)
    widget_by_label(at.number_input, 'Minimum Train Rows').set_value(5)
    widget_by_label(at.number_input, 'Minimum Test Rows').set_value(3)


def test_crosspred_pft_pft_page_trains_and_shows_detailed_analysis(select_menu):
    get_PFT_PFT_comb_interest_results.clear()  # isolate from other tests
    at = make_app()
    seed_session_state(at)
    select_menu(PageNames.CROSSPRED_PFT_PFT.value)

    at.run()  # first visit: no results yet
    assert not at.exception

    configure_widgets(at)
    widget_by_label(at.button, 'Calculate Cross Prediction: PFT-PFT').click()
    at.run()
    assert not at.exception
    # st.write wraps numbers in backticks, so only look for the count itself
    trained_models = [markdown.value for markdown in at.markdown
                      if 'Trained models' in markdown.value]
    assert any('3' in value for value in trained_models)
    # the detailed analysis multiselect is rendered but still empty
    assert any('Please select at least one trait to perform a detailed analysis.' in error.value
               for error in at.error)

    widget_by_label(
        at.multiselect, 'Select combination of traits for detailed analysis'
    ).set_value(['T_cw'])
    at.run()
    assert not at.exception
    assert 'Cross Prediction: PFT-PFT (with combination of interest)' in [
        header.value for header in at.header]
    assert 'Predictability scores' in [subheader.value for subheader in at.subheader]
    assert 'Gini importances' in [subheader.value for subheader in at.subheader]


def test_crosspred_pft_pft_page_without_enough_data(select_menu):
    at = make_app()
    seed_session_state(at)
    select_menu(PageNames.CROSSPRED_PFT_PFT.value)

    at.run()
    assert not at.exception

    configure_widgets(at)
    widget_by_label(at.number_input, 'Minimum Train Rows').set_value(100000)
    widget_by_label(at.number_input, 'Minimum Test Rows').set_value(100000)
    widget_by_label(at.button, 'Calculate Cross Prediction: PFT-PFT').click()
    at.run()
    assert not at.exception
    assert any('There are no trained models because of data availability!' in markdown.value
               for markdown in at.markdown)


def test_crosspred_pft_pft_page_when_detailed_analysis_cannot_train(monkeypatch, select_menu):
    """
    The PFT-PFT page is the only page that handles the case where the
    detailed analysis cannot train a model (model is None). It shows an
    error message instead.
    This cannot happen with the synthetic data above (there the detailed
    analysis can always train), so the model training itself is replaced
    by a stub that fails like it would with too little data.
    """
    get_PFT_PFT_comb_interest_results.clear()  # the stub must replace any cached result

    nan_results = {'R2': np.nan, 'R2_err': np.nan, 'R2_adj': np.nan,
                   'R2_adj_err': np.nan, 'r': np.nan, 'r_err': np.nan,
                   'importances': np.nan}
    monkeypatch.setattr(
        gm, 'cross_prediction_PFT_PFT_with_combination_of_interest',
        lambda *args, **kwargs: (nan_results, None))

    at = make_app()
    seed_session_state(at)
    select_menu(PageNames.CROSSPRED_PFT_PFT.value)

    at.run()
    assert not at.exception

    configure_widgets(at)
    widget_by_label(at.button, 'Calculate Cross Prediction: PFT-PFT').click()
    at.run()
    assert not at.exception
    trained_models = [markdown.value for markdown in at.markdown
                      if 'Trained models' in markdown.value]
    assert any('3' in value for value in trained_models)

    widget_by_label(
        at.multiselect, 'Select combination of traits for detailed analysis'
    ).set_value(['T_cw'])
    at.run()
    assert not at.exception
    assert any('The number of data points is less than the minimum required for the selected combination of traits.' in error.value
               for error in at.error)
