import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from leaf_gm_architecture.streamlit_utils import PageNames
from tests.conftest import (
    make_app,
    make_mosses_only_df,
    make_synthetic_global_df,
    widget_by_label,
)


def test_crossval_page_aggregates_on_first_visit_then_trains_and_predicts(select_menu):
    """
    First visit without data in the session: the page must load and
    aggregate the raw Excel data itself. Then train a small model and
    predict a gm value for one data point.
    """
    at = make_app()
    select_menu(PageNames.CROSSVAL.value)

    at.run()  # first visit: no session state yet
    assert not at.exception

    # only one trait, so that the test set of the split stays big enough
    # for the app's adjusted R2 formula (rows - traits - 1 > 0)
    widget_by_label(at.multiselect, 'Select Traits').set_value(['T_cw'])
    widget_by_label(at.number_input, 'Ensemble Size').set_value(2)
    widget_by_label(at.number_input, 'Minimum Rows').set_value(5)
    widget_by_label(at.selectbox, 'Select PFT Group').select('mosses')
    widget_by_label(at.button, 'Perform Analysis').click()
    at.run()
    assert not at.exception
    assert 'Predictability scores' in [subheader.value for subheader in at.subheader]
    assert 'Gini importances' in [subheader.value for subheader in at.subheader]
    assert len(at.dataframe) == 2

    # after training, the page offers a prediction for single data points
    widget_by_label(at.button, 'Predict gm').click()
    at.run()
    assert not at.exception
    assert any('Predicted gm value' in markdown.value for markdown in at.markdown)


def test_crossval_page_standalone_aggregates_raw_data():
    """
    Opened as a standalone page (not through the app entry point), the
    page finds no session state and loads and aggregates the raw Excel
    data itself. The PFT group chosen here combines several PFTs, which
    makes the page show the group description.
    """
    at = AppTest.from_string(
        'from leaf_gm_architecture.streamlit_crossval import app\n'
        'app()\n',
        default_timeout=300)
    at.run()  # first visit: no session state yet
    assert not at.exception

    widget_by_label(at.multiselect, 'Select Traits').set_value(['T_cw', 'Sc'])
    widget_by_label(at.number_input, 'Ensemble Size').set_value(2)
    widget_by_label(at.number_input, 'Minimum Rows').set_value(5)
    widget_by_label(at.selectbox, 'Select PFT Group').select('extended_ferns')
    widget_by_label(at.button, 'Perform Analysis').click()
    at.run()
    assert not at.exception
    assert any('which contains these PFTs' in markdown.value for markdown in at.markdown)
    assert len(at.dataframe) == 2


@pytest.mark.filterwarnings("ignore:Degrees of freedom <= 0:RuntimeWarning")
@pytest.mark.filterwarnings("ignore:invalid value encountered in scalar divide:RuntimeWarning")
def test_crossval_page_with_ensemble_size_1_has_nan_error_columns(select_menu):
    """
    Visitors can choose an ensemble size of 1 today. With a single
    repeat there is nothing to average over, so the three error columns
    (R2_err, R2_adj_err, r_err) are empty (NaN). This test records that
    behavior. (Ticket 0006 discusses disallowing ensemble size 1.)

    The two ignored RuntimeWarnings are the mechanism producing the
    NaNs (standard error of one single value); filtering them keeps the
    suite warning-free without hiding anything this test does not
    already assert on.
    """
    at = make_app()
    at.session_state['global_df'] = make_synthetic_global_df()
    at.session_state['aggregated_df'] = make_mosses_only_df()
    select_menu(PageNames.CROSSVAL.value)

    at.run()
    assert not at.exception

    widget_by_label(at.multiselect, 'Select Traits').set_value(['T_cw', 'Sc'])
    widget_by_label(at.number_input, 'Ensemble Size').set_value(1)
    widget_by_label(at.number_input, 'Minimum Rows').set_value(5)
    widget_by_label(at.selectbox, 'Select PFT Group').select('mosses')
    widget_by_label(at.button, 'Perform Analysis').click()
    at.run()
    assert not at.exception

    predictability_scores_df = at.dataframe[0].value
    assert list(predictability_scores_df.columns) == [
        'R2', 'R2_err', 'R2_adj', 'R2_adj_err', 'r', 'r_err']
    # the scores exist, the error columns are empty
    for score in ['R2', 'R2_adj', 'r']:
        assert np.isfinite(predictability_scores_df[score][0]), score
    for error in ['R2_err', 'R2_adj_err', 'r_err']:
        assert np.isnan(predictability_scores_df[error][0]), error
