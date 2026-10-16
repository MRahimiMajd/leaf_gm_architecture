from leaf_gm_architecture.streamlit_utils import PageNames
from tests.conftest import make_app, widget_by_label


def test_importing_the_app_does_not_start_it():
    """
    The app only starts when the file is run as a script (which is what
    the test browser does). Importing it as a module must not run main().
    """
    import leaf_gm_architecture.streamlit_app  # noqa: F401


def test_app_menu_simulation_and_about_page(seeded_random_forests, select_menu):
    """
    Visit all four pages of the app in one browser session, simulating
    the menu clicks. The first visit loads the real datasets. The later
    pages run with small settings so that the random forests train fast.
    """
    at = make_app()

    # first visit: web app starts on the About page and loads the data
    select_menu(PageNames.ABOUT.value)
    at.run()
    assert not at.exception
    assert at.title[0].value == 'leaf_gm_architecture'
    assert at.subheader[0].value == 'Early Test Version'

    # second visit (menu simulated): Cross Validation with a small plant
    # group, one trait and a small ensemble (one trait, so that the test
    # set stays big enough for the adjusted R2 formula)
    select_menu(PageNames.CROSSVAL.value)
    at.run()
    assert not at.exception
    widget_by_label(at.multiselect, 'Select Traits').set_value(['T_cw'])
    widget_by_label(at.number_input, 'Ensemble Size').set_value(2)
    widget_by_label(at.number_input, 'Minimum Rows').set_value(5)
    widget_by_label(at.selectbox, 'Select PFT Group').select('mosses')
    widget_by_label(at.button, 'Perform Analysis').click()
    at.run()
    assert not at.exception
    assert len(at.dataframe) == 2  # predictability scores + Gini importances

    # third visit (menu simulated): Cross Prediction global-PFT, with a
    # minimum rows setting so high that no model can be trained
    select_menu(PageNames.CROSSPRED_GLOBAL_PFT.value)
    at.run()
    assert not at.exception
    widget_by_label(at.number_input, 'Minimum Train Rows').set_value(100000)
    widget_by_label(at.number_input, 'Minimum Test Rows').set_value(100000)
    widget_by_label(at.button, 'Calculate Cross Prediction: global-PFT').click()
    at.run()
    assert not at.exception
    assert any('There are no trained models because of data availability!' in markdown.value
               for markdown in at.markdown)

    # fourth visit (menu simulated): Cross Prediction PFT-PFT, again with
    # no trainable data
    select_menu(PageNames.CROSSPRED_PFT_PFT.value)
    at.run()
    assert not at.exception
    widget_by_label(at.selectbox, 'Select PFT Group for Training').select('ferns')
    widget_by_label(at.selectbox, 'Select PFT Group for Testing').select('woody_evergreen_angiosperms')
    widget_by_label(at.number_input, 'Minimum Train Rows').set_value(100000)
    widget_by_label(at.number_input, 'Minimum Test Rows').set_value(100000)
    widget_by_label(at.button, 'Calculate Cross Prediction: PFT-PFT').click()
    at.run()
    assert not at.exception
    assert any('There are no trained models because of data availability!' in markdown.value
               for markdown in at.markdown)


def test_app_runs_with_the_real_menu():
    """
    All other tests simulate the menu, because the test browser cannot
    click it (it is a third-party component). This test runs the app
    once with the real menu add-on instead, so that its compatibility
    with the current Streamlit version is checked on every test run
    (ticket 0002). The menu stays on its default entry, the About page.
    """
    at = make_app()
    at.run()
    assert not at.exception
    component_elements = [element for element in at.main
                          if element.type == 'component_instance']
    assert len(component_elements) == 1
