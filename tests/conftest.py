import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import streamlit_option_menu
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from streamlit.testing.v1 import AppTest

import leaf_gm_architecture
import leaf_gm_architecture.functions as gm

APP_FILE = Path(leaf_gm_architecture.__file__).parent / 'streamlit_app.py'


@pytest.fixture
def seeded_random_forests(monkeypatch):
    """
    The app's model training draws real randomness (functions.py passes
    random_state=None to the train/test split and to the random forest).
    With the small datasets in these tests, the forest occasionally
    predicts the same value for every test point, for which scipy's
    pearsonr emits a warning - and the suite treats every warning as an
    error (see ticket 0001). This fixture pins the randomness to a fixed
    seed, which makes the affected tests deterministic. The seed is chosen
    so that every dataset the tests use trains without that warning
    (seed 10, by contrast, provably hits it). The production app keeps
    its unseeded behavior. Note the wrapper functions below *force* the
    seed: a functools.partial would silently lose it, because
    functions.py passes random_state=None explicitly.
    """
    def train_test_split_with_seed(X, y, **kwargs):
        kwargs['random_state'] = 1
        return train_test_split(X, y, **kwargs)

    def random_forest_with_seed(**kwargs):
        kwargs['random_state'] = 1
        return RandomForestRegressor(**kwargs)

    monkeypatch.setattr(gm, 'train_test_split', train_test_split_with_seed)
    monkeypatch.setattr(gm, 'RandomForestRegressor', random_forest_with_seed)


def make_app():
    """Start a test browser on the web app entry point."""
    # generous timeout, because training the random forests can take a
    # while on a busy machine (e.g. in CI)
    return AppTest.from_file(str(APP_FILE), default_timeout=300)


@pytest.fixture(scope='session')
def streamlit_server():
    """
    Start the web app in a real, headless Streamlit server and wait until it
    answers. Used by the HTTP smoke test and the browser menu test.
    The server runs with the repository root as working directory, because
    the app loads its dataset files by name from there.
    """
    repo_root = Path(__file__).resolve().parent.parent
    with socket.socket() as port_finder:
        port_finder.bind(('127.0.0.1', 0))
        port = port_finder.getsockname()[1]
    base_url = f'http://127.0.0.1:{port}'

    server = subprocess.Popen(
        [sys.executable, '-m', 'streamlit', 'run', str(APP_FILE),
         '--server.headless', 'true', '--server.address', '127.0.0.1',
         '--server.port', str(port)],
        cwd=repo_root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True)

    def stopped_server_output():
        return server.communicate(timeout=15)[0]

    try:
        for _ in range(60):
            if server.poll() is not None:
                raise RuntimeError(
                    'the streamlit server stopped during startup:\n'
                    + stopped_server_output())
            try:
                with urllib.request.urlopen(f'{base_url}/_stcore/health',
                                            timeout=1) as response:
                    if response.read() == b'ok':
                        break
            except OSError:
                pass
            time.sleep(1)
        else:
            raise RuntimeError('the streamlit server did not become '
                               'healthy within 60 s')
        yield base_url
    finally:
        server.terminate()
        server.communicate(timeout=15)


@pytest.fixture
def select_menu(monkeypatch):
    """
    AppTest cannot click the menu bar, because it is a third-party
    component (streamlit_option_menu). Every page test therefore
    simulates the menu choice by patching option_menu to return the
    selected page name, like the real menu would.
    """
    def select(page_value):
        monkeypatch.setattr(
            streamlit_option_menu, 'option_menu', lambda **kwargs: page_value)
    return select


def widget_by_label(widgets, label):
    """Return the single widget with the given label."""
    matching = [widget for widget in widgets if widget.label == label]
    assert len(matching) == 1, f'expected exactly one widget labeled {label!r}, found {len(matching)}'
    return matching[0]


def make_synthetic_global_df():
    """Raw dataset stand-in, only used to derive number input ranges."""
    rng = np.random.default_rng(1)
    return pd.DataFrame({
        'gm': rng.normal(0.10, 0.01, 30),
        'T_cw': rng.normal(0.20, 0.02, 30),
        'Sc': rng.normal(10.0, 1.0, 30),
    })


def make_mosses_only_df():
    """Aggregated dataset stand-in with one small, complete PFT group."""
    rng = np.random.default_rng(2)
    return pd.DataFrame({
        'plant_functional_type': 'mosses',
        'gm': rng.normal(0.10, 0.01, 12),
        'T_cw': rng.normal(0.20, 0.02, 12),
        'Sc': rng.normal(10.0, 1.0, 12),
    })


def make_synthetic_aggregated_df():
    """
    Small aggregated dataset with two plant functional types:
      - ferns: 30 rows, complete for gm/T_cw/Sc, only 8 rows with T_leaf
        and 2 rows with D_leaf
      - evergreen angiosperms: 40 rows, of which 20 complete for
        gm/T_cw/Sc, 15 rows with T_leaf and 3 rows with D_leaf

    This way some trait combinations have more training than test data,
    and some have too little data at all, so that all skip branches of
    the cross-prediction code can be reached by the page tests.
    """
    rng = np.random.default_rng(42)

    def block(pft, n):
        return pd.DataFrame({
            'plant_functional_type': pft,
            'gm': rng.normal(0.10, 0.01, n),
            'T_cw': rng.normal(0.20, 0.02, n),
            'Sc': rng.normal(10.0, 1.0, n),
            'T_leaf': np.nan,
            'D_leaf': np.nan,
        })

    ferns = block('ferns', 30)
    ferns.loc[range(8), 'T_leaf'] = rng.normal(0.10, 0.01, 8)
    ferns.loc[range(2), 'D_leaf'] = rng.normal(5.0, 0.5, 2)

    evergreens = block('evergreen angiosperms', 40)
    evergreens.loc[range(20, 40), 'T_cw'] = np.nan
    evergreens.loc[range(20, 40), 'Sc'] = np.nan
    evergreens.loc[range(15), 'T_leaf'] = rng.normal(0.10, 0.01, 15)
    evergreens.loc[range(3), 'D_leaf'] = rng.normal(5.0, 0.5, 3)

    return pd.concat([ferns, evergreens], ignore_index=True)
