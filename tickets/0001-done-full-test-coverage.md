# Ticket 0001 — Full test coverage before the library update

**Status:** done (drafted 2026-10-14, finished 2026-10-14)

**Phases:**
- [x] Phase 1 — this ticket
- [x] Phase 2 — switch on coverage measuring, record today's gaps
- [x] Phase 3 — write tests until 100 %

## Goal

We want automated tests for every line and every if/else branch in the code
base, so that we can be sure nothing breaks and no mathematical results change
when we update the dependencies of the app in ticket 0002.

## Background

A *test* is a small automatic check that runs part of the program and compares
the outcome with an expected result. *Coverage* measures how much of the
program the tests actually run: *line coverage* asks whether every line of code
runs at least once, *branch coverage* asks whether every decision is taken in
both directions (e.g. both the "enough data" and the "not enough data" path).

Today the repository has two test files. They cover the two main example
calculations from the paper. The three interactive pages of the web app and
several helper functions have no tests at all. That is why we cannot currently
prove that the app still works after a library update.

## Scope

We will only test the folder `src/leaf_gm_architecture/` (the web app),
but ignore `Bias_analyses/` (separate analysis script that's not part of the
web app.

## Plan

- **Setup (once):** a script `scripts/0001-setup-dev-environment.sh` creates a
  clean Python 3.11 test environment using `uv`.
- **Setup fix (2026-10-14):** with the pinned libraries, `pip install .`
  resolved pyarrow 26 (pulled in by streamlit's `pyarrow>=7.0`), which needs
  NumPy 2, but NumPy is pinned to 1.24.3, so even the existing tests failed
  with an ImportError (no parquet engine). Pinned `pyarrow = ">=7.0,<15"` in
  `pyproject.toml`.
- **Phase 2:** switch on coverage measuring (including branches) in
  `pyproject.toml` and set the requirement to 100 %, so future test runs fail
   if coverage drops. Measure today's baseline and record the gaps here.
- **Phase 3:** write tests until the report shows 100 % for lines and branches:
  - mathematical routines in `functions.py`, including all
    "not enough data" detours
  - small helpers in `streamlit_utils.py`
  - the three pages of the web app (using Streamlit's built-in test
    browser `AppTest`)
  - The menu bar is an add-on that this test browser cannot click,
    so menu-clicking will be simulated in the tests.
  - Keep the tests fast. Runtime is driven by three settings together:
    the ensemble size (how often models are retrained), the number of
    selected traits (the app tries every combination of them: with all
    10 traits that is over 1,000 combinations), and the minimum-rows
    fields. Tests therefore use small plant groups, only 2-3 traits,
    tight minimum-rows values, and 2-3 ensemble repeats.
  - One dedicated test runs with ensemble size 1, because visitors can
    choose that value in the app today. With a single repeat, the three
    error columns (R2_err, R2_adj_err, r_err) are empty (NaN); the test
    records that behavior.

## Hurdles

- The calculations are partly random (many small *random forests* are trained).
  The tests therefore use tiny settings and generous tolerances, like the two
  existing tests already do.
- The tests must pass on the *current*, old library versions, because this
  ticket changes no program behavior. Ticket 0002 then re-runs them on the new
  versions.
- **Found while covering the "not enough data" detours:** `RF_with_split`
  (functions.py) crashes with an `UnboundLocalError` instead of returning the
  NaN result, because `model` is never assigned in that branch
  (`RF_with_train_and_test_data` sets `model = None` correctly). Reachable
  from the Cross Validation page when a visitor sets "Minimum Rows" higher
  than the available data (defaults avoid it; PFT-PFT page even handles
  `model = None` explicitly). Decision (2026-10-14): ticket 0001 only
  documents the current behavior in a test; the fix belongs in ticket 0006.

## Baseline (measured 2026-10-14, branch coverage switched on)

Existing tests pass, gate fails at **31 %** (lines + branches):

| file | line+branch coverage | main gaps |
|---|---|---|
| `functions.py` | 62 % | "not enough data" detours in `RF_with_split`/`RF_with_train_and_test_data`, `trait_pairs_correlation`, `CV_with_PFT_of_interest`, `cross_prediction_global_PFT_with_combination_of_interest`, `available_PFT_pairs`, `cross_prediction_PFT_PFT*`, skip-`continue` branches |
| `streamlit_utils.py` | 0 % | nothing tested |
| `streamlit_app.py` | 0 % | nothing tested |
| `streamlit_crossval.py` | 0 % | nothing tested |
| `streamlit_crosspred_global_pft.py` | 0 % | nothing tested |
| `streamlit_crosspred_pft_pft.py` | 0 % | nothing tested |

## Result (2026-10-14, phase 3)

`pytest` reports **100 % line coverage and 100 % branch coverage** for
`src/leaf_gm_architecture/`, 31 tests pass warning-free in about 3 minutes
(on Python 3.11.10 with the pinned library versions). Any *new* warning
fails the suite, so the library update in ticket 0002 will surface changed
behavior immediately.

New test files:

| file | what it tests |
|---|---|
| `tests/test_functions_helpers.py` | all untested routines of `functions.py`, including the "not enough data" detours and the skip-`continue` branches (train/test sets crafted so both directions of every decision are taken) |
| `tests/test_streamlit_utils.py` | `PageNames`, the two dataset loaders, `dict_to_tables` with and without importances |
| `tests/test_streamlit_app.py` | visits all four pages in one session with simulated menu clicks; plus the import guard |
| `tests/test_page_crossval.py` | Cross Validation page: first visit aggregates the raw data, training + single-value prediction, a standalone visit (no session state), and the dedicated ensemble-size-1 test (NaN error columns) |
| `tests/test_page_crosspred_global_pft.py` | Cross Prediction global-PFT page: trained models, empty detailed analysis, no trainable data |
| `tests/test_page_crosspred_pft_pft.py` | Cross Prediction PFT-PFT page: trained models, detailed analysis, no trainable data, and the "model could not be trained" error message (training stubbed, since the synthetic data can always train) |
| `tests/conftest.py` | shared helpers: app starter, simulated menu clicks, small synthetic datasets |

Two technical hurdles worth remembering:

- Coverage must be given the folder as a *path* (`--cov=src/leaf_gm_architecture`),
  not as a package name: the test browser executes `streamlit_app.py` as a
  script (`__main__`), which coverage cannot attribute to a package name.
- The Cross Validation page has a first-visit branch that loads and aggregates
  the raw Excel data itself. It is unreachable through the normal app flow
  (the entry point always fills the session state first), so one test opens
  the page standalone to cover it.

The 100 % requirement is enforced by the pytest configuration itself
(`--cov-fail-under=100`), so the existing GitHub Actions check enforces it
automatically. One workflow change was needed: it now installs the app
*editable* (`pip install -e .`), because a regular install copies the package
into site-packages, where the coverage gate (which measures `src/`) cannot
see it.

## Done when

- `pytest` reports 100 % line coverage and 100 % branch coverage for
  `src/leaf_gm_architecture/`.
- All tests pass on Python 3.11 with today's library versions.
- The 100 % requirement is enforced automatically, including in the GitHub
  Actions check that runs on every pull request.

## Glossary

- **AppTest:** a tool that ships with Streamlit; it simulates a visitor who
  opens the app, sets fields and presses buttons, so pages can be tested
  without a real browser.
