# Ticket 0001 — Full test coverage before the library update

**Status:** reopened (flaky test found after the first completion;
drafted and finished 2026-10-14)

**Phases:**
- [x] Phase 1 — this ticket
- [x] Phase 2 — switch on coverage measuring, record today's gaps
- [x] Phase 3 — write tests until 100 %
- [x] Phase 4 (reopened) — demonstrate the flaky test live (red)
- [ ] Phase 5 (reopened) — pin the randomness in the tests (green)

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

## Reopened: the app menu test is a dice game

After the ticket had been declared done, `tests/test_streamlit_app.py::
test_app_menu_simulation_and_about_page` failed in an ordinary test run with
no code changed in between:

    FAILED tests/test_streamlit_app.py::test_app_menu_simulation_and_about_page
    - assert not ElementList(_list=[Exception(message='An input array is
    constant; the correlation coefficient is not defined.', ...

### What happened

1. The test runs the Cross Validation page with the *real* Knauer dataset,
   plant group "mosses", trait "T_cw": only **10 rows** of that group have
   both `gm` and `T_cw`.
2. `RF_with_split` (functions.py) hardcodes `random_state=None` for both the
   train/test split and the random forest, so every test run splits the data
   differently and grows different trees.
3. With ensemble size 2 and a 3-row test set, the forest sometimes predicts
   the *same* value for every test point. scipy's `pearsonr` cannot compute
   a correlation on a constant list and emits `ConstantInputWarning`.
4. This ticket deliberately set `filterwarnings = ["error"]` in
   pyproject.toml ("any new warning fails the suite"). The warning therefore
   becomes an exception inside the app run, and the test fails.

### Evidence

- Reproduction loop on the real data (10 mosses rows, ensemble 2, warnings
  as errors): the warning fired in **5 of 30 runs (~17 %)**.
- Seed demonstration: forcing `random_state=10` into the split and the
  forest reproduces the warning deterministically; `random_state=1` was
  verified clean for *every* dataset and setting the tests use (real mosses
  + T_cw, real extended ferns + T_cw/Sc, both synthetic page datasets).
- So the first completion was legitimate: the suite passed at commit time
  by luck. Re-running would usually pass and hide the problem again.

### Plan (phases 4 and 5)

- **Phase 4 (red):** a flaky failure cannot be captured in a deterministic
  new test, since the change below *is* test code. Deviation from the usual
  red/green mechanics, documented here: the red evidence is a live run of
  the flaky test until it fails, recorded in this ticket.
- **Phase 5 (green):** add a `seeded_random_forests` fixture to
  `tests/conftest.py` that monkeypatches the module-level names
  `gm.train_test_split` and `gm.RandomForestRegressor` so that they
  *force* `random_state=1`. (A plain `functools.partial` would silently
  lose the seed, because functions.py passes `random_state=None`
  explicitly.) Apply the fixture to all six tests that train forests:
  the app menu test, the three Cross Validation page tests and the two
  Cross Prediction page tests. Tests that only exercise the "not enough
  data" detours stay untouched.
- **Done when:** the full suite passes with 100 % line and branch coverage,
  no warning filters were added, and the two tests that hit the 10-row real
  mosses data pass several repeated runs in a row.

### Phase 4 (red): the flake, caught live

Running the single flaky test repeatedly until it failed
(`python -m pytest tests/test_streamlit_app.py::
test_app_menu_simulation_and_about_page`, 15 fresh pytest processes,
default configuration):

    run 1:  1 passed     run 6:  1 passed    run 11: 1 passed
    run 2:  1 passed     run 7:  1 passed    run 12: 1 passed
    run 3:  1 passed     run 8:  1 passed    run 13: 1 passed
    run 4:  1 failed     run 9:  1 passed    run 14: 1 failed
    run 5:  1 passed     run 10: 1 passed    run 15: 1 passed

**2 of 15 runs failed (13 %)**, matching the ~17 % measured in the
reproduction loop. The captured failure (run 4) is exactly the reported
one:

    tests/test_streamlit_app.py:40: AssertionError
    E       assert not ElementList(_list=[Exception(message='An input array
    is constant; the correlation coefficient is not defined.', ...
    2026-10-14 14:26:13.914 Uncaught app exception
    scipy.stats._warnings_errors.ConstantInputWarning: An input array is
    constant; the correlation coefficient is not defined.

(Technical note: single-test runs cannot reach the 100 % coverage gate, so
pytest exits non-zero even when the test passes; the runs above were
counted by the summary line, not the exit code.)

## Glossary

- **AppTest:** a tool that ships with Streamlit; it simulates a visitor who
  opens the app, sets fields and presses buttons, so pages can be tested
  without a real browser.
