# Ticket 0001 — Full test coverage before the library update

**Status:** todo (drafted 2026-10-14)

**Phases:**
- [x] Phase 1 — this ticket
- [ ] Phase 2 — switch on coverage measuring, record today's gaps
- [ ] Phase 3 — write tests until 100 %

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
  clean Python 3.11 test environment.
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
