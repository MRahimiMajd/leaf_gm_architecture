# Ticket 0002 — Update Streamlit so the app comes back online

**Status:** done in the repository, but still needs to be merged into
Milads repo and deployed.

**Phases:**
- [x] Phase 1 — this ticket
- [x] Phase 2 — version bump, record what breaks
- [x] Phase 3 — minimal fixes, smoke test, pull request

## Goal

The web app at
https://leaf-gm-architecture-bioinformatics-potsdam.streamlit.app/
was stopped by the hosting service (Streamlit Community Cloud) because it used
too much memory. The provider's own advice for this error starts with:
update Streamlit. The app has been running unchanged since spring 2024 on a
version of Streamlit that is now over two years old.

## Important limitation

This ticket restores service and modernizes the foundation. It does **not**
fix the underlying memory problems. Those are split into Tickets 0003-0006
and will be done one by one afterwards.

## Plan

- **Phase 2:** in `pyproject.toml`, change the Streamlit version from 1.33.0
  to the current release (1.65.0 as of 2026-10). Run the full test suite from
  Ticket 0001 and record every failure. The failures show exactly what the
  new version changes.
- **Phase 3:** fix the breakage with as few changes as possible, keeping
  coverage at 100 % (the fixes need tests too). Then start the app once
  without a browser as a smoke test.
- Only the Streamlit-related dependencies will be updated.
  The scientific libraries (pandas, numpy, scikit-learn, scipy, openpyxl)
  and Python 3.11 stay unchanged, *unless* the new Streamlit itself insists
  on a newer version of one of them.
  If that happens, we report it here and let the user decide.

## Phase 2 result (2026-10-16): the version bump broke nothing

Changed `pyproject.toml`: `streamlit from "1.33.0"` to `"1.65.0"` (most
current release) and synced the test environment with the
Ticket 0001 setup command (`uv pip install --python .venv/bin/python -e .`).
Only Streamlit and its web-server dependencies changed.

What changed in the environment:

- streamlit 1.33.0 -> 1.65.0, protobuf 4.25.9 -> 7.36.2
- new: anyio, h11, httptools, itsdangerous, python-multipart, starlette,
  uvicorn, websockets (the new Streamlit server stack)

Known blind spot: the test browser only *simulates* the menu choice
(`tests/conftest.py`). We should add Playwright test to verify it keeps
working.

## Phase 3 result (2026-10-16): no fixes needed — and the menu is now really tested

The expected breakage did not happen, so `src/` stayed untouched: this
ticket changes **zero lines of production code**. The phase's scope grew
by decision (review of phase 2): the suite now also proves that the menu
add-on really works, in three new layers:

1. **Unmocked test-browser run** (`test_app_runs_with_the_real_menu`):
   the app runs with the real menu add-on's Python side. Before, every
   test simulated the menu, so only its import was ever real.
2. **HTTP smoke test** (`tests/test_app_server.py`): a real headless
   Streamlit server (session fixture in `conftest.py`) answers on its
   health endpoint and serves the start page. Worth its own check
   because the update also replaced the server technology
   (tornado → uvicorn).
3. **Real-browser menu test** (`tests/test_menu_browser.py`): headless
   Chromium clicks 'Cross Validation' in the real menu and then back to
   'About'. This is the only test that exercises the menu exactly like a
   visitor does — the strongest available proof short of the live app.

New dev-only dependency: `playwright` (`^1.64`, optional, in the "dev"
extra of `pyproject.toml`) plus a user-level headless Chromium download
(~120 MB). Deliberately kept out of the base install: the hosting
service stopped the app for memory overuse, so the deployment stays
lean — the browser tooling is for development and CI only. New setup
script: `scripts/0002-install-browser-test-dependencies.sh`.

The pinned `streamlit_option_menu = "0.3.12"` stays: the real-browser
test proves rendering and clicks work on Streamlit 1.65.0, so the hurdle
from the plan is cleared without updating the add-on.

Evidence:

    33 passed, 1 xfailed in 317.56s (0:05:17)
    coverage: 100.00 % lines and branches, warnings-as-errors silent

The GitHub Actions workflow now installs the dev extra and headless
Chromium, so all three new layers also run on every pull request.

Remaining after the merge (hosting side, manual): the live app at
https://leaf-gm-architecture-bioinformatics-potsdam.streamlit.app/ must
show its normal pages again.

## Hurdles & risks

- The menu add-on (`streamlit_option_menu`) must still work with the new
  Streamlit, otherwise it gets updated too.

## Done when

- All tests pass on the new versions and coverage is still 100 %.
- The app starts headlessly as a smoke test.
- The pull request is merged and the live app shows its normal pages again.

## Glossary

- **pyproject.toml:** the file that lists which dependencies (libraries)
  the app needs, and in which exact versions.
- **Headless / smoke test:** starting the program once automatically and
  checking that it comes up, without clicking around by hand.
