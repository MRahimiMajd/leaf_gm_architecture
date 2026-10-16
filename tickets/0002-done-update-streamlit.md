# Ticket 0002 — Update Streamlit so the app comes back online

**Status:** in progress (drafted 2026-10-14)

**Phases:**
- [x] Phase 1 — this ticket
- [x] Phase 2 — version bump, record what breaks
- [ ] Phase 3 — minimal fixes, smoke test, pull request

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
