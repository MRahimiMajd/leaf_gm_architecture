#!/usr/bin/env bash
# Ticket 0002, setup (once per machine): install the test-only browser
# tooling for the real-browser menu test (playwright + a headless
# Chromium). The deployed app itself never needs them; playwright lives
# in the "dev" extra.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# 1. install the app with the "dev" extra (adds playwright, nothing else)
uv pip install --python .venv/bin/python -e '.[dev]'

# 2. download the headless Chromium browser (user-level, no root needed)
.venv/bin/playwright install chromium

# 3. if Chromium fails to launch later with messages about missing
#    system libraries, playwright's own installer can fetch them.
#    That step needs administrator rights, so it is NOT run here - run
#    it yourself if step 4 reports missing libraries:
#      sudo .venv/bin/playwright install-deps chromium

# 4. check that the browser really starts
.venv/bin/python - <<'PY'
from playwright.sync_api import sync_playwright
with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    browser.close()
print('Chromium starts. Run the tests with: .venv/bin/pytest')
PY
