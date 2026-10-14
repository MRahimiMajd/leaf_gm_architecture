#!/usr/bin/env bash
# Ticket 0001, setup (once): create a clean Python 3.11 test environment using uv.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# 1. make sure `uv` is available
if ! command -v uv >/dev/null 2>&1; then
    echo "error: uv is not installed. See https://docs.astral.sh/uv/ for install instructions." >&2
    exit 1
fi

# 2. install Python 3.11 (user-level)
uv python install 3.11

# 3. create the virtual environment
uv venv --python 3.11 .venv

# 4. install the app (editable, so coverage reports point at src/) plus
#    pytest and pytest-cov
uv pip install --python .venv/bin/python -e . pytest pytest-cov

echo
echo "Done. Run the tests with:"
echo "  .venv/bin/pytest"
