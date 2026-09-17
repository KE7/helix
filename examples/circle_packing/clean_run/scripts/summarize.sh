#!/usr/bin/env bash
# Summarize the circle-packing run (safe while it is still running).
#
#   .helix-run/bin/summarize.sh [--rerun-best]
set -euo pipefail
PRIVATE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "${PRIVATE}/protocol.env"
exec env -u ANTHROPIC_API_KEY -u ANTHROPIC_AUTH_TOKEN \
  uv run --no-project --python 3.12 python3 "${PRIVATE}/bin/summarize.py" "$@"
