#!/usr/bin/env bash
# Public showcase entrypoint: warm the Play view with the registry champions,
# then serve the API + static playground. The pull is best-effort — the UI
# still works read-only (Watch/Lab against MLflow) if the pull fails.
set -euo pipefail

python main.py pull-champions || echo "pull-champions failed — serving without local champions" >&2
exec python main.py serve --host 0.0.0.0 --port "${PORT:-8080}"
