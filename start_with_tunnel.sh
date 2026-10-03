#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

if [ -d ".venv" ]; then
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
fi

PORT="${GRADIO_PORT:-7860}"
export GRADIO_SHARE="${GRADIO_SHARE:-false}"

echo "Starting sentiment agent at http://127.0.0.1:${PORT}"
echo "Optional SSH tunnel from your local machine:"
echo "  ssh -L ${PORT}:localhost:${PORT} <user>@<server>"
echo

python main.py
