#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT="${APS_UI_PORT:-7860}"

cd "${SCRIPT_DIR}"
uv run apsagent.py --transport webrtc --port "${PORT}" "$@"
