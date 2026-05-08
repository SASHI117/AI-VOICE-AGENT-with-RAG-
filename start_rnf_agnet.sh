#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT="${RNF_UI_PORT:-7861}"

cd "${SCRIPT_DIR}"
uv run rnf_gemini/rnf_agnet.py --transport webrtc --port "${PORT}" "$@"
