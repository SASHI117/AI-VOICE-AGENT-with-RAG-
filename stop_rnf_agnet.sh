#!/usr/bin/env bash
set -euo pipefail

pkill -f "rnf_gemini/rnf_agnet.py --transport webrtc" || true
pkill -f "rnf_agnet.py --transport webrtc" || true
echo "rnf_agnet stopped (if it was running)."
