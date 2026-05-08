#!/usr/bin/env bash
set -euo pipefail

pkill -f "apsagent.py --transport webrtc" || true
echo "apsagent stopped (if it was running)."
