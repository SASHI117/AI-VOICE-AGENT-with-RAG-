#!/bin/bash
# =====================================================
# One-shot deploy: setup + build + deploy + print URL.
# Usage:  ./deploy_all.sh [tag]
# Example: ./deploy_all.sh v1
# =====================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TAG="${1:-v1}"

echo "🚀 Full deploy starting (tag: $TAG)"
echo ""

"$SCRIPT_DIR/1_setup_azure.sh"
echo ""

"$SCRIPT_DIR/2_build_image.sh" "$TAG"
echo ""

"$SCRIPT_DIR/3_deploy_container.sh" "$TAG"
echo ""

# Wait a moment for revision to come up
echo "⏳ Waiting 15s for container to start..."
sleep 15

"$SCRIPT_DIR/4_get_url.sh"
