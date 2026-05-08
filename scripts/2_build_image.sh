#!/bin/bash
# =====================================================
# Build the Docker image inside Azure Container Registry.
# No local Docker required — Azure builds it for you.
#
# Usage:  ./2_build_image.sh <tag>
# Example: ./2_build_image.sh v1
# =====================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

TAG="${1:-latest}"
KIT_ROOT="$(dirname "$SCRIPT_DIR")"
IMAGE_FULL="${ACR_NAME}.azurecr.io/${IMAGE_REPO}:${TAG}"

echo "================================================="
echo "  Build image"
echo "================================================="
echo "  Registry: $ACR_NAME.azurecr.io"
echo "  Image:    $IMAGE_REPO:$TAG"
echo "================================================="
echo ""

cd "$KIT_ROOT"

az acr build \
    --registry "$ACR_NAME" \
    --image "${IMAGE_REPO}:${TAG}" \
    --file Dockerfile \
    .

echo ""
echo "✅ Built and pushed: $IMAGE_FULL"
echo ""
echo "Next: ./scripts/3_deploy_container.sh $TAG"
