#!/bin/bash
# =====================================================
# Print the public URLs for your deployed agent.
# These are what you paste into Exotel / Smartflo dashboards.
# =====================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

if ! az containerapp show --name "$APP_NAME" --resource-group "$RESOURCE_GROUP" &>/dev/null; then
    echo "❌ App '$APP_NAME' not found. Have you run scripts 1 → 3 yet?"
    exit 1
fi

FQDN=$(az containerapp show \
    --name "$APP_NAME" \
    --resource-group "$RESOURCE_GROUP" \
    --query "properties.configuration.ingress.fqdn" \
    -o tsv | tr -d '\r')

REVISION=$(az containerapp show \
    --name "$APP_NAME" \
    --resource-group "$RESOURCE_GROUP" \
    --query "properties.latestRevisionName" \
    -o tsv | tr -d '\r')

IMAGE=$(az containerapp show \
    --name "$APP_NAME" \
    --resource-group "$RESOURCE_GROUP" \
    --query "properties.template.containers[0].image" \
    -o tsv | tr -d '\r')

echo "================================================="
echo "  ✅ Your agent is live!"
echo "================================================="
echo ""
echo "  Image:    $IMAGE"
echo "  Revision: $REVISION"
echo ""
echo "  Health check:"
echo "    https://$FQDN/health"
echo ""
echo "  Paste into Exotel:"
echo "    wss://$FQDN/ws/exotel"
echo ""
echo "  Paste into Smartflo:"
echo "    wss://$FQDN/ws/smartflo"
echo ""
echo "================================================="
echo ""
echo "Tail logs:"
echo "  az containerapp logs show --name $APP_NAME -g $RESOURCE_GROUP --follow"
