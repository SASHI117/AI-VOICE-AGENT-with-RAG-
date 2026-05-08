#!/bin/bash
# =====================================================
# First-time Azure setup
# Creates: resource group, container registry, container apps environment.
# Idempotent — safe to re-run.
# =====================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

echo "================================================="
echo "  Azure first-time setup"
echo "================================================="
echo "  Resource Group:    $RESOURCE_GROUP"
echo "  Location:          $LOCATION"
echo "  Container Registry: $ACR_NAME"
echo "  Apps Environment:  $ENV_NAME"
echo "================================================="
echo ""

# Verify logged in
if ! az account show &>/dev/null; then
    echo "❌ Not logged in to Azure. Run: az login"
    exit 1
fi

SUBSCRIPTION=$(az account show --query name -o tsv)
echo "✅ Using subscription: $SUBSCRIPTION"
echo ""

# 1. Resource group
echo "📦 [1/4] Resource group..."
if az group show --name "$RESOURCE_GROUP" &>/dev/null; then
    echo "   ✓ already exists"
else
    az group create --name "$RESOURCE_GROUP" --location "$LOCATION" --output none
    echo "   ✓ created"
fi
echo ""

# 2. Container registry
echo "🗄️  [2/4] Container registry..."
if az acr show --name "$ACR_NAME" --resource-group "$RESOURCE_GROUP" &>/dev/null; then
    echo "   ✓ already exists"
else
    az acr create \
        --name "$ACR_NAME" \
        --resource-group "$RESOURCE_GROUP" \
        --location "$LOCATION" \
        --sku Basic \
        --admin-enabled true \
        --output none
    echo "   ✓ created"
fi
echo ""

# 3. Container Apps extension
echo "🔧 [3/4] Container Apps CLI extension..."
az extension add --name containerapp --upgrade --only-show-errors --output none || true
echo "   ✓ ready"
echo ""

# Register required providers (idempotent)
echo "📝 Registering required resource providers..."
az provider register --namespace Microsoft.App --output none 2>/dev/null || true
az provider register --namespace Microsoft.OperationalInsights --output none 2>/dev/null || true
echo "   ✓ done"
echo ""

# 4. Container Apps environment
echo "🌐 [4/4] Container Apps environment..."
if az containerapp env show --name "$ENV_NAME" --resource-group "$RESOURCE_GROUP" &>/dev/null; then
    echo "   ✓ already exists"
else
    az containerapp env create \
        --name "$ENV_NAME" \
        --resource-group "$RESOURCE_GROUP" \
        --location "$LOCATION" \
        --output none
    echo "   ✓ created"
fi
echo ""

echo "================================================="
echo "  ✅ Azure setup complete"
echo "================================================="
echo ""
echo "Next: ./scripts/2_build_image.sh v1"
