#!/bin/bash
# =====================================================
# Deploy (or update) the Container App.
# - First run: creates the app, attaches env vars + secrets, sets up ingress
# - Subsequent runs: rolling update to the new image tag
#
# Usage:   ./3_deploy_container.sh <tag>
# Example: ./3_deploy_container.sh v1
# =====================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

TAG="${1:-latest}"
KIT_ROOT="$(dirname "$SCRIPT_DIR")"
ENV_FILE="$KIT_ROOT/.env"
IMAGE_FULL="${ACR_NAME}.azurecr.io/${IMAGE_REPO}:${TAG}"

if [ ! -f "$ENV_FILE" ]; then
    echo "❌ .env file not found at $ENV_FILE"
    echo "   Create it from .env.example before deploying."
    exit 1
fi

echo "================================================="
echo "  Deploy container"
echo "================================================="
echo "  App:    $APP_NAME"
echo "  Image:  $IMAGE_FULL"
echo "  CPU:    $CPU"
echo "  Memory: $MEMORY"
echo "  Replicas: $MIN_REPLICAS - $MAX_REPLICAS"
echo "================================================="
echo ""

# Read .env, split into secrets (anything matching *_KEY / *_TOKEN / *_SECRET / PASSWORD)
# and plain env vars. Secrets get stored encrypted in Container Apps.
SECRETS_ARGS=()
ENV_VAR_ARGS=()
SECRET_REFS=()

while IFS= read -r line || [ -n "$line" ]; do
    line="${line//$'\r'/}"
    # Skip comments and empty lines
    [[ "$line" =~ ^[[:space:]]*# ]] && continue
    [[ -z "${line// }" ]] && continue

    # Parse KEY=VALUE
    if [[ "$line" =~ ^([A-Z_][A-Z0-9_]*)=(.*)$ ]]; then
        KEY="${BASH_REMATCH[1]}"
        VAL="${BASH_REMATCH[2]}"
        # Trim surrounding quotes
        VAL="${VAL%\"}"
        VAL="${VAL#\"}"
        VAL="${VAL%\'}"
        VAL="${VAL#\'}"

        # Skip empty values
        [[ -z "$VAL" ]] && continue

        # Sensitive: route to Container App secret store
        if [[ "$KEY" == *_KEY || "$KEY" == *_TOKEN || "$KEY" == *_SECRET || "$KEY" == *PASSWORD* || "$KEY" == *CREDENTIAL* ]]; then
            SECRET_NAME=$(echo "$KEY" | tr '[:upper:]_' '[:lower:]-')
            SECRETS_ARGS+=("$SECRET_NAME=$VAL")
            SECRET_REFS+=("$KEY=secretref:$SECRET_NAME")
        else
            ENV_VAR_ARGS+=("$KEY=$VAL")
        fi
    fi
done < "$ENV_FILE"

ALL_ENV_VARS=("${ENV_VAR_ARGS[@]}" "${SECRET_REFS[@]}")

# Check if app already exists
if az containerapp show --name "$APP_NAME" --resource-group "$RESOURCE_GROUP" &>/dev/null; then
    echo "🔄 App exists — updating to new image..."

    # Update secrets first (separately, since update doesn't take both at once cleanly)
    if [ ${#SECRETS_ARGS[@]} -gt 0 ]; then
        echo "🔐 Updating secrets..."
        az containerapp secret set \
            --name "$APP_NAME" \
            --resource-group "$RESOURCE_GROUP" \
            --secrets "${SECRETS_ARGS[@]}" \
            --output none
    fi

    # Update image + env vars
    az containerapp update \
        --name "$APP_NAME" \
        --resource-group "$RESOURCE_GROUP" \
        --image "$IMAGE_FULL" \
        --set-env-vars "${ALL_ENV_VARS[@]}" \
        --output none

    echo "✅ Updated"
else
    echo "🆕 First deploy — creating app..."

    # Get ACR credentials for image pull
    ACR_USERNAME=$(az acr credential show --name "$ACR_NAME" --query username -o tsv | tr -d '\r')
    ACR_PASSWORD=$(az acr credential show --name "$ACR_NAME" --query "passwords[0].value" -o tsv | tr -d '\r')

    # Build args for create
    CREATE_ARGS=(
        --name "$APP_NAME"
        --resource-group "$RESOURCE_GROUP"
        --environment "$ENV_NAME"
        --image "$IMAGE_FULL"
        --target-port 8000
        --ingress external
        --registry-server "${ACR_NAME}.azurecr.io"
        --registry-username "$ACR_USERNAME"
        --registry-password "$ACR_PASSWORD"
        --cpu "$CPU"
        --memory "$MEMORY"
        --min-replicas "$MIN_REPLICAS"
        --max-replicas "$MAX_REPLICAS"
        --transport auto
    )

    if [ ${#SECRETS_ARGS[@]} -gt 0 ]; then
        CREATE_ARGS+=(--secrets "${SECRETS_ARGS[@]}")
    fi
    if [ ${#ALL_ENV_VARS[@]} -gt 0 ]; then
        CREATE_ARGS+=(--env-vars "${ALL_ENV_VARS[@]}")
    fi

    az containerapp create "${CREATE_ARGS[@]}" --output none

    echo "✅ Created"
fi

echo ""
echo "Next: ./scripts/4_get_url.sh"
