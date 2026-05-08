#!/bin/bash
# =====================================================
# Edit these names BEFORE running any deploy script.
# All scripts source this file.
# =====================================================

# Resource group — groups all Azure resources for this project
export RESOURCE_GROUP="rg-apsagent-voicebot-new"

# Region — pick one close to your callers
#   centralindia, southeastasia, westeurope, eastus, etc.
export LOCATION="centralindia"

# Container Registry — globally unique, lowercase, no hyphens, 5-50 chars
# Pick something like: pipecat<yourname><randomdigits>
export ACR_NAME="apsagentacr2026"

# Container Apps environment — runtime where the container lives
export ENV_NAME="apsagent-env"

# Container App name — your bot's identity
export APP_NAME="apsagent-app"

# Image repository name (inside the registry)
export IMAGE_REPO="apsagent-image"

# Resource sizing — defaults are fine for testing
export CPU="0.5"           # 0.25, 0.5, 0.75, 1.0, ...
export MEMORY="1.0Gi"      # 0.5Gi, 1.0Gi, 2.0Gi, ...
export MIN_REPLICAS="1"    # Set to 0 for scale-to-zero (cheaper but cold starts)
export MAX_REPLICAS="10"   # Auto-scale ceiling
