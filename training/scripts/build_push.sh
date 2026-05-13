#!/usr/bin/env bash
# build_push.sh — Build the training Docker image and push to Artifact Registry.
#
# Prerequisites (one-time):
#   gcloud auth configure-docker us-central1-docker.pkg.dev
#   gcloud artifacts repositories create training \
#     --repository-format=docker \
#     --location=us-central1 \
#     --description="YT sponsor training images"
#
# Usage:
#   bash training/scripts/build_push.sh            # build + push latest
#   bash training/scripts/build_push.sh --no-push  # build only (local test)

set -euo pipefail
source "$(dirname "$0")/common_vars.sh"

PUSH=true
if [[ "${1:-}" == "--no-push" ]]; then
  PUSH=false
  echo "⚠  --no-push: will build but not push to Artifact Registry."
fi

# Run from the repo root so COPY paths in the Dockerfile resolve correctly.
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
echo "Building from repo root: ${REPO_ROOT}"

docker build \
  --platform linux/amd64 \
  -f "${REPO_ROOT}/training/Dockerfile" \
  -t "${IMAGE_URI}" \
  "${REPO_ROOT}"

echo "✓ Image built: ${IMAGE_URI}"

if $PUSH; then
  docker push "${IMAGE_URI}"
  echo "✓ Image pushed: ${IMAGE_URI}"
else
  echo "Skipped push (--no-push)."
fi
