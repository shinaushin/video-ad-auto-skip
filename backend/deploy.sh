#!/usr/bin/env bash
# deploy.sh — Build, push, and deploy the sponsor prediction backend to Cloud Run.
#
# Usage:
#   bash backend/deploy.sh
#
# Prerequisites:
#   - gcloud auth login
#   - gcloud config set project sponsor-segment-skipper
#   - Artifact Registry repo already exists (same one used for training)
#   - Teacher checkpoint at: gs://yt-sponsor-cache/outputs/teacher/<TEACHER_RUN>/teacher_best.pt

set -euo pipefail

PROJECT_ID="sponsor-segment-skipper"
REGION="us-central1"
AR_REPO="us-central1-docker.pkg.dev/${PROJECT_ID}/training"
IMAGE_NAME="yt-sponsor-backend"
IMAGE_TAG="latest"
IMAGE_URI="${AR_REPO}/${IMAGE_NAME}:${IMAGE_TAG}"
SERVICE_NAME="yt-sponsor-predict"

TEACHER_RUN="${TEACHER_RUN:-yt-sponsor-teacher-both-mv3-20260601-024344}"
TEACHER_CKPT_GCS="gs://yt-sponsor-cache/outputs/teacher/${TEACHER_RUN}/teacher_best.pt"

echo "Building image: ${IMAGE_URI}"
# Build from repo root so training/src/ is in the Docker context
docker build \
  -f backend/Dockerfile \
  -t "${IMAGE_URI}" \
  .

echo "Pushing image …"
docker push "${IMAGE_URI}"

echo "Downloading teacher checkpoint to /tmp for Cloud Run startup …"
# The teacher checkpoint is baked into the image at build time for fast startup.
# Alternative: download at container start via TEACHER_CKPT_GCS env var.
# Here we use a startup script approach via Cloud Run environment variables.

echo "Deploying to Cloud Run …"
gcloud run deploy "${SERVICE_NAME}" \
  --image             "${IMAGE_URI}" \
  --region            "${REGION}" \
  --project           "${PROJECT_ID}" \
  --platform          managed \
  --allow-unauthenticated \
  --memory            4Gi \
  --cpu               4 \
  --timeout           300s \
  --concurrency       4 \
  --min-instances     0 \
  --max-instances     4 \
  --set-env-vars      "TEACHER_CKPT_PATH=/tmp/teacher_best.pt,DEVICE=cpu" \
  --set-env-vars      "ALLOWED_ORIGINS=https://www.youtube.com,https://m.youtube.com"

# Get the service URL
SERVICE_URL=$(gcloud run services describe "${SERVICE_NAME}" \
  --region "${REGION}" \
  --project "${PROJECT_ID}" \
  --format "value(status.url)")

echo ""
echo "✓ Deployed: ${SERVICE_URL}"
echo ""
echo "  Update BACKEND_URL in youtube-ml-sponsor-detector/content.js:"
echo "    const BACKEND_URL = \"${SERVICE_URL}\";"
echo ""
echo "  Test:"
echo "    curl -X POST ${SERVICE_URL}/predict \\"
echo "      -H 'Content-Type: application/json' \\"
echo "      -d '{\"video_id\": \"dQw4w9WgXcQ\"}'"
