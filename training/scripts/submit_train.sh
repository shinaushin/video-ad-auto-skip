#!/usr/bin/env bash
# submit_train.sh — Submit the teacher model training as a Vertex AI custom job.
#
# Usage:
#   bash training/scripts/submit_train.sh

set -euo pipefail
source "$(dirname "$0")/common_vars.sh"

JOB_NAME="yt-sponsor-teacher-$(date +%Y%m%d-%H%M%S)"
echo "Submitting teacher training job: ${JOB_NAME}"

gcloud ai custom-jobs create \
  --project="${PROJECT_ID}" \
  --region="${REGION}" \
  --display-name="${JOB_NAME}" \
  --worker-pool-spec="machine-type=${MACHINE_TYPE},accelerator-type=${ACCELERATOR_TYPE},accelerator-count=${ACCELERATOR_COUNT},container-image-uri=${IMAGE_URI}" \
  --args="training/src/train.py,--config,${CFG_TEACHER},--gcs-input,${GCS_EMBEDDINGS},--gcs-output,${GCS_TRAIN_OUT}/${JOB_NAME},--cloud-logging,--job-name,${JOB_NAME}"

echo ""
echo "✓ Job submitted: ${JOB_NAME}"
echo "  Monitor:    https://console.cloud.google.com/vertex-ai/training/custom-jobs?project=${PROJECT_ID}"
echo "  Checkpoint: ${GCS_TRAIN_OUT}/${JOB_NAME}/teacher_best.pt"
