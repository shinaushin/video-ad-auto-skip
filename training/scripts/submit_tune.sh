#!/usr/bin/env bash
# submit_tune.sh — Submit the Optuna hyperparameter search as a Vertex AI custom job.
#
# Usage:
#   bash training/scripts/submit_tune.sh

set -euo pipefail
source "$(dirname "$0")/common_vars.sh"

JOB_NAME="yt-sponsor-tune-$(date +%Y%m%d-%H%M%S)"
echo "Submitting tune job: ${JOB_NAME}"

gcloud ai custom-jobs create \
  --project="${PROJECT_ID}" \
  --region="${REGION}" \
  --display-name="${JOB_NAME}" \
  --worker-pool-spec="machine-type=${MACHINE_TYPE},accelerator-type=${ACCELERATOR_TYPE},accelerator-count=${ACCELERATOR_COUNT},container-image-uri=${IMAGE_URI}" \
  --args="training/src/tune.py,--config,${CFG_TUNE},--gcs-input,${GCS_EMBEDDINGS},--gcs-output,${GCS_TUNE_OUT}/${JOB_NAME},--gcs-study-db,${GCS_STUDY_DB},--cloud-logging,--job-name,${JOB_NAME}"

echo ""
echo "✓ Job submitted: ${JOB_NAME}"
echo "  Monitor: https://console.cloud.google.com/vertex-ai/training/custom-jobs?project=${PROJECT_ID}"
echo "  Logs:    https://console.cloud.google.com/logs/query?project=${PROJECT_ID}"
echo "  Results: ${GCS_TUNE_OUT}/${JOB_NAME}/"
