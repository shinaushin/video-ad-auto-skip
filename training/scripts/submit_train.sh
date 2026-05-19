#!/usr/bin/env bash
# submit_train.sh — Submit the teacher model training as a Vertex AI custom job.
#
# Usage:
#   bash training/scripts/submit_train.sh                          # default: both modalities
#   bash training/scripts/submit_train.sh --embed-mode text_only
#   bash training/scripts/submit_train.sh --embed-mode audio_only

set -euo pipefail
source "$(dirname "$0")/common_vars.sh"

# ── Parse arguments ───────────────────────────────────────────────────────
EMBED_MODE="both"
MIN_VOTES=3
while [[ $# -gt 0 ]]; do
  case "$1" in
    --embed-mode)
      EMBED_MODE="$2"
      shift 2
      ;;
    --min-votes)
      MIN_VOTES="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if [[ ! "${EMBED_MODE}" =~ ^(both|text_only|audio_only)$ ]]; then
  echo "Error: --embed-mode must be 'both', 'text_only', or 'audio_only'" >&2
  exit 1
fi

JOB_NAME="yt-sponsor-teacher-${EMBED_MODE}-mv${MIN_VOTES}-$(date +%Y%m%d-%H%M%S)"
echo "Submitting teacher training job: ${JOB_NAME}  (embed_mode=${EMBED_MODE}, min_votes=${MIN_VOTES})"

gcloud ai custom-jobs create \
  --project="${PROJECT_ID}" \
  --region="${REGION}" \
  --display-name="${JOB_NAME}" \
  --worker-pool-spec="machine-type=${MACHINE_TYPE},accelerator-type=${ACCELERATOR_TYPE},accelerator-count=${ACCELERATOR_COUNT},container-image-uri=${IMAGE_URI}" \
  --args="training/src/train.py,--config,${CFG_TEACHER},--gcs-input,${GCS_EMBEDDINGS},--gcs-output,${GCS_TRAIN_OUT}/${JOB_NAME},--cloud-logging,--job-name,${JOB_NAME},--embed-mode,${EMBED_MODE},--min-votes,${MIN_VOTES}"

echo ""
echo "✓ Job submitted: ${JOB_NAME}"
echo "  Monitor:    https://console.cloud.google.com/vertex-ai/training/custom-jobs?project=${PROJECT_ID}"
echo "  Checkpoint: ${GCS_TRAIN_OUT}/${JOB_NAME}/teacher_best.pt"
