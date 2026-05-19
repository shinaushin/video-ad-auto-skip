#!/usr/bin/env bash
# submit_tune.sh — Submit the Optuna hyperparameter search as a Vertex AI custom job.
#
# Usage:
#   bash training/scripts/submit_tune.sh                          # fresh start, both modalities
#   bash training/scripts/submit_tune.sh --embed-mode text_only
#   bash training/scripts/submit_tune.sh --embed-mode audio_only
#   bash training/scripts/submit_tune.sh --resume                 # resume from existing study DB

set -euo pipefail
source "$(dirname "$0")/common_vars.sh"

# ── Parse arguments ───────────────────────────────────────────────────────
EMBED_MODE="both"
MIN_VOTES=3
RESUME=false
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
    --resume)
      RESUME=true
      shift
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

JOB_NAME="yt-sponsor-tune-${EMBED_MODE}-mv${MIN_VOTES}-$(date +%Y%m%d-%H%M%S)"
echo "Submitting tune job: ${JOB_NAME}  (embed_mode=${EMBED_MODE}, min_votes=${MIN_VOTES}, resume=${RESUME})"

# Include --gcs-study-db only when resuming.
STUDY_DB_ARG=""
if $RESUME; then
  STUDY_DB_ARG=",--gcs-study-db,${GCS_STUDY_DB}"
fi

gcloud ai custom-jobs create \
  --project="${PROJECT_ID}" \
  --region="${REGION}" \
  --display-name="${JOB_NAME}" \
  --worker-pool-spec="machine-type=${MACHINE_TYPE},accelerator-type=${ACCELERATOR_TYPE},accelerator-count=${ACCELERATOR_COUNT},container-image-uri=${IMAGE_URI}" \
  --args="training/src/tune.py,--config,${CFG_TUNE},--gcs-input,${GCS_EMBEDDINGS},--gcs-output,${GCS_TUNE_OUT}/${JOB_NAME},--cloud-logging,--job-name,${JOB_NAME},--embed-mode,${EMBED_MODE},--min-votes,${MIN_VOTES}${STUDY_DB_ARG}"

echo ""
echo "✓ Job submitted: ${JOB_NAME}"
echo "  Monitor: https://console.cloud.google.com/vertex-ai/training/custom-jobs?project=${PROJECT_ID}"
echo "  Logs:    https://console.cloud.google.com/logs/query?project=${PROJECT_ID}"
echo "  Results: ${GCS_TUNE_OUT}/${JOB_NAME}/"
