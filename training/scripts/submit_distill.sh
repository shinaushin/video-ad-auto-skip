#!/usr/bin/env bash
# submit_distill.sh — Submit knowledge distillation as a Vertex AI custom job.
#
# Usage:
#   TEACHER_RUN=yt-sponsor-teacher-20260510-120000
#   bash training/scripts/submit_distill.sh

set -euo pipefail
source "$(dirname "$0")/common_vars.sh"

TEACHER_RUN="${TEACHER_RUN:-}"
if [[ -z "${TEACHER_RUN}" ]]; then
  echo "ERROR: Set TEACHER_RUN to the teacher job name before running."
  echo "  e.g.:  TEACHER_RUN=yt-sponsor-teacher-20260510-120000 bash submit_distill.sh"
  exit 1
fi

TEACHER_CKPT_GCS="${GCS_TRAIN_OUT}/${TEACHER_RUN}/teacher_best.pt"
LOCAL_TEACHER_CKPT="/tmp/teacher_best.pt"

JOB_NAME="yt-sponsor-distill-$(date +%Y%m%d-%H%M%S)"
echo "Submitting distillation job: ${JOB_NAME}"
echo "  Teacher checkpoint: ${TEACHER_CKPT_GCS}"

gcloud ai custom-jobs create \
  --project="${PROJECT_ID}" \
  --region="${REGION}" \
  --display-name="${JOB_NAME}" \
  --worker-pool-spec="machine-type=${MACHINE_TYPE},accelerator-type=${ACCELERATOR_TYPE},accelerator-count=${ACCELERATOR_COUNT},container-image-uri=${IMAGE_URI}" \
  --command="bash" \
  --args="-c,gcloud storage cp ${TEACHER_CKPT_GCS} ${LOCAL_TEACHER_CKPT} && python3 -u training/src/train.py --config ${CFG_DISTILL} --gcs-input ${GCS_EMBEDDINGS} --gcs-output ${GCS_DISTILL_OUT}/${JOB_NAME} --cloud-logging --job-name ${JOB_NAME}"

echo ""
echo "✓ Job submitted: ${JOB_NAME}"
echo "  Monitor:    https://console.cloud.google.com/vertex-ai/training/custom-jobs?project=${PROJECT_ID}"
echo "  Checkpoint: ${GCS_DISTILL_OUT}/${JOB_NAME}/student_best.pt"
