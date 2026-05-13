#!/usr/bin/env bash
# submit_eval_teacher.sh — Evaluate the teacher model on the held-out test split.
#
# Usage:
#   TEACHER_RUN=yt-sponsor-teacher-20260510-120000
#   bash training/scripts/submit_eval_teacher.sh

set -euo pipefail
source "$(dirname "$0")/common_vars.sh"

TEACHER_RUN="${TEACHER_RUN:-}"
if [[ -z "${TEACHER_RUN}" ]]; then
  echo "ERROR: Set TEACHER_RUN to the teacher job name before running."
  echo "  e.g.:  TEACHER_RUN=yt-sponsor-teacher-20260510-120000 bash submit_eval_teacher.sh"
  exit 1
fi

TEACHER_CKPT_GCS="${GCS_TRAIN_OUT}/${TEACHER_RUN}/teacher_best.pt"
LOCAL_TEACHER_CKPT="/tmp/teacher_best.pt"

JOB_NAME="yt-sponsor-eval-teacher-$(date +%Y%m%d-%H%M%S)"
echo "Submitting teacher eval job: ${JOB_NAME}"
echo "  Checkpoint: ${TEACHER_CKPT_GCS}"

gcloud ai custom-jobs create \
  --project="${PROJECT_ID}" \
  --region="${REGION}" \
  --display-name="${JOB_NAME}" \
  --worker-pool-spec="machine-type=${MACHINE_TYPE},accelerator-type=${ACCELERATOR_TYPE},accelerator-count=${ACCELERATOR_COUNT},container-image-uri=${IMAGE_URI}" \
  --command="bash" \
  --args="-c,gcloud storage cp ${TEACHER_CKPT_GCS} ${LOCAL_TEACHER_CKPT} && python3 -u training/src/eval.py --phase teacher --checkpoint ${LOCAL_TEACHER_CKPT} --config ${CFG_TEACHER} --gcs-input ${GCS_EMBEDDINGS} --gcs-output ${GCS_EVAL_OUT}/teacher/${TEACHER_RUN} --cloud-logging --job-name ${JOB_NAME}"

echo ""
echo "✓ Teacher eval job submitted: ${JOB_NAME}"
echo "  Report: ${GCS_EVAL_OUT}/teacher/${TEACHER_RUN}/eval_report.json"
echo ""
echo "  To download: gcloud storage cp ${GCS_EVAL_OUT}/teacher/${TEACHER_RUN}/eval_report.json ."
