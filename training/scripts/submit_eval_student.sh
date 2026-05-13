#!/usr/bin/env bash
# submit_eval_student.sh — Evaluate the student model on the held-out test split.
#
# Usage:
#   DISTILL_RUN=yt-sponsor-distill-20260510-180000
#   bash training/scripts/submit_eval_student.sh

set -euo pipefail
source "$(dirname "$0")/common_vars.sh"

DISTILL_RUN="${DISTILL_RUN:-}"
if [[ -z "${DISTILL_RUN}" ]]; then
  echo "ERROR: Set DISTILL_RUN to the distill job name before running."
  echo "  e.g.:  DISTILL_RUN=yt-sponsor-distill-20260510-180000 bash submit_eval_student.sh"
  exit 1
fi

STUDENT_CKPT_GCS="${GCS_DISTILL_OUT}/${DISTILL_RUN}/student_best.pt"
LOCAL_STUDENT_CKPT="/tmp/student_best.pt"

JOB_NAME="yt-sponsor-eval-student-$(date +%Y%m%d-%H%M%S)"
echo "Submitting student eval job: ${JOB_NAME}"
echo "  Checkpoint: ${STUDENT_CKPT_GCS}"

gcloud ai custom-jobs create \
  --project="${PROJECT_ID}" \
  --region="${REGION}" \
  --display-name="${JOB_NAME}" \
  --worker-pool-spec="machine-type=${MACHINE_TYPE},accelerator-type=${ACCELERATOR_TYPE},accelerator-count=${ACCELERATOR_COUNT},container-image-uri=${IMAGE_URI}" \
  --command="bash" \
  --args="-c,gcloud storage cp ${STUDENT_CKPT_GCS} ${LOCAL_STUDENT_CKPT} && python3 -u training/src/eval.py --phase student --checkpoint ${LOCAL_STUDENT_CKPT} --config ${CFG_DISTILL} --gcs-input ${GCS_EMBEDDINGS} --gcs-output ${GCS_EVAL_OUT}/student/${DISTILL_RUN} --cloud-logging --job-name ${JOB_NAME}"

echo ""
echo "✓ Student eval job submitted: ${JOB_NAME}"
echo "  Report: ${GCS_EVAL_OUT}/student/${DISTILL_RUN}/eval_report.json"
echo ""
echo "  To download: gcloud storage cp ${GCS_EVAL_OUT}/student/${DISTILL_RUN}/eval_report.json ."
