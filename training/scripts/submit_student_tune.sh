#!/usr/bin/env bash
# submit_student_tune.sh — Submit student distillation hyperparameter search as a Vertex AI job.
#
# Runs tune_student.py with Optuna TPE sampler over:
#   lr, kd_temperature, kd_alpha, focal_gamma, focal_alpha, batch_size
#
# Usage:
#   TEACHER_RUN=yt-sponsor-teacher-both-mv3-20260601-024344
#   bash training/scripts/submit_student_tune.sh
#
# After the job finishes, download the best params:
#   gsutil cat gs://yt-sponsor-cache/outputs/student_tune/<JOB_NAME>/student_best_params.json
# Then update training/conf/training/distill.yaml with the best values.

set -euo pipefail
source "$(dirname "$0")/common_vars.sh"

TEACHER_RUN="${TEACHER_RUN:-}"
if [[ -z "${TEACHER_RUN}" ]]; then
  echo "ERROR: Set TEACHER_RUN to the teacher job name before running."
  echo "  e.g.: TEACHER_RUN=yt-sponsor-teacher-both-mv3-20260601-024344 bash submit_student_tune.sh"
  exit 1
fi

TEACHER_CKPT_GCS="${GCS_TRAIN_OUT}/${TEACHER_RUN}/teacher_best.pt"
LOCAL_TEACHER_CKPT="/tmp/teacher_best.pt"
GCS_TUNE_OUT="${BUCKET}/outputs/student_tune"

JOB_NAME="yt-sponsor-student-tune-$(date +%Y%m%d-%H%M%S)"
echo "Submitting student tune job: ${JOB_NAME}"
echo "  Teacher checkpoint: ${TEACHER_CKPT_GCS}"
echo "  Results:            ${GCS_TUNE_OUT}/${JOB_NAME}/"

gcloud ai custom-jobs create \
  --project="${PROJECT_ID}" \
  --region="${REGION}" \
  --display-name="${JOB_NAME}" \
  --worker-pool-spec="machine-type=${MACHINE_TYPE},accelerator-type=${ACCELERATOR_TYPE},accelerator-count=${ACCELERATOR_COUNT},container-image-uri=${IMAGE_URI}" \
  --command="bash" \
  --args="-c,gcloud storage cp ${TEACHER_CKPT_GCS} ${LOCAL_TEACHER_CKPT} && python3 -u training/src/tune_student.py --teacher-ckpt ${LOCAL_TEACHER_CKPT} --cache-dir /tmp/embeddings_cache --output-dir /tmp/outputs/student_tune --gcs-input ${GCS_EMBEDDINGS} --gcs-output ${GCS_TUNE_OUT}/${JOB_NAME} --n-trials 20 --tune-epochs 30 --tune-patience 8"

echo ""
echo "✓ Job submitted: ${JOB_NAME}"
echo "  Monitor: https://console.cloud.google.com/vertex-ai/training/custom-jobs?project=${PROJECT_ID}"
echo ""
echo "  When done, fetch best params:"
echo "    gsutil cat ${GCS_TUNE_OUT}/${JOB_NAME}/student_best_params.json"
