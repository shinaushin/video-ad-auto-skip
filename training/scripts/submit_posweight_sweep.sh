#!/usr/bin/env bash
# submit_posweight_sweep.sh — Sweep pos_weight_mult on the teacher after dataset cleaning.
#
# Background: temporal consistency filtering removes isolated sponsor windows,
# shifting the effective class balance.  pos_weight_mult in BCEWithLogitsLoss
# is the parameter most sensitive to this shift, so we sweep it cheaply with
# Hydra --multirun rather than re-running a full Optuna search.
#
# Each trial runs a full teacher training job on Vertex AI with the same
# architecture and learning rate as the existing best checkpoint.
# Pick the trial with the highest val/f1 and use that pos_weight_mult for
# the teacher retrain before distillation.
#
# Usage:
#   bash training/scripts/submit_posweight_sweep.sh
#
# Output:
#   One Vertex AI custom job per pos_weight_mult value.
#   Checkpoints at: gs://yt-sponsor-cache/outputs/posweight_sweep/<job_name>/
#   TensorBoard:    gs://yt-sponsor-cache/outputs/posweight_sweep/<job_name>/tb/

set -euo pipefail
source "$(dirname "$0")/common_vars.sh"

# ── Sweep values ──────────────────────────────────────────────────────────────
# Four trials: below, at, and above the previous best (1.87).
POS_WEIGHTS=(1.5 2.0 2.5 3.0)

GCS_SWEEP_OUT="${BUCKET}/outputs/posweight_sweep"

echo "Submitting pos_weight_mult sweep: ${POS_WEIGHTS[*]}"
echo "  Base config: conf/model/teacher.yaml + conf/training/teacher.yaml"
echo "  Output root: ${GCS_SWEEP_OUT}"
echo ""

for PW in "${POS_WEIGHTS[@]}"; do
  JOB_NAME="yt-sponsor-posweight-${PW//./_}-$(date +%Y%m%d-%H%M%S)"

  echo "  Submitting: pos_weight_mult=${PW}  job=${JOB_NAME}"

  gcloud ai custom-jobs create \
    --project="${PROJECT_ID}" \
    --region="${REGION}" \
    --display-name="${JOB_NAME}" \
    --worker-pool-spec="machine-type=${MACHINE_TYPE},accelerator-type=${ACCELERATOR_TYPE},accelerator-count=${ACCELERATOR_COUNT},container-image-uri=${IMAGE_URI}" \
    --args="training/src/train_lightning.py,phase=teacher,model.pos_weight_mult=${PW},gcs.input=${GCS_EMBEDDINGS},gcs.output=${GCS_SWEEP_OUT}/${JOB_NAME},cloud_logging=true,job_name=${JOB_NAME}"

  # Small delay so the date-stamped job names don't collide.
  sleep 2
done

echo ""
echo "✓ ${#POS_WEIGHTS[@]} sweep jobs submitted."
echo ""
echo "  Monitor:  https://console.cloud.google.com/vertex-ai/training/custom-jobs?project=${PROJECT_ID}"
echo "  TensorBoard (compare all trials):"
echo "    tensorboard --logdir ${GCS_SWEEP_OUT}"
echo ""
echo "  Once jobs finish, pick the pos_weight_mult with the highest val/f1 and"
echo "  update training/conf/model/teacher.yaml before running submit_train.sh."
