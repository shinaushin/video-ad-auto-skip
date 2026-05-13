#!/usr/bin/env bash
# submit_export.sh — Export student_best.pt to ONNX as a Vertex AI custom job.
#
# Usage:
#   DISTILL_RUN=yt-sponsor-distill-20260510-180000
#   bash training/scripts/submit_export.sh

set -euo pipefail
source "$(dirname "$0")/common_vars.sh"

DISTILL_RUN="${DISTILL_RUN:-}"
if [[ -z "${DISTILL_RUN}" ]]; then
  echo "ERROR: Set DISTILL_RUN to the distill job name before running."
  echo "  e.g.:  DISTILL_RUN=yt-sponsor-distill-20260510-180000 bash submit_export.sh"
  exit 1
fi

STUDENT_CKPT_GCS="${GCS_DISTILL_OUT}/${DISTILL_RUN}/student_best.pt"
LOCAL_STUDENT_CKPT="/tmp/student_best.pt"

JOB_NAME="yt-sponsor-export-$(date +%Y%m%d-%H%M%S)"
echo "Submitting ONNX export job: ${JOB_NAME}"
echo "  Student checkpoint: ${STUDENT_CKPT_GCS}"

gcloud ai custom-jobs create \
  --project="${PROJECT_ID}" \
  --region="${REGION}" \
  --display-name="${JOB_NAME}" \
  --worker-pool-spec="machine-type=n1-standard-4,container-image-uri=${IMAGE_URI}" \
  --command="bash" \
  --args="-c,gcloud storage cp ${STUDENT_CKPT_GCS} ${LOCAL_STUDENT_CKPT} && python3 -u training/src/export_onnx.py --checkpoint ${LOCAL_STUDENT_CKPT} --out /tmp/model.onnx --validate && gcloud storage cp /tmp/model.onnx ${GCS_EXPORT_OUT}/model.onnx"

echo ""
echo "✓ Export job submitted: ${JOB_NAME}"
echo "  ONNX model will be at: ${GCS_EXPORT_OUT}/model.onnx"
echo ""
echo "  To download: gcloud storage cp ${GCS_EXPORT_OUT}/model.onnx extension/model.onnx"
