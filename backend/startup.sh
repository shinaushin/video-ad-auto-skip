#!/usr/bin/env bash
# startup.sh — Download teacher checkpoint from GCS then start the server.
# Used as the container CMD so the checkpoint is always fresh from GCS.
set -euo pipefail

CKPT_GCS="${TEACHER_CKPT_GCS:-gs://yt-sponsor-cache/outputs/teacher/yt-sponsor-teacher-both-mv3-20260601-024344/teacher_best.pt}"
CKPT_LOCAL="${TEACHER_CKPT_PATH:-/tmp/teacher_best.pt}"

if [ ! -f "${CKPT_LOCAL}" ]; then
  echo "Downloading teacher checkpoint: ${CKPT_GCS} → ${CKPT_LOCAL}"
  gcloud storage cp "${CKPT_GCS}" "${CKPT_LOCAL}"
  echo "Checkpoint ready."
else
  echo "Checkpoint already present at ${CKPT_LOCAL}."
fi

exec python main.py
