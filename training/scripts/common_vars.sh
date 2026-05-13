#!/usr/bin/env bash
# common_vars.sh — Shared GCP configuration for all training scripts.
# Source this file at the top of each submit/build script:
#   source "$(dirname "$0")/common_vars.sh"

# ── GCP project & region ──────────────────────────────────────────────────
export PROJECT_ID="sponsor-segment-skipper"
export REGION="us-central1"          # Vertex AI region
export ZONE="us-central1-a"

# ── GCS bucket ────────────────────────────────────────────────────────────
export BUCKET="gs://yt-sponsor-cache"
export GCS_EMBEDDINGS="${BUCKET}/embeddings"       # Phase 1 output (synced from Mac)
export GCS_TUNE_OUT="${BUCKET}/outputs/tune"       # tune.py results
export GCS_TRAIN_OUT="${BUCKET}/outputs/teacher"   # train.py (teacher) results
export GCS_DISTILL_OUT="${BUCKET}/outputs/distill" # distill results
export GCS_EXPORT_OUT="${BUCKET}/outputs/export"   # ONNX model
export GCS_EVAL_OUT="${BUCKET}/outputs/eval"       # eval_report.json (teacher + student)
export GCS_STUDY_DB="${BUCKET}/tune/optuna_study.db"  # Optuna DB (cross-job resume)

# ── Artifact Registry image ───────────────────────────────────────────────
export AR_REPO="us-central1-docker.pkg.dev/${PROJECT_ID}/training"
export IMAGE_NAME="yt-sponsor-train"
export IMAGE_TAG="latest"
export IMAGE_URI="${AR_REPO}/${IMAGE_NAME}:${IMAGE_TAG}"

# ── Vertex AI job machine spec ────────────────────────────────────────────
export MACHINE_TYPE="n1-standard-8"
export ACCELERATOR_TYPE="NVIDIA_TESLA_T4"
export ACCELERATOR_COUNT="1"

# ── Configs (stored in repo; referenced as local paths inside the container) ──
export CFG_TUNE="/app/training/configs/phase3_tune.json"
export CFG_TEACHER="/app/training/configs/phase3_teacher.json"
export CFG_DISTILL="/app/training/configs/phase4_distill.json"
export CFG_EXPORT="/app/training/configs/phase5_export.json"

echo "✓ GCP vars loaded  project=${PROJECT_ID}  region=${REGION}  image=${IMAGE_URI}"
