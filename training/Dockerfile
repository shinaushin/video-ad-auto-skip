# Dockerfile for Vertex AI custom training jobs.
#
# Base: PyTorch 2.1.2 + CUDA 11.8 (compatible with V100, T4, A100).
# Includes gcloud CLI for GCS I/O and all Python training dependencies.
#
# Build & push:
#   bash training/scripts/build_push.sh
#
# Run locally (for smoke-testing):
#   docker run --rm --gpus all \
#     -v ~/.config/gcloud:/root/.config/gcloud \
#     IMAGE \
#     training/src/train.py --config training/configs/phase3_teacher.json

FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime

# Prevent apt from prompting for timezone/keyboard input during package installs.
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# ── System deps & gcloud CLI ──────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        gnupg \
        ffmpeg \
        git \
    && echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] \
        https://packages.cloud.google.com/apt cloud-sdk main" \
        | tee /etc/apt/sources.list.d/google-cloud-sdk.list \
    && curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg \
        | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg \
    && apt-get update && apt-get install -y --no-install-recommends google-cloud-cli \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies ───────────────────────────────────────────────────
COPY training/requirements-vertex.txt /tmp/requirements-vertex.txt
RUN pip install --no-cache-dir -r /tmp/requirements-vertex.txt

# ── Training source code ──────────────────────────────────────────────────
WORKDIR /app
COPY training/ /app/training/

# training/src is the module root (scripts import each other without a package).
ENV PYTHONPATH=/app/training/src

# ── Entry point ───────────────────────────────────────────────────────────
# Vertex AI appends args from the job spec to this entrypoint.
# Example: python3 -u training/src/train.py --config gs://... --cloud-logging
ENTRYPOINT ["python3", "-u"]
CMD ["training/src/train.py", "--help"]
