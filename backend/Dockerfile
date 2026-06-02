FROM python:3.11-slim

# System deps: ffmpeg (required by yt-dlp audio postprocessor)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    apt-transport-https \
    ca-certificates \
    gnupg \
    curl \
    && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" \
       > /etc/apt/sources.list.d/google-cloud-sdk.list \
    && apt-get update && apt-get install -y --no-install-recommends google-cloud-cli \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (Docker layer caching)
COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy training/src so predictor.py can import models.py and load_teacher
COPY training/src/ ./training/src/

# Copy backend source
COPY backend/ ./

ENV PYTHONUNBUFFERED=1
ENV PORT=8080

# Teacher checkpoint is mounted or downloaded at runtime via TEACHER_CKPT_PATH.
# Default path matches the Cloud Run job startup script.
ENV TEACHER_CKPT_PATH=/tmp/teacher_best.pt
ENV DEVICE=cpu

EXPOSE 8080

COPY backend/startup.sh ./startup.sh
RUN chmod +x startup.sh

CMD ["bash", "startup.sh"]
