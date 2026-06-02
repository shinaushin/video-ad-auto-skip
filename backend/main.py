"""main.py — FastAPI backend for YouTube sponsor segment prediction.

Endpoints:
  POST /predict          — check cache; if miss, start background job
  GET  /result/{job_id} — poll for job completion
  GET  /health           — liveness check

Async job pattern:
  - POST /predict returns immediately with {job_id, status: "pending"}
    (or {status: "complete", segments: [...]} on a cache hit).
  - Background task runs the full pipeline (DistilBERT + Whisper + teacher).
  - Extension polls GET /result/{job_id} every 2s until status is "complete".
  - Completed predictions are written to Firestore cache so subsequent
    requests for the same video return instantly.

Configuration via environment variables:
  TEACHER_CKPT_PATH   Path to teacher_best.pt  (required)
  DEVICE              "cpu" | "cuda"            (default: "cpu")
  PORT                Server port               (default: 8080)
  ALLOWED_ORIGINS     Comma-separated CORS origins
                      (default: "https://www.youtube.com,https://m.youtube.com")
"""

from __future__ import annotations

import logging
import os
import traceback
from contextlib import asynccontextmanager

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from cache import PredictionCache
from predictor import SponsorPredictor

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s %(levelname)s %(name)s — %(message)s",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global state (loaded once at startup)
# ---------------------------------------------------------------------------

predictor: SponsorPredictor | None = None
cache:     PredictionCache  | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models and cache once on startup; release on shutdown."""
    global predictor, cache

    ckpt = os.environ.get("TEACHER_CKPT_PATH", "/tmp/teacher_best.pt")
    dev  = os.environ.get("DEVICE", "cpu")

    log.info("Starting up — loading models (ckpt=%s, device=%s) …", ckpt, dev)
    predictor = SponsorPredictor(teacher_ckpt=ckpt, device=dev)
    cache     = PredictionCache()
    log.info("Startup complete — ready to serve.")

    yield   # app is running

    log.info("Shutting down.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title       = "YouTube Sponsor Detector",
    description = "Predicts sponsor segments using a bimodal teacher model.",
    version     = "1.0.0",
    lifespan    = lifespan,
)

origins = os.environ.get(
    "ALLOWED_ORIGINS",
    "https://www.youtube.com,https://m.youtube.com",
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins     = origins,
    allow_credentials = False,
    allow_methods     = ["GET", "POST"],
    allow_headers     = ["Content-Type"],
)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    video_id: str

class Segment(BaseModel):
    start: float
    end:   float
    score: float

class PredictResponse(BaseModel):
    status:   str              # "complete" | "pending"
    job_id:   str | None = None
    segments: list[Segment] | None = None

class ResultResponse(BaseModel):
    status:   str              # "complete" | "pending" | "failed"
    segments: list[Segment] | None = None
    error:    str | None = None


# ---------------------------------------------------------------------------
# Background job
# ---------------------------------------------------------------------------

def _run_prediction_job(job_id: str, video_id: str) -> None:
    """Run the full pipeline in the background; write result to cache."""
    try:
        log.info("[%s] Job %s started.", video_id, job_id)
        segments = predictor.predict(video_id)
        cache.set_prediction(video_id, segments)
        cache.complete_job(job_id, segments)
        log.info("[%s] Job %s complete — %d segment(s).", video_id, job_id, len(segments))
    except Exception:
        err = traceback.format_exc()
        log.error("[%s] Job %s failed:\n%s", video_id, job_id, err)
        cache.fail_job(job_id, err[-500:])   # store last 500 chars of traceback


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest, background_tasks: BackgroundTasks):
    """Check cache; return segments immediately or start a background job.

    The extension should poll GET /result/{job_id} every 2s when status
    is "pending".
    """
    video_id = req.video_id.strip()
    if not video_id:
        raise HTTPException(status_code=400, detail="video_id is required")

    # ── Cache hit → return immediately ────────────────────────────────────
    cached = cache.get_prediction(video_id)
    if cached is not None:
        log.info("[%s] Cache hit — returning %d segment(s).", video_id, len(cached))
        return PredictResponse(
            status   = "complete",
            segments = [Segment(**s) for s in cached],
        )

    # ── Cache miss → queue background job ────────────────────────────────
    job_id = cache.create_job(video_id)
    background_tasks.add_task(_run_prediction_job, job_id, video_id)
    log.info("[%s] Job %s queued.", video_id, job_id)
    return PredictResponse(status="pending", job_id=job_id)


@app.get("/result/{job_id}", response_model=ResultResponse)
async def get_result(job_id: str):
    """Poll for job completion.

    Returns status "pending", "complete" (with segments), or "failed".
    """
    job = cache.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    status = job.get("status", "pending")

    if status == "complete":
        segs = job.get("segments") or []
        return ResultResponse(
            status   = "complete",
            segments = [Segment(**s) for s in segs],
        )

    if status == "failed":
        return ResultResponse(status="failed", error=job.get("error"))

    return ResultResponse(status="pending")


@app.get("/health")
async def health():
    """Liveness check — Cloud Run uses this to verify the container is ready."""
    return {
        "status":    "ok",
        "predictor": predictor is not None,
        "cache":     cache is not None,
    }


# ---------------------------------------------------------------------------
# Entry point (local dev)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host    = "0.0.0.0",
        port    = int(os.environ.get("PORT", 8080)),
        reload  = False,
        workers = 1,
    )
