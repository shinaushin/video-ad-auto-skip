"""cache.py — Firestore-backed prediction cache and job state store.

Two Firestore collections:
  predictions/{video_id}  — cached segment lists with TTL (7 days)
  jobs/{job_id}           — async job state: pending | complete | failed

Using Firestore (instead of in-memory dict) means the cache and job state
are shared across all Cloud Run instances and persist across restarts.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

log = logging.getLogger(__name__)

# Prediction cache TTL — SponsorBlock labels are stable over days.
PREDICTION_TTL_DAYS = 7


class PredictionCache:
    """Thread-safe Firestore wrapper for predictions and job state."""

    def __init__(self) -> None:
        try:
            from google.cloud import firestore
            self._db = firestore.Client()
            self._predictions = self._db.collection("predictions")
            self._jobs        = self._db.collection("jobs")
            log.info("Firestore client initialised.")
        except Exception as e:
            log.warning("Firestore unavailable (%s) — using in-memory fallback.", e)
            self._db          = None
            self._predictions = {}   # video_id → {segments, expires_at}
            self._jobs        = {}   # job_id   → {status, segments, error}

    # ── Prediction cache ──────────────────────────────────────────────────────

    def get_prediction(self, video_id: str) -> Optional[list[dict]]:
        """Return cached segments for video_id, or None on miss/expiry."""
        if self._db is None:
            entry = self._predictions.get(video_id)
            if entry and entry["expires_at"] > datetime.now(tz=timezone.utc):
                log.info("[cache] HIT (memory): %s", video_id)
                return entry["segments"]
            return None

        try:
            doc = self._predictions.document(video_id).get()
            if not doc.exists:
                return None
            data       = doc.to_dict()
            expires_at = data.get("expires_at")
            if expires_at and expires_at < datetime.now(tz=timezone.utc):
                log.info("[cache] EXPIRED: %s", video_id)
                return None
            log.info("[cache] HIT (Firestore): %s  (%d segments)", video_id, len(data["segments"]))
            return data["segments"]
        except Exception as e:
            log.warning("[cache] get_prediction error: %s", e)
            return None

    def set_prediction(self, video_id: str, segments: list[dict]) -> None:
        """Write segments to cache with TTL."""
        expires_at = datetime.now(tz=timezone.utc) + timedelta(days=PREDICTION_TTL_DAYS)

        if self._db is None:
            self._predictions[video_id] = {"segments": segments, "expires_at": expires_at}
            return

        try:
            self._predictions.document(video_id).set({
                "segments":   segments,
                "expires_at": expires_at,
                "updated_at": datetime.now(tz=timezone.utc),
            })
            log.info("[cache] SET: %s  (%d segments)", video_id, len(segments))
        except Exception as e:
            log.warning("[cache] set_prediction error: %s", e)

    # ── Job state ─────────────────────────────────────────────────────────────

    def create_job(self, video_id: str) -> str:
        """Create a new pending job; return job_id."""
        job_id = str(uuid.uuid4())
        data   = {
            "video_id":   video_id,
            "status":     "pending",
            "created_at": datetime.now(tz=timezone.utc),
            "segments":   None,
            "error":      None,
        }

        if self._db is None:
            self._jobs[job_id] = data
        else:
            try:
                self._jobs.document(job_id).set(data)
            except Exception as e:
                log.warning("[cache] create_job error: %s", e)
                self._jobs[job_id] = data   # fall back to memory

        log.info("[job] Created %s for video %s", job_id, video_id)
        return job_id

    def complete_job(self, job_id: str, segments: list[dict]) -> None:
        """Mark job as complete with its segments."""
        data = {
            "status":       "complete",
            "segments":     segments,
            "completed_at": datetime.now(tz=timezone.utc),
        }
        self._update_job(job_id, data)
        log.info("[job] Completed %s  (%d segments)", job_id, len(segments))

    def fail_job(self, job_id: str, error: str) -> None:
        """Mark job as failed with an error message."""
        data = {
            "status":     "failed",
            "error":      error,
            "failed_at":  datetime.now(tz=timezone.utc),
        }
        self._update_job(job_id, data)
        log.warning("[job] Failed %s: %s", job_id, error)

    def get_job(self, job_id: str) -> Optional[dict]:
        """Return job state dict, or None if not found."""
        if self._db is None:
            return self._jobs.get(job_id)

        try:
            doc = self._jobs.document(job_id).get()
            return doc.to_dict() if doc.exists else None
        except Exception as e:
            log.warning("[cache] get_job error: %s", e)
            return self._jobs.get(job_id)   # fall back to memory

    def _update_job(self, job_id: str, data: dict) -> None:
        if self._db is None:
            if job_id in self._jobs:
                self._jobs[job_id].update(data)
            return
        try:
            self._jobs.document(job_id).update(data)
        except Exception as e:
            log.warning("[cache] _update_job error: %s", e)
            if job_id in self._jobs:
                self._jobs[job_id].update(data)
