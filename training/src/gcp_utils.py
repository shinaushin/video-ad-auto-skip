"""gcp_utils.py — Optional GCP integration helpers.

Provides:
    setup_cloud_logging()   Attach a Cloud Logging handler to the Python root logger.
    MetricsReporter         Emit per-epoch training metrics to Cloud Monitoring.
    download_from_gcs()     rsync a GCS prefix to a local directory.
    upload_to_gcs()         rsync a local directory to a GCS prefix.
    load_config()           Load a JSON config from a local path or gs:// URI.

Every public function degrades gracefully when:
    - Running locally (outside GCP / no metadata server reachable)
    - GCP client libraries not installed
    - gcloud CLI not available
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _gcp_project() -> str:
    """Return the active GCP project ID, or empty string if undetermined."""
    # 1. Explicit env var (set in Vertex AI jobs via --env-vars)
    proj = os.environ.get("GOOGLE_CLOUD_PROJECT", "")
    if proj:
        return proj
    # 2. Metadata server (running on GCP VM / Vertex AI)
    try:
        req = urllib.request.Request(
            "http://metadata.google.internal/computeMetadata/v1/project/project-id",
            headers={"Metadata-Flavor": "Google"},
        )
        proj = urllib.request.urlopen(req, timeout=2).read().decode().strip()
        return proj
    except Exception:
        return ""


def _is_gcs_uri(path: str) -> bool:
    return str(path).startswith("gs://")


# ---------------------------------------------------------------------------
# Cloud Logging
# ---------------------------------------------------------------------------

def setup_cloud_logging(job_name: str = "yt-sponsor-training") -> bool:
    """Attach a Cloud Logging handler to the Python root logger.

    Safe to call outside GCP — returns False and logs a debug message instead
    of raising.

    Args:
        job_name: Log name shown in Cloud Logging (e.g. "yt-sponsor-tune").

    Returns:
        True if the handler was successfully attached.
    """
    try:
        import google.cloud.logging as cloud_logging  # type: ignore
        client = cloud_logging.Client()
        handler = cloud_logging.handlers.CloudLoggingHandler(client, name=job_name)
        handler.setLevel(logging.DEBUG)
        root = logging.getLogger()
        root.addHandler(handler)
        log.info("Cloud Logging handler attached  log_name=%s", job_name)
        return True
    except ImportError:
        log.debug("google-cloud-logging not installed — stdout only.")
        return False
    except Exception as exc:
        log.warning("Could not attach Cloud Logging handler: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Cloud Monitoring — custom metrics
# ---------------------------------------------------------------------------

_METRIC_PREFIX = "custom.googleapis.com/yt_sponsor"

# Metrics emitted per epoch:
#   yt_sponsor/train_loss, yt_sponsor/val_loss,
#   yt_sponsor/val_f1, yt_sponsor/val_precision, yt_sponsor/val_recall,
#   yt_sponsor/lr
#
# Labels on every series:
#   phase   — "teacher" | "distill" | "tune"
#   job     — display name of the Vertex AI job (or "local")
# Additional label for tune phase:
#   trial   — Optuna trial number (as string)


class MetricsReporter:
    """Emit per-epoch training metrics to Cloud Monitoring.

    Falls back to a no-op when the library is unavailable or the project
    cannot be determined.  Training is never interrupted by a metrics failure.
    """

    def __init__(self, phase: str, job_name: str = "local") -> None:
        self.phase = phase
        self.job_name = job_name
        self._client = None
        self._project_name = ""

        project = _gcp_project()
        if not project:
            log.debug("MetricsReporter: no project ID — custom metrics disabled.")
            return

        try:
            from google.cloud import monitoring_v3  # type: ignore
            self._client = monitoring_v3.MetricServiceClient()
            self._project_name = f"projects/{project}"
            log.info(
                "Cloud Monitoring enabled  project=%s  phase=%s  job=%s",
                project, phase, job_name,
            )
        except ImportError:
            log.debug("google-cloud-monitoring not installed — metrics disabled.")
        except Exception as exc:
            log.warning("MetricsReporter init failed: %s", exc)

    # ── Internal write helper ──────────────────────────────────────────────

    def _write_batch(self, name_value_pairs: list[tuple[str, float]], labels: dict) -> None:
        """Write a batch of GAUGE data points in a single API call."""
        if not self._client:
            return
        try:
            from google.cloud import monitoring_v3  # type: ignore

            now = time.time()
            seconds = int(now)
            nanos = int((now - seconds) * 1e9)
            interval = monitoring_v3.TimeInterval(
                end_time={"seconds": seconds, "nanos": nanos}
            )

            series_list = []
            for metric_name, value in name_value_pairs:
                point = monitoring_v3.Point(
                    interval=interval,
                    value=monitoring_v3.TypedValue(double_value=float(value)),
                )
                series = monitoring_v3.TimeSeries(
                    metric=monitoring_v3.Metric(
                        type=f"{_METRIC_PREFIX}/{metric_name}",
                        labels={k: str(v) for k, v in labels.items()},
                    ),
                    resource=monitoring_v3.MonitoredResource(type="global"),
                    points=[point],
                )
                series_list.append(series)

            self._client.create_time_series(
                name=self._project_name, time_series=series_list
            )
        except Exception as exc:
            # Never interrupt training for a metrics failure.
            log.debug("MetricsReporter._write_batch failed: %s", exc)

    # ── Public API ─────────────────────────────────────────────────────────

    def emit_epoch(
        self,
        epoch: int,
        *,
        train_loss: float = 0.0,
        val_loss: float = 0.0,
        val_f1: float = 0.0,
        val_precision: float = 0.0,
        val_recall: float = 0.0,
        lr: float = 0.0,
        trial: int = -1,
    ) -> None:
        """Emit all per-epoch metrics as a batched write.

        Args:
            epoch:         Current epoch number (1-indexed).
            train_loss:    Training loss for this epoch.
            val_loss:      Validation loss for this epoch.
            val_f1:        Validation F1 score.
            val_precision: Validation precision.
            val_recall:    Validation recall.
            lr:            Current learning rate.
            trial:         Optuna trial number (-1 = not a tune run).
        """
        labels: dict = {"phase": self.phase, "job": self.job_name}
        if trial >= 0:
            labels["trial"] = str(trial)

        pairs = [
            ("train_loss",    train_loss),
            ("val_loss",      val_loss),
            ("val_f1",        val_f1),
            ("val_precision", val_precision),
            ("val_recall",    val_recall),
            ("lr",            lr),
        ]
        self._write_batch(pairs, labels)
        log.debug(
            "Metrics emitted  epoch=%d  train_loss=%.4f  val_f1=%.4f",
            epoch, train_loss, val_f1,
        )


# ---------------------------------------------------------------------------
# GCS I/O via gcloud CLI
# ---------------------------------------------------------------------------

def _run_gsutil(args: list[str], desc: str) -> bool:
    """Run a gcloud storage command; return True on success."""
    cmd = ["gcloud", "storage"] + args
    log.info("%s: %s", desc, " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        log.error("%s failed:\n%s", desc, result.stderr)
        return False
    return True


def download_from_gcs(gcs_uri: str, local_dir: str | Path, delete_extra: bool = False) -> bool:
    """rsync a GCS prefix to a local directory.

    Args:
        gcs_uri:      Source GCS URI, e.g. ``gs://bucket/embeddings``.
        local_dir:    Destination local directory (created if absent).
        delete_extra: If True, delete local files not in GCS (rsync -d).

    Returns:
        True on success.
    """
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    args = ["rsync", "-r"]
    if delete_extra:
        args.append("-d")
    args += [gcs_uri, str(local_dir)]
    return _run_gsutil(args, f"download {gcs_uri} → {local_dir}")


def upload_to_gcs(local_dir: str | Path, gcs_uri: str, delete_extra: bool = False) -> bool:
    """rsync a local directory to a GCS prefix.

    Args:
        local_dir:    Source local directory.
        gcs_uri:      Destination GCS URI, e.g. ``gs://bucket/outputs/run1``.
        delete_extra: If True, delete GCS objects not in local dir (rsync -d).

    Returns:
        True on success.
    """
    args = ["rsync", "-r"]
    if delete_extra:
        args.append("-d")
    args += [str(local_dir), gcs_uri]
    return _run_gsutil(args, f"upload {local_dir} → {gcs_uri}")


def copy_from_gcs(gcs_uri: str, local_path: str | Path) -> bool:
    """Copy a single file from GCS.

    Args:
        gcs_uri:    Source GCS URI.
        local_path: Destination local file path.

    Returns:
        True on success.
    """
    Path(local_path).parent.mkdir(parents=True, exist_ok=True)
    return _run_gsutil(["cp", gcs_uri, str(local_path)], f"cp {gcs_uri} → {local_path}")


def copy_to_gcs(local_path: str | Path, gcs_uri: str) -> bool:
    """Copy a single file to GCS."""
    return _run_gsutil(["cp", str(local_path), gcs_uri], f"cp {local_path} → {gcs_uri}")


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(path_or_uri: str | Path) -> dict:
    """Load a JSON config from a local path or a gs:// URI.

    If a GCS URI is provided, the file is downloaded to /tmp and then read.

    Args:
        path_or_uri: Local file path or ``gs://bucket/path/config.json``.

    Returns:
        Parsed config dict.

    Raises:
        FileNotFoundError: If the local file does not exist.
        RuntimeError:      If a GCS download fails.
    """
    path_or_uri = str(path_or_uri)
    if _is_gcs_uri(path_or_uri):
        local_tmp = Path("/tmp") / "vertex_config.json"
        ok = copy_from_gcs(path_or_uri, local_tmp)
        if not ok:
            raise RuntimeError(f"Failed to download config from {path_or_uri}")
        return json.loads(local_tmp.read_text())
    return json.loads(Path(path_or_uri).read_text())
