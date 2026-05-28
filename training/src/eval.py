#!/usr/bin/env python3
"""Standalone evaluation script for teacher and student models.

Runs inference on the held-out test split and outputs an eval_report.json with
precision / recall / F1 at the default threshold (0.5) plus a full threshold
scan so you can pick the best operating point before deploying.

Usage — teacher:
    python3 training/src/eval.py \
        --phase teacher \
        --checkpoint /tmp/teacher_best.pt \
        --config training/configs/phase3_teacher.json \
        --gcs-input gs://yt-sponsor-cache/embeddings \
        --gcs-output gs://yt-sponsor-cache/outputs/eval/teacher/RUN_ID \
        --cloud-logging --job-name yt-sponsor-eval-teacher-...

Usage — student:
    python3 training/src/eval.py \
        --phase student \
        --checkpoint /tmp/student_best.pt \
        --config training/configs/phase4_distill.json \
        --gcs-input gs://yt-sponsor-cache/embeddings \
        --gcs-output gs://yt-sponsor-cache/outputs/eval/student/RUN_ID \
        --cloud-logging --job-name yt-sponsor-eval-student-...
"""

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# Local imports (PYTHONPATH=/app/training/src inside the container)
# ---------------------------------------------------------------------------
from data_pipeline import SponsorDataset
from models import build_student, load_teacher
from train import (
    FocalLoss,
    TeacherSequenceDataset,
    StudentWindowDataset,
    collate_teacher_sequences,
    _eval_metrics,
    _eval_teacher_with_thresholds,
)

try:
    from gcp_utils import (
        setup_cloud_logging,
        MetricsReporter,
        download_from_gcs,
        upload_to_gcs,
        load_config,
    )
    _HAS_GCP_UTILS = True
except ImportError:
    _HAS_GCP_UTILS = False

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Student threshold-scan evaluation
# ---------------------------------------------------------------------------

def _eval_student_with_thresholds(
    model: nn.Module,
    loader: DataLoader,
    device: str,
) -> tuple[dict, list[dict]]:
    """Evaluate the student model and scan decision thresholds.

    Returns:
        metrics          dict  (precision / recall / f1 / accuracy + tp/fp/fn/tn at 0.5)
        threshold_curve  list of {"threshold": t, "f1": f, "precision": p, "recall": r}
                         for t in 0.05 … 0.90 (step 0.05)
    """
    model.eval()
    all_probs:  list[float] = []
    all_labels: list[int]   = []

    with torch.no_grad():
        for keyword_vec, mfcc, hard_label, _teacher_logit, _context, _position in loader:
            keyword_vec = keyword_vec.to(device)
            mfcc        = mfcc.to(device)
            hard_label  = hard_label.to(device)

            student_logit = model(keyword_vec, mfcc).squeeze(-1)  # [B]
            probs  = torch.sigmoid(student_logit).cpu().tolist()
            labels = hard_label.long().cpu().tolist()
            all_probs.extend(probs)
            all_labels.extend(labels)

    # Metrics at default 0.5 threshold.
    preds_50 = [int(p > 0.5) for p in all_probs]
    metrics  = _eval_metrics(preds_50, all_labels)

    pos_probs = [p for p, l in zip(all_probs, all_labels) if l == 1]
    neg_probs = [p for p, l in zip(all_probs, all_labels) if l == 0]
    metrics["mean_prob_pos"] = float(sum(pos_probs) / max(len(pos_probs), 1))
    metrics["mean_prob_neg"] = float(sum(neg_probs) / max(len(neg_probs), 1))
    metrics["max_prob_pos"]  = float(max(pos_probs)) if pos_probs else 0.0

    # Threshold scan: 0.05 … 0.90 in steps of 0.05.
    curve: list[dict] = []
    for t_int in range(1, 19):
        t     = round(t_int / 20, 2)
        preds = [int(p > t) for p in all_probs]
        m     = _eval_metrics(preds, all_labels)
        curve.append({
            "threshold": t,
            "f1":        round(m["f1"],        4),
            "precision": round(m["precision"], 4),
            "recall":    round(m["recall"],    4),
        })

    return metrics, curve


# ---------------------------------------------------------------------------
# Phase-specific eval entry points
# ---------------------------------------------------------------------------

def eval_teacher(
    cfg: dict,
    checkpoint: str,
    device: str,
    output_dir: Path,
    reporter=None,
) -> dict:
    """Evaluate the teacher model on the held-out test split."""
    cache_dir = Path(cfg["cache_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    _, _, test_ids = SponsorDataset.train_val_test_split(cache_dir)
    log.info("Teacher eval — test split: %d videos", len(test_ids))

    seq_batch = int(cfg.get("seq_batch_size", 8))
    test_ds     = TeacherSequenceDataset(SponsorDataset(cache_dir, test_ids))
    test_loader = DataLoader(
        test_ds,
        batch_size=seq_batch,
        shuffle=False,
        collate_fn=collate_teacher_sequences,
        num_workers=0,
    )

    model     = load_teacher(checkpoint, device=device)
    criterion = nn.BCEWithLogitsLoss()

    t0 = time.time()
    test_loss, test_m, threshold_curve = _eval_teacher_with_thresholds(
        model, test_loader, criterion, device
    )
    elapsed = time.time() - t0

    best = max(threshold_curve, key=lambda x: x["f1"])
    log.info(
        "Teacher  thresh=0.50  loss=%.4f  p=%.3f  r=%.3f  f1=%.3f  (%.1fs)",
        test_loss, test_m["precision"], test_m["recall"], test_m["f1"], elapsed,
    )
    log.info(
        "Teacher  best thresh=%.2f  p=%.3f  r=%.3f  f1=%.3f  ← use at inference",
        best["threshold"], best["precision"], best["recall"], best["f1"],
    )

    report = {
        "phase":               "teacher",
        "checkpoint":          str(checkpoint),
        "n_test_videos":       len(test_ids),
        "test_loss":           round(test_loss, 6),
        "metrics_at_0.50":     test_m,
        "best_threshold":      best["threshold"],
        "best_threshold_f1":   best["f1"],
        "best_threshold_precision": best["precision"],
        "best_threshold_recall":    best["recall"],
        "threshold_curve":     threshold_curve,
    }

    if reporter is not None:
        reporter.emit_epoch(
            0,
            train_loss=test_loss,
            val_loss=test_loss,
            val_f1=test_m["f1"],
            val_precision=test_m.get("precision", 0.0),
            val_recall=test_m.get("recall", 0.0),
            lr=0.0,
        )

    report_path = output_dir / "eval_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    log.info("Teacher eval report → %s", report_path)
    return report


def eval_student(
    cfg: dict,
    checkpoint: str,
    device: str,
    output_dir: Path,
    reporter=None,
) -> dict:
    """Evaluate the student model on the held-out test split."""
    cache_dir = Path(cfg["cache_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    _, _, test_ids = SponsorDataset.train_val_test_split(cache_dir)
    log.info("Student eval — test split: %d videos", len(test_ids))

    batch_size = int(cfg.get("batch_size", 64))
    # Pass empty teacher_logits — not needed for inference.
    test_ds     = StudentWindowDataset(SponsorDataset(cache_dir, test_ids), teacher_logits={})
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    ckpt  = torch.load(checkpoint, map_location=device)
    model = build_student(device=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    t0 = time.time()
    test_m, threshold_curve = _eval_student_with_thresholds(model, test_loader, device)
    elapsed = time.time() - t0

    best = max(threshold_curve, key=lambda x: x["f1"])
    log.info(
        "Student  thresh=0.50  p=%.3f  r=%.3f  f1=%.3f  (%.1fs)",
        test_m["precision"], test_m["recall"], test_m["f1"], elapsed,
    )
    log.info(
        "Student  best thresh=%.2f  p=%.3f  r=%.3f  f1=%.3f  ← use at inference",
        best["threshold"], best["precision"], best["recall"], best["f1"],
    )

    report = {
        "phase":               "student",
        "checkpoint":          str(checkpoint),
        "n_test_videos":       len(test_ids),
        "metrics_at_0.50":     test_m,
        "best_threshold":      best["threshold"],
        "best_threshold_f1":   best["f1"],
        "best_threshold_precision": best["precision"],
        "best_threshold_recall":    best["recall"],
        "threshold_curve":     threshold_curve,
    }

    if reporter is not None:
        reporter.emit_epoch(
            0,
            train_loss=0.0,
            val_loss=0.0,
            val_f1=test_m["f1"],
            val_precision=test_m.get("precision", 0.0),
            val_recall=test_m.get("recall", 0.0),
            lr=0.0,
        )

    report_path = output_dir / "eval_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    log.info("Student eval report → %s", report_path)
    return report


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Evaluate teacher or student model on the held-out test split."
    )
    p.add_argument("--phase",       required=True, choices=["teacher", "student"])
    p.add_argument("--checkpoint",  required=True, type=str,
                   help="Local path to teacher_best.pt or student_best.pt")
    p.add_argument("--config",      required=True, type=str,
                   help="Path or gs:// URI to the phase JSON config")
    p.add_argument("--gcs-input",   type=str, default=None,
                   help="GCS URI to rsync embeddings from (gs://bucket/embeddings)")
    p.add_argument("--local-data",  type=str, default="/tmp/embeddings_cache",
                   help="Local directory to rsync embeddings into")
    p.add_argument("--gcs-output",  type=str, default=None,
                   help="GCS URI to upload eval_report.json to")
    p.add_argument("--cloud-logging", action="store_true",
                   help="Route logs to Cloud Logging")
    p.add_argument("--job-name",    type=str, default="yt-sponsor-eval",
                   help="Job name used in Cloud Logging / Monitoring labels")
    p.add_argument("--device",      type=str, default="auto")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    if args.cloud_logging and _HAS_GCP_UTILS:
        setup_cloud_logging(args.job_name)

    # Device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    log.info("Device: %s  |  phase: %s  |  checkpoint: %s", device, args.phase, args.checkpoint)

    # Config — supports gs:// URIs when gcp_utils is available.
    if _HAS_GCP_UTILS:
        cfg = load_config(args.config)
    else:
        cfg = json.loads(Path(args.config).read_text())

    # Download embeddings from GCS.
    if args.gcs_input and _HAS_GCP_UTILS:
        log.info("Syncing embeddings %s → %s", args.gcs_input, args.local_data)
        download_from_gcs(args.gcs_input, args.local_data)
        cfg["cache_dir"] = args.local_data

    gcs_output = args.gcs_output or os.environ.get("AIP_MODEL_DIR")
    output_dir = Path("/tmp/outputs/eval")

    reporter: Optional[object] = None
    if _HAS_GCP_UTILS:
        reporter = MetricsReporter(phase=f"eval_{args.phase}", job_name=args.job_name)

    if args.phase == "teacher":
        eval_teacher(cfg, args.checkpoint, device, output_dir, reporter)
    else:
        eval_student(cfg, args.checkpoint, device, output_dir, reporter)

    # Upload results.
    if gcs_output and _HAS_GCP_UTILS:
        log.info("Uploading results to %s", gcs_output)
        upload_to_gcs(str(output_dir), gcs_output)

    log.info("Evaluation complete.")


if __name__ == "__main__":
    main()
