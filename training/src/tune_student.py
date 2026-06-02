"""tune_student.py — Optuna hyperparameter search for the student distillation.

Mirrors the logging style of tune.py:
  - Per-epoch metrics logged at INFO level (visible in Vertex AI logs)
  - TensorBoard writer per trial (one sub-run per trial, comparable side-by-side)
  - GCS upload of TB logs after every epoch (viewable in real time)
  - add_hparams summary written at end of each trial

Datasets and teacher logits are loaded ONCE and shared across all trials.

Usage (Vertex AI / local):
    python tune_student.py \
        --teacher-ckpt /tmp/teacher_best.pt \
        --cache-dir    /tmp/embeddings_cache \
        --output-dir   /tmp/outputs/student_tune \
        --n-trials     20 \
        --tune-epochs  30

Outputs (written to --output-dir):
    student_tune_results.json   All trial results with params + val_f1
    student_best_params.json    Best-trial params ready to paste into distill.yaml
    student_study.db            SQLite study (resumable)
    tb/trial_NNN/               TensorBoard logs per trial
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    from torch.utils.tensorboard import SummaryWriter
    _HAS_TB = True
except ImportError:
    _HAS_TB = False

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError as exc:
    raise ImportError("optuna is required — run: pip install optuna>=3.0") from exc

try:
    from gcp_utils import download_from_gcs, upload_to_gcs
    _HAS_GCP_UTILS = True
except ImportError:
    _HAS_GCP_UTILS = False

from data_pipeline import SponsorDataset
from models import build_student, load_teacher
from train import (
    FocalLoss,
    StudentWindowDataset,
    make_balanced_sampler,
    kd_loss,
    _eval_metrics,
    collect_teacher_logits,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Single trial
# ---------------------------------------------------------------------------

def run_trial(
    trial:         optuna.Trial,
    train_ds:      StudentWindowDataset,
    val_ds:        StudentWindowDataset,
    device:        str,
    tune_epochs:   int,
    tune_patience: int,
    tb_dir:        Optional[str] = None,
    gcs_output:    Optional[str] = None,
) -> float:
    """Train the student for one Optuna trial; return best val_f1."""

    # ── Suggest hyperparameters ───────────────────────────────────────────
    lr              = trial.suggest_float("lr",              5e-5, 5e-4, log=True)
    kd_temperature  = trial.suggest_float("kd_temperature",  1.0,  3.5)
    kd_alpha        = trial.suggest_float("kd_alpha",        0.6,  1.0)
    focal_gamma     = trial.suggest_float("focal_gamma",     0.0,  2.0)
    focal_alpha_val = trial.suggest_float("focal_alpha",     0.25, 0.75)
    batch_size      = trial.suggest_categorical("batch_size", [64, 128, 256])

    log.info(
        "Trial %d | lr=%.2e  T=%.2f  alpha=%.2f  "
        "focal_gamma=%.2f  focal_alpha=%.2f  batch=%d",
        trial.number, lr, kd_temperature, kd_alpha,
        focal_gamma, focal_alpha_val, batch_size,
    )

    # ── DataLoaders ───────────────────────────────────────────────────────
    train_labels = [int(d[2].item()) for d in train_ds]
    sampler      = make_balanced_sampler(train_labels)
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,   num_workers=0)

    # ── Model ─────────────────────────────────────────────────────────────
    model     = build_student(device=device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=3, factor=0.5,
    )
    focal = FocalLoss(alpha=focal_alpha_val, gamma=focal_gamma)

    # ── TensorBoard writer (one sub-run per trial) ────────────────────────
    writer = None
    if _HAS_TB and tb_dir:
        writer = SummaryWriter(log_dir=f"{tb_dir}/trial_{trial.number:03d}")

    best_val_f1 = -1.0
    no_improve  = 0

    for epoch in range(1, tune_epochs + 1):
        t0 = time.time()

        # ── Train ─────────────────────────────────────────────────────────
        model.train()
        total_loss   = 0.0
        n_train      = 0
        train_probs:  list[float] = []
        train_labels_ep: list[int] = []

        for keyword_vec, mfcc, hard_label, teacher_logit, context_input, position_input, vote_weight in train_loader:
            keyword_vec    = keyword_vec.to(device)
            mfcc           = mfcc.to(device)
            hard_label     = hard_label.to(device)
            teacher_logit  = teacher_logit.to(device)
            context_input  = context_input.to(device)
            position_input = position_input.to(device)
            vote_weight    = vote_weight.to(device)

            student_logit = model(keyword_vec, mfcc, context_input, position_input).squeeze(-1)
            loss = kd_loss(
                student_logit, teacher_logit, hard_label,
                temperature    = kd_temperature,
                alpha          = kd_alpha,
                focal          = focal if kd_alpha < 1.0 else None,
                sample_weights = vote_weight,
            )
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item() * len(hard_label)
            n_train    += len(hard_label)
            train_probs.extend(torch.sigmoid(student_logit).detach().cpu().tolist())
            train_labels_ep.extend(hard_label.long().cpu().tolist())

        train_loss = total_loss / max(n_train, 1)
        train_f1   = _eval_metrics([int(p > 0.5) for p in train_probs], train_labels_ep)["f1"]

        # ── Val ───────────────────────────────────────────────────────────
        model.eval()
        val_probs:  list[float] = []
        val_labels: list[int]   = []

        with torch.no_grad():
            for keyword_vec, mfcc, hard_label, _, context_input, position_input, _ in val_loader:
                keyword_vec    = keyword_vec.to(device)
                mfcc           = mfcc.to(device)
                context_input  = context_input.to(device)
                position_input = position_input.to(device)
                student_logit  = model(keyword_vec, mfcc, context_input, position_input).squeeze(-1)
                val_probs.extend(torch.sigmoid(student_logit).cpu().tolist())
                val_labels.extend(hard_label.long().tolist())

        val_preds = [int(p > 0.5) for p in val_probs]
        val_m     = _eval_metrics(val_preds, val_labels)
        val_f1    = val_m["f1"]

        scheduler.step(val_f1)
        current_lr = optimizer.param_groups[0]["lr"]
        elapsed    = time.time() - t0

        # ── Per-epoch log (mirrors tune.py format) ────────────────────────
        log.info(
            "  Trial %3d | Epoch %2d/%d | "
            "train_loss=%.4f train_f1=%.3f | "
            "val_f1=%.3f p=%.3f r=%.3f | "
            "lr=%.2e  %.1fs",
            trial.number, epoch, tune_epochs,
            train_loss, train_f1,
            val_f1, val_m["precision"], val_m["recall"],
            current_lr, elapsed,
        )

        # ── TensorBoard ───────────────────────────────────────────────────
        if writer is not None:
            writer.add_scalar("train/loss",    train_loss,        epoch)
            writer.add_scalar("train/f1",      train_f1,          epoch)
            writer.add_scalar("val/f1",        val_f1,            epoch)
            writer.add_scalar("val/precision", val_m["precision"], epoch)
            writer.add_scalar("val/recall",    val_m["recall"],    epoch)
            writer.add_scalar("lr",            current_lr,        epoch)
            writer.flush()
            if gcs_output and _HAS_GCP_UTILS:
                upload_to_gcs(tb_dir, f"{gcs_output}/tb")

        # ── Optuna pruning ────────────────────────────────────────────────
        trial.report(val_f1, epoch)
        if trial.should_prune():
            log.info("  Trial %d pruned at epoch %d.", trial.number, epoch)
            if writer is not None:
                writer.close()
            raise optuna.exceptions.TrialPruned()

        # ── Early stopping ────────────────────────────────────────────────
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            no_improve  = 0
        else:
            no_improve += 1
            if no_improve >= tune_patience:
                log.info(
                    "  Trial %d early-stopped at epoch %d (patience=%d).",
                    trial.number, epoch, tune_patience,
                )
                break

    log.info("  Trial %d finished | best_val_f1=%.4f", trial.number, best_val_f1)

    # ── HParam summary (shows up in TensorBoard HParams tab) ─────────────
    if writer is not None:
        writer.add_hparams(
            {
                "lr":             lr,
                "kd_temperature": kd_temperature,
                "kd_alpha":       kd_alpha,
                "focal_gamma":    focal_gamma,
                "focal_alpha":    focal_alpha_val,
                "batch_size":     float(batch_size),
            },
            {"hparam/best_val_f1": best_val_f1},
        )
        writer.close()

    return best_val_f1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    p = argparse.ArgumentParser(description="Optuna hyperparameter sweep for student distillation.")
    p.add_argument("--teacher-ckpt", required=True, type=str)
    p.add_argument("--cache-dir",    required=True, type=Path)
    p.add_argument("--output-dir",   required=True, type=Path)
    p.add_argument("--gcs-input",    type=str,  default=None)
    p.add_argument("--gcs-output",   type=str,  default=None)
    p.add_argument("--n-trials",     type=int,  default=20)
    p.add_argument("--tune-epochs",  type=int,  default=30)
    p.add_argument("--tune-patience",type=int,  default=8)
    p.add_argument("--device",       type=str,  default="auto")
    p.add_argument("--seed",         type=int,  default=42)
    p.add_argument("--min-votes",    type=int,  default=0)
    args = p.parse_args()

    # ── GCS: download cache ───────────────────────────────────────────────
    if args.gcs_input and _HAS_GCP_UTILS:
        log.info("Downloading embeddings: %s → %s", args.gcs_input, args.cache_dir)
        download_from_gcs(args.gcs_input, str(args.cache_dir))

    # ── Setup ─────────────────────────────────────────────────────────────
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tb_dir = str(args.output_dir / "tb")

    device = ("cuda" if torch.cuda.is_available() else "cpu") \
        if args.device == "auto" else args.device
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    log.info(
        "Device: %s  Trials: %d  Epochs/trial: %d  TensorBoard: %s",
        device, args.n_trials, args.tune_epochs, tb_dir,
    )

    # ── Load data once ────────────────────────────────────────────────────
    train_ids, val_ids, _ = SponsorDataset.train_val_test_split(args.cache_dir)
    log.info(
        "Dataset split: %d train / %d val videos (shared across all trials)",
        len(train_ids), len(val_ids),
    )

    # ── Collect teacher logits once ───────────────────────────────────────
    log.info("Loading teacher: %s", args.teacher_ckpt)
    teacher = load_teacher(args.teacher_ckpt, device=device)
    teacher_logits: dict = {}
    for split_ids in (train_ids, val_ids):
        split_ds = SponsorDataset(args.cache_dir, split_ids, require_audio=False,
                                  min_votes=args.min_votes)
        teacher_logits.update(collect_teacher_logits(teacher, split_ds, device))
    del teacher
    log.info("Collected %d teacher logits.", len(teacher_logits))

    # ── Build datasets once ───────────────────────────────────────────────
    train_ds = StudentWindowDataset(
        SponsorDataset(args.cache_dir, train_ids, require_audio=False, min_votes=args.min_votes),
        teacher_logits,
    )
    val_ds = StudentWindowDataset(
        SponsorDataset(args.cache_dir, val_ids, require_audio=False, min_votes=args.min_votes),
        teacher_logits,
    )
    log.info("Train: %d windows  Val: %d windows", len(train_ds), len(val_ds))

    # ── Optuna study ──────────────────────────────────────────────────────
    study_path = args.output_dir / "student_study.db"
    study = optuna.create_study(
        study_name     = "student_distill",
        direction      = "maximize",
        sampler        = optuna.samplers.TPESampler(seed=args.seed),
        pruner         = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5),
        storage        = f"sqlite:///{study_path}",
        load_if_exists = True,
    )

    def objective(trial: optuna.Trial) -> float:
        return run_trial(
            trial, train_ds, val_ds, device,
            tune_epochs   = args.tune_epochs,
            tune_patience = args.tune_patience,
            tb_dir        = tb_dir,
            gcs_output    = args.gcs_output,
        )

    log.info("Starting Optuna sweep (%d trials) …", args.n_trials)
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=False)

    # ── Save results ──────────────────────────────────────────────────────
    results = [
        {
            "trial":   t.number,
            "val_f1":  t.value,
            "params":  t.params,
            "state":   str(t.state),
        }
        for t in study.trials
    ]
    (args.output_dir / "student_tune_results.json").write_text(json.dumps(results, indent=2))

    best = study.best_trial
    best_params = {**best.params, "best_val_f1": best.value}
    (args.output_dir / "student_best_params.json").write_text(json.dumps(best_params, indent=2))

    log.info("=" * 60)
    log.info("Best trial: %d  val_f1=%.4f", best.number, best.value)
    for k, v in best.params.items():
        log.info("  %-20s %s", k, v)
    log.info("=" * 60)

    if args.gcs_output and _HAS_GCP_UTILS:
        upload_to_gcs(str(args.output_dir), args.gcs_output)
        log.info("Results uploaded to %s", args.gcs_output)


if __name__ == "__main__":
    main()
