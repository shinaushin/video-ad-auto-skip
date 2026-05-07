"""tune.py — Optuna hyperparameter search for the teacher model.

Runs N trials of teacher training on the Kaggle GPU.  Each trial tests a
different set of hyperparameters suggested by Optuna's TPE sampler.
MedianPruner stops unpromising trials after epoch 10 if they're tracking
below the median of completed trials.

Datasets are loaded ONCE and shared across all trials to avoid redundant I/O.
No full checkpoint is saved per trial — only the best val_f1 is returned to
Optuna so it can guide the next suggestion.

Usage (inside Kaggle kernel):
    python tune.py --config /tmp/run_config.json

Config keys (JSON):
    cache_dir       str   Path to per-video .npz cache
    output_dir      str   Where to save results (tune_results.json, best_params.json)
    n_trials        int   Number of Optuna trials (default 20)
    tune_epochs     int   Max epochs per trial (default 40 — reduced from full 200)
    tune_patience   int   Early-stopping patience per trial (default 8)
    device          str   "cuda" | "cpu" | "auto"
    seed            int   (default 42)

Search space:
    lr               log-uniform [5e-5, 2e-3]
    weight_decay     log-uniform [1e-5, 1e-2]
    dropout          uniform     [0.0, 0.25]
    lstm_hidden      categorical [96, 128, 192, 256]
    pos_weight_mult  uniform     [0.5, 2.5]   (multiplied onto computed n_neg/n_pos)
    seq_batch_size   categorical [4, 8, 16]

Outputs (written to config["output_dir"]):
    tune_results.json      All trial results with params + val_f1
    best_params.json       Best-trial params ready to paste into phase3_teacher.json
    optuna_study.db        SQLite study (can resume if kernel is interrupted)
    run_manifest.json      run_id stamp for bridge freshness check
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError as exc:
    raise ImportError("optuna is required — run: pip install optuna>=3.0") from exc

from data_pipeline import SponsorDataset
from models import TeacherModel
from train import (
    TeacherSequenceDataset,
    collate_teacher_sequences,
    make_balanced_sampler,
    _run_teacher_epoch,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------


def objective(
    trial: optuna.Trial,
    train_ds: TeacherSequenceDataset,
    val_ds: TeacherSequenceDataset,
    n_pos: int,
    n_neg: int,
    device: str,
    tune_epochs: int,
    tune_patience: int,
) -> float:
    """Train a teacher model with trial-suggested hyperparameters; return best val_f1."""

    # ── Hyperparameter suggestions ─────────────────────────────────────────
    lr              = trial.suggest_float("lr",              5e-5, 2e-3,  log=True)
    weight_decay    = trial.suggest_float("weight_decay",    1e-5, 1e-2,  log=True)
    dropout         = trial.suggest_float("dropout",         0.0,  0.25)
    lstm_hidden     = trial.suggest_categorical("lstm_hidden", [96, 128, 192, 256])
    pos_weight_mult = trial.suggest_float("pos_weight_mult", 0.5,  2.5)
    seq_batch_size  = trial.suggest_categorical("seq_batch_size", [4, 8, 16])

    pos_weight_val = pos_weight_mult * (n_neg / max(n_pos, 1))

    log.info(
        "Trial %d | lr=%.2e wd=%.2e drop=%.2f lstm=%d pw=%.1f batch=%d",
        trial.number, lr, weight_decay, dropout, lstm_hidden, pos_weight_val, seq_batch_size,
    )

    # ── Build model with trial architecture ───────────────────────────────
    model     = TeacherModel(lstm_hidden=lstm_hidden, dropout=dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight_val], device=device)
    )

    # Rebuild loaders each trial (batch size may vary between trials).
    train_has_sponsor = [
        int(any(w["label"] for w in seq)) for seq in train_ds._sequences
    ]
    sampler = make_balanced_sampler(train_has_sponsor)
    train_loader = DataLoader(
        train_ds, batch_size=seq_batch_size, sampler=sampler,
        collate_fn=collate_teacher_sequences, num_workers=0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=seq_batch_size, shuffle=False,
        collate_fn=collate_teacher_sequences, num_workers=0,
    )

    # ── Short training loop with Optuna pruning ───────────────────────────
    best_val_f1 = 0.0
    no_improve  = 0

    for epoch in range(1, tune_epochs + 1):
        t0 = time.time()
        _run_teacher_epoch(
            model, train_loader, optimizer, criterion, device, train=True
        )
        val_loss, val_m = _run_teacher_epoch(
            model, val_loader, None, criterion, device, train=False
        )
        scheduler.step(val_loss)

        val_f1 = val_m["f1"]
        elapsed = time.time() - t0
        log.info(
            "  Trial %3d | Epoch %2d/%d | val_f1=%.3f p=%.3f r=%.3f p(+)=%.3f  %.1fs",
            trial.number, epoch, tune_epochs,
            val_f1, val_m["precision"], val_m["recall"], val_m["mean_prob_pos"], elapsed,
        )

        # Report to Optuna; allows MedianPruner to cut unpromising trials.
        trial.report(val_f1, epoch)
        if trial.should_prune():
            log.info("  Trial %d pruned at epoch %d.", trial.number, epoch)
            raise optuna.exceptions.TrialPruned()

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            no_improve  = 0
        else:
            no_improve += 1
            if no_improve >= tune_patience:
                log.info(
                    "  Trial %d early-stopped at epoch %d (patience %d).",
                    trial.number, epoch, tune_patience,
                )
                break

    log.info("  Trial %d finished | best_val_f1=%.4f", trial.number, best_val_f1)
    return best_val_f1


# ---------------------------------------------------------------------------
# Study runner
# ---------------------------------------------------------------------------


def run_tune(cfg: dict, device: str) -> Path:
    """Run the Optuna study and save results to output_dir."""
    cache_dir   = Path(cfg["cache_dir"])
    output_dir  = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    n_trials      = int(cfg.get("n_trials",      20))
    tune_epochs   = int(cfg.get("tune_epochs",   40))
    tune_patience = int(cfg.get("tune_patience",  8))
    seed          = int(cfg.get("seed",           42))

    # ── Load datasets once (shared across all trials) ─────────────────────
    train_ids, val_ids, _ = SponsorDataset.train_val_test_split(cache_dir)
    log.info(
        "Dataset split: %d train  %d val  (shared across %d trials)",
        len(train_ids), len(val_ids), n_trials,
    )

    train_ds = TeacherSequenceDataset(SponsorDataset(cache_dir, train_ids))
    val_ds   = TeacherSequenceDataset(SponsorDataset(cache_dir, val_ids))

    n_pos = sum(w["label"] for seq in train_ds._sequences for w in seq)
    n_neg = sum(1 - w["label"] for seq in train_ds._sequences for w in seq)
    base_pw = n_neg / max(n_pos, 1)
    log.info(
        "Class counts — pos: %d  neg: %d  base_pos_weight=%.1f",
        n_pos, n_neg, base_pw,
    )

    # ── Create Optuna study ───────────────────────────────────────────────
    # MedianPruner: don't prune before epoch 10 and require at least 3 completed
    # trials before any pruning decisions (no baseline to compare against).
    pruner  = optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=10)
    sampler = optuna.samplers.TPESampler(seed=seed)
    db_path = output_dir / "optuna_study.db"
    study   = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        study_name="teacher-hparam-search",
        storage=f"sqlite:///{db_path}",
        load_if_exists=True,   # resume if kernel is interrupted and restarted
    )

    log.info(
        "Starting Optuna study: %d trials | %d epochs/trial | patience=%d",
        n_trials, tune_epochs, tune_patience,
    )

    study.optimize(
        lambda trial: objective(
            trial, train_ds, val_ds, n_pos, n_neg, device, tune_epochs, tune_patience,
        ),
        n_trials=n_trials,
        catch=(RuntimeError, ValueError),  # log exceptions, don't abort the study
    )

    # ── Summarise ─────────────────────────────────────────────────────────
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned    = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    failed    = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]

    log.info(
        "Study complete: %d completed  %d pruned  %d failed",
        len(completed), len(pruned), len(failed),
    )

    if not completed:
        log.error("No trials completed — check the logs above for errors.")
        # Write an empty results file so the bridge doesn't crash on download.
        (output_dir / "tune_results.json").write_text("[]")
        (output_dir / "best_params.json").write_text("{}")
        return output_dir / "best_params.json"

    best = study.best_trial
    log.info("Best trial #%d: val_f1=%.4f  params=%s", best.number, best.value, best.params)

    # ── Save all trial results ─────────────────────────────────────────────
    all_results = []
    for t in sorted(study.trials, key=lambda t: t.number):
        all_results.append({
            "trial":  t.number,
            "state":  t.state.name,
            "val_f1": t.value,
            "params": t.params,
        })
    results_path = output_dir / "tune_results.json"
    results_path.write_text(json.dumps(all_results, indent=2))
    log.info("All trial results → %s", results_path)

    # ── Save best params as a patch for phase3_teacher.json ───────────────
    best_params = {
        "lr":               best.params.get("lr"),
        "weight_decay":     best.params.get("weight_decay"),
        "dropout":          best.params.get("dropout"),
        "lstm_hidden":      best.params.get("lstm_hidden"),
        "pos_weight_mult":  best.params.get("pos_weight_mult"),
        "seq_batch_size":   best.params.get("seq_batch_size"),
        "_best_val_f1":     best.value,
        "_best_trial":      best.number,
        "_note": (
            "Copy these values into phase3_teacher.json for the final full training run. "
            "'pos_weight_mult' multiplies the auto-computed n_neg/n_pos ratio; "
            "set cfg['pos_weight'] = pos_weight_mult * (n_neg/n_pos) in train.py "
            "or add a 'pos_weight_mult' key to the config."
        ),
    }
    best_path = output_dir / "best_params.json"
    best_path.write_text(json.dumps(best_params, indent=2))
    log.info("Best params → %s", best_path)

    # Print a human-readable summary to stdout for the Kaggle logs.
    print("\n" + "=" * 60)
    print("OPTUNA RESULTS SUMMARY")
    print("=" * 60)
    print(f"  Trials completed : {len(completed)}")
    print(f"  Trials pruned    : {len(pruned)}")
    print(f"  Best val_f1      : {best.value:.4f}  (trial #{best.number})")
    print(f"  Best params:")
    for k, v in best.params.items():
        print(f"    {k:20s} = {v}")
    print("=" * 60 + "\n")

    return best_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    p = argparse.ArgumentParser(description="Optuna hyperparameter search for teacher model.")
    p.add_argument("--config", required=True, type=Path, help="Phase config JSON path.")
    args = p.parse_args()

    cfg = json.loads(args.config.read_text())

    device_pref = cfg.get("device", "auto")
    device = "cuda" if (device_pref == "auto" and torch.cuda.is_available()) else (
        "cpu" if device_pref == "auto" else device_pref
    )
    log.info("Phase: tune  Device: %s", device)

    seed = int(cfg.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)

    run_tune(cfg, device)


if __name__ == "__main__":
    main()
