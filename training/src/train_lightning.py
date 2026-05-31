"""train_lightning.py — Hydra + PyTorch Lightning training entry point.

Replaces manual job submission with a single config-driven command.  The old
train.py is left untouched for backward compatibility with existing Vertex
submit scripts.

Usage — local:
    # Teacher (uses conf/config.yaml defaults)
    python train_lightning.py

    # Student distillation
    python train_lightning.py phase=distill

    # Quick smoke test (1 batch train + val, no fit)
    python train_lightning.py +trainer.fast_dev_run=true

    # Hyperparameter sweep (replaces tune.py + submit_tune.sh)
    python train_lightning.py --multirun training.lr=1e-4,2e-4,5e-4

    # Override individual keys inline
    python train_lightning.py training.lr=5e-5 model.dropout=0.3

Usage — Vertex AI:
    Pass the same command as the python_module arg in the Vertex job spec.
    Set gcs.input / gcs.output in the config or as CLI overrides:
        python train_lightning.py \\
            gcs.input=gs://yt-sponsor-cache/embeddings \\
            gcs.output=gs://yt-sponsor-cache/outputs/run1

Hydra config tree:
    training/conf/
        config.yaml           ← defaults list + top-level keys
        model/
            teacher.yaml      ← lstm_hidden, lstm_layers, dropout, ...
            student.yaml      ← placeholder (student arch is fixed)
        training/
            teacher.yaml      ← lr, epochs, patience, seq_batch_size, ...
            distill.yaml      ← kd_alpha, kd_temperature, batch_size, ...
        data/
            default.yaml      ← cache_dir, output_dir
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import hydra
import lightning as L
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import TensorBoardLogger
from omegaconf import DictConfig, OmegaConf

try:
    from gcp_utils import download_from_gcs, setup_cloud_logging, upload_to_gcs
    _HAS_GCP_UTILS = True
except ImportError:
    _HAS_GCP_UTILS = False


class GCSTensorBoardSync(L.Callback):
    """Upload TensorBoard logs to GCS after every validation epoch.

    This replicates the per-epoch upload that train.py does so TensorBoard
    is viewable in real time while the job is still running on Vertex AI.
    Without this, logs only appear in GCS after the entire job completes.
    """

    def __init__(self, local_tb_dir: str, gcs_tb_dir: str) -> None:
        self.local_tb_dir = local_tb_dir
        self.gcs_tb_dir   = gcs_tb_dir

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        if _HAS_GCP_UTILS:
            upload_to_gcs(self.local_tb_dir, self.gcs_tb_dir)

from lightning_modules import (
    StudentDataModule,
    StudentLightningModule,
    TeacherDataModule,
    TeacherLightningModule,
)

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    log.info("Effective config:\n%s", OmegaConf.to_yaml(cfg))

    # ── Cloud logging (Vertex AI) ──────────────────────────────────────────
    if cfg.get("cloud_logging", False) and _HAS_GCP_UTILS:
        setup_cloud_logging(cfg.get("job_name", "yt-sponsor-train"))

    # ── GCS: download embeddings cache ────────────────────────────────────
    gcs_input  = cfg.get("gcs", {}).get("input")
    local_data = cfg.get("gcs", {}).get("local_data", "/tmp/embeddings_cache")
    if gcs_input and _HAS_GCP_UTILS:
        log.info("Downloading embeddings: %s → %s", gcs_input, local_data)
        if not download_from_gcs(gcs_input, local_data):
            raise RuntimeError(f"Failed to download embeddings from {gcs_input}")
        # Point data config at the local copy so DataModules don't need GCS awareness.
        OmegaConf.update(cfg, "data.cache_dir", local_data, merge=True)

    # ── Reproducibility ───────────────────────────────────────────────────
    L.seed_everything(int(cfg.get("seed", 42)), workers=True)

    # ── Output directory ──────────────────────────────────────────────────
    output_dir = Path(cfg.data.output_dir) / cfg.phase
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── TensorBoard logger ────────────────────────────────────────────────
    # Writes to <output_dir>/tb/<job_name>/ so multiple runs under the same
    # output_dir can be compared by pointing tensorboard at <output_dir>/tb/.
    tb_logger = TensorBoardLogger(
        save_dir = str(output_dir),
        name     = "tb",
        version  = cfg.get("job_name", "run"),
    )
    log.info("TensorBoard logs → %s/tb/%s", output_dir, cfg.get("job_name", "run"))

    # ── Callbacks ─────────────────────────────────────────────────────────
    # ModelCheckpoint saves the best model by val/f1.
    # EarlyStopping stops when val/f1 stops improving.
    # LearningRateMonitor writes lr to TensorBoard each epoch.
    # GCSTensorBoardSync uploads TB logs to GCS after each val epoch so
    # TensorBoard is viewable in real time on Vertex AI (not just after job ends).
    local_tb_dir = str(output_dir / "tb" / cfg.get("job_name", "run"))
    ckpt_filename = f"{cfg.phase}_best-{{epoch:02d}}-{{val_f1:.3f}}"
    callbacks = [
        ModelCheckpoint(
            dirpath   = str(output_dir),
            filename  = ckpt_filename,
            monitor   = "val/f1",
            mode      = "max",
            save_top_k = 1,
            verbose   = True,
        ),
        EarlyStopping(
            monitor = "val/f1",
            mode    = "max",
            patience= int(cfg.training.patience),
            verbose = True,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    gcs_output = cfg.get("gcs", {}).get("output") or os.environ.get("AIP_MODEL_DIR", "")
    if gcs_output and _HAS_GCP_UTILS:
        callbacks.append(
            GCSTensorBoardSync(
                local_tb_dir = local_tb_dir,
                gcs_tb_dir   = f"{gcs_output}/tb",
            )
        )
        log.info("GCS TensorBoard sync enabled → %s/tb", gcs_output)

    # ── Trainer ───────────────────────────────────────────────────────────
    # Extra Trainer kwargs can be injected via Hydra CLI:
    #   +trainer.fast_dev_run=true     ← 1 batch smoke test
    #   +trainer.limit_train_batches=5 ← quick local sanity run
    trainer_overrides = dict(
        OmegaConf.to_container(cfg.get("trainer", {}), resolve=True)
    )
    trainer = L.Trainer(
        max_epochs          = int(cfg.training.epochs),
        logger              = tb_logger,
        callbacks           = callbacks,
        gradient_clip_val   = 1.0,
        log_every_n_steps   = 1,
        enable_progress_bar = True,
        deterministic       = True,
        **trainer_overrides,
    )

    # ── Phase dispatch ─────────────────────────────────────────────────────
    if cfg.phase == "teacher":
        datamodule = TeacherDataModule(cfg)
        module     = TeacherLightningModule(cfg)

        trainer.fit(module, datamodule=datamodule)

        # Run threshold scan on the held-out test set using the best checkpoint.
        # The best_threshold attribute is populated by on_test_epoch_end and can
        # be read from module after this call.
        if not trainer_overrides.get("fast_dev_run", False):
            trainer.test(module, datamodule=datamodule, ckpt_path="best")
            log.info(
                "Best inference threshold: %.2f  (copy to training.json or pass at inference)",
                module.best_threshold,
            )

    elif cfg.phase == "distill":
        datamodule = StudentDataModule(cfg)
        module     = StudentLightningModule(cfg)
        trainer.fit(module, datamodule=datamodule)

    elif cfg.phase == "baseline":
        # Baseline evaluation doesn't use Lightning — delegate to legacy helper.
        from train import evaluate_baseline
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        cfg_dict.setdefault("cache_dir",  str(cfg.data.cache_dir))
        cfg_dict.setdefault("output_dir", str(output_dir))
        evaluate_baseline(cfg_dict)
        return

    else:
        raise ValueError(
            f"Unknown phase: {cfg.phase!r}. Expected: teacher | distill | baseline"
        )

    # ── GCS: upload final results ─────────────────────────────────────────
    if gcs_output and _HAS_GCP_UTILS:
        log.info("Uploading outputs: %s → %s", output_dir, gcs_output)
        upload_to_gcs(str(output_dir), gcs_output)


if __name__ == "__main__":
    main()
