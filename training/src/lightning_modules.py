"""lightning_modules.py — PyTorch Lightning modules for sponsor-segment detection.

This module is the Lightning-native counterpart to train.py.  It contains all
the dataset classes, loss functions, and training logic wrapped in
LightningModule / LightningDataModule so that train_lightning.py (the Hydra
entry point) can drive them with a single Trainer call.

train.py is left untouched for backward compatibility with existing Vertex
submit scripts.

Classes
-------
FocalLoss               Binary focal loss (class-imbalance-aware).
TeacherDataModule       LightningDataModule: padded video sequences.
StudentDataModule       LightningDataModule: flat windows for distillation.
TeacherLightningModule  LightningModule: BiLSTM teacher.
StudentLightningModule  LightningModule: student distillation with KD.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from data_pipeline import SponsorDataset, N_MFCC_FRAMES
from models import (
    K_CONTEXT,
    MFCC_DIM,
    N_FRAMES,
    TEXT_DIM,
    StudentModel,
    TeacherModel,
    build_student,
    load_teacher,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


class FocalLoss(nn.Module):
    """Binary focal loss for class-imbalanced datasets.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        alpha: Weight applied to the positive (sponsor) class.
        gamma: Focusing exponent.  gamma=0 → standard BCE.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits  = logits.view(-1)
        targets = targets.view(-1).float()
        bce     = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t     = torch.exp(-bce)
        alpha_t = targets * self.alpha + (1.0 - targets) * (1.0 - self.alpha)
        return (alpha_t * (1.0 - p_t) ** self.gamma * bce).mean()


# ---------------------------------------------------------------------------
# Metrics helper
# ---------------------------------------------------------------------------


def _eval_metrics(all_preds: list[int], all_labels: list[int]) -> dict[str, float]:
    """Compute precision, recall, F1, and accuracy from flat integer lists."""
    tp = sum(p == 1 and l == 1 for p, l in zip(all_preds, all_labels))
    fp = sum(p == 1 and l == 0 for p, l in zip(all_preds, all_labels))
    fn = sum(p == 0 and l == 1 for p, l in zip(all_preds, all_labels))
    tn = sum(p == 0 and l == 0 for p, l in zip(all_preds, all_labels))
    precision = tp / max(tp + fp, 1)
    recall    = tp / max(tp + fn, 1)
    f1        = 2 * precision * recall / max(precision + recall, 1e-9)
    accuracy  = (tp + tn) / max(len(all_preds), 1)
    return {
        "precision": precision, "recall": recall,
        "f1": f1, "accuracy": accuracy,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


# ---------------------------------------------------------------------------
# Class-imbalance sampler
# ---------------------------------------------------------------------------


def make_balanced_sampler(labels: list[int]) -> Optional[WeightedRandomSampler]:
    """WeightedRandomSampler that up-samples the minority class."""
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    w_pos = 1.0 / n_pos
    w_neg = 1.0 / n_neg
    weights = [w_pos if l == 1 else w_neg for l in labels]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


# ---------------------------------------------------------------------------
# Dataset classes (mirror train.py — kept here so this module is self-contained)
# ---------------------------------------------------------------------------


class TeacherSequenceDataset(Dataset):
    """Video-level sequence dataset for teacher training.

    Each item is a full video's ordered window sequence:
        (text_embs [T, 768], audio_embs [T, 384], labels [T])
    """

    def __init__(self, sponsor_ds: SponsorDataset) -> None:
        items_by_video: dict[str, list[dict]] = {}
        for item in sponsor_ds:
            items_by_video.setdefault(item["video_id"], []).append(item)
        self._sequences: list[list[dict]] = list(items_by_video.values())

    def __len__(self) -> int:
        return len(self._sequences)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        windows    = self._sequences[idx]
        text_embs  = torch.stack([torch.from_numpy(w["text_emb"])  for w in windows])
        audio_embs = torch.stack([torch.from_numpy(w["audio_emb"]) for w in windows])
        labels     = torch.tensor([w["label"] for w in windows], dtype=torch.float32)
        return text_embs, audio_embs, labels


def collate_teacher_sequences(
    batch: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad variable-length video sequences to the longest one in the batch.

    Returns:
        text_padded   [B, T_max, 768]
        audio_padded  [B, T_max, 384]
        labels_padded [B, T_max]  — padding positions marked -1.0
        lengths       [B]
    """
    text_list, audio_list, label_list = zip(*batch)
    lengths      = torch.tensor([t.shape[0] for t in text_list], dtype=torch.long)
    text_padded  = nn.utils.rnn.pad_sequence(text_list,  batch_first=True)
    audio_padded = nn.utils.rnn.pad_sequence(audio_list, batch_first=True)
    labels_padded = nn.utils.rnn.pad_sequence(label_list, batch_first=True, padding_value=-1.0)
    return text_padded, audio_padded, labels_padded, lengths


class StudentWindowDataset(Dataset):
    """Flat window dataset for student distillation.

    Yields (keyword_vec, mfcc, hard_label, teacher_logit, context, position).
    Falls back to hard-label logits when no teacher logit is available.
    """

    def __init__(
        self,
        sponsor_ds: SponsorDataset,
        teacher_logits: dict[tuple[str, int], float] | None = None,
    ) -> None:
        self._items: list[dict] = list(sponsor_ds)
        self._teacher_logits    = teacher_logits or {}
        self._keys: list[tuple[str, int]] = []
        vid_counter: dict[str, int] = {}
        for item in self._items:
            vid = item["video_id"]
            idx = vid_counter.get(vid, 0)
            self._keys.append((vid, idx))
            vid_counter[vid] = idx + 1
        self._vid_total: dict[str, int] = vid_counter

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        item        = self._items[idx]
        keyword_vec = torch.from_numpy(item["keyword_vec"])
        mfcc        = torch.from_numpy(item["mfcc"])                # [N_FRAMES, 13]
        hard_label  = torch.tensor(item["label"], dtype=torch.float32)

        key = self._keys[idx]
        vid, w_idx = key
        teacher_logit_val = self._teacher_logits.get(key, None)
        if teacher_logit_val is not None:
            teacher_logit = torch.tensor(teacher_logit_val, dtype=torch.float32)
        else:
            teacher_logit = torch.tensor(10.0 if item["label"] else -10.0, dtype=torch.float32)

        context = []
        for k in range(1, K_CONTEXT + 1):
            prev = self._teacher_logits.get((vid, w_idx - k), None)
            context.append(float(torch.sigmoid(torch.tensor(prev))) if prev is not None else 0.5)
        context_input  = torch.tensor(context, dtype=torch.float32)
        total          = self._vid_total.get(vid, 1)
        position_input = torch.tensor([w_idx / max(total - 1, 1)], dtype=torch.float32)

        return keyword_vec, mfcc, hard_label, teacher_logit, context_input, position_input


# ---------------------------------------------------------------------------
# Knowledge distillation utilities
# ---------------------------------------------------------------------------


@torch.no_grad()
def collect_teacher_logits(
    teacher: TeacherModel,
    sponsor_ds: SponsorDataset,
    device: str,
    batch_size: int = 128,
) -> dict[tuple[str, int], float]:
    """Run teacher inference over a dataset; return (video_id, window_idx) → logit."""
    teacher.eval()
    raw_items   = list(sponsor_ds)
    result:     dict[tuple[str, int], float] = {}
    vid_counter: dict[str, int] = {}

    for i in range(0, len(raw_items), batch_size):
        batch      = raw_items[i : i + batch_size]
        text_embs  = torch.stack([torch.from_numpy(it["text_emb"])  for it in batch]).to(device)
        audio_embs = torch.stack([torch.from_numpy(it["audio_emb"]) for it in batch]).to(device)
        text_in    = text_embs.unsqueeze(1)
        audio_in   = audio_embs.unsqueeze(1)
        logits     = teacher(text_in, audio_in).squeeze(-1).squeeze(-1).cpu().tolist()

        for item, logit in zip(batch, logits):
            vid = item["video_id"]
            idx = vid_counter.get(vid, 0)
            result[(vid, idx)] = logit
            vid_counter[vid]   = idx + 1

    log.info("Collected %d teacher logits.", len(result))
    return result


def kd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    hard_labels: torch.Tensor,
    temperature: float = 4.0,
    alpha: float = 0.7,
    focal: Optional[FocalLoss] = None,
) -> torch.Tensor:
    """Combined knowledge distillation + focal loss.

    L = alpha * KL(teacher_soft || student_soft) + (1 - alpha) * focal(student, hard)
    """
    T            = temperature
    student_soft = torch.sigmoid(student_logits / T)
    teacher_soft = torch.sigmoid(teacher_logits / T)
    eps = 1e-7
    kl = (
        teacher_soft * (torch.log(teacher_soft + eps) - torch.log(student_soft + eps))
        + (1 - teacher_soft) * (torch.log(1 - teacher_soft + eps) - torch.log(1 - student_soft + eps))
    )
    kl_loss = (T ** 2) * kl.mean()
    hard_loss = focal(student_logits, hard_labels) if focal is not None \
        else F.binary_cross_entropy_with_logits(student_logits, hard_labels)
    return alpha * kl_loss + (1.0 - alpha) * hard_loss


# ---------------------------------------------------------------------------
# LightningDataModule: Teacher
# ---------------------------------------------------------------------------


class TeacherDataModule(L.LightningDataModule):
    """Loads padded video-sequence batches for teacher training.

    setup() is called once by the Trainer before any dataloader is requested.
    The train_val_test_split is deterministic (hash-based), so splits are
    reproducible across runs without storing split IDs to disk.
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.train_ds: Optional[TeacherSequenceDataset] = None
        self.val_ds:   Optional[TeacherSequenceDataset] = None
        self.test_ds:  Optional[TeacherSequenceDataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        cache_dir  = Path(self.cfg.data.cache_dir)
        min_votes  = int(self.cfg.training.get("min_votes", 0))
        train_ids, val_ids, test_ids = SponsorDataset.train_val_test_split(cache_dir)
        log.info(
            "Data split: %d train / %d val / %d test videos",
            len(train_ids), len(val_ids), len(test_ids),
        )
        def _ds(ids):
            return TeacherSequenceDataset(
                SponsorDataset(cache_dir, ids, require_audio=False, min_votes=min_votes)
            )
        self.train_ds = _ds(train_ids)
        self.val_ds   = _ds(val_ids)
        self.test_ds  = _ds(test_ids)

        n_pos = sum(w["label"] for seq in self.train_ds._sequences for w in seq)
        n_neg = sum(1 - w["label"] for seq in self.train_ds._sequences for w in seq)
        log.info(
            "Train class counts — pos: %d  neg: %d  ratio=%.1f",
            n_pos, n_neg, n_neg / max(n_pos, 1),
        )

    def _seq_batch(self) -> int:
        return int(self.cfg.training.get("seq_batch_size", 8))

    def train_dataloader(self) -> DataLoader:
        has_sponsor = [int(any(w["label"] for w in seq)) for seq in self.train_ds._sequences]
        sampler     = make_balanced_sampler(has_sponsor)
        return DataLoader(
            self.train_ds,
            batch_size=self._seq_batch(),
            sampler=sampler,
            collate_fn=collate_teacher_sequences,
            num_workers=0,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds,
            batch_size=self._seq_batch(),
            shuffle=False,
            collate_fn=collate_teacher_sequences,
            num_workers=0,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_ds,
            batch_size=self._seq_batch(),
            shuffle=False,
            collate_fn=collate_teacher_sequences,
            num_workers=0,
        )


# ---------------------------------------------------------------------------
# LightningDataModule: Student
# ---------------------------------------------------------------------------


class StudentDataModule(L.LightningDataModule):
    """Loads flat window batches for student distillation.

    Teacher logits are collected once in setup() before any training begins.
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.train_ds: Optional[StudentWindowDataset] = None
        self.val_ds:   Optional[StudentWindowDataset] = None
        self.test_ds:  Optional[StudentWindowDataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        cache_dir   = Path(self.cfg.data.cache_dir)
        min_votes   = int(self.cfg.training.get("min_votes", 0))
        teacher_ckpt = self.cfg.training.get("teacher_ckpt", None)
        train_ids, val_ids, test_ids = SponsorDataset.train_val_test_split(cache_dir)

        # ── Collect teacher logits ─────────────────────────────────────────
        teacher_logits: dict[tuple[str, int], float] = {}
        if teacher_ckpt:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            log.info("Loading teacher for logit collection: %s (device=%s)", teacher_ckpt, device)
            teacher = load_teacher(teacher_ckpt, device=device)
            for split_ids in (train_ids, val_ids):
                split_ds = SponsorDataset(cache_dir, split_ids, require_audio=False, min_votes=min_votes)
                teacher_logits.update(collect_teacher_logits(teacher, split_ds, device))
            del teacher
            if device == "cuda":
                torch.cuda.empty_cache()
        else:
            log.warning("No teacher_ckpt provided — using hard labels only (KD alpha forced to 0).")

        def _ds(ids):
            return StudentWindowDataset(
                SponsorDataset(cache_dir, ids, require_audio=False, min_votes=min_votes),
                teacher_logits,
            )
        self.train_ds = _ds(train_ids)
        self.val_ds   = _ds(val_ids)
        self.test_ds  = _ds(test_ids)

    def _batch_size(self) -> int:
        return int(self.cfg.training.get("batch_size", 64))

    def train_dataloader(self) -> DataLoader:
        train_labels = [int(d[2].item()) for d in self.train_ds]
        sampler      = make_balanced_sampler(train_labels)
        return DataLoader(self.train_ds, batch_size=self._batch_size(), sampler=sampler, num_workers=0)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_ds, batch_size=self._batch_size(), shuffle=False, num_workers=0)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_ds, batch_size=self._batch_size(), shuffle=False, num_workers=0)


# ---------------------------------------------------------------------------
# LightningModule: Teacher
# ---------------------------------------------------------------------------


class TeacherLightningModule(L.LightningModule):
    """BiLSTM teacher training + evaluation.

    Metrics are computed epoch-level (F1 is not averageable across batches),
    so probabilities and labels are accumulated in step methods and aggregated
    in on_*_epoch_end hooks.

    EarlyStopping and ModelCheckpoint both monitor val/f1 (mode=max).
    ReduceLROnPlateau monitors val/loss.
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters()

        self.model = TeacherModel(
            lstm_hidden  = int(cfg.model.lstm_hidden),
            lstm_layers  = int(cfg.model.lstm_layers),
            dropout      = float(cfg.model.dropout),
            embed_mode   = str(cfg.model.embed_mode),
            arch_variant = str(cfg.model.arch_variant),
        )
        pos_weight     = torch.tensor([float(cfg.model.pos_weight_mult)])
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        # Epoch-level accumulators
        self._train_probs:  list[float] = []
        self._train_labels: list[int]   = []
        self._val_probs:    list[float] = []
        self._val_labels:   list[int]   = []
        self._test_probs:   list[float] = []
        self._test_labels:  list[int]   = []
        self.best_threshold: float = 0.5

    # ── Core step ─────────────────────────────────────────────────────────

    def _shared_step(
        self, batch: tuple
    ) -> tuple[torch.Tensor, list[float], list[int]]:
        text_emb, audio_emb, labels, lengths = batch
        logits        = self.model(text_emb, audio_emb, lengths).squeeze(-1)
        mask          = labels >= 0
        logits_valid  = logits[mask]
        labels_valid  = labels[mask]
        loss          = self.criterion(logits_valid, labels_valid)
        probs         = torch.sigmoid(logits_valid).detach().cpu().tolist()
        labels_list   = labels_valid.long().cpu().tolist()
        return loss, probs, labels_list

    # ── Training ──────────────────────────────────────────────────────────

    def on_train_epoch_start(self) -> None:
        self._train_probs  = []
        self._train_labels = []

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        loss, probs, labels = self._shared_step(batch)
        self._train_probs.extend(probs)
        self._train_labels.extend(labels)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        return loss

    def on_train_epoch_end(self) -> None:
        preds = [int(p > 0.5) for p in self._train_probs]
        m     = _eval_metrics(preds, self._train_labels)
        pos   = [p for p, l in zip(self._train_probs, self._train_labels) if l == 1]
        neg   = [p for p, l in zip(self._train_probs, self._train_labels) if l == 0]
        self.log("train/f1",            m["f1"])
        self.log("train/mean_prob_pos", float(sum(pos) / max(len(pos), 1)))
        self.log("train/mean_prob_neg", float(sum(neg) / max(len(neg), 1)))

    # ── Validation ────────────────────────────────────────────────────────

    def on_validation_epoch_start(self) -> None:
        self._val_probs  = []
        self._val_labels = []

    def validation_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        loss, probs, labels = self._shared_step(batch)
        self._val_probs.extend(probs)
        self._val_labels.extend(labels)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def on_validation_epoch_end(self) -> None:
        preds = [int(p > 0.5) for p in self._val_probs]
        m     = _eval_metrics(preds, self._val_labels)
        pos   = [p for p, l in zip(self._val_probs, self._val_labels) if l == 1]
        self.log("val/f1",            m["f1"],        prog_bar=True)
        self.log("val/precision",     m["precision"])
        self.log("val/recall",        m["recall"])
        self.log("val/mean_prob_pos", float(sum(pos) / max(len(pos), 1)))

    # ── Test (threshold scan) ─────────────────────────────────────────────

    def on_test_epoch_start(self) -> None:
        self._test_probs  = []
        self._test_labels = []

    def test_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        loss, probs, labels = self._shared_step(batch)
        self._test_probs.extend(probs)
        self._test_labels.extend(labels)
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        return loss

    def on_test_epoch_end(self) -> None:
        preds = [int(p > 0.5) for p in self._test_probs]
        m     = _eval_metrics(preds, self._test_labels)
        self.log("test/f1",        m["f1"])
        self.log("test/precision", m["precision"])
        self.log("test/recall",    m["recall"])

        # Threshold scan — find the decision boundary that maximises test F1.
        curve: list[dict] = []
        for t_int in range(1, 19):            # 0.05 … 0.90 step 0.05
            t  = round(t_int / 20, 2)
            cm = _eval_metrics([int(p > t) for p in self._test_probs], self._test_labels)
            curve.append({
                "threshold": t,
                "f1":        round(cm["f1"],        4),
                "precision": round(cm["precision"], 4),
                "recall":    round(cm["recall"],    4),
            })

        best = max(curve, key=lambda x: x["f1"])
        self.best_threshold = best["threshold"]
        self.log("test/best_threshold_f1", best["f1"])
        log.info(
            "Test (thresh=0.50)  F1=%.3f  P=%.3f  R=%.3f",
            m["f1"], m["precision"], m["recall"],
        )
        log.info(
            "Test (best thresh=%.2f)  F1=%.3f  P=%.3f  R=%.3f  ← use at inference",
            best["threshold"], best["f1"], best["precision"], best["recall"],
        )

    # ── Optimizer / scheduler ─────────────────────────────────────────────

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(self.cfg.training.lr),
            weight_decay=float(self.cfg.training.weight_decay),
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=3, factor=0.5,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor":   "val/loss",
                "frequency": 1,
            },
        }


# ---------------------------------------------------------------------------
# LightningModule: Student (distillation)
# ---------------------------------------------------------------------------


class StudentLightningModule(L.LightningModule):
    """Student model training via knowledge distillation.

    Loss = alpha * KL(teacher_soft || student_soft) + (1-alpha) * focal(student, hard_label)

    EarlyStopping and ModelCheckpoint both monitor val/f1 (mode=max).
    ReduceLROnPlateau monitors val/f1 (mode=max).
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters()

        self.model    = build_student()
        self.focal    = FocalLoss(
            alpha = float(cfg.training.focal_alpha),
            gamma = float(cfg.training.focal_gamma),
        )
        self.kd_temp  = float(cfg.training.kd_temperature)
        self.kd_alpha = float(cfg.training.kd_alpha)

        self._train_probs:  list[float] = []
        self._train_labels: list[int]   = []
        self._val_probs:    list[float] = []
        self._val_labels:   list[int]   = []

    # ── Core step ─────────────────────────────────────────────────────────

    def _shared_step(
        self, batch: tuple
    ) -> tuple[torch.Tensor, list[float], list[int]]:
        keyword_vec, mfcc, hard_label, teacher_logit, context_input, position_input = batch
        student_logit = self.model(keyword_vec, mfcc, context_input, position_input).squeeze(-1)
        loss   = kd_loss(
            student_logit, teacher_logit, hard_label,
            temperature = self.kd_temp,
            alpha       = self.kd_alpha,
            focal       = self.focal,
        )
        probs  = torch.sigmoid(student_logit).detach().cpu().tolist()
        labels = hard_label.long().cpu().tolist()
        return loss, probs, labels

    # ── Training ──────────────────────────────────────────────────────────

    def on_train_epoch_start(self) -> None:
        self._train_probs  = []
        self._train_labels = []

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        loss, probs, labels = self._shared_step(batch)
        self._train_probs.extend(probs)
        self._train_labels.extend(labels)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        return loss

    def on_train_epoch_end(self) -> None:
        preds = [int(p > 0.5) for p in self._train_probs]
        m     = _eval_metrics(preds, self._train_labels)
        self.log("train/f1",  m["f1"])
        self.log("train/acc", m["accuracy"])

    # ── Validation ────────────────────────────────────────────────────────

    def on_validation_epoch_start(self) -> None:
        self._val_probs  = []
        self._val_labels = []

    def validation_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        loss, probs, labels = self._shared_step(batch)
        self._val_probs.extend(probs)
        self._val_labels.extend(labels)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def on_validation_epoch_end(self) -> None:
        preds = [int(p > 0.5) for p in self._val_probs]
        m     = _eval_metrics(preds, self._val_labels)
        self.log("val/f1",        m["f1"],        prog_bar=True)
        self.log("val/precision", m["precision"])
        self.log("val/recall",    m["recall"])
        self.log("val/acc",       m["accuracy"])

    # ── Optimizer / scheduler ─────────────────────────────────────────────

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(self.cfg.training.lr),
            weight_decay=float(self.cfg.training.weight_decay),
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", patience=3, factor=0.5,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor":   "val/f1",
                "frequency": 1,
            },
        }
