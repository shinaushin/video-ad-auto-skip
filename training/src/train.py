"""train.py — Training loop for teacher and student models.

Modes:
    teacher   Train TeacherModel on DistilBERT + Whisper embeddings from cache.
    distill   Train StudentModel using teacher soft labels (knowledge distillation)
              plus hard ground-truth labels.
    baseline  Evaluate keyword heuristic accuracy against SponsorBlock ground truth.

Usage:
    python train.py --config configs/phase3_teacher.json
    python train.py --config configs/phase4_distill.json
    python train.py --config configs/phase2_baseline.json

Config keys (JSON):
    phase           str   "teacher" | "distill" | "baseline"
    cache_dir       str   Path to per-video .npz cache
    output_dir      str   Where to save checkpoints and training_log.json
    teacher_ckpt    str   (distill only) Path to trained teacher checkpoint
    epochs          int   Max training epochs (default 30)
    batch_size      int   Windows per batch (default 64)
    lr              float Learning rate (default 1e-3)
    weight_decay    float AdamW weight decay (default 1e-4)
    patience        int   Early stopping patience in epochs (default 5)
    kd_temperature  float KD temperature — higher = softer teacher labels (default 4.0)
    kd_alpha        float Weight of KD loss vs. hard CE loss (default 0.7)
    focal_gamma     float Focal loss gamma (default 2.0)
    focal_alpha     float Focal loss positive weight (default 0.25)
    device          str   "cuda" | "cpu" (default: auto-detect)
    seed            int   (default 42)
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import time
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from data_pipeline import SponsorDataset, MFCC_DIM, N_MFCC_FRAMES, TEXT_DIM
from models import (
    DISTILBERT_DIM,
    N_FRAMES,
    WHISPER_DIM,
    StudentModel,
    TeacherModel,
    build_student,
    build_teacher,
    load_teacher,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Focal loss
# ---------------------------------------------------------------------------


class FocalLoss(nn.Module):
    """Binary focal loss for class-imbalanced datasets.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        alpha: Weight applied to the positive class (sponsor=1).
               Use a value < 0.5 to down-weight the dominant positive loss
               when there are many more sponsor windows than non-sponsor ones,
               or > 0.5 to up-weight the rare positive class.
        gamma: Focusing exponent.  gamma=0 → standard BCE.  gamma=2 is a
               common default that reduces the loss contribution of
               easy-to-classify examples.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss.

        Args:
            logits:  [N] or [N, 1]  pre-sigmoid logits.
            targets: [N] or [N, 1]  binary float targets (0.0 or 1.0).

        Returns:
            Scalar mean focal loss.
        """
        logits = logits.view(-1)
        targets = targets.view(-1).float()

        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = torch.exp(-bce)  # = sigmoid(logits) when target=1 else 1-sigmoid(logits)
        alpha_t = targets * self.alpha + (1.0 - targets) * (1.0 - self.alpha)
        focal = alpha_t * (1.0 - p_t) ** self.gamma * bce
        return focal.mean()


# ---------------------------------------------------------------------------
# Dataset wrappers (torch Dataset from SponsorDataset)
# ---------------------------------------------------------------------------


class TeacherWindowDataset(Dataset):
    """Flat window dataset for teacher training: (text_emb, audio_emb) → label.

    Each item is a single window (no sequence context).  The BiLSTM in the
    teacher handles temporal context inside the forward pass, but for batching
    purposes we treat each window as independent during training.
    """

    def __init__(self, sponsor_ds: SponsorDataset) -> None:
        self._items: list[dict] = list(sponsor_ds)

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        item = self._items[idx]
        text_emb = torch.from_numpy(item["text_emb"])     # [768]
        audio_emb = torch.from_numpy(item["audio_emb"])   # [384]
        label = torch.tensor(item["label"], dtype=torch.float32)
        return text_emb, audio_emb, label


class StudentWindowDataset(Dataset):
    """Flat window dataset for student training: (keyword_vec, dummy_mfcc) → label.

    During distillation the ``teacher_logit`` field is populated by running the
    teacher model over the same window embeddings.  When available it overrides
    the hard label for the KD loss.

    Note: the student is trained on keyword vectors + MFCC.  The MFCC frames
    come from the extension's real-time capturer; during offline training we
    cannot easily reproduce the exact per-video MFCC sequence.  We therefore
    use a deterministic-noise surrogate derived from the Whisper embeddings so
    the CNN learns *structure* from the embeddings rather than random noise.
    This is replaced with real MFCC at inference time.
    """

    def __init__(
        self,
        sponsor_ds: SponsorDataset,
        teacher_logits: dict[tuple[str, int], float] | None = None,
    ) -> None:
        self._items: list[dict] = list(sponsor_ds)
        self._teacher_logits = teacher_logits or {}
        # Pre-index items by (video_id, window_idx) for teacher logit lookup.
        self._keys: list[tuple[str, int]] = []
        vid_counter: dict[str, int] = {}
        for item in self._items:
            vid = item["video_id"]
            idx = vid_counter.get(vid, 0)
            self._keys.append((vid, idx))
            vid_counter[vid] = idx + 1

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        item = self._items[idx]
        keyword_vec = torch.from_numpy(item["keyword_vec"])   # [64]

        # Construct surrogate MFCC from Whisper embedding: project 384 → 13, tile N_FRAMES.
        audio_emb = torch.from_numpy(item["audio_emb"])       # [384]
        # Simple deterministic projection: take 13 linearly-spaced values from 384 dims,
        # then add small per-frame positional variation.
        indices = torch.linspace(0, 383, 13).long()
        frame_base = audio_emb[indices]                        # [13]
        # Tile to [N_FRAMES, 13] with small Gaussian jitter (std=0.01) for diversity.
        torch.manual_seed(idx)  # reproducible per-sample
        noise = torch.randn(N_FRAMES, 13) * 0.01
        mfcc = frame_base.unsqueeze(0).expand(N_FRAMES, -1) + noise  # [N_FRAMES, 13]

        hard_label = torch.tensor(item["label"], dtype=torch.float32)

        # Teacher soft label: logit from teacher inference (or fall back to hard label).
        key = self._keys[idx]
        teacher_logit_val = self._teacher_logits.get(key, None)
        if teacher_logit_val is not None:
            teacher_logit = torch.tensor(teacher_logit_val, dtype=torch.float32)
        else:
            # No teacher logit available — encode hard label as logit.
            teacher_logit = torch.tensor(10.0 if item["label"] else -10.0, dtype=torch.float32)

        return keyword_vec, mfcc, hard_label, teacher_logit


# ---------------------------------------------------------------------------
# Class-imbalance sampler
# ---------------------------------------------------------------------------


def make_balanced_sampler(labels: list[int]) -> WeightedRandomSampler:
    """Create a WeightedRandomSampler that up-samples the minority class."""
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None  # type: ignore
    w_pos = 1.0 / n_pos
    w_neg = 1.0 / n_neg
    weights = [w_pos if l == 1 else w_neg for l in labels]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


# ---------------------------------------------------------------------------
# Teacher training
# ---------------------------------------------------------------------------


def _run_teacher_epoch(
    model: TeacherModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    criterion: FocalLoss,
    device: str,
    train: bool,
) -> tuple[float, float]:
    """Run one epoch; return (mean_loss, accuracy)."""
    model.train(train)
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.set_grad_enabled(train):
        for text_emb, audio_emb, labels in loader:
            text_emb = text_emb.to(device)    # [B, 768]
            audio_emb = audio_emb.to(device)  # [B, 384]
            labels = labels.to(device)         # [B]

            # Teacher expects [batch, seq_len, dim]; treat each window as seq_len=1.
            text_in = text_emb.unsqueeze(1)    # [B, 1, 768]
            audio_in = audio_emb.unsqueeze(1)  # [B, 1, 384]
            logits = model(text_in, audio_in)  # [B, 1, 1]
            logits = logits.squeeze(-1).squeeze(-1)  # [B]

            loss = criterion(logits, labels)

            if train and optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item() * len(labels)
            preds = (torch.sigmoid(logits) > 0.5).long()
            correct += (preds == labels.long()).sum().item()
            total += len(labels)

    return total_loss / max(total, 1), correct / max(total, 1)


def train_teacher(cfg: dict, device: str) -> Path:
    """Train the teacher model and return the checkpoint path."""
    cache_dir = Path(cfg["cache_dir"])
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    epochs = int(cfg.get("epochs", 30))
    batch_size = int(cfg.get("batch_size", 64))
    lr = float(cfg.get("lr", 1e-3))
    weight_decay = float(cfg.get("weight_decay", 1e-4))
    patience = int(cfg.get("patience", 5))
    focal_gamma = float(cfg.get("focal_gamma", 2.0))
    focal_alpha = float(cfg.get("focal_alpha", 0.25))

    train_ids, val_ids, test_ids = SponsorDataset.train_val_test_split(cache_dir)

    train_ds = TeacherWindowDataset(SponsorDataset(cache_dir, train_ids))
    val_ds = TeacherWindowDataset(SponsorDataset(cache_dir, val_ids))

    train_labels = [int(d[2].item()) for d in train_ds]
    sampler = make_balanced_sampler(train_labels)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = build_teacher(device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)

    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_ckpt = output_dir / "teacher_best.pt"

    training_log: dict = {
        "phase": "teacher",
        "epochs": [],
        "best_epoch": 0,
        "best_val_loss": float("inf"),
    }

    log.info("Starting teacher training for %d epochs", epochs)
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_acc = _run_teacher_epoch(
            model, train_loader, optimizer, criterion, device, train=True
        )
        val_loss, val_acc = _run_teacher_epoch(
            model, val_loader, None, criterion, device, train=False
        )
        scheduler.step(val_loss)

        elapsed = time.time() - t0
        log.info(
            "Epoch %2d/%d  train_loss=%.4f  train_acc=%.3f  val_loss=%.4f  val_acc=%.3f  %.1fs",
            epoch, epochs, train_loss, train_acc, val_loss, val_acc, elapsed,
        )

        ep_entry = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        }
        training_log["epochs"].append(ep_entry)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            training_log["best_epoch"] = epoch
            training_log["best_val_loss"] = best_val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "config": cfg,
                },
                best_ckpt,
            )
            log.info("  ✓ Best checkpoint saved (val_loss=%.4f)", val_loss)
        else:
            epochs_no_improve += 1
            log.info("  No improvement (%d/%d)", epochs_no_improve, patience)
            if epochs_no_improve >= patience:
                log.info("Early stopping triggered.")
                break

    log.path = str(best_ckpt)
    log_path = output_dir / "training_log.json"
    log_path.write_text(json.dumps(training_log, indent=2))
    log.info("Training log written to %s", log_path)
    return best_ckpt


# ---------------------------------------------------------------------------
# Knowledge distillation — collect teacher logits
# ---------------------------------------------------------------------------


@torch.no_grad()
def collect_teacher_logits(
    teacher: TeacherModel,
    sponsor_ds: SponsorDataset,
    device: str,
    batch_size: int = 128,
) -> dict[tuple[str, int], float]:
    """Run teacher inference over the dataset; return (video_id, window_idx) → logit."""
    teacher.eval()
    raw_items = list(sponsor_ds)
    result: dict[tuple[str, int], float] = {}
    vid_counter: dict[str, int] = {}

    for i in range(0, len(raw_items), batch_size):
        batch = raw_items[i : i + batch_size]
        text_embs = torch.stack([torch.from_numpy(it["text_emb"]) for it in batch]).to(device)
        audio_embs = torch.stack([torch.from_numpy(it["audio_emb"]) for it in batch]).to(device)

        text_in = text_embs.unsqueeze(1)    # [B, 1, 768]
        audio_in = audio_embs.unsqueeze(1)  # [B, 1, 384]
        logits = teacher(text_in, audio_in).squeeze(-1).squeeze(-1).cpu().tolist()

        for item, logit in zip(batch, logits):
            vid = item["video_id"]
            idx = vid_counter.get(vid, 0)
            result[(vid, idx)] = logit
            vid_counter[vid] = idx + 1

    log.info("Collected %d teacher logits", len(result))
    return result


# ---------------------------------------------------------------------------
# Knowledge distillation loss
# ---------------------------------------------------------------------------


def kd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    hard_labels: torch.Tensor,
    temperature: float = 4.0,
    alpha: float = 0.7,
    focal: FocalLoss | None = None,
) -> torch.Tensor:
    """Combined knowledge distillation + focal loss.

    L = alpha * KL_div(student_soft, teacher_soft) + (1 - alpha) * focal(student, hard_label)

    Args:
        student_logits:  [N] pre-sigmoid student output.
        teacher_logits:  [N] pre-sigmoid teacher output.
        hard_labels:     [N] binary hard labels.
        temperature:     Softening temperature T (applied to both student and teacher logits).
        alpha:           Weight on the KD term vs. hard label term.
        focal:           FocalLoss instance; uses standard BCE if None.
    """
    # Soften both distributions.
    T = temperature
    student_soft = torch.sigmoid(student_logits / T)   # [N]
    teacher_soft = torch.sigmoid(teacher_logits / T)   # [N]

    # KL divergence between Bernoulli distributions (binary case).
    # KL(teacher || student) = sum(teacher * log(teacher/student))
    eps = 1e-7
    kl = teacher_soft * (torch.log(teacher_soft + eps) - torch.log(student_soft + eps)) \
       + (1 - teacher_soft) * (torch.log(1 - teacher_soft + eps) - torch.log(1 - student_soft + eps))
    kl_loss = (T ** 2) * kl.mean()  # scale by T^2 to preserve gradient magnitude

    # Hard label loss.
    if focal is not None:
        hard_loss = focal(student_logits, hard_labels)
    else:
        hard_loss = F.binary_cross_entropy_with_logits(student_logits, hard_labels)

    return alpha * kl_loss + (1.0 - alpha) * hard_loss


# ---------------------------------------------------------------------------
# Student training (distillation)
# ---------------------------------------------------------------------------


def _run_student_epoch(
    model: StudentModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    focal: FocalLoss,
    device: str,
    train: bool,
    kd_temp: float,
    kd_alpha: float,
) -> tuple[float, float]:
    model.train(train)
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.set_grad_enabled(train):
        for keyword_vec, mfcc, hard_label, teacher_logit in loader:
            keyword_vec = keyword_vec.to(device)    # [B, 64]
            mfcc = mfcc.to(device)                  # [B, N_FRAMES, 13]
            hard_label = hard_label.to(device)      # [B]
            teacher_logit = teacher_logit.to(device) # [B]

            student_logit = model(keyword_vec, mfcc).squeeze(-1)  # [B]

            loss = kd_loss(
                student_logit, teacher_logit, hard_label,
                temperature=kd_temp,
                alpha=kd_alpha,
                focal=focal,
            )

            if train and optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item() * len(hard_label)
            preds = (torch.sigmoid(student_logit) > 0.5).long()
            correct += (preds == hard_label.long()).sum().item()
            total += len(hard_label)

    return total_loss / max(total, 1), correct / max(total, 1)


def train_distill(cfg: dict, device: str) -> Path:
    """Train the student model via knowledge distillation from the teacher."""
    cache_dir = Path(cfg["cache_dir"])
    output_dir = Path(cfg["output_dir"])
    teacher_ckpt = cfg.get("teacher_ckpt")
    output_dir.mkdir(parents=True, exist_ok=True)

    epochs = int(cfg.get("epochs", 30))
    batch_size = int(cfg.get("batch_size", 64))
    lr = float(cfg.get("lr", 1e-3))
    weight_decay = float(cfg.get("weight_decay", 1e-4))
    patience = int(cfg.get("patience", 5))
    kd_temp = float(cfg.get("kd_temperature", 4.0))
    kd_alpha = float(cfg.get("kd_alpha", 0.7))
    focal_gamma = float(cfg.get("focal_gamma", 2.0))
    focal_alpha = float(cfg.get("focal_alpha", 0.25))

    train_ids, val_ids, test_ids = SponsorDataset.train_val_test_split(cache_dir)

    # Collect teacher logits over train + val sets.
    teacher_logits: dict[tuple[str, int], float] = {}
    if teacher_ckpt:
        log.info("Loading teacher for logit collection: %s", teacher_ckpt)
        teacher = load_teacher(teacher_ckpt, device=device)
        for split_ids in (train_ids, val_ids):
            split_ds = SponsorDataset(cache_dir, split_ids)
            teacher_logits.update(collect_teacher_logits(teacher, split_ds, device))
        del teacher
        if device == "cuda":
            torch.cuda.empty_cache()
    else:
        log.warning("No teacher checkpoint provided — using hard labels only (alpha→0)")
        kd_alpha = 0.0

    train_ds = StudentWindowDataset(SponsorDataset(cache_dir, train_ids), teacher_logits)
    val_ds = StudentWindowDataset(SponsorDataset(cache_dir, val_ids), teacher_logits)

    train_labels = [int(d[2].item()) for d in train_ds]
    sampler = make_balanced_sampler(train_labels)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = build_student(device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    focal = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)

    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_ckpt = output_dir / "student_best.pt"

    training_log: dict = {
        "phase": "distill",
        "epochs": [],
        "best_epoch": 0,
        "best_val_loss": float("inf"),
    }

    log.info("Starting distillation training for %d epochs (T=%.1f alpha=%.2f)", epochs, kd_temp, kd_alpha)
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_acc = _run_student_epoch(
            model, train_loader, optimizer, focal, device, train=True,
            kd_temp=kd_temp, kd_alpha=kd_alpha,
        )
        val_loss, val_acc = _run_student_epoch(
            model, val_loader, None, focal, device, train=False,
            kd_temp=kd_temp, kd_alpha=kd_alpha,
        )
        scheduler.step(val_loss)

        elapsed = time.time() - t0
        log.info(
            "Epoch %2d/%d  train_loss=%.4f  train_acc=%.3f  val_loss=%.4f  val_acc=%.3f  %.1fs",
            epoch, epochs, train_loss, train_acc, val_loss, val_acc, elapsed,
        )

        ep_entry = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        }
        training_log["epochs"].append(ep_entry)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            training_log["best_epoch"] = epoch
            training_log["best_val_loss"] = best_val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "config": cfg,
                },
                best_ckpt,
            )
            log.info("  ✓ Best checkpoint saved (val_loss=%.4f)", val_loss)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                log.info("Early stopping triggered.")
                break

    log_path = output_dir / "training_log.json"
    log_path.write_text(json.dumps(training_log, indent=2))
    log.info("Training log written to %s", log_path)
    return best_ckpt


# ---------------------------------------------------------------------------
# Baseline evaluation
# ---------------------------------------------------------------------------


def evaluate_baseline(cfg: dict) -> None:
    """Evaluate the keyword heuristic against ground-truth labels."""
    cache_dir = Path(cfg["cache_dir"])
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    _, _, test_ids = SponsorDataset.train_val_test_split(cache_dir)
    test_ds = SponsorDataset(cache_dir, test_ids)

    _WEIGHTS = {0: 3.0, 1: 1.5, 2: 1.5, 3: 0.5}
    _GROUPS = [g for (_, g) in [(None, 0)] * 16 + [(None, 1)] * 16 + [(None, 2)] * 16 + [(None, 3)] * 16]

    tp = fp = tn = fn = 0
    for item in test_ds:
        kv = item["keyword_vec"]
        score = sum(kv[i] * _WEIGHTS[g] for i, (_, g) in enumerate(zip(kv, _GROUPS)))
        pred = 1 if score >= 2.0 else 0
        label = item["label"]
        if pred == 1 and label == 1:
            tp += 1
        elif pred == 1 and label == 0:
            fp += 1
        elif pred == 0 and label == 1:
            fn += 1
        else:
            tn += 1

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-7)

    results = {
        "phase": "baseline",
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": (tp + tn) / max(tp + fp + tn + fn, 1),
    }
    log.info("Baseline  P=%.3f  R=%.3f  F1=%.3f", precision, recall, f1)

    log_path = output_dir / "training_log.json"
    log_path.write_text(json.dumps(results, indent=2))
    log.info("Baseline results written to %s", log_path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    p = argparse.ArgumentParser(description="Train teacher or student model.")
    p.add_argument("--config", required=True, type=Path, help="Path to phase config JSON.")
    args = p.parse_args()

    cfg = json.loads(args.config.read_text())
    phase = cfg.get("phase", "teacher")

    # Device selection.
    device_pref = cfg.get("device", "auto")
    if device_pref == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = device_pref
    log.info("Phase: %s  Device: %s", phase, device)

    # Reproducibility.
    seed = int(cfg.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)

    if phase == "teacher":
        train_teacher(cfg, device)
    elif phase == "distill":
        train_distill(cfg, device)
    elif phase == "baseline":
        evaluate_baseline(cfg)
    else:
        raise ValueError(f"Unknown phase: {phase!r}. Expected: teacher | distill | baseline")


if __name__ == "__main__":
    main()
