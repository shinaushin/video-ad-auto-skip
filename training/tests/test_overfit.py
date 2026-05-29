"""test_overfit.py — Overfit sanity checks for teacher and student training loops.

A correctly implemented model must be able to *memorize* a tiny synthetic
dataset and drive training loss close to zero.  Failure means there is a bug
in the model architecture, loss function, optimizer setup, or data pipeline.

These tests use synthetic data only — no real cache files needed.
They are intentionally slow (~5–15 s each) but still fast enough for a
pre-push check.  Mark with -m slow to skip them in a quick CI run.

Run:
    pytest training/tests/test_overfit.py -v
    pytest training/tests/test_overfit.py -v -s   # show per-epoch output
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pytest

from data_pipeline import SponsorDataset
from models import build_student, build_teacher, K_CONTEXT, N_FRAMES, MFCC_DIM, TEXT_DIM
from lightning_modules import (
    FocalLoss,
    TeacherSequenceDataset,
    StudentWindowDataset,
    collate_teacher_sequences,
    kd_loss,
)

DEVICE = "cpu"


# ---------------------------------------------------------------------------
# Teacher overfit
# ---------------------------------------------------------------------------

class TestTeacherOverfit:
    """The BiLSTM teacher should memorize a tiny synthetic dataset."""

    def test_overfit_synthetic(self, fake_cache_dir_large):
        """Training loss must drop by ≥ 90% within 200 epochs on 6 synthetic videos."""
        cache_dir = fake_cache_dir_large
        video_ids = [p.stem for p in sorted(cache_dir.glob("*.npz"))]

        ds       = SponsorDataset(cache_dir, video_ids)
        train_ds = TeacherSequenceDataset(ds)

        loader = DataLoader(
            train_ds,
            batch_size=len(train_ds),       # full-batch GD — easiest to memorize
            collate_fn=collate_teacher_sequences,
            shuffle=False,
        )

        model     = build_teacher(device=DEVICE).train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        criterion = nn.BCEWithLogitsLoss()

        first_loss = None
        final_loss = None

        for epoch in range(200):
            total_loss = 0.0
            n_valid    = 0
            for text_emb, audio_emb, labels, lengths in loader:
                logits = model(text_emb, audio_emb, lengths).squeeze(-1)
                mask         = labels >= 0
                logits_valid = logits[mask]
                labels_valid = labels[mask]
                loss = criterion(logits_valid, labels_valid)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item() * mask.sum().item()
                n_valid    += mask.sum().item()

            epoch_loss = total_loss / max(n_valid, 1)
            if first_loss is None:
                first_loss = epoch_loss
            final_loss = epoch_loss

            if epoch_loss < 0.02:
                break   # converged early

        reduction = (first_loss - final_loss) / max(first_loss, 1e-9)
        assert reduction >= 0.90, (
            f"Teacher failed to overfit synthetic data.\n"
            f"  Initial loss: {first_loss:.4f}\n"
            f"  Final loss:   {final_loss:.4f}\n"
            f"  Reduction:    {reduction*100:.1f}%  (need ≥ 90%)\n"
            "This suggests a bug in TeacherModel, collate_teacher_sequences, or the loss."
        )


# ---------------------------------------------------------------------------
# Student overfit
# ---------------------------------------------------------------------------

class TestStudentOverfit:
    """The student should memorize a tiny synthetic dataset via KD loss."""

    def _make_teacher_logits(self, ds: SponsorDataset) -> dict:
        """Generate synthetic teacher logits: ±5 aligned with hard labels."""
        logits = {}
        vid_counter: dict[str, int] = {}
        for item in ds:
            vid = item["video_id"]
            idx = vid_counter.get(vid, 0)
            # Strong logit: easy for student to distill
            logits[(vid, idx)] = 5.0 if item["label"] else -5.0
            vid_counter[vid] = idx + 1
        return logits

    def test_overfit_synthetic(self, fake_cache_dir_large):
        """KD training loss must drop by ≥ 80% within 100 epochs on 6 synthetic videos."""
        cache_dir = fake_cache_dir_large
        video_ids = [p.stem for p in sorted(cache_dir.glob("*.npz"))]

        ds             = SponsorDataset(cache_dir, video_ids)
        teacher_logits = self._make_teacher_logits(ds)

        # Re-load so the dataset iteration is fresh for StudentWindowDataset
        ds2      = SponsorDataset(cache_dir, video_ids)
        train_ds = StudentWindowDataset(ds2, teacher_logits)

        loader = DataLoader(train_ds, batch_size=len(train_ds), shuffle=False)

        model     = build_student(device=DEVICE).train()
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
        focal     = FocalLoss(alpha=0.5, gamma=0.0)  # gamma=0 = BCE for simplicity

        first_loss = None
        final_loss = None

        for epoch in range(100):
            total_loss = 0.0
            n = 0
            for keyword_vec, mfcc, hard_label, teacher_logit, context, position in loader:
                student_logit = model(keyword_vec, mfcc, context, position).squeeze(-1)
                loss = kd_loss(
                    student_logit, teacher_logit, hard_label,
                    temperature=2.0,
                    alpha=0.5,
                    focal=focal,
                )
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item() * len(hard_label)
                n          += len(hard_label)

            epoch_loss = total_loss / max(n, 1)
            if first_loss is None:
                first_loss = epoch_loss
            final_loss = epoch_loss

            if epoch_loss < 0.05:
                break

        reduction = (first_loss - final_loss) / max(first_loss, 1e-9)
        assert reduction >= 0.80, (
            f"Student failed to overfit synthetic data.\n"
            f"  Initial loss: {first_loss:.4f}\n"
            f"  Final loss:   {final_loss:.4f}\n"
            f"  Reduction:    {reduction*100:.1f}%  (need ≥ 80%)\n"
            "This suggests a bug in StudentModel, StudentWindowDataset, or kd_loss."
        )

    def test_kd_loss_decreases_with_correct_logits(self):
        """kd_loss decreases when student logits move toward teacher logits."""
        B = 32
        teacher_logits = torch.full((B,),  5.0)   # teacher says "sponsor"
        hard_labels    = torch.ones(B)
        focal          = FocalLoss(alpha=0.5, gamma=0.0)

        # Start from student logits that are neutral (0.0)
        student_logits = torch.zeros(B, requires_grad=True)
        loss_before = kd_loss(student_logits, teacher_logits, hard_labels, focal=focal)

        # After one gradient step the loss should be lower
        grad = torch.autograd.grad(loss_before, student_logits)[0]
        student_logits_after = student_logits - 0.5 * grad
        loss_after = kd_loss(student_logits_after.detach(), teacher_logits, hard_labels, focal=focal)

        assert loss_after < loss_before, (
            f"kd_loss did not decrease after gradient step: "
            f"before={loss_before:.4f}  after={loss_after:.4f}"
        )
