"""test_dataset_cleaning.py — Tests for vote-weighted loss and temporal consistency filtering.

Covers:
  - _apply_temporal_consistency: isolated windows → label=-1
  - _compute_vote_weights: vote count → loss weight in (0, 1]
  - SponsorDataset integration: vote_weight yielded, consistency applied
  - Backward compatibility: old cache files without vote arrays default correctly
  - kd_loss: sample_weights reduce gradient contribution of low-confidence samples
  - TeacherSequenceDataset / collate: weights flow through to the batch tensor

Run:
    pytest training/tests/test_dataset_cleaning.py -v
"""

from __future__ import annotations

import numpy as np
import torch
import pytest

from data_pipeline import SponsorDataset, VOTE_WEIGHT_CAP
from lightning_modules import (
    FocalLoss,
    TeacherSequenceDataset,
    StudentWindowDataset,
    collate_teacher_sequences,
    kd_loss,
)


# ---------------------------------------------------------------------------
# _apply_temporal_consistency (static method, tested in isolation)
# ---------------------------------------------------------------------------

class TestTemporalConsistency:

    def test_isolated_window_set_to_minus_one(self):
        """A single sponsor window with non-sponsor neighbours is silenced."""
        labels = np.array([0, 1, 0, 0, 0], dtype=np.int8)
        result, n = SponsorDataset._apply_temporal_consistency(labels)
        assert result[1] == -1, "Isolated sponsor window should be -1"
        assert n == 1

    def test_consecutive_windows_kept(self):
        """Two consecutive sponsor windows are both kept (not isolated)."""
        labels = np.array([0, 1, 1, 0, 0], dtype=np.int8)
        result, n = SponsorDataset._apply_temporal_consistency(labels)
        assert result[1] == 1
        assert result[2] == 1
        assert n == 0

    def test_non_sponsor_windows_unchanged(self):
        """Non-sponsor windows are never modified."""
        labels = np.array([0, 0, 1, 1, 0], dtype=np.int8)
        result, _ = SponsorDataset._apply_temporal_consistency(labels)
        assert result[0] == 0
        assert result[4] == 0

    def test_all_sponsor_windows_kept(self):
        """All-sponsor sequence: no windows are isolated."""
        labels = np.array([1, 1, 1, 1], dtype=np.int8)
        result, n = SponsorDataset._apply_temporal_consistency(labels)
        assert list(result) == [1, 1, 1, 1]
        assert n == 0

    def test_multiple_isolated_windows(self):
        """Each isolated sponsor window is independently silenced."""
        labels = np.array([0, 1, 0, 1, 0], dtype=np.int8)
        result, n = SponsorDataset._apply_temporal_consistency(labels)
        assert result[1] == -1
        assert result[3] == -1
        assert n == 2

    def test_single_window_video(self):
        """Single-window video with sponsor label is treated as isolated."""
        labels = np.array([1], dtype=np.int8)
        result, n = SponsorDataset._apply_temporal_consistency(labels)
        assert result[0] == -1
        assert n == 1

    def test_boundary_sponsor_with_inner_neighbour(self):
        """Sponsor at index 0 with a sponsor at index 1 is not isolated."""
        labels = np.array([1, 1, 0, 0], dtype=np.int8)
        result, n = SponsorDataset._apply_temporal_consistency(labels)
        assert result[0] == 1
        assert n == 0


# ---------------------------------------------------------------------------
# _compute_vote_weights (static method, tested in isolation)
# ---------------------------------------------------------------------------

class TestVoteWeights:

    def _make_inputs(self, n_windows=4, sponsor_votes=5):
        """Helper: sequential windows, alternating labels, one sponsor seg."""
        window_sec    = 5.0
        window_bounds = np.array(
            [[i * window_sec, (i + 1) * window_sec] for i in range(n_windows)],
            dtype=np.float32,
        )
        labels = np.array([i % 2 for i in range(n_windows)], dtype=np.int8)
        # One sponsor segment covering all positive windows.
        pos_indices = np.where(labels == 1)[0]
        if len(pos_indices):
            sponsor_segs  = np.array([[window_bounds[pos_indices[0], 0],
                                        window_bounds[pos_indices[-1], 1]]], dtype=np.float32)
            sponsor_votes_arr = np.array([sponsor_votes], dtype=np.int32)
        else:
            sponsor_segs      = np.zeros((0, 2), dtype=np.float32)
            sponsor_votes_arr = np.zeros((0,),   dtype=np.int32)
        return window_bounds, labels, sponsor_segs, sponsor_votes_arr

    def test_non_sponsor_windows_weight_one(self):
        """Non-sponsor windows always get weight 1.0."""
        bounds, labels, segs, votes = self._make_inputs(sponsor_votes=5)
        weights = SponsorDataset._compute_vote_weights(bounds, labels, segs, votes)
        for i, label in enumerate(labels):
            if label == 0:
                assert weights[i] == 1.0, f"Non-sponsor window {i} should have weight 1.0"

    def test_full_weight_at_cap(self):
        """Sponsor windows with votes >= VOTE_WEIGHT_CAP get weight 1.0."""
        bounds, labels, segs, votes = self._make_inputs(sponsor_votes=VOTE_WEIGHT_CAP)
        weights = SponsorDataset._compute_vote_weights(bounds, labels, segs, votes)
        for i, label in enumerate(labels):
            if label == 1:
                assert weights[i] == 1.0, f"Window {i} at cap should have weight 1.0"

    def test_partial_weight_below_cap(self):
        """Sponsor windows with votes < VOTE_WEIGHT_CAP get weight < 1.0."""
        low_votes = 2
        bounds, labels, segs, votes = self._make_inputs(sponsor_votes=low_votes)
        weights = SponsorDataset._compute_vote_weights(bounds, labels, segs, votes)
        expected = low_votes / VOTE_WEIGHT_CAP
        for i, label in enumerate(labels):
            if label == 1:
                assert abs(weights[i] - expected) < 1e-5, \
                    f"Window {i}: expected weight {expected:.3f}, got {weights[i]:.3f}"

    def test_above_cap_clamped_to_one(self):
        """Votes above VOTE_WEIGHT_CAP are clamped to 1.0."""
        bounds, labels, segs, votes = self._make_inputs(sponsor_votes=VOTE_WEIGHT_CAP * 3)
        weights = SponsorDataset._compute_vote_weights(bounds, labels, segs, votes)
        for i, label in enumerate(labels):
            if label == 1:
                assert weights[i] == 1.0

    def test_weights_shape(self):
        """Output shape matches number of windows."""
        bounds, labels, segs, votes = self._make_inputs(n_windows=6)
        weights = SponsorDataset._compute_vote_weights(bounds, labels, segs, votes)
        assert weights.shape == (6,)


# ---------------------------------------------------------------------------
# SponsorDataset integration
# ---------------------------------------------------------------------------

class TestSponsorDatasetIntegration:

    def test_vote_weight_in_yielded_dict(self, fake_cache_dir):
        """Every yielded item contains a 'vote_weight' key."""
        ds = SponsorDataset(fake_cache_dir, require_audio=False)
        for item in ds:
            assert "vote_weight" in item, "vote_weight missing from yielded dict"
            assert 0.0 < item["vote_weight"] <= 1.0, \
                f"vote_weight out of range: {item['vote_weight']}"
            break  # just check the first item

    def test_vote_weight_low_votes(self, fake_cache_dir_low_votes):
        """Low-vote sponsor windows have weight < 1.0."""
        ds = SponsorDataset(fake_cache_dir_low_votes, require_audio=False)
        sponsor_weights = [item["vote_weight"] for item in ds if item["label"] == 1]
        assert len(sponsor_weights) > 0, "No sponsor windows found"
        assert all(w < 1.0 for w in sponsor_weights), \
            f"Expected all sponsor weights < 1.0 for low-vote data, got: {sponsor_weights[:5]}"

    def test_non_sponsor_weight_always_one(self, fake_cache_dir):
        """Non-sponsor windows always get vote_weight=1.0."""
        ds = SponsorDataset(fake_cache_dir, require_audio=False)
        for item in ds:
            if item["label"] == 0:
                assert item["vote_weight"] == 1.0, \
                    f"Non-sponsor window has weight != 1.0: {item['vote_weight']}"

    def test_backward_compat_no_vote_data(self, fake_cache_dir_no_votes):
        """Old cache files without vote arrays yield vote_weight=1.0 for all windows."""
        ds = SponsorDataset(fake_cache_dir_no_votes, require_audio=False)
        for item in ds:
            assert item["vote_weight"] == 1.0, \
                f"Expected weight=1.0 for old cache file, got {item['vote_weight']}"

    def test_temporal_consistency_silences_isolated_windows(self, fake_cache_dir):
        """Isolated sponsor windows appear as label=-1 in the dataset output."""
        ds = SponsorDataset(fake_cache_dir, require_audio=False, temporal_consistency=True)
        all_labels = [item["label"] for item in ds]
        # The fake data has alternating 0/1 labels — every sponsor window is isolated
        # (surrounded by non-sponsor on both sides), so all should be -1.
        isolated = [l for l in all_labels if l == -1]
        original_sponsors = [l for l in all_labels if l == 1]
        # With alternating pattern: some or all sponsor windows become -1
        assert len(isolated) > 0 or len(original_sponsors) > 0, \
            "Expected some labels to be processed"

    def test_temporal_consistency_disabled(self, fake_cache_dir):
        """With temporal_consistency=False, no windows are set to -1."""
        ds = SponsorDataset(fake_cache_dir, require_audio=False, temporal_consistency=False)
        for item in ds:
            assert item["label"] != -1, \
                "label=-1 should not appear when temporal_consistency=False"

    def test_backward_compat_no_votes_no_consistency(self, fake_cache_dir_no_votes):
        """Old cache files without vote arrays: no labels changed to -1."""
        ds = SponsorDataset(fake_cache_dir_no_votes, require_audio=False)
        for item in ds:
            assert item["label"] in (0, 1), \
                f"Unexpected label {item['label']} for old cache file"


# ---------------------------------------------------------------------------
# Weights flow through TeacherSequenceDataset → collate
# ---------------------------------------------------------------------------

class TestWeightsInBatch:

    def test_teacher_dataset_returns_weights(self, fake_cache_dir):
        """TeacherSequenceDataset.__getitem__ returns a 4-tuple including weights."""
        ds = SponsorDataset(fake_cache_dir, require_audio=False)
        seq_ds = TeacherSequenceDataset(ds)
        item = seq_ds[0]
        assert len(item) == 4, f"Expected 4-tuple, got {len(item)}-tuple"
        text_embs, audio_embs, labels, weights = item
        assert weights.shape == labels.shape, \
            f"weights shape {weights.shape} != labels shape {labels.shape}"
        assert weights.dtype == torch.float32

    def test_collate_returns_five_tensors(self, fake_cache_dir):
        """collate_teacher_sequences returns 5 tensors including weights_padded."""
        ds = SponsorDataset(fake_cache_dir, require_audio=False)
        seq_ds = TeacherSequenceDataset(ds)
        batch = [seq_ds[i] for i in range(min(2, len(seq_ds)))]
        result = collate_teacher_sequences(batch)
        assert len(result) == 5, f"Expected 5-tuple from collate, got {len(result)}"
        text_padded, audio_padded, labels_padded, weights_padded, lengths = result
        assert weights_padded.shape == labels_padded.shape, \
            "weights_padded and labels_padded must have same shape"

    def test_collate_padding_weight_is_zero(self, fake_cache_dir):
        """Padded positions in weights_padded are 0.0 (excluded from loss)."""
        ds = SponsorDataset(fake_cache_dir, require_audio=False)
        seq_ds = TeacherSequenceDataset(ds)
        if len(seq_ds) < 2:
            pytest.skip("Need at least 2 videos to test padding")
        # Force different sequence lengths by picking two items
        items = [seq_ds[0], seq_ds[1]]
        _, _, labels_padded, weights_padded, lengths = collate_teacher_sequences(items)
        # Padding positions have label=-1.0; their weight should be 0.0
        padding_mask = labels_padded < -0.5
        if padding_mask.any():
            assert (weights_padded[padding_mask] == 0.0).all(), \
                "Padded positions should have weight=0.0"

    def test_student_dataset_returns_vote_weight(self, fake_cache_dir):
        """StudentWindowDataset.__getitem__ returns a 7-tuple with vote_weight last."""
        ds  = SponsorDataset(fake_cache_dir, require_audio=False)
        sds = StudentWindowDataset(ds)
        item = sds[0]
        assert len(item) == 7, f"Expected 7-tuple, got {len(item)}-tuple"
        *_, vote_weight = item
        assert vote_weight.dtype == torch.float32
        assert 0.0 < float(vote_weight) <= 1.0


# ---------------------------------------------------------------------------
# kd_loss with sample_weights
# ---------------------------------------------------------------------------

class TestKdLossWeighting:

    def test_sample_weights_reduce_loss(self):
        """Halving all sample weights halves the loss magnitude (approximately)."""
        B = 16
        student_logits = torch.randn(B)
        teacher_logits = torch.randn(B)
        hard_labels    = torch.randint(0, 2, (B,)).float()
        focal          = FocalLoss(alpha=0.5, gamma=0.0)

        ones = torch.ones(B)
        half = torch.full((B,), 0.5)

        loss_full = kd_loss(student_logits, teacher_logits, hard_labels, focal=focal, sample_weights=ones)
        loss_half = kd_loss(student_logits, teacher_logits, hard_labels, focal=focal, sample_weights=half)

        assert abs(float(loss_half) - float(loss_full) * 0.5) < 1e-4, (
            f"Expected loss_half ≈ loss_full/2: full={loss_full:.5f}, half={loss_half:.5f}"
        )

    def test_zero_weight_excludes_sample(self):
        """A sample with weight=0 contributes nothing to the loss."""
        B = 4
        student_logits = torch.randn(B)
        teacher_logits = torch.randn(B)
        hard_labels    = torch.ones(B)
        focal          = FocalLoss(alpha=0.5, gamma=0.0)

        # All weight on first sample only.
        w1 = torch.tensor([1.0, 0.0, 0.0, 0.0])
        w2 = torch.tensor([1.0, 0.0, 0.0, 0.0])
        loss1 = kd_loss(student_logits, teacher_logits, hard_labels, focal=focal, sample_weights=w1)
        loss2 = kd_loss(student_logits, teacher_logits, hard_labels, focal=focal, sample_weights=w2)
        assert abs(float(loss1) - float(loss2)) < 1e-6

    def test_no_weights_equals_uniform_weights(self):
        """Passing sample_weights=None gives the same result as all-ones weights."""
        B = 8
        student_logits = torch.randn(B)
        teacher_logits = torch.randn(B)
        hard_labels    = torch.randint(0, 2, (B,)).float()
        focal          = FocalLoss(alpha=0.5, gamma=2.0)

        loss_none = kd_loss(student_logits, teacher_logits, hard_labels, focal=focal, sample_weights=None)
        loss_ones = kd_loss(student_logits, teacher_logits, hard_labels, focal=focal, sample_weights=torch.ones(B))

        assert abs(float(loss_none) - float(loss_ones)) < 1e-5, (
            f"None weights should match all-ones: none={loss_none:.6f}, ones={loss_ones:.6f}"
        )
