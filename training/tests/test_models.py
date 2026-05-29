"""test_models.py — Shape and smoke tests for TeacherModel and StudentModel.

These tests run on CPU with random weights — no checkpoint, no real data.
They catch:
  - Wrong output shapes
  - Transpose bugs in MFCCConvBranch (Conv1d expects [B, C, L])
  - Broken embed_mode branches (text_only / audio_only / both)
  - Variable-length packed-sequence handling in the BiLSTM
  - predict_proba output in [0, 1]
  - Factory functions (build_teacher, build_student, load_teacher)

Run:
    pytest training/tests/test_models.py -v
"""

from __future__ import annotations

import torch
import pytest

from models import (
    DISTILBERT_DIM,
    K_CONTEXT,
    MFCC_DIM,
    N_FRAMES,
    TEXT_DIM,
    WHISPER_DIM,
    MFCCConvBranch,
    KeywordTextBranch,
    StudentModel,
    TeacherModel,
    build_student,
    build_teacher,
    load_student,
    load_teacher,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DEVICE = "cpu"
B      = 2    # batch size used throughout
SEQ    = 5    # sequence length for teacher (windows per video)


def _teacher_batch(batch: int = B, seq: int = SEQ) -> tuple:
    """Return (text_embs, audio_embs, lengths) for teacher forward."""
    text   = torch.randn(batch, seq, DISTILBERT_DIM)
    audio  = torch.randn(batch, seq, WHISPER_DIM)
    lengths = torch.tensor([seq] * batch, dtype=torch.long)
    return text, audio, lengths


def _student_batch(batch: int = B) -> tuple:
    """Return all four student inputs."""
    text     = torch.randn(batch, TEXT_DIM)
    audio    = torch.randn(batch, N_FRAMES, MFCC_DIM)
    context  = torch.zeros(batch, K_CONTEXT)
    position = torch.rand(batch, 1)
    return text, audio, context, position


# ---------------------------------------------------------------------------
# Teacher model
# ---------------------------------------------------------------------------

class TestTeacherModel:

    def test_output_shape_default(self):
        """TeacherModel with default args returns [B, SEQ, 1]."""
        model = build_teacher(device=DEVICE).train()
        text, audio, lengths = _teacher_batch()
        out = model(text, audio, lengths)
        assert out.shape == (B, SEQ, 1), f"Expected ({B}, {SEQ}, 1), got {out.shape}"

    def test_output_shape_no_lengths(self):
        """Forward works without lengths (no packed-sequence)."""
        model = build_teacher(device=DEVICE).train()
        text, audio, _ = _teacher_batch()
        out = model(text, audio)   # lengths=None
        assert out.shape == (B, SEQ, 1)

    def test_variable_lengths(self):
        """Padded positions are masked — output shape still [B, SEQ, 1]."""
        model = build_teacher(device=DEVICE).eval()
        text, audio, _ = _teacher_batch(batch=3, seq=10)
        lengths = torch.tensor([10, 7, 4], dtype=torch.long)
        out = model(text, audio, lengths)
        assert out.shape == (3, 10, 1)

    @pytest.mark.parametrize("embed_mode", ["both", "text_only", "audio_only"])
    def test_embed_modes(self, embed_mode: str):
        """All three embed_mode branches produce the correct output shape."""
        model = TeacherModel(embed_mode=embed_mode).to(DEVICE).eval()
        text, audio, lengths = _teacher_batch()
        out = model(text, audio, lengths)
        assert out.shape == (B, SEQ, 1), \
            f"embed_mode={embed_mode!r}: expected ({B},{SEQ},1), got {out.shape}"

    def test_predict_proba_range(self):
        """predict_proba returns values in [0, 1]."""
        model = build_teacher(device=DEVICE).eval()
        text, audio, lengths = _teacher_batch()
        with torch.no_grad():
            proba = model.predict_proba(text, audio, lengths)
        assert proba.shape == (B, SEQ, 1)
        assert proba.min() >= 0.0 and proba.max() <= 1.0, \
            f"predict_proba out of range: min={proba.min():.4f} max={proba.max():.4f}"

    def test_gradients_flow(self):
        """Loss.backward() produces non-None gradients for all parameters."""
        model = build_teacher(device=DEVICE).train()
        text, audio, lengths = _teacher_batch()
        out  = model(text, audio, lengths).squeeze(-1)          # [B, SEQ]
        loss = out.mean()
        loss.backward()
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"

    def test_load_teacher_no_checkpoint(self):
        """load_teacher(None) returns a model with default architecture."""
        model = load_teacher(None, device=DEVICE)
        assert isinstance(model, TeacherModel)

    @pytest.mark.parametrize("arch_variant", ["v1", "v2", "v3"])
    def test_arch_variants(self, arch_variant: str):
        """All arch_variants produce the correct output shape."""
        model = TeacherModel(arch_variant=arch_variant).to(DEVICE).eval()
        text, audio, lengths = _teacher_batch()
        out = model(text, audio, lengths)
        assert out.shape == (B, SEQ, 1), \
            f"arch_variant={arch_variant!r}: got {out.shape}"


# ---------------------------------------------------------------------------
# Student model sub-modules
# ---------------------------------------------------------------------------

class TestStudentSubModules:

    def test_keyword_text_branch_shape(self):
        """KeywordTextBranch maps [B, TEXT_DIM] → [B, 32]."""
        branch = KeywordTextBranch(TEXT_DIM, out_dim=32).eval()
        x   = torch.randn(B, TEXT_DIM)
        out = branch(x)
        assert out.shape == (B, 32), f"Expected ({B}, 32), got {out.shape}"

    def test_mfcc_conv_branch_shape(self):
        """MFCCConvBranch maps [B, N_FRAMES, MFCC_DIM] → [B, 32]."""
        branch = MFCCConvBranch(MFCC_DIM, out_dim=32).eval()
        x   = torch.randn(B, N_FRAMES, MFCC_DIM)
        out = branch(x)
        assert out.shape == (B, 32), f"Expected ({B}, 32), got {out.shape}"

    def test_mfcc_branch_wrong_order_fails(self):
        """Passing [B, MFCC_DIM, N_FRAMES] (wrong axis order) should raise."""
        branch = MFCCConvBranch(MFCC_DIM, out_dim=32).eval()
        x_wrong = torch.randn(B, MFCC_DIM, N_FRAMES)   # transposed — wrong
        with pytest.raises(Exception):
            branch(x_wrong)

    def test_mfcc_branch_gradient(self):
        """MFCCConvBranch gradients flow back through all conv layers."""
        branch = MFCCConvBranch(MFCC_DIM, out_dim=32).train()
        x   = torch.randn(B, N_FRAMES, MFCC_DIM, requires_grad=True)
        out = branch(x).sum()
        out.backward()
        assert x.grad is not None


# ---------------------------------------------------------------------------
# Student model
# ---------------------------------------------------------------------------

class TestStudentModel:

    def test_output_shape(self):
        """StudentModel returns [B, 1]."""
        model = build_student(device=DEVICE).eval()
        text, audio, context, position = _student_batch()
        with torch.no_grad():
            out = model(text, audio, context, position)
        assert out.shape == (B, 1), f"Expected ({B}, 1), got {out.shape}"

    def test_batch_size_one(self):
        """Works with batch_size=1 (the normal extension call)."""
        model = build_student(device=DEVICE).eval()
        text, audio, context, position = _student_batch(batch=1)
        with torch.no_grad():
            out = model(text, audio, context, position)
        assert out.shape == (1, 1)

    def test_predict_proba_range(self):
        """predict_proba output is in [0, 1]."""
        model = build_student(device=DEVICE).eval()
        text, audio, context, position = _student_batch()
        with torch.no_grad():
            proba = model.predict_proba(text, audio, context, position)
        assert proba.shape == (B, 1)
        assert proba.min() >= 0.0 and proba.max() <= 1.0, \
            f"predict_proba out of range: min={proba.min():.4f} max={proba.max():.4f}"

    def test_score_returns_scalar(self):
        """score() convenience wrapper returns a Python float."""
        model = build_student(device=DEVICE).eval()
        text, audio, context, position = _student_batch(batch=1)
        with torch.no_grad():
            s = model.score(text, audio, context, position)
        assert isinstance(s, float), f"Expected float, got {type(s)}"
        assert 0.0 <= s <= 1.0

    def test_gradients_flow(self):
        """Loss.backward() produces non-None gradients for all parameters."""
        model = build_student(device=DEVICE).train()
        text, audio, context, position = _student_batch()
        out  = model(text, audio, context, position).squeeze(-1)
        loss = out.mean()
        loss.backward()
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"

    def test_load_student_no_checkpoint(self):
        """load_student(None) returns a StudentModel with random weights."""
        model = load_student(None, device=DEVICE)
        assert isinstance(model, StudentModel)

    def test_zero_keyword_vec_still_runs(self):
        """All-zero keyword vector (no sponsor keywords) runs without error."""
        model = build_student(device=DEVICE).eval()
        text     = torch.zeros(1, TEXT_DIM)
        audio    = torch.zeros(1, N_FRAMES, MFCC_DIM)
        context  = torch.zeros(1, K_CONTEXT)
        position = torch.zeros(1, 1)
        with torch.no_grad():
            out = model(text, audio, context, position)
        assert out.shape == (1, 1)
        assert 0.0 <= float(out[0, 0]) <= 1.0  # after sigmoid — no NaN

    def test_context_at_boundaries(self):
        """Context values of 0 and 1 (extreme priors) don't cause NaN."""
        model = build_student(device=DEVICE).eval()
        text, audio, _, position = _student_batch(batch=2)
        ctx_zero = torch.zeros(2, K_CONTEXT)
        ctx_one  = torch.ones(2, K_CONTEXT)
        with torch.no_grad():
            out_zero = model(text, audio, ctx_zero, position)
            out_one  = model(text, audio, ctx_one,  position)
        assert not torch.isnan(out_zero).any(), "NaN with context=0"
        assert not torch.isnan(out_one).any(),  "NaN with context=1"
