"""models.py — Teacher and Student model architectures for sponsor-segment detection.

Teacher (used on Kaggle for training):
    DistilBERT[CLS] → Linear(768→384)
    Whisper-tiny encoder → mean-pool → Linear(384→384)
    CrossAttention(384) fuses the two branches  [v1: single layer; v2: 2-layer + FFN]
    BiLSTM(384 hidden, 2 layers) over the window sequence
    Linear(768→1) classifier (sigmoid output)

Student (ships in the Chrome extension via ONNX Runtime Web):
    Text branch  : Linear(128→32)                           (keyword indicator vector input)
    Audio branch : Conv1d(1, 32, k=3) → ReLU → Conv1d(32, 64, k=3) → ReLU → AdaptiveAvgPool1d(1)
                   → Linear(64→32)                          (MFCC frame sequence input)
    Fusion MLP   : Linear(68→32) → ReLU → Linear(32→16) → ReLU → Linear(16→1) (sigmoid)
                   (68 = 32 text + 32 audio + 3 context + 1 position)

Input shapes (student ONNX):
    text_input      float32 [batch, 128]              keyword indicator vector
    audio_input     float32 [batch, N_FRAMES, 13]     MFCC frame sequence (N_FRAMES=30)
    context_input   float32 [batch, K_CONTEXT]        last K sigmoid outputs (K=3)
    position_input  float32 [batch, 1]                relative window position in video [0, 1]

Output:
    float32 [batch, 1]   (sigmoid probability — sponsor confidence)
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Keyword indicator vector dimension (must match feature-extractor.js).
TEXT_DIM = 128

#: MFCC coefficient dimension per frame.
MFCC_DIM = 13

#: Number of buffered MFCC frames for the CNN (must match N_MFCC_FRAMES in extension).
N_FRAMES = 30

#: Number of previous window predictions fed as context to the student fusion MLP.
K_CONTEXT = 3

#: DistilBERT hidden size.
DISTILBERT_DIM = 768

#: Whisper-tiny encoder hidden size.
WHISPER_DIM = 384

#: Shared projection dimension for the teacher cross-attention.
PROJ_DIM = 384


# ---------------------------------------------------------------------------
# Teacher model
# ---------------------------------------------------------------------------


class CrossAttentionFusion(nn.Module):
    """Single-layer cross-attention: text queries, audio keys/values.  (arch_variant='v1')

    Inputs:
        text_proj   [batch, 1, PROJ_DIM]   DistilBERT projection (query)
        audio_proj  [batch, 1, PROJ_DIM]   Whisper projection (key/value)

    Output:
        [batch, 1, PROJ_DIM]   context vector
    """

    def __init__(self, dim: int = PROJ_DIM, n_heads: int = 4) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=n_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, text_proj: torch.Tensor, audio_proj: torch.Tensor) -> torch.Tensor:
        # text_proj  → query
        # audio_proj → key and value
        out, _ = self.attn(query=text_proj, key=audio_proj, value=audio_proj)
        return self.norm(out + text_proj)  # residual


class CrossAttentionBlock(nn.Module):
    """One cross-attention transformer block: Attn → Add&Norm → FFN → Add&Norm.

    Used as the building block for DeepCrossAttentionFusion (arch_variant='v2' / 'v3').

    Deliberately has NO dropout:
    - CrossAttentionFusion (v1) has no internal dropout and reaches 0.82 F1.
    - Adding dropout in the cross-attention pathway compounds with the dropout
      already present in the BiLSTM and classifier, over-regularising the model.
    - With a single-token sequence (one window at a time), attention-weight dropout
      randomly zeros the only weight, killing the entire attention output.
    Regularisation is left entirely to the BiLSTM and classifier layers.
    """

    def __init__(self, dim: int, n_heads: int, ffn_dim: int) -> None:
        super().__init__()
        self.attn  = nn.MultiheadAttention(dim, n_heads, dropout=0.0, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.ffn   = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, dim),
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, query: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(query, kv, kv)
        query = self.norm1(query + attn_out)
        query = self.norm2(query + self.ffn(query))
        return query


class DeepCrossAttentionFusion(nn.Module):
    """Stacked cross-attention layers (arch_variant='v2').

    Stacks n_layers CrossAttentionFusion modules (the same proven v1 module).
    No FFN sublayer — depth comes purely from stacking attention+norm blocks.

    Inputs:
        text_proj   [batch, 1, PROJ_DIM]   DistilBERT projection (query)
        audio_proj  [batch, 1, PROJ_DIM]   Whisper projection (key/value)

    Output:
        [batch, 1, PROJ_DIM]
    """

    def __init__(
        self,
        dim: int = PROJ_DIM,
        n_heads: int = 4,
        n_layers: int = 2,
        **kwargs,  # absorb unused args for forward-compat
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            CrossAttentionFusion(dim=dim, n_heads=n_heads)
            for _ in range(n_layers)
        ])

    def forward(self, text_proj: torch.Tensor, audio_proj: torch.Tensor) -> torch.Tensor:
        x = text_proj
        for layer in self.layers:
            x = layer(x, audio_proj)
        return x


class DeepCrossAttentionFFN(nn.Module):
    """Stacked cross-attention blocks with FFN sublayers (arch_variant='v3').

    Proper transformer-style cross-attention: each block is
    Attn → Add&Norm → FFN → Add&Norm, stacked n_layers times.
    No dropout inside the blocks — see CrossAttentionBlock docstring.

    Inputs:
        text_proj   [batch, 1, PROJ_DIM]   DistilBERT projection (query)
        audio_proj  [batch, 1, PROJ_DIM]   Whisper projection (key/value)

    Output:
        [batch, 1, PROJ_DIM]
    """

    def __init__(
        self,
        dim: int = PROJ_DIM,
        n_heads: int = 4,
        ffn_dim: Optional[int] = None,
        n_layers: int = 2,
        **kwargs,  # absorb unused args for forward-compat
    ) -> None:
        super().__init__()
        ffn_dim = ffn_dim or dim * 2
        self.blocks = nn.ModuleList([
            CrossAttentionBlock(dim, n_heads, ffn_dim)
            for _ in range(n_layers)
        ])

    def forward(self, text_proj: torch.Tensor, audio_proj: torch.Tensor) -> torch.Tensor:
        x = text_proj
        for block in self.blocks:
            x = block(x, audio_proj)
        return x


class TeacherModel(nn.Module):
    """Bimodal teacher model: DistilBERT + Whisper-tiny + CrossAttention + BiLSTM.

    Processes a *sequence* of windows for a single video. The BiLSTM captures
    temporal context (sponsor reads typically span multiple consecutive windows).

    Forward inputs:
        text_embs    float32 [batch, seq_len, DISTILBERT_DIM]
        audio_embs   float32 [batch, seq_len, WHISPER_DIM]
        lengths      int64   [batch]   actual sequence lengths (for packing)

    Forward output:
        logits       float32 [batch, seq_len, 1]   (pre-sigmoid)
    """

    def __init__(
        self,
        distilbert_dim: int = DISTILBERT_DIM,
        whisper_dim: int = WHISPER_DIM,
        proj_dim: int = PROJ_DIM,
        lstm_hidden: int = 192,
        lstm_layers: int = 2,
        dropout: float = 0.1,
        embed_mode: str = "both",
        arch_variant: str = "v1",
        n_cross_attn_layers: int = 2,
        cross_attn_ffn_dim: Optional[int] = None,
    ) -> None:
        """
        Args:
            embed_mode: Which modalities to use.
                "both"       — cross-attention fusion of text + audio (default).
                "text_only"  — text projection → BiLSTM; audio branch unused.
                "audio_only" — audio projection → BiLSTM; text branch unused.
            arch_variant: Cross-attention architecture version.
                "v1" — original single-layer CrossAttentionFusion (default, no FFN).
                "v2" — stacked CrossAttentionBlocks with FFN sublayers (deeper fusion).
            n_cross_attn_layers: Number of stacked blocks for v2 (default 2).
            cross_attn_ffn_dim:  FFN hidden dim for v2 blocks (default proj_dim * 2).
        """
        super().__init__()

        assert embed_mode in ("both", "text_only", "audio_only"), (
            f"embed_mode must be 'both', 'text_only', or 'audio_only'; got {embed_mode!r}"
        )
        assert arch_variant in ("v1", "v2", "v3"), (
            f"arch_variant must be 'v1', 'v2', or 'v3'; got {arch_variant!r}"
        )
        self.embed_mode   = embed_mode
        self.arch_variant = arch_variant

        # Project both modalities to the same dimension.
        # (Both projections are always instantiated so checkpoints stay compatible
        #  regardless of mode; unused branches are simply not called at forward time.)
        self.text_proj = nn.Sequential(
            nn.Linear(distilbert_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.ReLU(),
        )
        self.audio_proj = nn.Sequential(
            nn.Linear(whisper_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.ReLU(),
        )

        # Cross-attention fusion (text attends to audio) — only used in "both" mode.
        # v1: original single-layer fusion (matches all existing checkpoints).
        # v2: deeper stacked blocks with FFN sublayers; dropout tied to model dropout.
        if arch_variant == "v3":
            # Proper transformer blocks: Attn → Add&Norm → FFN → Add&Norm, no dropout.
            self.cross_attn: nn.Module = DeepCrossAttentionFFN(
                dim=proj_dim,
                n_heads=4,
                ffn_dim=cross_attn_ffn_dim,
                n_layers=n_cross_attn_layers,
            )
        elif arch_variant == "v2":
            # Simpler: stacked CrossAttentionFusion layers, no FFN.
            self.cross_attn = DeepCrossAttentionFusion(
                dim=proj_dim,
                n_heads=4,
                n_layers=n_cross_attn_layers,
            )
        else:
            # v1: original single-layer fusion.
            self.cross_attn = CrossAttentionFusion(dim=proj_dim, n_heads=4)

        # BiLSTM for temporal context over the window sequence.
        self.bilstm = nn.LSTM(
            input_size=proj_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        # Classifier head: BiLSTM output is 2×lstm_hidden.
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        text_embs: torch.Tensor,
        audio_embs: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch, seq_len, _ = text_embs.shape

        if self.embed_mode == "text_only":
            # Project text directly; skip audio and cross-attention.
            fused = self.text_proj(text_embs)                  # [batch, seq_len, proj_dim]
        elif self.embed_mode == "audio_only":
            # Project audio directly; skip text and cross-attention.
            fused = self.audio_proj(audio_embs)                # [batch, seq_len, proj_dim]
        else:
            # "both": project both modalities then fuse via cross-attention.
            t = self.text_proj(text_embs)                      # [batch, seq_len, proj_dim]
            a = self.audio_proj(audio_embs)                    # [batch, seq_len, proj_dim]
            # Apply cross-attention per window (reshape to [batch*seq, 1, proj_dim]).
            t_flat = t.view(batch * seq_len, 1, -1)
            a_flat = a.view(batch * seq_len, 1, -1)
            fused_flat = self.cross_attn(t_flat, a_flat)       # [batch*seq, 1, proj_dim]
            fused = fused_flat.view(batch, seq_len, -1)        # [batch, seq_len, proj_dim]

        # BiLSTM temporal modelling.
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                fused, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            lstm_out, _ = self.bilstm(packed)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        else:
            lstm_out, _ = self.bilstm(fused)

        # Classify each window.
        logits = self.classifier(lstm_out)  # [batch, seq_len, 1]
        return logits

    def predict_proba(
        self,
        text_embs: torch.Tensor,
        audio_embs: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return sigmoid probabilities [batch, seq_len, 1]."""
        return torch.sigmoid(self.forward(text_embs, audio_embs, lengths))


# ---------------------------------------------------------------------------
# Student model (ships in browser via ONNX)
# ---------------------------------------------------------------------------


class KeywordTextBranch(nn.Module):
    """Lightweight text branch for the student model.

    Input:  float32 [batch, 64]  keyword indicator vector
    Output: float32 [batch, 32]
    """

    def __init__(self, in_dim: int = TEXT_DIM, out_dim: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MFCCConvBranch(nn.Module):
    """1-D CNN over the MFCC frame sequence for the student audio branch.

    Processes an [N_FRAMES, MFCC_DIM] sequence with 1-D convolutions along
    the time axis, then adaptive-average-pools to a fixed-size embedding.

    Architecture (3 conv layers, 64→128→128 channels):
        Conv1d(13→64, k=3) → BN → ReLU
        Conv1d(64→128, k=3) → BN → ReLU
        Conv1d(128→128, k=3) → BN → ReLU   ← extra layer vs. original
        AdaptiveAvgPool1d(1) → [batch, 128]
        Linear(128→64) → ReLU → Linear(64→out_dim)

    Input:  float32 [batch, N_FRAMES, MFCC_DIM]  (13 MFCC coefficients per frame)
    Output: float32 [batch, out_dim]
    """

    def __init__(
        self,
        mfcc_dim: int = MFCC_DIM,
        out_dim: int = 32,
    ) -> None:
        super().__init__()
        # Conv1d expects [batch, channels, length].
        # Treat MFCC_DIM as input channels, N_FRAMES as the sequence length.
        self.conv = nn.Sequential(
            nn.Conv1d(mfcc_dim, 64,  kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64,       128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128,      128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)  # → [batch, 128, 1]
        self.proj = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, N_FRAMES, MFCC_DIM]
        x = x.permute(0, 2, 1)              # → [batch, MFCC_DIM, N_FRAMES]
        x = self.conv(x)                    # → [batch, 128, N_FRAMES]
        x = self.pool(x).squeeze(-1)        # → [batch, 128]
        return self.proj(x)                 # → [batch, out_dim]


class StudentModel(nn.Module):
    """Lightweight student model — designed for ONNX Runtime Web inference.

    Accepts the same two-input interface that feature-extractor.js provides:
        text_input   float32 [batch, 64]          keyword indicator vector
        audio_input  float32 [batch, N_FRAMES, 13] MFCC frame sequence

    Architecture:
        text_branch  → [batch, 32]
        audio_branch → [batch, 32]
        concat       → [batch, 64]
        fusion MLP:  Linear(64→32) → ReLU → Linear(32→16) → ReLU → Linear(16→1)

    Output:
        logits   float32 [batch, 1]   (pre-sigmoid — apply sigmoid at inference time)
    """

    def __init__(self, dropout: float = 0.2) -> None:
        super().__init__()
        self.text_branch = KeywordTextBranch(TEXT_DIM, 32)
        self.audio_branch = MFCCConvBranch(MFCC_DIM, 32)
        # Fusion MLP input: 32 (text) + 32 (audio) + K_CONTEXT (context) + 1 (position) = 68
        fusion_in = 32 + 32 + K_CONTEXT + 1
        self.fusion = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(fusion_in, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(
        self,
        text_input: torch.Tensor,
        audio_input: torch.Tensor,
        context_input: torch.Tensor,
        position_input: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            text_input:     [batch, TEXT_DIM]         keyword indicator vector
            audio_input:    [batch, N_FRAMES, MFCC_DIM]
            context_input:  [batch, K_CONTEXT]        last K sigmoid predictions
            position_input: [batch, 1]                relative window position [0, 1]

        Returns:
            logits [batch, 1]
        """
        text_feat  = self.text_branch(text_input)    # [batch, 32]
        audio_feat = self.audio_branch(audio_input)  # [batch, 32]
        combined   = torch.cat(
            [text_feat, audio_feat, context_input, position_input], dim=-1
        )                                            # [batch, 68]
        return self.fusion(combined)                 # [batch, 1]

    def predict_proba(
        self,
        text_input: torch.Tensor,
        audio_input: torch.Tensor,
        context_input: torch.Tensor,
        position_input: torch.Tensor,
    ) -> torch.Tensor:
        """Return sigmoid probabilities [batch, 1]."""
        return torch.sigmoid(self.forward(text_input, audio_input, context_input, position_input))

    @torch.no_grad()
    def score(
        self,
        text_input: torch.Tensor,
        audio_input: torch.Tensor,
        context_input: torch.Tensor,
        position_input: torch.Tensor,
    ) -> float:
        """Return a single float confidence score (convenience wrapper)."""
        return float(self.predict_proba(text_input, audio_input, context_input, position_input).squeeze())


# ---------------------------------------------------------------------------
# Model factory helpers
# ---------------------------------------------------------------------------


def build_teacher(pretrained: bool = False, device: str = "cpu") -> TeacherModel:
    """Instantiate the teacher model.

    Args:
        pretrained: (unused for now; placeholder for loading a checkpoint).
        device:     Target device string.

    Returns:
        A TeacherModel in eval mode on ``device``.
    """
    model = TeacherModel().to(device)
    return model


def build_student(device: str = "cpu") -> StudentModel:
    """Instantiate the student model.

    Returns:
        A StudentModel in eval mode on ``device``.
    """
    return StudentModel().to(device)


def load_teacher(checkpoint_path: str | None, device: str = "cpu") -> TeacherModel:
    """Load a TeacherModel from a .pt checkpoint file.

    Reads the ``config`` dict stored inside the checkpoint to reconstruct the
    exact architecture (lstm_hidden, lstm_layers, dropout, arch_variant, …)
    before loading weights.  Falls back to defaults if config is absent.
    """
    if checkpoint_path:
        state = torch.load(checkpoint_path, map_location=device)
        cfg = state.get("config", {})
        model = TeacherModel(
            lstm_hidden=int(cfg.get("lstm_hidden", 192)),
            lstm_layers=int(cfg.get("lstm_layers", 1)),
            dropout=float(cfg.get("dropout", 0.1)),
            embed_mode=cfg.get("embed_mode", "both"),
            arch_variant=cfg.get("arch_variant", "v1"),
        ).to(device)
        model.load_state_dict(state["model_state_dict"])
    else:
        model = build_teacher(device=device)
    model.eval()
    return model


def load_student(checkpoint_path: str | None, device: str = "cpu") -> StudentModel:
    """Load a StudentModel from a .pt checkpoint file."""
    model = build_student(device=device)
    if checkpoint_path:
        state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state["model_state_dict"])
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Quick architecture smoke test
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Teacher
    teacher = build_teacher(device=device).train()
    B, SEQ = 2, 20
    t_emb = torch.randn(B, SEQ, DISTILBERT_DIM).to(device)
    a_emb = torch.randn(B, SEQ, WHISPER_DIM).to(device)
    lengths = torch.tensor([SEQ, SEQ - 5]).to(device)
    logits_t = teacher(t_emb, a_emb, lengths)
    print(f"Teacher output: {logits_t.shape}")  # [2, 20, 1]
    assert logits_t.shape == (B, SEQ, 1), "Teacher shape mismatch"

    # Student
    student = build_student(device=device).train()
    text_in     = torch.randn(B, TEXT_DIM).to(device)
    audio_in    = torch.randn(B, N_FRAMES, MFCC_DIM).to(device)
    context_in  = torch.zeros(B, K_CONTEXT).to(device)
    position_in = torch.rand(B, 1).to(device)
    logits_s = student(text_in, audio_in, context_in, position_in)
    print(f"Student output: {logits_s.shape}")  # [2, 1]
    assert logits_s.shape == (B, 1), "Student shape mismatch"

    n_teacher = sum(p.numel() for p in teacher.parameters())
    n_student = sum(p.numel() for p in student.parameters())
    print(f"Teacher params: {n_teacher:,}")
    print(f"Student params: {n_student:,}")
    print("Smoke test passed.")
