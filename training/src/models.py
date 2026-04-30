"""models.py — Teacher and Student model architectures for sponsor-segment detection.

Teacher (used on Kaggle for training):
    DistilBERT[CLS] → Linear(768→384)
    Whisper-tiny encoder → mean-pool → Linear(384→384)
    CrossAttention(384) fuses the two branches
    BiLSTM(384 hidden, 2 layers) over the window sequence
    Linear(768→1) classifier (sigmoid output)

Student (ships in the Chrome extension via ONNX Runtime Web):
    Text branch  : Linear(64→32)                            (keyword indicator vector input)
    Audio branch : Conv1d(1, 32, k=3) → ReLU → Conv1d(32, 64, k=3) → ReLU → AdaptiveAvgPool1d(1)
                   → Linear(64→32)                          (MFCC frame sequence input)
    Fusion MLP   : Linear(64→32) → ReLU → Linear(32→16) → ReLU → Linear(16→1) (sigmoid)

Input shapes (student ONNX):
    text_input   float32 [batch, 64]
    audio_input  float32 [batch, N_FRAMES, 13]   (N_FRAMES = 30, MFCC channels = 13)

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
TEXT_DIM = 64

#: MFCC coefficient dimension per frame.
MFCC_DIM = 13

#: Number of buffered MFCC frames for the CNN (must match N_MFCC_FRAMES in extension).
N_FRAMES = 30

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
    """Single-head cross-attention: text queries, audio keys/values.

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
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        # Project both modalities to the same dimension.
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

        # Cross-attention fusion (text attends to audio).
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

        # Project modalities.  [batch, seq_len, proj_dim]
        t = self.text_proj(text_embs)
        a = self.audio_proj(audio_embs)

        # Apply cross-attention per window (reshape to [batch*seq, 1, proj_dim]).
        t_flat = t.view(batch * seq_len, 1, -1)
        a_flat = a.view(batch * seq_len, 1, -1)
        fused_flat = self.cross_attn(t_flat, a_flat)           # [batch*seq, 1, proj_dim]
        fused = fused_flat.view(batch, seq_len, -1)            # [batch, seq_len, proj_dim]

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

    Input:  float32 [batch, N_FRAMES, MFCC_DIM]  (13 MFCC coefficients per frame)
    Output: float32 [batch, 32]
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
            nn.Conv1d(mfcc_dim, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)  # → [batch, 64, 1]
        self.proj = nn.Sequential(
            nn.Linear(64, out_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, N_FRAMES, MFCC_DIM]
        x = x.permute(0, 2, 1)              # → [batch, MFCC_DIM, N_FRAMES]
        x = self.conv(x)                    # → [batch, 64, N_FRAMES]
        x = self.pool(x).squeeze(-1)        # → [batch, 64]
        return self.proj(x)                 # → [batch, 32]


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
        self.fusion = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(
        self,
        text_input: torch.Tensor,
        audio_input: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            text_input:   [batch, 64]
            audio_input:  [batch, N_FRAMES, 13]

        Returns:
            logits [batch, 1]
        """
        text_feat = self.text_branch(text_input)    # [batch, 32]
        audio_feat = self.audio_branch(audio_input) # [batch, 32]
        combined = torch.cat([text_feat, audio_feat], dim=-1)  # [batch, 64]
        return self.fusion(combined)                # [batch, 1]

    def predict_proba(
        self,
        text_input: torch.Tensor,
        audio_input: torch.Tensor,
    ) -> torch.Tensor:
        """Return sigmoid probabilities [batch, 1]."""
        return torch.sigmoid(self.forward(text_input, audio_input))

    @torch.no_grad()
    def score(self, text_input: torch.Tensor, audio_input: torch.Tensor) -> float:
        """Return a single float confidence score (convenience wrapper)."""
        return float(self.predict_proba(text_input, audio_input).squeeze())


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
    """Load a TeacherModel from a .pt checkpoint file."""
    model = build_teacher(device=device)
    if checkpoint_path:
        state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state["model_state_dict"])
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
    text_in = torch.randn(B, TEXT_DIM).to(device)
    audio_in = torch.randn(B, N_FRAMES, MFCC_DIM).to(device)
    logits_s = student(text_in, audio_in)
    print(f"Student output: {logits_s.shape}")  # [2, 1]
    assert logits_s.shape == (B, 1), "Student shape mismatch"

    n_teacher = sum(p.numel() for p in teacher.parameters())
    n_student = sum(p.numel() for p in student.parameters())
    print(f"Teacher params: {n_teacher:,}")
    print(f"Student params: {n_student:,}")
    print("Smoke test passed.")
