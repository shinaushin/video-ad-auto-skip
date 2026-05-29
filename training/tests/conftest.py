"""conftest.py — Shared pytest fixtures for the training test suite.

All tests import from training/src/ — conftest.py adds it to sys.path so
tests can be run from any working directory:

    pytest training/tests/
    pytest training/tests/ -v
    pytest training/tests/test_models.py -k "teacher"
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# sys.path — make training/src importable regardless of cwd
# ---------------------------------------------------------------------------

_TESTS_DIR = Path(__file__).resolve().parent
_SRC_DIR   = _TESTS_DIR.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

# Constants are imported here so fixtures can reference them without
# re-importing in every test file.
from models import DISTILBERT_DIM, K_CONTEXT, MFCC_DIM, N_FRAMES, TEXT_DIM, WHISPER_DIM  # noqa: E402


# ---------------------------------------------------------------------------
# Fake .npz factory
# ---------------------------------------------------------------------------

def make_fake_npz(
    path: Path,
    n_windows: int = 8,
    seed: int = 0,
    text_dim: int = TEXT_DIM,
    mfcc_dim: int = MFCC_DIM,
    n_mfcc_frames: int = N_FRAMES,
) -> None:
    """Write a synthetic per-video .npz matching the real SponsorDataset schema.

    Positive windows (label=1) have embeddings shifted by +2 so the model has
    a separable signal — necessary for overfit tests to converge.
    """
    rng    = np.random.RandomState(seed)
    labels = np.array([i % 2 for i in range(n_windows)], dtype=np.int8)

    text_embs  = rng.randn(n_windows, DISTILBERT_DIM).astype(np.float32)
    audio_embs = rng.randn(n_windows, WHISPER_DIM).astype(np.float32)
    text_embs[labels == 1]  += 2.0
    audio_embs[labels == 1] += 2.0

    # Keyword vecs: positives get a non-zero feature at index 0
    keyword_vecs = np.zeros((n_windows, text_dim), dtype=np.float32)
    keyword_vecs[labels == 1, 0] = 1.0

    # MFCC: positives have a slight energy bump
    mfcc = rng.randn(n_windows, n_mfcc_frames, mfcc_dim).astype(np.float32)
    mfcc[labels == 1] += 0.5

    np.savez(
        path,
        segments         = rng.rand(n_windows, 2).astype(np.float32),
        audio_embs       = audio_embs,
        text_embs        = text_embs,
        text_keyword_vecs= keyword_vecs,
        mfcc_features    = mfcc,
        labels           = labels,
        video_duration   = np.float32(float(n_windows) * 5.0),
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def fake_cache_dir(tmp_path: Path) -> Path:
    """Temporary directory with 4 synthetic .npz files (2 windows each)."""
    for i in range(4):
        make_fake_npz(tmp_path / f"video_{i:04d}.npz", n_windows=6, seed=i)
    return tmp_path


@pytest.fixture()
def fake_cache_dir_large(tmp_path: Path) -> Path:
    """Temporary directory with 6 synthetic .npz files (12 windows each).

    Larger dataset used by overfit tests — more windows = more gradient signal.
    """
    for i in range(6):
        make_fake_npz(tmp_path / f"video_{i:04d}.npz", n_windows=12, seed=i)
    return tmp_path
