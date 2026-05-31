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
    include_vote_data: bool = True,
    sponsor_votes: int = 5,
) -> None:
    """Write a synthetic per-video .npz matching the real SponsorDataset schema.

    Positive windows (label=1) have embeddings shifted by +2 so the model has
    a separable signal — necessary for overfit tests to converge.

    Args:
        include_vote_data: If True, writes ``sponsor_segs`` and
            ``sponsor_seg_votes`` arrays so vote-weighting and temporal
            consistency filtering code paths are exercised.  Set False to
            simulate old cache files that pre-date vote tracking.
        sponsor_votes: Community vote count assigned to each synthetic sponsor
            segment.  Affects the per-window vote_weight computed by
            SponsorDataset (weight = min(1.0, votes / VOTE_WEIGHT_CAP=10)).
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

    # Window boundaries: each window is WINDOW_SEC=5s wide, sequential.
    window_sec = 5.0
    segments = np.array(
        [[i * window_sec, (i + 1) * window_sec] for i in range(n_windows)],
        dtype=np.float32,
    )

    arrays = dict(
        segments          = segments,
        audio_embs        = audio_embs,
        text_embs         = text_embs,
        text_keyword_vecs = keyword_vecs,
        mfcc_features     = mfcc,
        labels            = labels,
        video_duration    = np.float32(float(n_windows) * window_sec),
    )

    if include_vote_data:
        # Build one sponsor_seg per run of consecutive positive windows.
        # This mirrors the real SponsorBlock schema and lets _compute_vote_weights
        # and _apply_temporal_consistency run on the fake data.
        sponsor_segs  = []
        sponsor_votes_list = []
        i = 0
        while i < n_windows:
            if labels[i] == 1:
                start = segments[i, 0]
                while i < n_windows and labels[i] == 1:
                    i += 1
                end = segments[i - 1, 1]
                sponsor_segs.append([start, end])
                sponsor_votes_list.append(sponsor_votes)
            else:
                i += 1

        if sponsor_segs:
            arrays["sponsor_segs"]      = np.array(sponsor_segs,       dtype=np.float32)
            arrays["sponsor_seg_votes"] = np.array(sponsor_votes_list, dtype=np.int32)
        else:
            # No sponsor windows in this video — write empty arrays.
            arrays["sponsor_segs"]      = np.zeros((0, 2), dtype=np.float32)
            arrays["sponsor_seg_votes"] = np.zeros((0,),   dtype=np.int32)

    np.savez(path, **arrays)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def fake_cache_dir(tmp_path: Path) -> Path:
    """Temporary directory with 4 synthetic .npz files (6 windows each).

    Includes vote data (sponsor_votes=5) so vote-weighting and temporal
    consistency filtering code paths are exercised.
    """
    for i in range(4):
        make_fake_npz(tmp_path / f"video_{i:04d}.npz", n_windows=6, seed=i)
    return tmp_path


@pytest.fixture()
def fake_cache_dir_large(tmp_path: Path) -> Path:
    """Temporary directory with 6 synthetic .npz files (12 windows each).

    Larger dataset used by overfit tests — more windows = more gradient signal.
    Includes vote data so the full cleaning pipeline runs during setup().
    """
    for i in range(6):
        make_fake_npz(tmp_path / f"video_{i:04d}.npz", n_windows=12, seed=i)
    return tmp_path


@pytest.fixture()
def fake_cache_dir_no_votes(tmp_path: Path) -> Path:
    """Temporary directory with old-style .npz files lacking vote arrays.

    Used to verify backward compatibility: SponsorDataset must default to
    vote_weight=1.0 and skip consistency filtering gracefully.
    """
    for i in range(4):
        make_fake_npz(
            tmp_path / f"video_{i:04d}.npz",
            n_windows=6,
            seed=i,
            include_vote_data=False,
        )
    return tmp_path


@pytest.fixture()
def fake_cache_dir_low_votes(tmp_path: Path) -> Path:
    """Cache with sponsor segments that have only 2 votes (below VOTE_WEIGHT_CAP=10).

    Used to verify that low-vote sponsor windows receive weight < 1.0.
    """
    for i in range(4):
        make_fake_npz(
            tmp_path / f"video_{i:04d}.npz",
            n_windows=6,
            seed=i,
            sponsor_votes=2,
        )
    return tmp_path
