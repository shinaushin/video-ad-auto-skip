"""test_teacher_overfit.py — Sanity-check the teacher training loop.

A correctly implemented model should be able to *memorize* a tiny dataset
and drive the training loss very close to 0.  If it can't, something is
wrong with the model architecture, loss function, or data pipeline.

Run from the repo root:
    python training/tests/test_teacher_overfit.py

Optional flags:
    --n-videos   N    Number of synthetic videos to create  (default: 3)
    --n-windows  N    Windows (time-steps) per video        (default: 8)
    --epochs     N    Max training epochs                    (default: 300)
    --threshold  F    Loss value considered "near zero"      (default: 0.05)
    --verbose        Print loss every 25 epochs
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# Ensure training/src/ is on the import path regardless of cwd.
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_SRC = _HERE.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from data_pipeline import SponsorDataset          # noqa: E402
from models import build_teacher                   # noqa: E402
from train import (                                # noqa: E402
    TeacherSequenceDataset,
    collate_teacher_sequences,
    FocalLoss,
    _run_teacher_epoch,
)


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

def _make_fake_npz(path: Path, n_windows: int = 8, seed: int = 0) -> None:
    """Write a synthetic per-video .npz that matches the real data schema.

    Labels alternate 0/1 so the model must learn a non-trivial pattern
    (pure majority-class prediction won't get loss → 0).
    """
    rng = np.random.RandomState(seed)
    labels = np.array([i % 2 for i in range(n_windows)], dtype=np.int8)

    # Make the embeddings slightly different for label=0 vs label=1 so the
    # model has a real signal to latch onto.
    text_embs  = rng.randn(n_windows, 768).astype(np.float32)
    audio_embs = rng.randn(n_windows, 384).astype(np.float32)
    text_embs[labels == 1]  += 2.0   # shift positives → easily separable
    audio_embs[labels == 1] += 2.0

    np.savez(
        path,
        segments=rng.rand(n_windows, 2).astype(np.float32),
        audio_embs=audio_embs,
        text_embs=text_embs,
        text_keyword_vecs=rng.rand(n_windows, 64).astype(np.float32),
        labels=labels,
        video_duration=np.float32(float(n_windows) * 5.0),
    )


# ---------------------------------------------------------------------------
# Core overfit test
# ---------------------------------------------------------------------------

def run_overfit_test(
    n_videos: int = 3,
    n_windows: int = 8,
    epochs: int = 300,
    threshold: float = 0.05,
    verbose: bool = False,
    real_cache_dir: Path | None = None,
) -> None:
    print(f"\n{'='*60}")
    print(f"  Teacher overfit test")
    print(f"  threshold={threshold}  device=cpu")
    print(f"{'='*60}")

    with tempfile.TemporaryDirectory(prefix="teacher_overfit_") as tmp:
        # --- 1. Real or synthetic data ----------------------------------------
        if real_cache_dir is not None:
            # Pick n_videos real .npz files from the cache directory.
            all_npz = sorted(real_cache_dir.glob("*.npz"))
            if not all_npz:
                print(f"  ERROR: no .npz files found in {real_cache_dir}")
                sys.exit(1)
            chosen = all_npz[:n_videos]
            video_ids = [f.stem for f in chosen]
            cache_dir = real_cache_dir

            # Report label distribution so we can see the real class balance.
            total_windows = sponsor_windows = 0
            for f in chosen:
                data = np.load(f)
                labels = data["labels"]
                total_windows   += len(labels)
                sponsor_windows += int(labels.sum())
            print(f"  Using {len(chosen)} real video(s) from {real_cache_dir}")
            print(f"  Windows: {total_windows} total  "
                  f"({sponsor_windows} sponsor / {total_windows - sponsor_windows} non-sponsor)")
        else:
            cache_dir = Path(tmp)
            video_ids = [f"fake_{i:04d}" for i in range(n_videos)]
            for i, vid in enumerate(video_ids):
                _make_fake_npz(cache_dir / f"{vid}.npz", n_windows=n_windows, seed=i)
            total_windows = n_videos * n_windows
            print(f"  Using {n_videos} synthetic .npz file(s) ({n_windows} windows each)")

        print(f"  epochs={epochs}")
        print()

        # --- 2. Build dataset (all selected videos → training only) ----------
        ds       = SponsorDataset(cache_dir, video_ids)
        train_ds = TeacherSequenceDataset(ds)

        # Full-batch gradient descent: every example seen every epoch.
        loader = DataLoader(
            train_ds,
            batch_size=len(train_ds),
            collate_fn=collate_teacher_sequences,
            shuffle=False,
        )

        # --- 3. Model, optimizer, loss ----------------------------------------
        device    = "cpu"
        model     = build_teacher(device=device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        # γ=0 → pure BCE (simpler gradient signal, easier to memorize).
        criterion = FocalLoss(alpha=0.5, gamma=0.0)

        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Model params: {n_params:,}  |  Training windows: {total_windows}")
        print(f"  Param/data ratio: {n_params / max(total_windows, 1):,.0f}x  (should memorize easily)")
        print()

        # --- 4. Training loop -------------------------------------------------
        first_loss: float | None = None
        last_loss:  float = float("inf")

        for epoch in range(1, epochs + 1):
            last_loss, metrics = _run_teacher_epoch(
                model, loader, optimizer, criterion, device, train=True
            )
            if first_loss is None:
                first_loss = last_loss
                print(f"  Epoch   1  loss={last_loss:.4f}  "
                      f"acc={metrics['accuracy']:.3f}  f1={metrics['f1']:.3f}")

            if verbose and epoch % 25 == 0:
                print(f"  Epoch {epoch:3d}  loss={last_loss:.6f}  "
                      f"acc={metrics['accuracy']:.3f}  f1={metrics['f1']:.3f}")

            if last_loss < threshold:
                print(f"  Epoch {epoch:3d}  loss={last_loss:.6f}  "
                      f"← converged below threshold {threshold}")
                break

        # --- 5. Results -------------------------------------------------------
        print()
        print(f"  Initial loss : {first_loss:.4f}")
        print(f"  Final loss   : {last_loss:.6f}  (threshold: {threshold})")
        reduction = (first_loss - last_loss) / max(first_loss, 1e-9) * 100
        print(f"  Reduction    : {reduction:.1f}%")
        print()

        if last_loss < threshold:
            print("  ✓  PASS — model memorised the training data as expected.")
        else:
            print(f"  ✗  FAIL — loss did not reach {threshold} after {epochs} epochs.")
            print("     This suggests a bug in the model, loss, or data pipeline.")
            sys.exit(1)

    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Overfit-sanity-check for the teacher training loop.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Synthetic data (no real cache needed):
  python test_teacher_overfit.py --verbose

  # 1 real video from the master cache:
  python test_teacher_overfit.py --cache-dir training/cache/embeddings --n-videos 1 --verbose

  # 3 real videos:
  python test_teacher_overfit.py --cache-dir training/cache/embeddings --n-videos 3 --verbose
""",
    )
    p.add_argument("--cache-dir", type=Path, default=None,
                   help="Path to a directory of real .npz files. "
                        "When set, uses real data instead of synthetic.")
    p.add_argument("--n-videos",  type=int,   default=3,
                   help="Number of videos to use (default 3). "
                        "With --cache-dir, picks the first N .npz files.")
    p.add_argument("--n-windows", type=int,   default=8,
                   help="Windows per synthetic video (ignored with --cache-dir, default 8).")
    p.add_argument("--epochs",    type=int,   default=300,  help="max epochs (default 300)")
    p.add_argument("--threshold", type=float, default=0.05, help="loss target (default 0.05)")
    p.add_argument("--verbose",   action="store_true",      help="print loss every 25 epochs")
    args = p.parse_args()

    run_overfit_test(
        n_videos=args.n_videos,
        n_windows=args.n_windows,
        epochs=args.epochs,
        threshold=args.threshold,
        verbose=args.verbose,
        real_cache_dir=args.cache_dir,
    )


if __name__ == "__main__":
    main()
