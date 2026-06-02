"""inspect_teacher_logits.py — Inspect the distribution of teacher logits.

Runs the teacher over a sample of cached videos and prints statistics about
the logit distribution.  If logits are all concentrated near zero the KD
soft labels give the student very little gradient signal.

Usage:
    python training/src/inspect_teacher_logits.py \
        --checkpoint /path/to/teacher_best.pt \
        --cache-dir  training/cache/embeddings \
        --n-videos   200
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from data_pipeline import SponsorDataset
from models import load_teacher


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, type=str)
    p.add_argument("--cache-dir",  required=True, type=Path)
    p.add_argument("--n-videos",   type=int, default=200)
    p.add_argument("--device",     type=str, default="cpu")
    args = p.parse_args()

    device = args.device
    print(f"\nLoading teacher from: {args.checkpoint}")
    teacher = load_teacher(args.checkpoint, device=device)
    teacher.eval()

    all_ids = [p.stem for p in sorted(args.cache_dir.glob("*.npz"))]
    sample_ids = all_ids[: args.n_videos]
    print(f"Running inference on {len(sample_ids)} videos …\n")

    ds = SponsorDataset(args.cache_dir, sample_ids, require_audio=False)

    all_logits:  list[float] = []
    pos_logits:  list[float] = []
    neg_logits:  list[float] = []

    items = list(ds)
    batch_size = 128
    vid_counter: dict[str, int] = {}

    for i in range(0, len(items), batch_size):
        batch = items[i : i + batch_size]
        text  = torch.stack([torch.from_numpy(it["text_emb"])  for it in batch]).to(device)
        audio = torch.stack([torch.from_numpy(it["audio_emb"]) for it in batch]).to(device)
        with torch.no_grad():
            logits = teacher(
                text.unsqueeze(1), audio.unsqueeze(1)
            ).squeeze(-1).squeeze(-1).cpu().tolist()

        for item, logit in zip(batch, logits):
            all_logits.append(logit)
            if item["label"] == 1:
                pos_logits.append(logit)
            elif item["label"] == 0:
                neg_logits.append(logit)

    all_logits = np.array(all_logits, dtype=np.float32)
    pos_logits = np.array(pos_logits, dtype=np.float32) if pos_logits else np.array([])
    neg_logits = np.array(neg_logits, dtype=np.float32) if neg_logits else np.array([])

    def stats(arr, name):
        if len(arr) == 0:
            print(f"  {name}: (empty)")
            return
        probs = 1 / (1 + np.exp(-arr))
        print(f"  {name} (n={len(arr):,})")
        print(f"    logit  mean={arr.mean():.3f}  std={arr.std():.3f}  "
              f"min={arr.min():.3f}  max={arr.max():.3f}")
        print(f"    prob   mean={probs.mean():.3f}  "
              f"pct>0.5={100*(probs>0.5).mean():.1f}%  "
              f"pct>0.6={100*(probs>0.6).mean():.1f}%")

    print("─" * 60)
    print("Logit / probability distribution")
    print("─" * 60)
    stats(all_logits, "ALL windows")
    print()
    stats(pos_logits, "SPONSOR windows (label=1)")
    print()
    stats(neg_logits, "NON-SPONSOR windows (label=0)")

    print()
    print("─" * 60)
    print("Separation check")
    print("─" * 60)
    if len(pos_logits) and len(neg_logits):
        sep = pos_logits.mean() - neg_logits.mean()
        print(f"  Mean logit gap (pos - neg): {sep:.3f}")
        if sep < 1.0:
            print("  ⚠  Gap < 1.0 — teacher logits are not well separated.")
            print("     KD soft labels will give the student weak gradient signal.")
        elif sep < 2.0:
            print("  ~  Gap 1–2 — moderate separation. KD should work but signal is limited.")
        else:
            print("  ✓  Gap > 2.0 — good separation. KD soft labels are informative.")
    print()


if __name__ == "__main__":
    main()
