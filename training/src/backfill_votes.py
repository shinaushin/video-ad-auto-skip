"""backfill_votes.py — Add per-segment vote data to existing .npz cache files.

The data pipeline was updated to store ``sponsor_segs`` [M, 2] and
``sponsor_seg_votes`` [M] in each video's .npz file so that training can
re-label windows at any min_votes threshold without re-running the full
embedding pipeline.  This script retroactively adds those arrays to cache
files that were written before the update.

Usage:
    python backfill_votes.py \\
        --csv  training/data/sponsorTimes.csv \\
        --cache-dir training/cache/embeddings

    # Dry run (report what would change, write nothing):
    python backfill_votes.py --csv ... --cache-dir ... --dry-run

    # Only update files that are already missing vote arrays (default behaviour):
    python backfill_votes.py --csv ... --cache-dir ...

    # Force re-write even for files that already have vote arrays:
    python backfill_votes.py --csv ... --cache-dir ... --force

After running this script, pass ``--min-votes N`` to submit_train.sh or
submit_tune.sh to filter labels at training time.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import tempfile
from pathlib import Path

import numpy as np

from data_pipeline import parse_sponsorblock_csv

log = logging.getLogger(__name__)


def backfill(
    csv_path: Path,
    cache_dir: Path,
    dry_run: bool = False,
    force: bool = False,
) -> None:
    """Iterate over all .npz files in ``cache_dir`` and inject vote arrays."""
    # Parse all segments (min_votes=1) to get votes for every segment.
    log.info("Parsing SponsorBlock CSV (min_votes=1): %s", csv_path)
    sponsor_map = parse_sponsorblock_csv(csv_path, min_votes=1)
    log.info("Loaded %d videos from CSV", len(sponsor_map))

    npz_files = sorted(cache_dir.glob("*.npz"))
    log.info("Found %d .npz files in %s", len(npz_files), cache_dir)

    updated = 0
    already_done = 0
    missing_from_csv = 0
    errors = 0

    for npz_path in npz_files:
        video_id = npz_path.stem
        try:
            data = np.load(npz_path)
            existing_keys = set(data.files)

            has_vote_arrays = "sponsor_segs" in existing_keys and "sponsor_seg_votes" in existing_keys

            if has_vote_arrays and not force:
                already_done += 1
                log.debug("Already has vote arrays — skipping: %s", video_id)
                continue

            # Build the vote arrays from the CSV.
            segs = sponsor_map.get(video_id, [])  # list of (start, end, votes)
            sponsor_segs_arr = np.array(
                [[s, e] for (s, e, v) in segs], dtype=np.float32
            ).reshape(-1, 2)
            sponsor_seg_votes_arr = np.array(
                [v for (s, e, v) in segs], dtype=np.int32
            )

            if len(segs) == 0 and video_id not in sponsor_map:
                missing_from_csv += 1
                log.debug("Video not in CSV (no sponsor segments recorded): %s", video_id)
                # Still write empty arrays so we know this file has been backfilled.

            if dry_run:
                log.info(
                    "[dry-run] Would update %s: %d segs  votes=%s",
                    video_id, len(segs),
                    str(sponsor_seg_votes_arr[:5].tolist()) + ("…" if len(segs) > 5 else ""),
                )
                updated += 1
                continue

            # Build the updated npz dict.
            new_data = {k: data[k] for k in existing_keys}
            new_data["sponsor_segs"] = sponsor_segs_arr
            new_data["sponsor_seg_votes"] = sponsor_seg_votes_arr

            # Write atomically via a temp file in the same directory.
            tmp_fd, tmp_path = tempfile.mkstemp(dir=cache_dir, suffix=".tmp.npz")
            try:
                import os
                os.close(tmp_fd)
                np.savez_compressed(tmp_path, **new_data)
                shutil.move(tmp_path, npz_path)
            except Exception:
                try:
                    import os
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise

            updated += 1
            if updated % 100 == 0:
                log.info("Progress: %d updated so far…", updated)

        except Exception as exc:
            log.warning("Failed to update %s: %s", video_id, exc)
            errors += 1

    action = "Would update" if dry_run else "Updated"
    log.info(
        "Done. %s=%d  already_done=%d  missing_from_csv=%d  errors=%d",
        action, updated, already_done, missing_from_csv, errors,
    )
    if dry_run:
        log.info("(Dry run — no files were written)")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    p = argparse.ArgumentParser(
        description="Backfill per-segment vote data into existing .npz cache files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--csv", required=True, type=Path,
        help="Path to sponsorTimes.csv",
    )
    p.add_argument(
        "--cache-dir", required=True, type=Path,
        help="Directory containing per-video .npz cache files",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Report what would be changed without writing any files.",
    )
    p.add_argument(
        "--force", action="store_true",
        help="Re-write vote arrays even for files that already have them.",
    )
    args = p.parse_args()

    if not args.csv.exists():
        p.error(f"CSV not found: {args.csv}")
    if not args.cache_dir.is_dir():
        p.error(f"Cache directory not found: {args.cache_dir}")

    backfill(
        csv_path=args.csv,
        cache_dir=args.cache_dir,
        dry_run=args.dry_run,
        force=args.force,
    )


if __name__ == "__main__":
    main()
