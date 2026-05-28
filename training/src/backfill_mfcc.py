#!/usr/bin/env python3
"""backfill_mfcc.py — Add real MFCC features to existing Phase 1 cache files.

Iterates over every .npz in the cache directory.  For files that are missing
the ``mfcc_features`` array (i.e. they pre-date compute_mfcc_features()), this
script:

  1. Downloads the audio for that video via yt-dlp (to a temp directory).
  2. Computes real MFCC features at 1 fps with a 30-frame rolling buffer.
  3. Patches the .npz in-place by re-writing it with the new array appended.

Files that already contain ``mfcc_features`` are skipped unless --force is
passed.

Usage:
    python3 training/src/backfill_mfcc.py \\
        --cache  /tmp/embeddings_cache \\
        --workers 4 \\
        [--cookies path/to/cookies.txt] \\
        [--cookies-from-browser chrome] \\
        [--force]

    # Dry-run: just report how many files need backfilling
    python3 training/src/backfill_mfcc.py --cache /tmp/embeddings_cache --dry-run

On Vertex AI (GCS-backed cache) you will typically:
  1. gsutil -m rsync -r gs://yt-sponsor-cache/embeddings /tmp/embeddings_cache
  2. python3 training/src/backfill_mfcc.py --cache /tmp/embeddings_cache --workers 4
  3. gsutil -m rsync -r /tmp/embeddings_cache gs://yt-sponsor-cache/embeddings
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Inline constants (must match data_pipeline.py)
# ---------------------------------------------------------------------------
MFCC_FPS     = 1
N_MFCC_FRAMES = 30
MFCC_DIM     = 13

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Audio download (minimal — mirrors data_pipeline._download_audio)
# ---------------------------------------------------------------------------

def _download_audio(
    video_id: str,
    tmp_dir: Path,
    cookies_path: Optional[Path],
    cookies_from_browser: Optional[str],
    sleep_interval: float = 0.0,
) -> Optional[Path]:
    """Download audio-only stream for ``video_id`` into ``tmp_dir``.

    Returns the path to the downloaded WAV file, or None on failure.
    """
    url = f"https://www.youtube.com/watch?v={video_id}"
    out_template = str(tmp_dir / "audio.%(ext)s")
    cmd = [
        "yt-dlp",
        "--format", "bestaudio/best",
        "--extract-audio", "--audio-format", "wav",
        "--audio-quality", "0",
        "--output", out_template,
        "--no-playlist",
    ]
    if cookies_path:
        cmd += ["--cookies", str(cookies_path)]
    if cookies_from_browser:
        cmd += ["--cookies-from-browser", cookies_from_browser]
    if sleep_interval > 0:
        cmd += ["--sleep-interval", str(sleep_interval)]
    cmd += ["--", url]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    except FileNotFoundError:
        log.error("yt-dlp not found on PATH — install with: pip install yt-dlp  or  brew install yt-dlp")
        return None
    except subprocess.TimeoutExpired:
        log.warning("[%s] yt-dlp timed out", video_id)
        return None

    if result.returncode != 0:
        # Log the first line of stderr so failures are diagnosable
        stderr_first = (result.stderr or "").strip().splitlines()
        reason = stderr_first[0] if stderr_first else "(no stderr)"
        log.warning("[%s] yt-dlp failed (rc=%d): %s", video_id, result.returncode, reason)
        return None

    # Find the downloaded file
    candidates = list(tmp_dir.glob("audio.*"))
    if not candidates:
        return None

    audio_path = candidates[0]

    # Convert to 16 kHz mono WAV if not already
    wav_path = tmp_dir / "audio_16k.wav"
    ffmpeg_cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(audio_path),
        "-ac", "1", "-ar", "16000", "-f", "wav",
        str(wav_path),
    ]
    res = subprocess.run(ffmpeg_cmd, capture_output=True, timeout=120)
    if res.returncode == 0 and wav_path.exists():
        return wav_path
    # Fall back to whatever yt-dlp wrote
    return audio_path


# ---------------------------------------------------------------------------
# MFCC computation (mirrors data_pipeline.compute_mfcc_features)
# ---------------------------------------------------------------------------

def compute_mfcc_for_windows(
    audio_path: Path,
    window_starts: np.ndarray,   # float32 [N] — t_start for each window
    n_mfcc: int    = MFCC_DIM,
    fps: int       = MFCC_FPS,
    n_frames: int  = N_MFCC_FRAMES,
) -> np.ndarray:
    """Return float32 [N, n_frames, n_mfcc].  Zeros on librosa failure."""
    N = len(window_starts)
    result = np.zeros((N, n_frames, n_mfcc), dtype=np.float32)

    try:
        import librosa
    except ImportError:
        log.error("librosa is not installed.  Run: pip install librosa")
        sys.exit(1)

    try:
        y, sr = librosa.load(str(audio_path), sr=16000, mono=True)
    except Exception as exc:
        log.warning("librosa.load failed for %s: %s", audio_path, exc)
        return result

    hop_length = sr // fps  # 16 000 samples per frame → 1 frame/sec

    try:
        mfcc_all = librosa.feature.mfcc(
            y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length,
        ).T.astype(np.float32)  # [T, n_mfcc]
    except Exception as exc:
        log.warning("MFCC computation failed: %s", exc)
        return result

    T = len(mfcc_all)

    for i, t_start in enumerate(window_starts):
        end_frame   = int(float(t_start) * fps)
        start_frame = end_frame - n_frames
        if end_frame <= 0:
            continue
        valid_start = max(0, start_frame)
        valid_end   = min(T, end_frame)
        if valid_end <= valid_start:
            continue
        chunk = mfcc_all[valid_start:valid_end]
        k = len(chunk)
        result[i, n_frames - k:] = chunk

    return result


# ---------------------------------------------------------------------------
# Per-file worker
# ---------------------------------------------------------------------------

def backfill_one(
    npz_path: Path,
    cookies_path: Optional[Path],
    cookies_from_browser: Optional[str],
    sleep_interval: float,
    force: bool,
) -> str:
    """Process one npz file.  Returns a status string for logging."""
    video_id = npz_path.stem

    # Load existing data
    try:
        data = dict(np.load(npz_path, allow_pickle=False))
    except Exception as exc:
        return f"LOAD_ERROR ({exc})"

    # Skip if already has MFCC and we're not forcing
    if "mfcc_features" in data and not force:
        return "SKIP (already has mfcc_features)"

    # Read window starts from stored segments [N, 2]
    if "segments" not in data:
        return "SKIP (no segments array)"
    window_starts = data["segments"][:, 0]  # float32 [N]

    # Download audio to a temp dir
    with tempfile.TemporaryDirectory(prefix=f"backfill_{video_id}_") as tmp:
        tmp_dir = Path(tmp)
        audio_path = _download_audio(
            video_id, tmp_dir, cookies_path, cookies_from_browser, sleep_interval,
        )
        if audio_path is None:
            return "AUDIO_FAIL (yt-dlp — see WARNING log above)"

        mfcc_features = compute_mfcc_for_windows(audio_path, window_starts)

    # Check we got non-zero data
    if not np.any(mfcc_features):
        return "MFCC_ZERO (librosa produced all zeros)"

    # Patch the npz in-place: write to a temp file then rename atomically
    data["mfcc_features"] = mfcc_features
    tmp_out = npz_path.with_suffix(".npz.tmp")
    try:
        np.savez_compressed(tmp_out, **data)
        tmp_out.rename(npz_path)
    except Exception as exc:
        tmp_out.unlink(missing_ok=True)
        return f"WRITE_ERROR ({exc})"

    return f"OK  mfcc={mfcc_features.shape}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    p = argparse.ArgumentParser(
        description="Backfill real MFCC features into existing Phase 1 cache .npz files."
    )
    p.add_argument("--cache",   required=True, type=Path,
                   help="Cache directory containing .npz files")
    p.add_argument("--workers", type=int, default=2,
                   help="Parallel workers for audio download + MFCC (default 2)")
    p.add_argument("--cookies", type=Path, default=None,
                   help="Netscape cookies.txt for yt-dlp auth")
    p.add_argument("--cookies-from-browser", dest="cookies_from_browser",
                   type=str, default=None,
                   help="Browser name to extract cookies from (e.g. chrome)")
    p.add_argument("--sleep-interval", dest="sleep_interval",
                   type=float, default=1.0,
                   help="Seconds to sleep between yt-dlp requests (default 1.0)")
    p.add_argument("--force", action="store_true",
                   help="Re-compute MFCC even for files that already have it")
    p.add_argument("--dry-run", action="store_true",
                   help="Just count how many files need backfilling, then exit")
    args = p.parse_args()

    cache_dir = args.cache
    if not cache_dir.is_dir():
        log.error("Cache directory not found: %s", cache_dir)
        sys.exit(1)

    all_npz = sorted(cache_dir.glob("*.npz"))
    log.info("Found %d .npz files in %s", len(all_npz), cache_dir)

    # Filter to files that need processing
    todo: list[Path] = []
    for path in all_npz:
        try:
            with np.load(path, allow_pickle=False) as d:
                has_mfcc = "mfcc_features" in d.files
        except Exception:
            has_mfcc = False
        if not has_mfcc or args.force:
            todo.append(path)

    log.info("%d / %d files need MFCC backfill%s",
             len(todo), len(all_npz),
             " (--force active)" if args.force else "")

    if args.dry_run:
        log.info("Dry-run — exiting without making changes.")
        return

    if not todo:
        log.info("Nothing to do.")
        return

    ok = skip = fail = 0

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                backfill_one,
                path,
                args.cookies,
                args.cookies_from_browser,
                args.sleep_interval,
                args.force,
            ): path
            for path in todo
        }
        for i, fut in enumerate(as_completed(futures), 1):
            path   = futures[fut]
            status = fut.result()
            prefix = status.split()[0]
            if prefix == "OK":
                ok += 1
            elif prefix == "SKIP":
                skip += 1
            else:
                fail += 1
            log.info("[%d/%d] %s  %s", i, len(todo), path.stem, status)

    log.info("Backfill complete — OK=%d  SKIP=%d  FAIL=%d", ok, skip, fail)


if __name__ == "__main__":
    main()
