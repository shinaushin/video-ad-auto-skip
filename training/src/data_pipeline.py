"""data_pipeline.py — SponsorBlock dataset preparation for sponsor-segment detection.

Downloads and caches per-video embeddings for the teacher model training pipeline.

Pipeline per video:
  1. Parse sponsorTimes.csv → filter sponsor segments (votes ≥ 3, not hidden, duration ≥ 60s)
  2. Download audio-only stream with yt-dlp
  3. Slice each labelled segment (and equal-length non-sponsor windows) with ffmpeg
  4. Compute Whisper-tiny encoder embeddings for each segment (mean-pooled over time)
  5. Fetch YouTube captions and compute DistilBERT [CLS] embeddings per 5-second window
  6. Save per-video .npz to CACHE_DIR, then delete the raw audio

Usage (standalone):
    python data_pipeline.py --csv sponsorTimes.csv --out cache/ --videos 300 --workers 4

Output per video  <CACHE_DIR>/<videoId>.npz:
    segments           float32 [N, 2]     window start/end seconds
    audio_embs         float32 [N, 384]   Whisper mean-pooled encoder embeddings
    text_embs          float32 [N, 768]   DistilBERT [CLS] embeddings
    text_keyword_vecs  float32 [N, 128]   keyword indicator vectors (matching feature-extractor.js)
    labels             int8    [N]        1 = sponsor, 0 = non-sponsor (built with pipeline's min_votes)
    video_duration     float32 scalar
    sponsor_segs       float32 [M, 2]     all SponsorBlock segment start/end times (min_votes=1)
    sponsor_seg_votes  int32   [M]        community vote count per segment

Requirements:
    pip install torch transformers yt-dlp numpy tqdm
    ffmpeg must be on PATH
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterator

import numpy as np

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Minimum SponsorBlock community votes to trust a segment.
MIN_VOTES = 3

#: Minimum video duration in seconds (skip shorts / music clips).
MIN_VIDEO_DURATION_SEC = 60.0

#: Length of each analysis window fed to the model (seconds).
WINDOW_SEC = 5.0

#: MFCC frame rate matches feature-extractor.js: 1 frame per second.
MFCC_FPS = 1

#: Number of MFCC frames buffered for the CNN audio branch.
N_MFCC_FRAMES = 30

#: Sponsor segments with this many votes or more get full loss weight (1.0).
#: Segments below this are downweighted proportionally.
VOTE_WEIGHT_CAP = 10

#: Whisper-tiny encoder hidden size.
WHISPER_DIM = 384

#: DistilBERT hidden size.
DISTILBERT_DIM = 768

#: Keyword patterns matching feature-extractor.js exactly (64 patterns, 4 groups).
#: group 0 → intro phrases (weight 3)
#: group 1 → CTA language  (weight 1.5)
#: group 2 → offer/discount (weight 1.5)
#: group 3 → product endorsement (weight 0.5)
_KEYWORD_PATTERNS: list[tuple[re.Pattern, int]] = []

_RAW_PATTERNS: list[tuple[str, int]] = [
    # group 0 — intro phrases
    (r"\bsponsored\s+by\b", 0), (r"\bthis\s+video\s+is\s+sponsored\b", 0),
    (r"\bbrought\s+to\s+you\s+by\b", 0), (r"\bour\s+sponsor\s+today\b", 0),
    (r"\btoday'?s\s+sponsor\b", 0), (r"\bpartnered\s+with\b", 0),
    (r"\bin\s+partnership\s+with\b", 0), (r"\bpaid\s+partnership\b", 0),
    (r"\baffiliate\b", 0), (r"\bpaid\s+promotion\b", 0),
    (r"\bproud\s+to\s+partner\b", 0), (r"\bthanks?\s+to\s+\w+\s+for\s+sponsoring\b", 0),
    (r"\bsponsor\s+of\s+(this|today'?s)\b", 0), (r"\bcheck\s+out\s+our\s+sponsor\b", 0),
    (r"\bspecial\s+thanks?\s+to\s+our\s+sponsor\b", 0), (r"\bsupported\s+by\b", 0),
    # group 1 — CTA language
    (r"\buse\s+code\b", 1), (r"\bpromo\s+code\b", 1),
    (r"\bdiscount\s+code\b", 1), (r"\bcoupon\s+code\b", 1),
    (r"\blink\s+in\s+(the\s+)?description\b", 1), (r"\blink\s+below\b", 1),
    (r"\bclick\s+the\s+link\b", 1), (r"\bsign\s+up\b", 1),
    (r"\bfree\s+trial\b", 1), (r"\btry\s+it\s+for\s+free\b", 1),
    (r"\bget\s+started\b", 1), (r"\bdownload\s+(the\s+)?(app|it|now)\b", 1),
    (r"\bvisit\s+\w+\s*\.\s*(com|io|org|net)\b", 1), (r"\bgo\s+to\b", 1),
    (r"\bhead\s+(over\s+)?to\b", 1), (r"\bcheck\s+(it\s+)?out\b", 1),
    # group 2 — offer / discount language
    (r"\b\d+\s*%\s*off\b", 2), (r"\bfirst\s+\d+\s*(days?|months?|weeks?)\s+free\b", 2),
    (r"\bexclusive\s+(offer|deal|discount)\b", 2), (r"\blimited\s+(time\s+)?(offer|deal)\b", 2),
    (r"\bspecial\s+(offer|deal|discount|price)\b", 2), (r"\bsave\s+\d+\b", 2),
    (r"\bno\s+credit\s+card\s+required\b", 2), (r"\bcancel\s+anytime\b", 2),
    (r"\bmoney.back\s+guarantee\b", 2), (r"\bfor\s+a\s+limited\s+time\b", 2),
    (r"\bget\s+\d+\s*%\s*off\b", 2), (r"\bfree\s+shipping\b", 2),
    (r"\bfree\s+month\b", 2), (r"\btrial\s+period\b", 2),
    (r"\bno\s+charge\b", 2), (r"\bon\s+sale\b", 2),
    # group 3 — product endorsement
    (r"\bapp\b", 3), (r"\bplatform\b", 3),
    (r"\bservice\b", 3), (r"\bsoftware\b", 3),
    (r"\bproduct\b", 3), (r"\bsubscription\b", 3),
    (r"\btool\b", 3), (r"\bsolution\b", 3),
    (r"\bpremium\b", 3), (r"\bpro\s+plan\b", 3),
    (r"\bplan\b", 3), (r"\baccount\b", 3),
    (r"\bwebsite\b", 3), (r"\bonline\b", 3),
    (r"\bdigital\b", 3), (r"\btech\b", 3),

    # group 4 [64–79] — extended sponsor intro variants (weight 2)
    (r"\bworking with\b", 4), (r"\bcollab(?:oration)?\s+with\b", 4),
    (r"\bbrand\s+(?:deal|collab|partnership)\b", 4), (r"\bendors(?:ed|ement|es)\b", 4),
    (r"\bgifted\s+by\b", 4), (r"\bsponsorship\b", 4),
    (r"\bcommercial\s+(?:break|message)\b", 4), (r"\bword\s+from\s+(?:our|today'?s)?\s*sponsor\b", 4),
    (r"\bmessage\s+from\s+(?:our|today'?s)\b", 4), (r"\bquick\s+(?:word|message|break)\s+from\b", 4),
    (r"\bpaid\s+collaboration\b", 4), (r"\bthis\s+(?:post|video)\s+(?:is\s+)?(?:an?\s+)?ad\b", 4),
    (r"\bproud\s+to\s+work\s+with\b", 4), (r"\bnew\s+sponsor\b", 4),
    (r"\blong.?time\s+sponsor\b", 4), (r"\bofficial\s+partner\b", 4),

    # group 5 [80–95] — extended CTA variants (weight 1)
    (r"\bshop\s+now\b", 5), (r"\bbuy\s+now\b", 5),
    (r"\border\s+(?:now|today)\b", 5), (r"\bget\s+yours?\b", 5),
    (r"\bclaim\s+(?:your|the)\b", 5), (r"\bgrab\s+yours?\b", 5),
    (r"\bfollow\s+(?:the\s+)?link\b", 5), (r"\bjoin\s+(?:now|today)\b", 5),
    (r"\bregister\s+(?:now|today)\b", 5), (r"\bbook\s+(?:a\s+)?(?:demo|call|consultation)\b", 5),
    (r"\bstart\s+(?:your\s+)?(?:free\s+)?trial\b", 5), (r"\bactivate\s+(?:your|the)\b", 5),
    (r"\bunlock\s+(?:your|the)\b", 5), (r"\bexclusive\s+link\b", 5),
    (r"\btap\s+(?:the\s+)?link\b", 5), (r"\bswipe\s+up\b", 5),

    # group 6 [96–111] — extended offer language (weight 1)
    (r"\bhalf\s+(?:off|price)\b", 6), (r"\bfree\s+shipping\b", 6),
    (r"\bno\s+credit\s+card\s+(?:needed|required|necessary)\b", 6), (r"\brisk.?free\b", 6),
    (r"\bat\s+no\s+(?:cost|charge|fee)\b", 6), (r"\bintroductory\s+(?:price|offer|rate)\b", 6),
    (r"\bearly\s+(?:access|bird)\b", 6), (r"\blifetime\s+(?:deal|access|membership)\b", 6),
    (r"\bfor\s+(?:just|only)\s+\$?\d", 6), (r"\bstarting\s+(?:at|from)\s+\$?\d", 6),
    (r"\bper\s+(?:month|year|week)\b", 6), (r"\bfree\s+(?:forever|for\s+life)\b", 6),
    (r"\b\d+.day\s+(?:free\s+)?trial\b", 6), (r"\bno\s+hidden\s+fees?\b", 6),
    (r"\bguaranteed\b", 6), (r"\bbest\s+price\b", 6),

    # group 7 [112–127] — specific product/service categories (weight 0.5)
    (r"\bonline\s+course\b", 7), (r"\bmaster\s*class\b", 7),
    (r"\bweb\s+hosting\b", 7), (r"\bdomain\s+name\b", 7),
    (r"\bpassword\s+manager\b", 7), (r"\bantivirus\b", 7),
    (r"\bcloud\s+storage\b", 7), (r"\bstreaming\s+(?:service|platform)\b", 7),
    (r"\binvesting\s+(?:app|platform)\b", 7), (r"\bfitness\s+(?:app|tracker|plan)\b", 7),
    (r"\bprotein\s+(?:powder|shake)\b", 7), (r"\bsleep\s+(?:app|tracker|aid)\b", 7),
    (r"\bbusiness\s+(?:tool|software)\b", 7), (r"\be.?commerce\b", 7),
    (r"\bsaas\b", 7), (r"\bfintech\b", 7),
]

# Compile and pad to exactly 128 entries.
for _pat, _grp in _RAW_PATTERNS[:128]:
    _KEYWORD_PATTERNS.append((re.compile(_pat, re.IGNORECASE), _grp))
while len(_KEYWORD_PATTERNS) < 128:
    _KEYWORD_PATTERNS.append((re.compile(r"(?!x)x"), 7))  # never-match filler


def keyword_vector(text: str) -> np.ndarray:
    """Return a 128-dim float32 indicator vector (1.0 if pattern matched)."""
    vec = np.zeros(128, dtype=np.float32)
    for i, (pat, _grp) in enumerate(_KEYWORD_PATTERNS):
        if pat.search(text):
            vec[i] = 1.0
    return vec


# ---------------------------------------------------------------------------
# SponsorBlock CSV parsing
# ---------------------------------------------------------------------------


def parse_sponsorblock_csv(
    csv_path: Path,
    category: str = "sponsor",
    min_votes: int = 1,
) -> dict[str, list[tuple[float, float, int]]]:
    """Parse sponsorTimes.csv and return a mapping videoId → list of (start, end, votes) tuples.

    By default (min_votes=1) this returns ALL non-negative-vote segments so that
    per-segment vote counts are preserved in the npz cache.  Pass min_votes > 1
    to pre-filter at parse time (e.g. for audit purposes); the data pipeline
    always parses with min_votes=1 and re-filters at training time using the
    stored vote arrays.

    Filters:
      * category == ``category``
      * votes >= ``min_votes``
      * hidden/shadowhidden == 0
      * actionType == 'skip' or empty

    Args:
        csv_path:  Path to sponsorTimes.csv (downloaded from SponsorBlock mirrors).
        category:  Segment category to keep (default: "sponsor").
        min_votes: Minimum community votes threshold (default 1 — keep all).

    Returns:
        Dict mapping videoId strings to sorted lists of (start_sec, end_sec, votes) triples.
    """
    log.info("Parsing SponsorBlock CSV: %s", csv_path)
    segments: dict[str, list[tuple[float, float, int]]] = {}

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        has_hidden = "hidden" in fieldnames
        has_action = "actionType" in fieldnames

        for row in reader:
            if row.get("category", "") != category:
                continue
            try:
                votes = int(row.get("votes", "0"))
            except ValueError:
                votes = 0
            if votes < min_votes:
                continue
            if has_hidden and row.get("hidden", "0") not in ("0", ""):
                continue
            if has_action and row.get("actionType", "skip") not in ("skip", ""):
                continue

            vid = row.get("videoID", "").strip()
            if not vid:
                continue
            try:
                start = float(row.get("startTime", "0"))
                end = float(row.get("endTime", "0"))
            except ValueError:
                continue
            if end <= start or end - start < 1.0:
                continue

            segments.setdefault(vid, []).append((start, end, votes))

    # Sort segments within each video.
    for vid in segments:
        segments[vid].sort()

    log.info("Found %d videos with %s segments", len(segments), category)
    return segments


# ---------------------------------------------------------------------------
# Caption / transcript fetching via yt-dlp
# ---------------------------------------------------------------------------


def fetch_video_captions(
    video_id: str,
    out_dir: Path | None = None,
    cookies_path: Path | None = None,
    cookies_from_browser: str | None = None,
) -> list[tuple[float, float, str]]:
    """Fetch auto-generated English captions via yt-dlp; return (start, end, text) tuples.

    Uses yt-dlp --write-auto-subs to download a .vtt file into a temp directory
    (or ``out_dir`` if provided), then parses it.  This is far more reliable than
    scraping the YouTube watch-page HTML, which breaks whenever YouTube restructures
    their page JavaScript.

    Returns an empty list if captions are unavailable or yt-dlp fails.
    """
    import contextlib

    @contextlib.contextmanager
    def _tmp_or_provided(d: Path | None):
        if d is not None:
            yield d
        else:
            with tempfile.TemporaryDirectory(prefix="yt_caps_") as t:
                yield Path(t)

    with _tmp_or_provided(out_dir) as work_dir:
        out_tmpl = str(work_dir / f"{video_id}.%(ext)s")
        cmd = [
            "yt-dlp",
            "--quiet", "--no-warnings",
            "--skip-download",           # captions only — no audio
            "--write-auto-subs",         # prefer auto-generated captions
            "--write-subs",              # fall back to manual subs if ASR unavailable
            "--sub-lang", "en",
            "--sub-format", "vtt",
            "--js-runtimes", "node",
            "--remote-components", "ejs:github",
            "-o", out_tmpl,
        ]
        if cookies_from_browser:
            cmd += ["--cookies-from-browser", cookies_from_browser]
        elif cookies_path is not None and cookies_path.exists():
            cmd += ["--cookies", str(cookies_path)]
        cmd += ["--", f"https://www.youtube.com/watch?v={video_id}"]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            log.debug("yt-dlp caption fetch failed for %s (rc=%d): %s",
                      video_id, result.returncode, (result.stderr or result.stdout)[:200])
            return []

        # yt-dlp writes  <id>.en.vtt  or  <id>.en-US.vtt  etc.
        vtt_files = list(work_dir.glob(f"{video_id}*.vtt"))
        if not vtt_files:
            log.debug("No VTT file found for %s after yt-dlp run", video_id)
            return []

        return _parse_vtt(vtt_files[0])


def _parse_vtt(vtt_path: Path) -> list[tuple[float, float, str]]:
    """Parse a WebVTT file; return (start_sec, end_sec, text) tuples.

    Handles YouTube's auto-generated VTT format, which includes duplicate
    lines with word-level timestamps in ``<HH:MM:SS.mmm>`` tags — these are
    stripped, and back-to-back duplicate cues are deduplicated.
    """
    def _ts_to_sec(ts: str) -> float:
        """Convert HH:MM:SS.mmm or MM:SS.mmm to seconds."""
        parts = ts.strip().split(":")
        if len(parts) == 3:
            return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
        elif len(parts) == 2:
            return float(parts[0]) * 60 + float(parts[1])
        return float(parts[0])

    try:
        raw = vtt_path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        log.debug("VTT read failed for %s: %s", vtt_path, exc)
        return []

    cues: list[tuple[float, float, str]] = []
    prev_text = None

    lines = raw.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        # Look for timestamp lines: "00:00:05.000 --> 00:00:10.000"
        if "-->" in line:
            ts_parts = line.split("-->")
            if len(ts_parts) == 2:
                try:
                    start = _ts_to_sec(ts_parts[0].strip())
                    # Strip positioning tags from end timestamp (e.g. "00:05.000 align:start")
                    end_str = ts_parts[1].strip().split()[0]
                    end = _ts_to_sec(end_str)
                except (ValueError, IndexError):
                    i += 1
                    continue

                # Collect text lines until blank line or next timestamp.
                i += 1
                text_lines = []
                while i < len(lines) and lines[i].strip() and "-->" not in lines[i]:
                    text_lines.append(lines[i].strip())
                    i += 1

                text = " ".join(text_lines)
                # Strip inline timestamp tags: <00:00:05.000>, <c>, </c>, etc.
                text = re.sub(r"<[^>]+>", "", text)
                # Unescape HTML entities.
                text = (text.replace("&amp;", "&").replace("&#39;", "'")
                            .replace("&quot;", '"').replace("&lt;", "<")
                            .replace("&gt;", ">").replace("&nbsp;", " "))
                text = text.strip()

                # YouTube ASR emits the same line repeatedly as words arrive;
                # skip exact duplicates of the previous cue.
                if text and text != prev_text and end > start:
                    cues.append((start, end, text))
                    prev_text = text
                continue
        i += 1

    log.debug("Parsed %d caption cues from %s", len(cues), vtt_path.name)
    return cues


# ---------------------------------------------------------------------------
# Window building: align captions to WINDOW_SEC boundaries
# ---------------------------------------------------------------------------


def build_windows(
    cues: list[tuple[float, float, str]],
    video_duration: float,
    sponsor_segments: list[tuple[float, float]],
) -> list[dict]:
    """Divide video into WINDOW_SEC windows; compute label and keyword vector for each.

    A window is labelled sponsor=1 if it overlaps any sponsor segment by ≥ 50%.

    Returns a list of dicts:
        t_start, t_end, text, keyword_vec (np.ndarray 64), label (int)
    """
    n_windows = max(1, int(video_duration / WINDOW_SEC))
    windows = []

    for i in range(n_windows):
        t_start = i * WINDOW_SEC
        t_end = min(t_start + WINDOW_SEC, video_duration)

        # Collect caption text overlapping this window.
        texts = [
            c_text
            for (c_start, c_end, c_text) in cues
            if c_end > t_start and c_start < t_end
        ]
        combined_text = " ".join(texts)

        # Label: sponsor if > 50% overlap with any sponsor segment.
        window_len = t_end - t_start
        label = 0
        for (s_start, s_end) in sponsor_segments:
            overlap = max(0.0, min(t_end, s_end) - max(t_start, s_start))
            if overlap / window_len >= 0.5:
                label = 1
                break

        windows.append(
            {
                "t_start": t_start,
                "t_end": t_end,
                "text": combined_text,
                "keyword_vec": keyword_vector(combined_text),
                "label": label,
            }
        )

    return windows


# ---------------------------------------------------------------------------
# Audio download + Whisper embedding
# ---------------------------------------------------------------------------


def _download_audio(video_id: str, out_dir: Path, cookies_path: Path | None = None,
                    cookies_from_browser: str | None = None,
                    sleep_interval: float = 0.0) -> Path | None:
    """Download audio-only stream with yt-dlp; return path to opus/webm file."""
    out_template = str(out_dir / f"{video_id}.%(ext)s")
    cmd = [
        "yt-dlp",
        "--quiet",
        "--no-warnings",
        "--js-runtimes", "node",          # needed for YouTube n-challenge signature decryption
        "--remote-components", "ejs:github",  # downloads challenge solver script from GitHub
        "-f", "bestaudio[ext=webm]/bestaudio",
        "-o", out_template,
    ]
    if cookies_from_browser:
        cmd += ["--cookies-from-browser", cookies_from_browser]
    elif cookies_path is not None and cookies_path.exists():
        cmd += ["--cookies", str(cookies_path)]
    if sleep_interval > 0:
        cmd += ["--sleep-requests", str(sleep_interval),
                "--sleep-interval", str(sleep_interval)]
    cmd += ["--", f"https://www.youtube.com/watch?v={video_id}"]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        log.warning("yt-dlp failed for %s (rc=%d): %s", video_id, result.returncode,
                    (result.stderr or result.stdout)[:300])
        return None
    # Find the downloaded file.
    for ext in ("webm", "opus", "m4a", "mp3", "ogg"):
        p = out_dir / f"{video_id}.{ext}"
        if p.exists():
            return p
    # Glob fallback.
    matches = list(out_dir.glob(f"{video_id}.*"))
    return matches[0] if matches else None


def _slice_audio_segment(
    audio_path: Path,
    start_sec: float,
    end_sec: float,
    out_path: Path,
) -> bool:
    """Slice a segment from ``audio_path`` using ffmpeg."""
    duration = end_sec - start_sec
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-ss", str(start_sec),
        "-t", str(duration),
        "-i", str(audio_path),
        "-ac", "1", "-ar", "16000", "-f", "wav",
        str(out_path),
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=60)
    return result.returncode == 0 and out_path.exists()


def _get_video_duration(audio_path: Path) -> float | None:
    """Return duration in seconds using ffprobe."""
    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_format", str(audio_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        return None
    try:
        info = json.loads(result.stdout)
        return float(info["format"]["duration"])
    except (json.JSONDecodeError, KeyError, ValueError):
        return None


def compute_whisper_embeddings(
    audio_path: Path,
    windows: list[dict],
    model,
    processor,
    device: str = "cpu",
) -> np.ndarray:
    """Compute Whisper-tiny encoder embeddings for each window; return float32 [N, 384].

    For each window we slice the corresponding audio, run it through the
    Whisper encoder, and mean-pool across the time dimension.
    If a slice fails we use a zero vector.

    Args:
        audio_path: Full downloaded audio file.
        windows:    List of window dicts with t_start / t_end.
        model:      WhisperModel (encoder-only usage).
        processor:  WhisperProcessor for feature extraction.
        device:     "cuda" or "cpu".

    Returns:
        Float32 array of shape [N, WHISPER_DIM].
    """
    import torch

    N = len(windows)
    embs = np.zeros((N, WHISPER_DIM), dtype=np.float32)

    _logged_audio_load_error = False  # log first failure only to avoid spam

    with tempfile.TemporaryDirectory(prefix="whisper_slice_") as tmp:
        tmp_dir = Path(tmp)
        for i, w in enumerate(windows):
            slice_path = tmp_dir / f"slice_{i}.wav"
            ok = _slice_audio_segment(audio_path, w["t_start"], w["t_end"], slice_path)
            if not ok:
                if i == 0:
                    log.warning("ffmpeg slice failed for window 0 of %s — check ffmpeg install", audio_path.name)
                continue

            audio_data = None
            try:
                import soundfile as sf
                audio_data, sr = sf.read(str(slice_path))
                if audio_data.ndim > 1:
                    audio_data = audio_data.mean(axis=1)
            except Exception as sf_exc:
                try:
                    import librosa
                    audio_data, sr = librosa.load(str(slice_path), sr=16000, mono=True)
                except Exception as lb_exc:
                    if not _logged_audio_load_error:
                        log.warning(
                            "Cannot read audio slices — soundfile error: %s | librosa error: %s. "
                            "Install soundfile: pip install soundfile",
                            sf_exc, lb_exc,
                        )
                        _logged_audio_load_error = True
                    continue

            if audio_data is None:
                continue

            try:
                inputs = processor(
                    audio_data,
                    sampling_rate=16000,
                    return_tensors="pt",
                    padding="max_length",  # pad to 3000 mel frames (Whisper's fixed input size)
                )
                input_features = inputs["input_features"].to(device)
                with torch.no_grad():
                    encoder_out = model.encoder(input_features)
                    # encoder_out.last_hidden_state: [1, T, 384]
                    # T=1500 always (Whisper fixed input), but only the first
                    # n_real_frames correspond to actual audio — the rest is
                    # silence padding.  Pool only over real frames to avoid
                    # diluting the embedding with zeros.
                    # Whisper mel: 100 frames/sec; encoder downsamples by 2x → 50/sec.
                    audio_duration_sec = len(audio_data) / 16000
                    n_real_frames = max(1, min(int(audio_duration_sec * 50),
                                               encoder_out.last_hidden_state.shape[1]))
                    emb = encoder_out.last_hidden_state[0, :n_real_frames, :].mean(dim=0)
                embs[i] = emb.cpu().float().numpy()
            except Exception as exc:
                if i == 0:
                    log.warning("Whisper encoder failed for window 0: %s", exc, exc_info=True)
                else:
                    log.debug("Whisper encoding failed for window %d: %s", i, exc)

    return embs


def compute_distilbert_embeddings(
    windows: list[dict],
    model,
    tokenizer,
    device: str = "cpu",
    batch_size: int = 32,
) -> np.ndarray:
    """Compute DistilBERT [CLS] embeddings for each window; return float32 [N, 768].

    Args:
        windows:    List of window dicts with a ``text`` key.
        model:      DistilBertModel.
        tokenizer:  DistilBertTokenizer / AutoTokenizer.
        device:     "cuda" or "cpu".
        batch_size: Number of windows to encode at once.

    Returns:
        Float32 array of shape [N, DISTILBERT_DIM].
    """
    import torch

    N = len(windows)
    embs = np.zeros((N, DISTILBERT_DIM), dtype=np.float32)

    for batch_start in range(0, N, batch_size):
        batch = windows[batch_start : batch_start + batch_size]
        texts = [w["text"] if w["text"].strip() else "[UNK]" for w in batch]
        try:
            enc = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            with torch.no_grad():
                out = model(**enc)
                # DistilBERT: last_hidden_state[:, 0, :] = [CLS]
                cls_emb = out.last_hidden_state[:, 0, :].cpu().float().numpy()
            embs[batch_start : batch_start + len(batch)] = cls_emb
        except Exception as exc:
            log.debug("DistilBERT batch %d failed: %s", batch_start, exc)

    return embs


# ---------------------------------------------------------------------------
# MFCC feature extraction
# ---------------------------------------------------------------------------

#: Number of MFCC coefficients (must match feature-extractor.js n_mfcc).
MFCC_DIM = 13


def compute_mfcc_features(
    audio_path: Path,
    windows: list[dict],
    n_mfcc: int = MFCC_DIM,
    fps: int = MFCC_FPS,
    n_frames: int = N_MFCC_FRAMES,
) -> np.ndarray:
    """Compute MFCC features for each window; return float32 [N, n_frames, n_mfcc].

    For each window we extract the ``n_frames`` MFCC frames ending at
    ``t_start``, matching the rolling buffer that the Chrome extension maintains
    in real-time at 1 frame/second.  Frames earlier than the start of the audio
    are zero-padded (matching the extension's behaviour for the first 30 s).

    Args:
        audio_path: Path to the full downloaded WAV (16 kHz mono).
        windows:    List of window dicts with a ``t_start`` key (seconds).
        n_mfcc:     Number of MFCC coefficients (default 13).
        fps:        MFCC frame rate in Hz (default 1 — matches extension).
        n_frames:   Rolling buffer length in frames (default 30).

    Returns:
        Float32 array of shape [N, n_frames, n_mfcc].  Zero rows for windows
        whose audio could not be processed.
    """
    N = len(windows)
    mfcc_features = np.zeros((N, n_frames, n_mfcc), dtype=np.float32)

    try:
        import librosa
    except ImportError:
        log.warning("librosa not installed — MFCC features will be all zeros. "
                    "Run: pip install librosa")
        return mfcc_features

    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y, sr = librosa.load(str(audio_path), sr=16000, mono=True)
    except Exception as exc:
        log.warning("librosa.load failed for %s: %s", audio_path.name, exc)
        return mfcc_features

    # hop_length for exactly 1 frame/sec at sr=16000.
    hop_length = sr // fps  # 16 000 samples/frame

    try:
        # librosa returns [n_mfcc, T]; we transpose to [T, n_mfcc].
        mfcc_all = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc,
                                         hop_length=hop_length).T.astype(np.float32)
    except Exception as exc:
        log.warning("MFCC computation failed for %s: %s", audio_path.name, exc)
        return mfcc_features

    T = len(mfcc_all)

    for i, w in enumerate(windows):
        # The extension's buffer at decision time holds frames for
        # [t_start - n_frames, t_start).  Frame index = floor(t * fps).
        end_frame   = int(w["t_start"] * fps)   # exclusive upper bound
        start_frame = end_frame - n_frames        # inclusive lower bound

        if end_frame <= 0:
            # Window is at the very start — no history yet; leave as zeros.
            continue

        valid_start = max(0, start_frame)
        valid_end   = min(T, end_frame)

        if valid_end <= valid_start:
            continue

        chunk = mfcc_all[valid_start:valid_end]  # [K, n_mfcc]
        k = len(chunk)
        # Place chunk at the END of the buffer so index [-1] is the most
        # recent frame, matching the extension's ring-buffer layout.
        mfcc_features[i, n_frames - k :] = chunk

    return mfcc_features


# ---------------------------------------------------------------------------
# Per-video processing
# ---------------------------------------------------------------------------


def process_video(
    video_id: str,
    sponsor_segments: list[tuple[float, float, int]],
    cache_dir: Path,
    whisper_model,
    whisper_processor,
    distilbert_model,
    distilbert_tokenizer,
    device: str = "cpu",
    skip_audio: bool = False,
    cookies_path: Path | None = None,
    cookies_from_browser: str | None = None,
    sleep_interval: float = 0.0,
    force: bool = False,
    force_zero_audio: bool = False,
    min_votes: int = MIN_VOTES,
) -> bool:
    """Process one video end-to-end and write <cache_dir>/<video_id>.npz.

    Returns True on success.
    """
    out_path = cache_dir / f"{video_id}.npz"
    if out_path.exists() and not force:
        if force_zero_audio:
            # Only re-process if audio embeddings are all zero.
            data = np.load(out_path)
            if np.any(data["audio_embs"]):
                log.debug("Cache hit (good audio): %s", video_id)
                return True
            log.info("Re-processing %s: audio embeddings are all zero", video_id)
        else:
            log.debug("Cache hit: %s", video_id)
            return True

    # Fetch captions via yt-dlp (auth params shared with audio download).
    cues = fetch_video_captions(
        video_id,
        cookies_path=cookies_path,
        cookies_from_browser=cookies_from_browser,
    )

    with tempfile.TemporaryDirectory(prefix=f"ytdl_{video_id}_") as tmp:
        tmp_dir = Path(tmp)
        audio_path: Path | None = None
        video_duration = 0.0

        if not skip_audio:
            audio_path = _download_audio(video_id, tmp_dir, cookies_path=cookies_path,
                                         cookies_from_browser=cookies_from_browser,
                                         sleep_interval=sleep_interval)
            if audio_path is None:
                log.warning("Audio download failed: %s", video_id)
                return False
            dur = _get_video_duration(audio_path)
            if dur is None or dur < MIN_VIDEO_DURATION_SEC:
                log.debug("Skipping short/unparseable video: %s (%.1fs)", video_id, dur or 0)
                return False
            video_duration = dur
        else:
            # Text-only mode: estimate duration from last cue.
            video_duration = max((c[1] for c in cues), default=0.0) + WINDOW_SEC

        if video_duration < MIN_VIDEO_DURATION_SEC:
            return False

        # Split into qualifying (meets min_votes) and all segments.
        # Labeling uses only qualifying segments; we store all for later re-filtering.
        qualifying_segs = [(s, e) for (s, e, v) in sponsor_segments if v >= min_votes]
        sponsor_segs_arr = np.array(
            [[s, e] for (s, e, v) in sponsor_segments], dtype=np.float32
        ).reshape(-1, 2)
        sponsor_seg_votes_arr = np.array(
            [v for (s, e, v) in sponsor_segments], dtype=np.int32
        )

        # Build windows (labels based on qualifying segments only).
        windows = build_windows(cues, video_duration, qualifying_segs)
        if not windows:
            return False

        N = len(windows)

        # Keyword feature vectors (always computed — no network needed).
        keyword_vecs = np.stack([w["keyword_vec"] for w in windows])  # [N, 64]
        labels = np.array([w["label"] for w in windows], dtype=np.int8)
        t_bounds = np.array(
            [[w["t_start"], w["t_end"]] for w in windows], dtype=np.float32
        )

        # Whisper embeddings (requires audio).
        if audio_path is not None and whisper_model is not None:
            audio_embs = compute_whisper_embeddings(
                audio_path, windows, whisper_model, whisper_processor, device
            )
        else:
            audio_embs = np.zeros((N, WHISPER_DIM), dtype=np.float32)

        # Real MFCC features for student model (requires audio).
        # Shape: [N, N_MFCC_FRAMES, MFCC_DIM] — matches the extension's rolling buffer.
        if audio_path is not None:
            mfcc_features = compute_mfcc_features(audio_path, windows)
        else:
            mfcc_features = np.zeros((N, N_MFCC_FRAMES, MFCC_DIM), dtype=np.float32)

        # Spot-check: log first window's audio embedding snippet so we can
        # visually confirm embeddings are non-zero.
        first_nonzero = next(
            (i for i in range(len(audio_embs)) if np.any(audio_embs[i])), None
        )
        if first_nonzero is not None:
            snippet = audio_embs[first_nonzero, :8]
            log.info(
                "  audio_emb[win=%d] snippet: [%s]",
                first_nonzero,
                ", ".join(f"{v:.4f}" for v in snippet),
            )
        else:
            log.warning("  audio_embs: ALL ZERO for %s — Whisper may have failed", video_id)

        # DistilBERT embeddings.
        if distilbert_model is not None:
            text_embs = compute_distilbert_embeddings(
                windows, distilbert_model, distilbert_tokenizer, device
            )
        else:
            text_embs = np.zeros((N, DISTILBERT_DIM), dtype=np.float32)

        # Spot-check: log first window's text embedding snippet.
        first_nonconst = next(
            (i for i in range(len(text_embs)) if np.any(text_embs[i])), None
        )
        if first_nonconst is not None:
            snippet = text_embs[first_nonconst, :8]
            log.info(
                "  text_emb[win=%d] snippet: [%s]",
                first_nonconst,
                ", ".join(f"{v:.4f}" for v in snippet),
            )
            # Also warn if all windows are identical (caption fetch likely failed).
            if np.allclose(text_embs, text_embs[0]):
                log.warning(
                    "  text_embs: all windows IDENTICAL for %s — captions may be empty ([UNK] fallback)",
                    video_id,
                )
        else:
            log.warning("  text_embs: ALL ZERO for %s — DistilBERT may have failed", video_id)

        # Save.
        np.savez_compressed(
            out_path,
            segments=t_bounds,
            audio_embs=audio_embs,
            text_embs=text_embs,
            text_keyword_vecs=keyword_vecs,
            labels=labels,
            video_duration=np.float32(video_duration),
            # Real MFCC features for student model.
            # mfcc_features:      float32 [N, N_MFCC_FRAMES, MFCC_DIM]
            mfcc_features=mfcc_features,
            # Per-segment vote data — stored for runtime re-labeling at training time.
            # sponsor_segs:       float32 [M, 2]  all sponsor segment start/end times
            # sponsor_seg_votes:  int32   [M]     community vote count per segment
            sponsor_segs=sponsor_segs_arr,
            sponsor_seg_votes=sponsor_seg_votes_arr,
        )
        log.info("Saved %s  (N=%d, sponsors=%d, segs=%d)", out_path.name, N, labels.sum(), len(sponsor_segs_arr))

    return True


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------


def run_batch(
    csv_path: Path,
    cache_dir: Path,
    n_videos: int = 300,
    workers: int = 2,
    device: str = "cpu",
    skip_audio: bool = False,
    seed: int = 42,
    extra_skip: set[str] | None = None,
    cookies_path: Path | None = None,
    cookies_from_browser: str | None = None,
    sleep_interval: float = 0.0,
    force: bool = False,
    force_zero_audio: bool = False,
    min_votes: int = MIN_VOTES,
) -> None:
    """Process up to ``n_videos`` random videos and cache their embeddings.

    Args:
        csv_path:       Path to sponsorTimes.csv.
        cache_dir:      Directory to write per-video .npz files.
        n_videos:       Number of videos to process.
        workers:        Parallel audio-download threads (yt-dlp tasks).
        device:         PyTorch device string ("cuda" or "cpu").
        skip_audio:     Skip audio download/Whisper (text-only mode, faster).
        seed:           Random seed for video sampling.
        cookies_path:   Path to Netscape cookies.txt for yt-dlp authentication.
        sleep_interval: Seconds to sleep between yt-dlp HTTP requests (reduces 429s).
    """
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Parse SponsorBlock CSV with min_votes=1 to capture ALL segments with their
    # vote counts.  We store the full vote data in each npz so that the training
    # loop can re-label windows at any min_votes threshold without re-running the
    # pipeline.  Video selection still respects the caller's min_votes threshold
    # (only include videos that have at least one qualifying sponsor segment).
    sponsor_map = parse_sponsorblock_csv(csv_path, min_votes=1)

    # Sample videos — only include those with ≥1 segment meeting the threshold.
    all_ids = [
        vid for vid, segs in sponsor_map.items()
        if any(v >= min_votes for (_, _, v) in segs)
    ]
    random.seed(seed)
    random.shuffle(all_ids)
    # Skip already-cached (unless --force / --force-zero-audio) and any IDs passed via --skip-ids.
    skip_set = set(extra_skip) if extra_skip else set()
    if force:
        uncached = [v for v in all_ids if v not in skip_set]
        log.info("--force: will overwrite all existing cached files")
    elif force_zero_audio:
        # Include all videos (zero-audio check happens inside process_video).
        uncached = [v for v in all_ids if v not in skip_set]
        log.info("--force-zero-audio: will re-process videos with all-zero audio embeddings")
    else:
        uncached = [v for v in all_ids
                    if not (cache_dir / f"{v}.npz").exists() and v not in skip_set]
    target = uncached[:n_videos]

    log.info("Processing %d videos (%d already cached, %d skipped by ID list)",
             len(target), len(all_ids) - len(uncached) - len(skip_set & set(all_ids)),
             len(skip_set & set(all_ids)))

    # Load models once (heavy — done before threading).
    whisper_model = whisper_processor = None
    distilbert_model = distilbert_tokenizer = None

    if not skip_audio:
        log.info("Loading Whisper-tiny encoder…")
        try:
            from transformers import WhisperModel, WhisperProcessor
            whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
            whisper_model = WhisperModel.from_pretrained("openai/whisper-tiny")
            try:
                whisper_model = whisper_model.to(device)
                # Smoke-test: verify the device actually works for Whisper ops.
                import torch
                _dummy = torch.zeros(1, 80, 3000, device=device)
                with torch.no_grad():
                    whisper_model.encoder(_dummy)
                log.info("Whisper loaded on %s ✓", device)
            except Exception as device_exc:
                log.warning(
                    "Whisper failed on device=%s (%s) — falling back to CPU. "
                    "This is normal for MPS; Whisper uses ops not fully supported there.",
                    device, device_exc,
                )
                whisper_model = whisper_model.to("cpu")
                device = "cpu"
                log.info("Whisper loaded on cpu ✓ (DistilBERT will also use cpu)")
            whisper_model.eval()
        except Exception as exc:
            log.error(
                "Whisper failed to load entirely — audio embeddings will be skipped. "
                "Error: %s", exc, exc_info=True,
            )
            skip_audio = True

    log.info("Loading DistilBERT…")
    try:
        from transformers import AutoModel, AutoTokenizer
        distilbert_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        distilbert_model = AutoModel.from_pretrained("distilbert-base-uncased").to(device)
        distilbert_model.eval()
    except Exception as exc:
        log.warning("DistilBERT not available — skipping text embeddings: %s", exc)

    success = 0
    errors = 0

    # Process sequentially (models are not thread-safe for inference).
    # Use thread pool only for caption/page fetches; model inference is sequential.
    for i, video_id in enumerate(target):
        log.info("[%d/%d] %s", i + 1, len(target), video_id)
        try:
            ok = process_video(
                video_id=video_id,
                sponsor_segments=sponsor_map[video_id],
                cache_dir=cache_dir,
                whisper_model=whisper_model,
                whisper_processor=whisper_processor,
                distilbert_model=distilbert_model,
                distilbert_tokenizer=distilbert_tokenizer,
                device=device,
                skip_audio=skip_audio,
                cookies_path=cookies_path,
                cookies_from_browser=cookies_from_browser,
                sleep_interval=sleep_interval,
                force=force,
                force_zero_audio=force_zero_audio,
                min_votes=min_votes,
            )
            if ok:
                success += 1
            else:
                errors += 1
        except Exception as exc:
            log.warning("Unhandled error for %s: %s", video_id, exc)
            errors += 1

        if (i + 1) % 50 == 0:
            log.info("Progress: %d done, %d ok, %d errors", i + 1, success, errors)

    log.info("Batch complete: %d ok  %d errors  (total cached: %d)", success, errors, len(list(cache_dir.glob("*.npz"))))


# ---------------------------------------------------------------------------
# Dataset loader (used by train.py)
# ---------------------------------------------------------------------------


class SponsorDataset:
    """Iterable dataset over cached .npz files.

    Yields dicts with keys:
        keyword_vec  float32 [128]
        audio_emb    float32 [WHISPER_DIM]              (from Whisper mean-pool)
        text_emb     float32 [DISTILBERT_DIM]
        mfcc         float32 [N_MFCC_FRAMES, MFCC_DIM]  (zeros for pre-MFCC cache files)
        label        int8 scalar  (-1 = ignored by loss, used for isolated windows)
        vote_weight  float32      per-window loss weight in (0, 1] based on SB vote count
        video_id     str

    Args:
        min_votes: If > 0 and the npz contains ``sponsor_segs`` / ``sponsor_seg_votes``
                   arrays, window labels are recomputed using only segments with
                   vote count ≥ min_votes.  Falls back to stored labels for files that
                   pre-date the vote arrays (backward-compatible).
        temporal_consistency: If True, isolated sponsor windows (label=1 with no
                   sponsor neighbor on either side) are set to label=-1 so the loss
                   ignores them.  These are almost always boundary artefacts or
                   mislabels — real sponsor reads span ≥ 2 consecutive windows.
    """

    def __init__(
        self,
        cache_dir: Path,
        video_ids: list[str] | None = None,
        require_audio: bool = True,
        require_text: bool = True,
        min_votes: int = 0,
        temporal_consistency: bool = True,
    ) -> None:
        self.cache_dir = cache_dir
        self.require_audio = require_audio
        self.require_text = require_text
        self.min_votes = min_votes
        self.temporal_consistency = temporal_consistency

        if video_ids is not None:
            self.video_ids = video_ids
        else:
            self.video_ids = [p.stem for p in sorted(cache_dir.glob("*.npz"))]

        # Flatten all windows into a list of (video_id, window_index) pairs.
        self._index: list[tuple[str, int]] = []
        # Per-video recomputed label arrays (only populated when min_votes > 0).
        self._recomputed_labels: dict[str, np.ndarray] = {}
        # Per-video consistency-filtered labels (isolated sponsors set to -1).
        self._consistency_labels: dict[str, np.ndarray] = {}
        # Per-video per-window vote weights in (0, 1].
        self._vote_weights: dict[str, np.ndarray] = {}

        skipped_audio = 0
        n_recomputed  = 0
        n_isolated    = 0

        for vid in self.video_ids:
            path = cache_dir / f"{vid}.npz"
            if not path.exists():
                continue
            data = np.load(path)
            # Skip videos whose audio embeddings are all-zero (failed yt-dlp download).
            if self.require_audio and not np.any(data["audio_embs"]):
                log.debug("Skipping %s: all-zero audio embeddings (failed download)", vid)
                skipped_audio += 1
                continue
            n = len(data["labels"])

            # ── Re-label windows from stored vote data if requested ───────
            if min_votes > 0 and "sponsor_segs" in data.files and "sponsor_seg_votes" in data.files:
                recomputed = self._recompute_labels(
                    window_bounds=data["segments"],
                    sponsor_segs=data["sponsor_segs"],
                    sponsor_seg_votes=data["sponsor_seg_votes"],
                    min_votes=min_votes,
                )
                self._recomputed_labels[vid] = recomputed
                n_recomputed += 1

            # Effective labels used for both consistency filtering and vote weights.
            effective_labels = self._recomputed_labels.get(vid, data["labels"])

            # ── Temporal consistency filtering ────────────────────────────
            # Isolated sponsor windows (no sponsor neighbour) → label=-1 (ignored).
            if temporal_consistency and len(effective_labels) > 1:
                filtered, n_vid_isolated = self._apply_temporal_consistency(effective_labels)
                if n_vid_isolated > 0:
                    self._consistency_labels[vid] = filtered
                    n_isolated += n_vid_isolated

            # ── Vote-based loss weights ───────────────────────────────────
            # Sponsor windows: weight = min(1.0, max_votes / VOTE_WEIGHT_CAP)
            # Non-sponsor windows: weight = 1.0 (absence of label is reliable)
            if "sponsor_segs" in data.files and "sponsor_seg_votes" in data.files:
                self._vote_weights[vid] = self._compute_vote_weights(
                    window_bounds=data["segments"],
                    labels=effective_labels,
                    sponsor_segs=data["sponsor_segs"],
                    sponsor_seg_votes=data["sponsor_seg_votes"],
                )

            self._index.extend((vid, i) for i in range(n))

        n_videos = len(self.video_ids) - skipped_audio
        if skipped_audio:
            log.info("SponsorDataset: skipped %d videos with all-zero audio embeddings", skipped_audio)
        if min_votes > 0:
            log.info(
                "SponsorDataset: recomputed labels for %d/%d videos (min_votes=%d); "
                "%d videos used stored labels (pre-date vote arrays)",
                n_recomputed, n_videos, min_votes, n_videos - n_recomputed,
            )
        if temporal_consistency and n_isolated > 0:
            log.info(
                "SponsorDataset: temporal consistency — ignored %d isolated sponsor windows "
                "across %d videos (set to label=-1; not included in loss)",
                n_isolated, len(self._consistency_labels),
            )
        log.info("SponsorDataset: %d windows across %d videos", len(self._index), n_videos)

    @staticmethod
    def _recompute_labels(
        window_bounds: np.ndarray,
        sponsor_segs: np.ndarray,
        sponsor_seg_votes: np.ndarray,
        min_votes: int,
    ) -> np.ndarray:
        """Recompute window labels from per-segment vote data.

        A window is labelled 1 if it overlaps any qualifying sponsor segment
        (votes ≥ min_votes) by ≥ 50%.

        Args:
            window_bounds:      float32 [N, 2]  window start/end times.
            sponsor_segs:       float32 [M, 2]  all sponsor segment start/end times.
            sponsor_seg_votes:  int32   [M]     community votes per segment.
            min_votes:          Minimum votes to qualify a segment for labeling.

        Returns:
            int8 [N] label array.
        """
        qualifying = sponsor_segs[sponsor_seg_votes >= min_votes]  # [K, 2]
        labels = np.zeros(len(window_bounds), dtype=np.int8)
        for i, (t_start, t_end) in enumerate(window_bounds):
            window_len = float(t_end - t_start)
            if window_len <= 0:
                continue
            for (s_start, s_end) in qualifying:
                overlap = max(0.0, min(float(t_end), float(s_end)) - max(float(t_start), float(s_start)))
                if overlap / window_len >= 0.5:
                    labels[i] = 1
                    break
        return labels

    @staticmethod
    def _apply_temporal_consistency(
        labels: np.ndarray,
    ) -> tuple[np.ndarray, int]:
        """Set isolated sponsor windows to -1 (ignored by the loss).

        A window is "isolated" if it is labelled sponsor (1) but has no
        sponsor-labelled neighbour on either side.  Real sponsor reads span
        ≥ 2 consecutive 5-second windows (≥ 10 s); single windows are almost
        always boundary artefacts or mislabels.

        Returns:
            filtered    int8 [N]  label array with isolated windows set to -1.
            n_isolated  int       number of windows that were silenced.
        """
        result     = labels.copy().astype(np.int8)
        n          = len(result)
        n_isolated = 0
        for i in range(n):
            if labels[i] != 1:
                continue
            has_sponsor_prev = (i > 0     and labels[i - 1] == 1)
            has_sponsor_next = (i < n - 1 and labels[i + 1] == 1)
            if not has_sponsor_prev and not has_sponsor_next:
                result[i] = -1
                n_isolated += 1
        return result, n_isolated

    @staticmethod
    def _compute_vote_weights(
        window_bounds: np.ndarray,
        labels: np.ndarray,
        sponsor_segs: np.ndarray,
        sponsor_seg_votes: np.ndarray,
    ) -> np.ndarray:
        """Per-window loss weight based on SponsorBlock vote count.

        Non-sponsor windows (label=0):  weight = 1.0  (absence is reliable)
        Sponsor windows     (label=1):  weight = min(1.0, max_votes / VOTE_WEIGHT_CAP)

        A segment with ≥ VOTE_WEIGHT_CAP votes gets full weight; one with
        fewer votes is downweighted proportionally so it contributes less to
        the gradient without being discarded entirely.
        """
        weights = np.ones(len(window_bounds), dtype=np.float32)
        for i, (t_start, t_end) in enumerate(window_bounds):
            if labels[i] != 1:
                continue
            window_len = float(t_end - t_start)
            if window_len <= 0:
                continue
            max_votes = 0
            for j, (s_start, s_end) in enumerate(sponsor_segs):
                overlap = max(
                    0.0,
                    min(float(t_end), float(s_end)) - max(float(t_start), float(s_start)),
                )
                if overlap / window_len >= 0.5:
                    max_votes = max(max_votes, int(sponsor_seg_votes[j]))
            weights[i] = min(1.0, max_votes / max(VOTE_WEIGHT_CAP, 1))
        return weights

    def __len__(self) -> int:
        return len(self._index)

    def __iter__(self) -> Iterator[dict]:
        # Cache open file handles to avoid re-opening on every window.
        _cache: dict[str, dict] = {}
        for vid, idx in self._index:
            if vid not in _cache:
                path = self.cache_dir / f"{vid}.npz"
                arr = np.load(path)
                _cache[vid] = {k: arr[k] for k in arr.files}
                if len(_cache) > 200:  # evict old entries
                    oldest = next(iter(_cache))
                    del _cache[oldest]
            data = _cache[vid]
            kw = data["text_keyword_vecs"][idx].astype(np.float32)
            # Pad old 64-dim cache files to 128-dim (new dims fire as 0 for old videos).
            if kw.shape[0] < 128:
                kw = np.concatenate([kw, np.zeros(128 - kw.shape[0], dtype=np.float32)])

            # Label priority: consistency-filtered > min_votes-recomputed > stored.
            if vid in self._consistency_labels:
                label = int(self._consistency_labels[vid][idx])
            elif vid in self._recomputed_labels:
                label = int(self._recomputed_labels[vid][idx])
            else:
                label = int(data["labels"][idx])

            # Vote-based loss weight: 1.0 for non-sponsor or when vote data unavailable.
            vote_weight = float(self._vote_weights[vid][idx]) if vid in self._vote_weights else 1.0

            # Real MFCC features — zeros for old cache files that pre-date
            # compute_mfcc_features() (backfill with backfill_mfcc.py).
            if "mfcc_features" in data:
                mfcc = data["mfcc_features"][idx].astype(np.float32)
            else:
                mfcc = np.zeros((N_MFCC_FRAMES, MFCC_DIM), dtype=np.float32)

            yield {
                "keyword_vec": kw,
                "audio_emb":   data["audio_embs"][idx].astype(np.float32),
                "text_emb":    data["text_embs"][idx].astype(np.float32),
                "mfcc":        mfcc,
                "label":       label,
                "vote_weight": vote_weight,
                "video_id":    vid,
            }

    @staticmethod
    def train_val_test_split(
        cache_dir: Path,
        train_frac: float = 0.70,
        val_frac: float = 0.15,
        seed: int = 42,
    ) -> tuple[list[str], list[str], list[str]]:
        """Split video IDs into train / val / test sets.

        Split is by video ID (not by window) to avoid data leakage.
        """
        all_ids = sorted(p.stem for p in cache_dir.glob("*.npz"))
        rng = random.Random(seed)
        rng.shuffle(all_ids)
        n = len(all_ids)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)
        train_ids = all_ids[:n_train]
        val_ids = all_ids[n_train : n_train + n_val]
        test_ids = all_ids[n_train + n_val :]
        log.info("Split: train=%d  val=%d  test=%d", len(train_ids), len(val_ids), len(test_ids))
        return train_ids, val_ids, test_ids


# ---------------------------------------------------------------------------
# MFCC backfill (adds mfcc_features to existing cache files without
# recomputing Whisper / DistilBERT embeddings)
# ---------------------------------------------------------------------------


def backfill_mfcc(
    cache_dir: Path,
    cookies_path: Path | None = None,
    cookies_from_browser: str | None = None,
    sleep_interval: float = 1.0,
    workers: int = 2,
    force: bool = False,
) -> None:
    """Add real MFCC features to existing .npz files that lack them.

    For each cached video that is missing ``mfcc_features``:
      1. Re-downloads the audio with yt-dlp (same flags as the main pipeline).
      2. Computes real MFCC at 1 fps with a 30-frame rolling buffer.
      3. Patches the .npz in-place (atomic rename).

    Videos that already have ``mfcc_features`` are skipped unless *force* is True.
    Videos whose audio download fails are skipped with a WARNING (their npz is
    unchanged, so they continue to yield zero MFCC during training).

    Args:
        cache_dir:              Directory containing .npz files.
        cookies_path:           Netscape cookies.txt for yt-dlp (optional).
        cookies_from_browser:   Browser name to pull cookies from (e.g. "chrome").
        sleep_interval:         Seconds between yt-dlp downloads.
        workers:                Parallel download threads.
        force:                  Re-compute MFCC even for files that already have it.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    all_npz = sorted(cache_dir.glob("*.npz"))
    log.info("Found %d .npz files in %s", len(all_npz), cache_dir)

    todo: list[Path] = []
    for path in all_npz:
        try:
            with np.load(path) as d:
                has_mfcc = "mfcc_features" in d.files
        except Exception:
            has_mfcc = False
        if not has_mfcc or force:
            todo.append(path)

    log.info(
        "%d / %d files need MFCC backfill%s",
        len(todo), len(all_npz),
        " (--force active)" if force else "",
    )

    if not todo:
        log.info("Nothing to do — all files already have mfcc_features.")
        return

    ok = skip = fail = 0

    def _process_one(npz_path: Path) -> str:
        video_id = npz_path.stem
        try:
            existing = dict(np.load(npz_path, allow_pickle=False))
        except Exception as exc:
            return f"LOAD_ERROR ({exc})"

        if "segments" not in existing:
            return "SKIP (no segments array)"

        with tempfile.TemporaryDirectory(prefix=f"mfcc_{video_id}_") as tmp:
            tmp_dir = Path(tmp)
            audio_path = _download_audio(
                video_id, tmp_dir,
                cookies_path=cookies_path,
                cookies_from_browser=cookies_from_browser,
                sleep_interval=sleep_interval,
            )
            if audio_path is None:
                return "AUDIO_FAIL"

            # Build minimal window list (just t_start needed for MFCC).
            segs = existing["segments"]  # [N, 2]  float32
            windows = [{"t_start": float(s), "t_end": float(e)} for s, e in segs]
            mfcc_features = compute_mfcc_features(audio_path, windows)

        if not np.any(mfcc_features):
            return "MFCC_ZERO (librosa produced all zeros)"

        existing["mfcc_features"] = mfcc_features
        # np.savez_compressed always appends .npz, so the temp file must end in .npz
        # so the rename source matches what numpy actually wrote.
        tmp_out = npz_path.with_name(npz_path.stem + "._tmp.npz")
        try:
            np.savez_compressed(str(tmp_out), **existing)
            tmp_out.rename(npz_path)
        except Exception as exc:
            tmp_out.unlink(missing_ok=True)
            return f"WRITE_ERROR ({exc})"

        return f"OK  shape={mfcc_features.shape}"

    n = len(todo)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_process_one, p): p for p in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            path   = futures[fut]
            try:
                status = fut.result()
            except Exception as exc:
                status = f"EXCEPTION ({exc})"
            prefix = status.split()[0]
            if prefix == "OK":
                ok += 1
            elif prefix == "SKIP":
                skip += 1
            else:
                fail += 1
                log.warning("[%d/%d] %s  %s", i, n, path.stem, status)
                continue
            log.info("[%d/%d] %s  %s", i, n, path.stem, status)

            if sleep_interval > 0 and prefix == "OK":
                time.sleep(sleep_interval)

    log.info("Backfill complete — OK=%d  SKIP=%d  FAIL=%d  (%.0f%% success)",
             ok, skip, fail, 100 * ok / max(ok + fail, 1))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    p = argparse.ArgumentParser(
        description="Prepare SponsorBlock embeddings cache for model training."
    )
    p.add_argument("--csv", required=False, default=None, type=Path,
                   help="Path to sponsorTimes.csv (not required with --mfcc-only)")
    p.add_argument("--out", required=True, type=Path, help="Cache output directory")
    p.add_argument("--videos", type=int, default=300, help="Max videos to process")
    p.add_argument("--workers", type=int, default=2, help="Parallel workers (currently unused — sequential)")
    p.add_argument("--device", default="cpu", help="PyTorch device (cpu or cuda)")
    p.add_argument("--skip-audio", action="store_true", help="Skip yt-dlp download and Whisper (text-only mode)")
    p.add_argument("--seed", type=int, default=42, help="Random seed for video sampling")
    p.add_argument("--skip-ids", type=Path, default=None,
                   help="Path to a text file of already-processed video IDs (one per line) to skip.")
    p.add_argument("--cookies", type=Path, default=None,
                   help="Path to Netscape cookies.txt file for yt-dlp authentication.")
    p.add_argument("--cookies-from-browser", type=str, default=None,
                   dest="cookies_from_browser",
                   help="Browser to read cookies from directly, e.g. 'chrome', 'safari', 'firefox'. "
                        "More reliable than --cookies for YouTube bot checks.")
    p.add_argument("--sleep", type=float, default=0.0,
                   help="Seconds to sleep between yt-dlp HTTP requests (helps reduce 429 rate limiting).")
    p.add_argument("--force", action="store_true",
                   help="Re-process and overwrite all already-cached .npz files.")
    p.add_argument("--force-zero-audio", action="store_true",
                   help="Re-process only cached videos whose audio embeddings are all zero. "
                        "Skips videos that already have good audio. Use this to resume after fixing Whisper.")
    p.add_argument("--min-votes", type=int, default=MIN_VOTES,
                   help=f"Minimum SponsorBlock community votes to trust a segment (default: {MIN_VOTES}).")
    p.add_argument("--mfcc-only", action="store_true",
                   help="Backfill mode: add real MFCC features to existing cache files without "
                        "recomputing Whisper/DistilBERT. --out must point to the cache directory. "
                        "--csv is not required in this mode.")
    args = p.parse_args()

    # --mfcc-only: patch existing cache without touching embeddings
    if args.mfcc_only:
        backfill_mfcc(
            cache_dir=args.out,
            cookies_path=args.cookies,
            cookies_from_browser=args.cookies_from_browser,
            sleep_interval=args.sleep,
            workers=args.workers,
            force=args.force,
        )
        return

    if args.csv is None:
        p.error("--csv is required unless --mfcc-only is set")

    extra_skip: set[str] = set()
    if args.skip_ids and args.skip_ids.exists():
        extra_skip = {line.strip() for line in args.skip_ids.read_text().splitlines() if line.strip()}
        log.info("Skipping %d pre-processed video IDs from %s", len(extra_skip), args.skip_ids)

    run_batch(
        csv_path=args.csv,
        cache_dir=args.out,
        n_videos=args.videos,
        workers=args.workers,
        device=args.device,
        skip_audio=args.skip_audio,
        seed=args.seed,
        extra_skip=extra_skip,
        cookies_path=args.cookies,
        cookies_from_browser=args.cookies_from_browser,
        sleep_interval=args.sleep,
        force=args.force,
        force_zero_audio=args.force_zero_audio,
        min_votes=args.min_votes,
    )


if __name__ == "__main__":
    main()
