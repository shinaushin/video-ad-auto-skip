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
    segments          float32 [N, 2]     sponsor segment start/end seconds
    audio_embs        float32 [N, 384]   Whisper mean-pooled encoder embeddings
    text_embs         float32 [N, 768]   DistilBERT [CLS] embeddings
    text_keyword_vecs float32 [N, 64]    keyword indicator vectors (matching feature-extractor.js)
    labels            int8    [N]        1 = sponsor, 0 = non-sponsor
    video_duration    float32 scalar

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
import urllib.request
import xml.etree.ElementTree as ET
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
]

# Compile and pad to exactly 64 entries.
for _pat, _grp in _RAW_PATTERNS[:64]:
    _KEYWORD_PATTERNS.append((re.compile(_pat, re.IGNORECASE), _grp))
while len(_KEYWORD_PATTERNS) < 64:
    _KEYWORD_PATTERNS.append((re.compile(r"(?!x)x"), 3))  # never-match filler


def keyword_vector(text: str) -> np.ndarray:
    """Return a 64-dim float32 indicator vector (1.0 if pattern matched)."""
    vec = np.zeros(64, dtype=np.float32)
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
    min_votes: int = MIN_VOTES,
) -> dict[str, list[tuple[float, float]]]:
    """Parse sponsorTimes.csv and return a mapping videoId → list of (start, end) tuples.

    Filters:
      * category == ``category``
      * votes >= ``min_votes``
      * hidden/shadowhidden == 0
      * actionType == 'skip' or empty

    Args:
        csv_path:  Path to sponsorTimes.csv (downloaded from SponsorBlock mirrors).
        category:  Segment category to keep (default: "sponsor").
        min_votes: Minimum community votes threshold.

    Returns:
        Dict mapping videoId strings to sorted lists of (start_sec, end_sec) pairs.
    """
    log.info("Parsing SponsorBlock CSV: %s", csv_path)
    segments: dict[str, list[tuple[float, float]]] = {}

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

            segments.setdefault(vid, []).append((start, end))

    # Sort segments within each video.
    for vid in segments:
        segments[vid].sort()

    log.info("Found %d videos with %s segments", len(segments), category)
    return segments


# ---------------------------------------------------------------------------
# Caption / transcript fetching (reuses logic from content.js)
# ---------------------------------------------------------------------------


def _fetch_caption_url(video_id: str) -> str | None:
    """Return the timedtext XML URL for a YouTube video, or None."""
    url = f"https://www.youtube.com/watch?v={video_id}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        html = urllib.request.urlopen(req, timeout=10).read().decode("utf-8", errors="replace")
    except Exception as exc:
        log.debug("Failed to fetch watch page for %s: %s", video_id, exc)
        return None

    # Extract ytInitialPlayerResponse JSON blob.
    match = re.search(r"ytInitialPlayerResponse\s*=\s*(\{.+?\});\s*(?:var\s|</script>)", html, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(1))
    except json.JSONDecodeError:
        return None

    try:
        tracks = data["captions"]["playerCaptionsTracklistRenderer"]["captionTracks"]
    except (KeyError, TypeError):
        return None

    # Prefer English auto-generated, then English, then first available.
    best = None
    for track in tracks:
        lang = track.get("languageCode", "")
        kind = track.get("kind", "")
        base_url = track.get("baseUrl", "")
        if lang == "en" and kind == "asr":
            return base_url
        if lang == "en" and best is None:
            best = base_url
    return best or (tracks[0].get("baseUrl") if tracks else None)


def fetch_video_captions(video_id: str) -> list[tuple[float, float, str]]:
    """Return a list of (start_sec, end_sec, text) tuples for a video.

    Returns an empty list if captions are unavailable.
    """
    caption_url = _fetch_caption_url(video_id)
    if not caption_url:
        return []

    try:
        data = urllib.request.urlopen(caption_url, timeout=10).read()
        root = ET.fromstring(data)
    except Exception as exc:
        log.debug("Caption fetch failed for %s: %s", video_id, exc)
        return []

    cues: list[tuple[float, float, str]] = []
    for elem in root.iter("text"):
        try:
            start = float(elem.get("start", "0"))
            dur = float(elem.get("dur", "0"))
        except ValueError:
            continue
        text = re.sub(r"<[^>]+>", "", elem.text or "")
        text = text.replace("&#39;", "'").replace("&amp;", "&").replace("&quot;", '"')
        text = text.strip()
        if text:
            cues.append((start, start + dur, text))
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


def _download_audio(video_id: str, out_dir: Path) -> Path | None:
    """Download audio-only stream with yt-dlp; return path to opus/webm file."""
    out_template = str(out_dir / f"{video_id}.%(ext)s")
    cmd = [
        "yt-dlp",
        "--quiet",
        "--no-warnings",
        "-f", "bestaudio[ext=webm]/bestaudio",
        "-o", out_template,
        "--",
        f"https://www.youtube.com/watch?v={video_id}",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        log.debug("yt-dlp failed for %s: %s", video_id, result.stderr[:200])
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

    with tempfile.TemporaryDirectory(prefix="whisper_slice_") as tmp:
        tmp_dir = Path(tmp)
        for i, w in enumerate(windows):
            slice_path = tmp_dir / f"slice_{i}.wav"
            ok = _slice_audio_segment(audio_path, w["t_start"], w["t_end"], slice_path)
            if not ok:
                continue
            try:
                import soundfile as sf
                audio_data, sr = sf.read(str(slice_path))
                if audio_data.ndim > 1:
                    audio_data = audio_data.mean(axis=1)
            except Exception:
                # Fall back to librosa if soundfile unavailable.
                try:
                    import librosa
                    audio_data, sr = librosa.load(str(slice_path), sr=16000, mono=True)
                except Exception:
                    continue

            try:
                inputs = processor(
                    audio_data,
                    sampling_rate=16000,
                    return_tensors="pt",
                    padding=True,
                )
                input_features = inputs["input_features"].to(device)
                with torch.no_grad():
                    encoder_out = model.model.encoder(input_features)
                    # encoder_out.last_hidden_state: [1, T, 384]
                    emb = encoder_out.last_hidden_state.mean(dim=1).squeeze(0)
                embs[i] = emb.cpu().float().numpy()
            except Exception as exc:
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
# Per-video processing
# ---------------------------------------------------------------------------


def process_video(
    video_id: str,
    sponsor_segments: list[tuple[float, float]],
    cache_dir: Path,
    whisper_model,
    whisper_processor,
    distilbert_model,
    distilbert_tokenizer,
    device: str = "cpu",
    skip_audio: bool = False,
) -> bool:
    """Process one video end-to-end and write <cache_dir>/<video_id>.npz.

    Returns True on success.
    """
    out_path = cache_dir / f"{video_id}.npz"
    if out_path.exists():
        log.debug("Cache hit: %s", video_id)
        return True

    # Fetch captions.
    cues = fetch_video_captions(video_id)

    with tempfile.TemporaryDirectory(prefix=f"ytdl_{video_id}_") as tmp:
        tmp_dir = Path(tmp)
        audio_path: Path | None = None
        video_duration = 0.0

        if not skip_audio:
            audio_path = _download_audio(video_id, tmp_dir)
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

        # Build windows.
        windows = build_windows(cues, video_duration, sponsor_segments)
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

        # DistilBERT embeddings.
        if distilbert_model is not None:
            text_embs = compute_distilbert_embeddings(
                windows, distilbert_model, distilbert_tokenizer, device
            )
        else:
            text_embs = np.zeros((N, DISTILBERT_DIM), dtype=np.float32)

        # Save.
        np.savez_compressed(
            out_path,
            segments=t_bounds,
            audio_embs=audio_embs,
            text_embs=text_embs,
            text_keyword_vecs=keyword_vecs,
            labels=labels,
            video_duration=np.float32(video_duration),
        )
        log.info("Saved %s  (N=%d, sponsors=%d)", out_path.name, N, labels.sum())

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
) -> None:
    """Process up to ``n_videos`` random videos and cache their embeddings.

    Args:
        csv_path:   Path to sponsorTimes.csv.
        cache_dir:  Directory to write per-video .npz files.
        n_videos:   Number of videos to process.
        workers:    Parallel audio-download threads (yt-dlp tasks).
        device:     PyTorch device string ("cuda" or "cpu").
        skip_audio: Skip audio download/Whisper (text-only mode, faster).
        seed:       Random seed for video sampling.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Parse SponsorBlock CSV.
    sponsor_map = parse_sponsorblock_csv(csv_path)

    # Sample videos.
    all_ids = list(sponsor_map.keys())
    random.seed(seed)
    random.shuffle(all_ids)
    # Skip already-cached.
    uncached = [v for v in all_ids if not (cache_dir / f"{v}.npz").exists()]
    target = uncached[:n_videos]

    log.info("Processing %d videos (%d already cached)", len(target), len(all_ids) - len(uncached))

    # Load models once (heavy — done before threading).
    whisper_model = whisper_processor = None
    distilbert_model = distilbert_tokenizer = None

    if not skip_audio:
        log.info("Loading Whisper-tiny encoder…")
        try:
            from transformers import WhisperModel, WhisperProcessor
            whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
            whisper_model = WhisperModel.from_pretrained("openai/whisper-tiny").to(device)
            whisper_model.eval()
        except Exception as exc:
            log.warning("Whisper not available — skipping audio embeddings: %s", exc)
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
        keyword_vec  float32 [64]
        audio_emb    float32 [WHISPER_DIM]   (from Whisper mean-pool)
        text_emb     float32 [DISTILBERT_DIM]
        label        int8 scalar
        video_id     str
    """

    def __init__(
        self,
        cache_dir: Path,
        video_ids: list[str] | None = None,
        require_audio: bool = True,
        require_text: bool = True,
    ) -> None:
        self.cache_dir = cache_dir
        self.require_audio = require_audio
        self.require_text = require_text

        if video_ids is not None:
            self.video_ids = video_ids
        else:
            self.video_ids = [p.stem for p in sorted(cache_dir.glob("*.npz"))]

        # Flatten all windows into a list of (video_id, window_index) pairs.
        self._index: list[tuple[str, int]] = []
        for vid in self.video_ids:
            path = cache_dir / f"{vid}.npz"
            if not path.exists():
                continue
            data = np.load(path)
            n = len(data["labels"])
            self._index.extend((vid, i) for i in range(n))

        log.info("SponsorDataset: %d windows across %d videos", len(self._index), len(self.video_ids))

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
            yield {
                "keyword_vec": data["text_keyword_vecs"][idx].astype(np.float32),
                "audio_emb": data["audio_embs"][idx].astype(np.float32),
                "text_emb": data["text_embs"][idx].astype(np.float32),
                "label": int(data["labels"][idx]),
                "video_id": vid,
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
    p.add_argument("--csv", required=True, type=Path, help="Path to sponsorTimes.csv")
    p.add_argument("--out", required=True, type=Path, help="Cache output directory")
    p.add_argument("--videos", type=int, default=300, help="Max videos to process")
    p.add_argument("--workers", type=int, default=2, help="Parallel workers (currently unused — sequential)")
    p.add_argument("--device", default="cpu", help="PyTorch device (cpu or cuda)")
    p.add_argument("--skip-audio", action="store_true", help="Skip yt-dlp download and Whisper (text-only mode)")
    p.add_argument("--seed", type=int, default=42, help="Random seed for video sampling")
    args = p.parse_args()

    run_batch(
        csv_path=args.csv,
        cache_dir=args.out,
        n_videos=args.videos,
        workers=args.workers,
        device=args.device,
        skip_audio=args.skip_audio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
