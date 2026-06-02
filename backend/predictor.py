"""predictor.py — Full bimodal sponsor segment prediction pipeline.

Replicates the training data pipeline at inference time:
  1. Fetch YouTube captions → group into 5s windows
  2. Download audio with yt-dlp → run Whisper encoder per window → audio_embs [N, 384]
  3. Run DistilBERT on each window's caption text → text_embs [N, 768]
  4. Run teacher BiLSTM on full sequence → logits [N]
  5. Apply threshold → merge adjacent windows → return [{start, end, score}]

Models are loaded once on startup and reused across requests.
"""

from __future__ import annotations

import logging
import os
import re
import sys
import tempfile
import warnings
from pathlib import Path
from typing import Optional
from urllib.parse import urlencode
from xml.etree import ElementTree

import numpy as np
import torch

log = logging.getLogger(__name__)

# ── Constants matching training pipeline and extension ────────────────────────
WINDOW_SEC      = 5.0   # caption grouping window
TEACHER_THRESH  = 0.50  # decision boundary (best threshold from teacher eval)
PAD_BEFORE      = 1.5   # seconds padding added before each segment
PAD_AFTER       = 2.0   # seconds padding added after each segment
MERGE_GAP_SEC   = 8.0   # max gap between adjacent windows before splitting
MIN_SEGMENT_SEC = 8.0   # minimum segment duration (shorter = likely false positive)
SAMPLE_RATE     = 16_000


# ---------------------------------------------------------------------------
# SponsorPredictor
# ---------------------------------------------------------------------------

class SponsorPredictor:
    """Loads all three models once; predict() is thread-safe (read-only inference)."""

    def __init__(self, teacher_ckpt: str, device: str = "cpu") -> None:
        self.device = device
        log.info("Loading models on device=%s …", device)
        self._load_distilbert()
        self._load_whisper()
        self._load_teacher(teacher_ckpt)
        log.info("All models loaded.")

    # ── Model loading ─────────────────────────────────────────────────────────

    def _load_distilbert(self) -> None:
        from transformers import DistilBertModel, DistilBertTokenizerFast
        self._tokenizer  = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
        self._distilbert = DistilBertModel.from_pretrained("distilbert-base-uncased").to(self.device)
        self._distilbert.eval()
        log.info("DistilBERT loaded.")

    def _load_whisper(self) -> None:
        from transformers import AutoProcessor, WhisperModel
        self._whisper_processor = AutoProcessor.from_pretrained("openai/whisper-tiny")
        self._whisper = WhisperModel.from_pretrained("openai/whisper-tiny").to(self.device)
        self._whisper.eval()
        log.info("Whisper-tiny loaded.")

    def _load_teacher(self, ckpt_path: str) -> None:
        # Add training/src to path so we can import our model classes.
        _src = str(Path(__file__).resolve().parent.parent / "training" / "src")
        if _src not in sys.path:
            sys.path.insert(0, _src)
        from models import load_teacher
        self._teacher = load_teacher(ckpt_path, device=self.device)
        self._teacher.eval()
        log.info("Teacher loaded from %s", ckpt_path)

    # ── Public API ────────────────────────────────────────────────────────────

    def predict(self, video_id: str) -> list[dict]:
        """Run the full pipeline for a video; return sponsor segment dicts.

        Returns:
            List of {"start": float, "end": float, "score": float} dicts,
            sorted by start time.  Empty list if no sponsor segments detected.
        """
        log.info("[%s] Starting prediction pipeline.", video_id)

        # ── 1. Captions → windows ─────────────────────────────────────────
        cues    = self._fetch_captions(video_id)
        windows = self._group_into_windows(cues)

        if not windows:
            log.warning("[%s] No caption windows — text embeddings will be zero.", video_id)
            # Still run audio-only inference if we have windows from duration.
            # Fall back: create windows based on video duration.
            windows = self._windows_from_duration(video_id)

        n = len(windows)
        log.info("[%s] %d windows (%.0f s of content).", video_id, n, n * WINDOW_SEC)

        # ── 2. Text embeddings ────────────────────────────────────────────
        text_embs = self._compute_text_embeddings(windows)    # [N, 768]

        # ── 3. Audio embeddings ───────────────────────────────────────────
        audio_embs = self._compute_audio_embeddings(video_id, windows)  # [N, 384]

        # ── 4. Teacher inference ──────────────────────────────────────────
        logits = self._run_teacher(text_embs, audio_embs)     # [N]

        # ── 5. Threshold → merge → return ─────────────────────────────────
        segments = self._merge_segments(windows, logits)
        log.info("[%s] %d sponsor segment(s) detected.", video_id, len(segments))
        return segments

    # ── Caption fetching ──────────────────────────────────────────────────────

    def _fetch_captions(self, video_id: str) -> list[dict]:
        """Fetch YouTube timed-text XML and return parsed cues.

        Tries the timedtext API endpoint directly; falls back to yt-dlp
        if the direct fetch fails (e.g. bot detection).
        """
        import urllib.request

        # Try direct timedtext API first (fastest, no auth needed for auto-captions).
        params = urlencode({"v": video_id, "lang": "en", "fmt": "xml"})
        url    = f"https://www.youtube.com/api/timedtext?{params}"
        try:
            req  = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                xml_bytes = resp.read()
            cues = self._parse_timedtext_xml(xml_bytes.decode("utf-8"))
            if cues:
                log.info("[caption] Fetched %d cues via timedtext API.", len(cues))
                return cues
        except Exception as e:
            log.warning("[caption] Direct fetch failed (%s); trying yt-dlp.", e)

        # yt-dlp fallback.
        try:
            import yt_dlp
            ydl_opts = {"skip_download": True, "quiet": True}
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(
                    f"https://www.youtube.com/watch?v={video_id}", download=False
                )
            subtitles = info.get("automatic_captions", {}) or info.get("subtitles", {})
            en = subtitles.get("en", [])
            for fmt in en:
                if fmt.get("ext") == "xml":
                    req = urllib.request.Request(
                        fmt["url"], headers={"User-Agent": "Mozilla/5.0"}
                    )
                    with urllib.request.urlopen(req, timeout=15) as resp:
                        cues = self._parse_timedtext_xml(resp.read().decode("utf-8"))
                    if cues:
                        log.info("[caption] Fetched %d cues via yt-dlp.", len(cues))
                        return cues
        except Exception as e:
            log.warning("[caption] yt-dlp caption fetch failed: %s", e)

        return []

    @staticmethod
    def _parse_timedtext_xml(xml: str) -> list[dict]:
        try:
            root = ElementTree.fromstring(xml)
            cues = []
            for node in root.findall(".//text"):
                start = float(node.get("start", 0))
                dur   = float(node.get("dur", 2))
                text  = re.sub(r"<[^>]+>", "", node.text or "").strip()
                if text:
                    cues.append({"start": start, "dur": dur, "text": text})
            return cues
        except Exception:
            return []

    def _group_into_windows(self, cues: list[dict]) -> list[dict]:
        """Group caption cues into WINDOW_SEC-wide windows."""
        if not cues:
            return []
        windows: list[dict] = []
        w_start, w_cues = cues[0]["start"], []
        for cue in cues:
            if cue["start"] - w_start >= WINDOW_SEC and w_cues:
                last = w_cues[-1]
                windows.append({
                    "start":  w_start,
                    "end":    last["start"] + last["dur"],
                    "text":   " ".join(c["text"] for c in w_cues),
                })
                w_start, w_cues = cue["start"], []
            w_cues.append(cue)
        if w_cues:
            last = w_cues[-1]
            windows.append({
                "start": w_start,
                "end":   last["start"] + last["dur"],
                "text":  " ".join(c["text"] for c in w_cues),
            })
        return windows

    def _windows_from_duration(self, video_id: str) -> list[dict]:
        """Create empty windows when no captions are available."""
        try:
            import yt_dlp
            with yt_dlp.YoutubeDL({"skip_download": True, "quiet": True}) as ydl:
                info     = ydl.extract_info(
                    f"https://www.youtube.com/watch?v={video_id}", download=False
                )
                duration = float(info.get("duration", 0))
            n = max(1, int(duration / WINDOW_SEC))
            return [
                {"start": i * WINDOW_SEC, "end": (i + 1) * WINDOW_SEC, "text": ""}
                for i in range(n)
            ]
        except Exception as e:
            log.warning("Could not determine video duration: %s", e)
            return []

    # ── Text embeddings ───────────────────────────────────────────────────────

    def _compute_text_embeddings(self, windows: list[dict]) -> np.ndarray:
        """Run DistilBERT on each window's text; return [N, 768]."""
        texts = [w["text"] if w["text"] else "[PAD]" for w in windows]
        n     = len(texts)
        embs  = np.zeros((n, 768), dtype=np.float32)

        batch_size = 32
        with torch.no_grad():
            for i in range(0, n, batch_size):
                batch = texts[i : i + batch_size]
                enc   = self._tokenizer(
                    batch,
                    truncation    = True,
                    max_length    = 512,
                    padding       = True,
                    return_tensors= "pt",
                ).to(self.device)
                out  = self._distilbert(**enc)
                cls  = out.last_hidden_state[:, 0, :].cpu().numpy()  # [B, 768]
                embs[i : i + len(batch)] = cls

        log.info("Text embeddings: %s", embs.shape)
        return embs

    # ── Audio embeddings ──────────────────────────────────────────────────────

    def _compute_audio_embeddings(
        self, video_id: str, windows: list[dict]
    ) -> np.ndarray:
        """Download audio, run Whisper encoder per window; return [N, 384]."""
        n    = len(windows)
        embs = np.zeros((n, 384), dtype=np.float32)

        with tempfile.TemporaryDirectory() as tmp:
            audio_path = self._download_audio(video_id, tmp)
            if audio_path is None:
                log.warning("[%s] Audio download failed — using zero audio embeddings.", video_id)
                return embs

            try:
                import librosa
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    audio, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
            except Exception as e:
                log.warning("Audio load failed: %s", e)
                return embs

            audio_len = len(audio)
            batch_size = 16

            with torch.no_grad():
                for batch_start in range(0, n, batch_size):
                    batch_end = min(batch_start + batch_size, n)
                    chunks    = []
                    for j in range(batch_start, batch_end):
                        w     = windows[j]
                        start = int(w["start"] * SAMPLE_RATE)
                        end   = int(w["end"]   * SAMPLE_RATE)
                        chunk = audio[start:end] if end <= audio_len else audio[start:]
                        # Whisper expects exactly 30s of audio; pad or trim.
                        target = SAMPLE_RATE * 30
                        if len(chunk) < target:
                            chunk = np.pad(chunk, (0, target - len(chunk)))
                        else:
                            chunk = chunk[:target]
                        chunks.append(chunk)

                    inputs = self._whisper_processor(
                        chunks,
                        sampling_rate   = SAMPLE_RATE,
                        return_tensors  = "pt",
                    ).to(self.device)

                    enc_out = self._whisper.encoder(
                        inputs.input_features
                    ).last_hidden_state                          # [B, T, 384]
                    pooled  = enc_out.mean(dim=1).cpu().numpy()  # [B, 384]
                    embs[batch_start:batch_end] = pooled

        log.info("Audio embeddings: %s", embs.shape)
        return embs

    def _download_audio(self, video_id: str, output_dir: str) -> Optional[str]:
        """Download audio-only stream with yt-dlp; return local path or None."""
        import yt_dlp
        out_tmpl = str(Path(output_dir) / "audio.%(ext)s")
        ydl_opts = {
            "format":           "bestaudio/best",
            "outtmpl":          out_tmpl,
            "quiet":            True,
            "no_warnings":      True,
            "postprocessors":   [{
                "key":            "FFmpegExtractAudio",
                "preferredcodec": "wav",
            }],
        }
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([f"https://www.youtube.com/watch?v={video_id}"])
            # Find the downloaded file.
            wav = Path(output_dir) / "audio.wav"
            if wav.exists():
                return str(wav)
            # yt-dlp may use a different extension.
            for f in Path(output_dir).iterdir():
                if f.suffix in (".wav", ".m4a", ".webm", ".mp4"):
                    return str(f)
        except Exception as e:
            log.warning("Audio download failed for %s: %s", video_id, e)
        return None

    # ── Teacher inference ─────────────────────────────────────────────────────

    def _run_teacher(
        self, text_embs: np.ndarray, audio_embs: np.ndarray
    ) -> np.ndarray:
        """Run teacher BiLSTM; return sigmoid probabilities [N]."""
        n          = text_embs.shape[0]
        text_t     = torch.from_numpy(text_embs).unsqueeze(0).to(self.device)   # [1, N, 768]
        audio_t    = torch.from_numpy(audio_embs).unsqueeze(0).to(self.device)  # [1, N, 384]
        lengths    = torch.tensor([n], dtype=torch.long).to(self.device)

        with torch.no_grad():
            logits = self._teacher(text_t, audio_t, lengths)   # [1, N, 1]
            probs  = torch.sigmoid(logits).squeeze().cpu().numpy()

        if probs.ndim == 0:
            probs = probs.reshape(1)

        return probs.astype(np.float32)

    # ── Segment merging ───────────────────────────────────────────────────────

    def _merge_segments(
        self, windows: list[dict], probs: np.ndarray
    ) -> list[dict]:
        """Apply threshold; merge adjacent candidate windows into segments.

        Mirrors the mergeWindowsIntoSegments logic in content.js so extension
        and backend produce comparable segment boundaries.
        """
        candidates = [
            {"time": w["start"], "endTime": w["end"], "score": float(p)}
            for w, p in zip(windows, probs)
            if p >= TEACHER_THRESH
        ]
        if not candidates:
            return []

        # Sort and merge windows within MERGE_GAP_SEC of each other.
        candidates.sort(key=lambda x: x["time"])
        groups: list[list[dict]] = [[candidates[0]]]
        for c in candidates[1:]:
            if c["time"] - groups[-1][-1]["endTime"] <= MERGE_GAP_SEC:
                groups[-1].append(c)
            else:
                groups.append([c])

        segments = []
        for group in groups:
            start = max(0.0, group[0]["time"]    - PAD_BEFORE)
            end   =          group[-1]["endTime"] + PAD_AFTER
            score = sum(g["score"] for g in group) / len(group)
            if (end - start) >= MIN_SEGMENT_SEC:
                segments.append({
                    "start": round(start, 2),
                    "end":   round(end,   2),
                    "score": round(score, 4),
                })

        return segments
