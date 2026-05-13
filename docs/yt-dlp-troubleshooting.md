# yt-dlp Troubleshooting — Known Failure Cases

Accumulated during Phase 1 data collection for the sponsor-segment-skipper project.

---

## 1. All audio embeddings are zero (silent Whisper fallback)

**Symptom:** Phase 1 completes without errors. `.npz` files are saved. But `audio_embs` arrays are all zeros for every video. Training dataset loads 0 windows when `require_audio=True`.

**Root cause:** `run_batch` in `data_pipeline.py` loaded Whisper successfully but then failed silently when calling `model.model.encoder(...)` — the correct attribute on `WhisperModel` is `model.encoder(...)`. The `AttributeError` was swallowed by a broad `except Exception` that logged only at DEBUG level, leaving every window's embedding as zeros.

**Fix:** Changed `model.model.encoder(input_features)` → `model.encoder(input_features)` in `compute_whisper_embeddings`. Also added WARNING-level logging for the first encoding failure per video so it surfaces immediately.

---

## 2. Whisper input length mismatch (padding error)

**Symptom:**
```
ValueError: Whisper expects the mel input features to be of length 3000, but found 497.
Make sure to pad the input mel features to 3000.
```

**Root cause:** `WhisperProcessor` was called with `padding=True`, which pads to the longest sequence in the batch. Since each window is processed individually, there's nothing to pad against — a 5-second window produces ~497 mel frames instead of the required 3000.

**Fix:** Changed `padding=True` → `padding="max_length"` so the processor always pads to Whisper's fixed 3000-frame input size.

**Secondary fix:** Mean-pooling was then being done over all 1500 encoder output frames, of which ~1250 were silence. Changed to pool only over `n_real_frames = int(audio_duration_sec * 50)` frames (Whisper encoder outputs 50 frames/sec after 2× downsampling), avoiding dilution by silence padding.

---

## 3. Audio downloads fail after N videos (rate limiting)

**Symptom:** First ~75 videos download successfully. Then all subsequent videos return:
```
WARNING Audio download failed: <video_id>
```
Downloads fail quickly (2–3 seconds each), indicating an immediate HTTP error rather than a timeout.

**Root cause:** YouTube rate-limited or temporarily blocked the IP/session after too many rapid requests. The `cookies.txt` session token may also have been rotated by YouTube mid-run, compounding the issue.

**Fix:**
- Wait 15–30 minutes for the block to lift.
- Restart with `--sleep 2` to add a 2-second pause between yt-dlp requests.
- Added `--force-zero-audio` flag so the pipeline skips already-good videos and only re-processes those with zero audio embeddings.

---

## 4. cookies.txt becomes invalid mid-run (bot check)

**Symptom:**
```
ERROR: [youtube] <id>: Sign in to confirm you're not a bot.
Use --cookies-from-browser or --cookies for the authentication.
```

**Root cause:** `cookies.txt` files exported from a browser extension go stale when YouTube rotates session tokens. After enough requests or elapsed time the exported cookies are rejected.

**Fix:** Switched from `--cookies cookies.txt` to `--cookies-from-browser chrome`, which reads live session cookies directly from Chrome's cookie store each run. Added `--cookies-from-browser` as a first-class CLI argument in `data_pipeline.py` (takes precedence over `--cookies` when both are provided).

---

## 5. macOS permissions block Safari cookie access

**Symptom:**
```
ERROR: [Errno 1] Operation not permitted: '/Users/<user>/Library/Cookies/Cookies.binarycookies'
```

**Root cause:** macOS TCC (Transparency, Consent, and Control) blocks Terminal from reading Safari's cookie store without explicit Full Disk Access permission.

**Fix (option A):** Use Chrome instead: `--cookies-from-browser chrome`.

**Fix (option B):** Grant Terminal Full Disk Access via System Settings → Privacy & Security → Full Disk Access → add Terminal.

---

## 6. n-challenge JS solving fails — only images available

**Symptom:**
```
WARNING: [youtube] x46MVIGS2KI: n challenge solving failed: Some formats may be missing.
WARNING: Only images are available for download. use --list-formats to see them
ERROR: [youtube] x46MVIGS2KI: Requested format is not available.
```

**Root cause:** YouTube's n-challenge (signature decryption for stream URLs) requires a JavaScript runtime. yt-dlp found Node.js but skipped the remote challenge solver script (`ejs`) because `--remote-components` was not enabled. Without it, only non-challenged formats (thumbnails/images) are served.

**Fix:** Added two flags to the yt-dlp subprocess in `_download_audio`:
```python
"--js-runtimes", "node",
"--remote-components", "ejs:github",
```
`ejs:github` tells yt-dlp to download the challenge solver script from the yt-dlp GitHub releases on first use and cache it locally.

**Prerequisites:** Node.js must be installed (`brew install node`).

---

## Summary — recommended Phase 1 invocation

```bash
python3 training/src/data_pipeline.py \
  --csv training/outputs/data/sponsorTimes.csv \
  --out training/cache/embeddings \
  --videos 999999 \
  --device mps \
  --cookies-from-browser chrome \
  --sleep 2 \
  --force-zero-audio   # use --force for a full re-run from scratch
```

To resume after an interruption without re-processing already-good videos, use `--force-zero-audio`. To re-process everything from scratch, use `--force`.
