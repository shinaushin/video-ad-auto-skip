# Video Ad Auto-Skip

Three Chrome extensions for handling YouTube ads, plus a benchmark suite for measuring detection accuracy.

## Extensions

### 1. YouTube Ad Auto-Skipper (`youtube-ad-skipper/`)

Automatically handles YouTube's native pre-roll, mid-roll, and overlay ads.

**What it does:**

- Clicks the "Skip Ad" button the instant it becomes clickable
- Mutes audio during the countdown while waiting for the skip button
- Speeds unskippable ads to 16x playback (they finish in under a second)
- Handles back-to-back double-ad stacks
- Closes overlay banners and "video will play after ad" prompts
- Displays a red badge on the extension icon showing how many ads were skipped per tab

**How it works:**

A `MutationObserver` watches the DOM for new elements. When YouTube injects the skip button, the script detects it via CSS class selectors and a text-based heuristic fallback (matches "Skip Ad" in 15+ languages). If `.click()` is swallowed by YouTube's handlers, it dispatches a full pointer/mouse event sequence as a fallback. A 500ms polling interval catches edge cases where buttons transition from disabled to enabled without new DOM nodes.

**Files:**

| File | Purpose |
|------|---------|
| `manifest.json` | Manifest V3 config, registers content script and service worker |
| `content.js` | Core ad detection, skip logic, mute/speedup, overlay closing |
| `background.js` | Service worker for per-tab skip counter badge |
| `icons/` | Extension icons (16, 48, 128px) |

---

### 2. YouTube Sponsor Speeder (`youtube-sponsor-speeder/`)

Detects and speeds through **built-in** sponsor reads that are part of the video content itself (e.g., "This video is sponsored by..."). These are not YouTube ads — they're baked into the video by the creator.

**What it does:**

- Detects sponsor segments using transcript analysis (primary) or audio/visual analysis (fallback)
- Speeds through detected segments at 16x with audio muted
- Restores normal playback and volume when the segment ends
- Shows an orange on-screen indicator ("⏩ Sponsor (Transcript) · 16×") during speedup
- Handles YouTube's SPA navigation (video changes without page reload)

**Detection — Transcript analysis (primary):**

Extracts the caption track URL from YouTube's `ytInitialPlayerResponse`, fetches the timed-text XML, and scores each cue against 20+ sponsor keyword patterns (e.g., "sponsored by," "use code," "link in the description," "percent off"). Patterns are weighted: strong patterns (very likely sponsor language) score 3 points, weak patterns (need corroboration) score 1. Nearby high-scoring cues are clustered into segments with configurable padding and merge gaps.

**Detection — Audio/Visual analysis (fallback):**

When captions are unavailable, the `AVAnalyzer` class activates. It samples video frames every second onto a 64×36 canvas and computes a colour histogram fingerprint (8 bins × 3 RGB channels). Simultaneously, it monitors audio frequency bands via the Web Audio API, tracking the energy ratio between low and high frequencies. Each 5-second window is compared to a sliding baseline (median of the last 12 windows) using chi-squared distance (visual) and energy-ratio distance (audio). Windows exceeding a combined threshold (60% visual + 40% audio weight) are flagged and merged into segments. This catches sponsor reads that cut to product demos, app screenshots, or branded graphics — but will miss the "creator just keeps talking to camera" style.

**Files:**

| File | Purpose |
|------|---------|
| `manifest.json` | Manifest V3 config, loads av-analyzer.js before content.js |
| `content.js` | Transcript fetching, keyword scoring, playback control, AV fallback integration |
| `av-analyzer.js` | `AVAnalyzer` class — canvas frame sampling, Web Audio frequency analysis, baseline comparison |
| `icons/` | Extension icons (16, 48, 128px) — orange to distinguish from the ad skipper |

---

## Benchmark (`youtube-sponsor-speeder/test/`)

Measures sponsor detection accuracy against [SponsorBlock's](https://sponsor.ajay.app/) community-maintained database of timestamped sponsor segments.

**Database mode** — large-scale testing against SponsorBlock's full dataset:

```bash
cd youtube-sponsor-speeder/test

# Sample 50 random videos from the full SponsorBlock database
python3 benchmark.py --db

# Sample 500 videos with 5 parallel caption fetches, less output
python3 benchmark.py --db --sample 500 --workers 5 --quiet

# Go big
python3 benchmark.py --db --sample 2000 --workers 10 --quiet
```

This downloads `sponsorTimes.csv` (~2-4 GB) from SponsorBlock's public mirrors and caches it locally in `.cache/`. It parses all rows, filters for high-confidence segments (votes ≥ 3, not hidden, videos over 60s), randomly samples N videos, fetches their YouTube captions, runs our detection, and compares.

**API mode** — test specific videos:

```bash
python3 benchmark.py VIDEO_ID1 VIDEO_ID2 VIDEO_ID3
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--db` | off | Use full SponsorBlock database instead of per-video API |
| `--sample N` | 50 | Number of videos to sample in database mode |
| `--workers N` | 3 | Parallel YouTube caption fetches |
| `--quiet` | off | One-line-per-video output instead of detailed breakdown |

**Metrics reported:**

- **Precision** — of the time we flagged as sponsor, how much actually was
- **Recall** — of the real sponsor time, how much we caught
- **IoU** — intersection-over-union, overall overlap quality
- **Segment-level coverage** — per ground-truth segment, what percentage was covered

Results are saved to `benchmark-results.json` for further analysis.

**Requirements:** Python 3.7+. Zero pip dependencies (stdlib only).

---

## Installation

Both extensions use Chrome's Manifest V3 and are loaded as unpacked extensions:

1. Open `chrome://extensions` in Chrome
2. Enable **Developer mode** (top-right toggle)
3. Click **Load unpacked**
4. Select either `youtube-ad-skipper/` or `youtube-sponsor-speeder/`
5. Repeat for the other extension if desired

Both extensions work on `youtube.com` and `m.youtube.com`. They are independent and can run side by side.

**Incognito:** Chrome disables extensions in incognito by default. To enable, go to `chrome://extensions` → extension Details → toggle "Allow in Incognito."

---

## Configuration

Both extensions have tunable constants at the top of their `content.js` files. There is no settings UI — edit the values directly and reload the extension.

**Ad Skipper (`youtube-ad-skipper/content.js`):**

| Constant | Default | Description |
|----------|---------|-------------|
| `UNSKIPPABLE_PLAYBACK_RATE` | `16` | Playback speed for unskippable ads |
| `POLL_INTERVAL_MS` | `500` | Fallback polling interval (ms) |

**Sponsor Speeder (`youtube-sponsor-speeder/content.js`):**

| Constant | Default | Description |
|----------|---------|-------------|
| `SPONSOR_SPEED` | `16` | Playback speed during sponsor segments |
| `PADDING_BEFORE` | `1.5` | Seconds of padding before detected segment |
| `PADDING_AFTER` | `2.0` | Seconds of padding after detected segment |
| `MERGE_GAP_SEC` | `8` | Max gap (seconds) between cues before they're split into separate segments |
| `MIN_KEYWORD_HITS` | `2` | Minimum strong-pattern-equivalent hits to count as a real segment |
| `POLL_MS` | `250` | How often playback position is checked (ms) |

**AV Analyzer (`youtube-sponsor-speeder/av-analyzer.js`):**

| Constant | Default | Description |
|----------|---------|-------------|
| `COMBINED_THRESHOLD` | `0.45` | Combined visual+audio score to flag a window |
| `VISUAL_WEIGHT` | `0.6` | Weight of visual signal in combined score |
| `AUDIO_WEIGHT` | `0.4` | Weight of audio signal in combined score |
| `WINDOW_SEC` | `5` | Length of each analysis window (seconds) |
| `BASELINE_WINDOW` | `12` | Number of recent windows forming the baseline |
| `MIN_SEGMENT_SEC` | `12` | Minimum segment duration to report (filters false positives) |

---

## Limitations

**Ad Skipper:** YouTube periodically changes its DOM structure and CSS class names. The extension covers multiple known selectors and includes a text-based fallback, but may need selector updates over time. If the skip button stops being clicked, check the browser console for `[Ad Skipper]` logs.

**Sponsor Speeder — Transcript mode:** Depends on captions being available (most videos have auto-generated ones). The keyword patterns are English-focused. Detection accuracy varies — see the benchmark tool to measure it against SponsorBlock's ground truth.

**Sponsor Speeder — AV mode:** This is a best-effort fallback. It catches sponsor reads with visually distinct content (product shots, screen recordings, branded graphics) but cannot detect the common "creator just keeps talking to camera" style. It also consumes more CPU than transcript mode due to continuous canvas draws and Web Audio processing.

**Mobile / Smart TV:** These are Chrome desktop extensions. They do not work on the YouTube mobile app or smart TV apps. See the note in the repo about network-level alternatives like Pi-hole.

---

### 3. YouTube ML Sponsor Detector (`youtube-ml-sponsor-detector/`)

An experimental alternative to the Sponsor Speeder that uses a **bimodal ML model** — combining keyword text features and MFCC audio features — instead of pattern matching and heuristic thresholds. Designed as a testable implementation of the project plan's student model architecture, runnable immediately via built-in calibrated weights and upgradeable to a trained ONNX model once training is complete.

**What it does:**

- Detects in-video sponsor segments using a two-stage pipeline (text pre-detection + real-time bimodal inference)
- When confidence > 0.85 and the segment end is known: **seeks** the video directly past the end — instantaneous skip, no fast-forwarding
- Falls back to 16× speedthrough only when a segment end time isn't available (real-time-only detection with no pre-detected boundaries)
- Shows a purple indicator top-left ("⏭ Skipped (ML 87% · heuristic)") — distinct from the Sponsor Speeder's orange indicator, so both can run side-by-side for A/B comparison
- Upgrades to a trained ONNX model by dropping `model.onnx` + `ort.min.js` into the folder

**How it works:**

The detection pipeline runs in two stages per video:

*Stage 1 — Upfront text pre-detection:* Fetches the caption timed-text XML, groups cues into 5-second windows, and extracts a 64-dimensional keyword indicator vector for each window. Each of the 64 dimensions corresponds to a specific regex pattern covering sponsor intro phrases (group 0, weighted 3×), call-to-action language (group 1, 1.5×), offer/discount language (group 2, 1.5×), and product endorsement language (group 3, 0.5×). The keyword vector is fed to the ML model with zeroed audio features. Windows scoring above a relaxed pre-detection threshold are merged into candidate segments.

*Stage 2 — Real-time bimodal inference:* Every second, the MFCC extractor captures one frame from the Web Audio API's AnalyserNode. The pipeline applies a mel filterbank (26 triangular bands, 80–8000 Hz) to the FFT power spectrum, computes log mel energies, then DCT-II → 13 MFCC coefficients. The coefficients are mean-subtracted against a 60-second rolling baseline (delta-MFCCs), capturing *changes* in the acoustic environment rather than absolute values. The 64-dim text vector and 13-dim MFCC vector are concatenated and passed through the MLP. The final score is a 75/25 blend of the ML score and a cosine-distance audio anomaly score.

**The MLP architecture (student model):** `[77 → 32 → 16 → 1]` with ReLU activations and sigmoid output. Layer 1 neurons are grouped by feature type: intro-phrase neurons (0–7), CTA neurons (8–15), offer neurons (16–23), product-language neurons (24–27), and audio-anomaly neurons (28–31). Layer 2 neurons detect combinations: intro-phrase alone (high score), CTA+offer co-occurrence (high score), audio anomaly alone (moderate), and audio+product reinforcement. The built-in heuristic weights produce scores > 0.65 for "sponsored by" + any CTA, and > 0.5 for a clear audio shift.

**Two inference backends:**

| Backend | When used | Accuracy |
|---------|-----------|----------|
| Built-in heuristic MLP | Always (no setup needed) | Pattern-match quality — good baseline |
| ONNX Runtime Web | When `model.onnx` + `ort.min.js` present | Trained model quality |

**Files:**

| File | Purpose |
|------|---------|
| `manifest.json` | Manifest V3 config; lists `feature-extractor.js`, `ml-detector.js`, `content.js` |
| `feature-extractor.js` | `KeywordFeatureExtractor` (64-dim keyword vector) + `MFCCExtractor` (13-dim MFCC + anomaly score) |
| `ml-detector.js` | `MLSponsorDetector` — MLP inference engine with ONNX path and heuristic fallback |
| `content.js` | Two-stage detection pipeline, playback control, on-screen indicator |
| `model.onnx` | *(not included — drop in after training)* Trained student model |
| `ort.min.js` | *(not included — download separately)* ONNX Runtime Web |

**Plugging in the trained ONNX model:**

Once training on the SponsorBlock dataset is complete (see project plan, Phases 1–4):

1. Export the student model: `torch.onnx.export(student, dummy_input, "model.onnx", input_names=["input"], output_names=["output"])`
2. Download `ort.min.js` from `https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js`
3. Copy both files into `youtube-ml-sponsor-detector/`
4. Add `"ort.min.js"` to `content_scripts` in `manifest.json` *before* `feature-extractor.js`
5. Reload the extension — the indicator will show "ONNX" instead of "heuristic"

**Expected input to the ONNX model:** a single float32 tensor of shape `[1, 77]` — the first 64 values are the keyword indicator vector, the last 13 are the mean-pooled delta-MFCC vector.

---

## Configuration (ML Sponsor Detector)

Constants in `youtube-ml-sponsor-detector/content.js`:

| Constant | Default | Description |
|----------|---------|-------------|
| `SEEK_BUFFER` | `1.0` | Seconds added after segment end before seeking (landing buffer) |
| `SPEED_FALLBACK` | `16` | Playback speed used in fallback speed mode (no known segment end) |
| `POLL_MS` | `1000` | Real-time inference poll interval (ms) |
| `WINDOW_SEC` | `5` | Caption window size for text features (seconds) |
| `PAD_BEFORE` | `1.5` | Seconds of padding before detected segment |
| `PAD_AFTER` | `2.0` | Seconds of padding after detected segment |
| `ML_WEIGHT` | `0.75` | Weight of ML score in the blended real-time score |
| `AUDIO_WEIGHT` | `0.25` | Weight of raw audio anomaly in the blended score |
| `PRE_DETECT_BOOST` | `0.85` | Entry threshold multiplier inside a pre-detected region |

Constants in `youtube-ml-sponsor-detector/ml-detector.js`:

| Constant | Default | Description |
|----------|---------|-------------|
| `ENTRY_THRESHOLD` | `0.85` | Minimum blended score to seek/speed through a segment |
| `EXIT_THRESHOLD` | `0.50` | Score must fall below this (with 0 consecutive frames) to exit speed mode |
| `MIN_CONSECUTIVE_FRAMES` | `2` | Consecutive frames required above threshold before triggering |

**Relationship to the existing Sponsor Speeder:** The two extensions are fully independent and can run simultaneously — useful for comparing detection results on the same video. The ML Detector uses a purple top-left indicator; the Sponsor Speeder uses an orange top-right indicator. Both use the same `video.playbackRate` mechanism but maintain separate state.
