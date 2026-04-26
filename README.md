# Video Ad Auto-Skip

Two Chrome extensions for handling YouTube ads, plus a benchmark suite for measuring detection accuracy.

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
