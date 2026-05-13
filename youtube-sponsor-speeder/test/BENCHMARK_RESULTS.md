# Sponsor Detection Benchmark Results

## Full-Database Run — May 2026

### Configuration

| Parameter | Value |
|---|---|
| Dataset | SponsorBlock `sponsorTimes.csv` (full DB, no sampling) |
| Filter | `category=sponsor`, `votes≥1`, not hidden, video duration≥60s |
| Videos tested | 63,783 |
| Caption source | yt-dlp + browser cookies, pre-cached |
| Git commit | `5125403` → `e55ddec` |

### Aggregate Results

```
Total videos tested:     63,783
  Successful benchmarks: 39,701
  No captions available: 24,082
  Errors:                0
```

| Metric | Score |
|---|---|
| **Average Precision** | **57.2%** |
| **Average Recall** | **61.3%** |
| **Average IoU** | **36.9%** |

**Segment-level coverage:**

| | Count | % |
|---|---|---|
| Ground truth segments | 53,474 | — |
| Correctly covered (≥50% overlap) | 30,818 | 57.6% |
| False positives | 0 | — |

> Note: false positives count is 0 because no videos in the DB have zero ground-truth segments (the DB is filtered to only include videos *with* sponsor annotations, so there are no true negatives to generate false positives against).

### Caption Coverage

37.8% of videos (24,082 / 63,783) had no captions available. These are undetectable regardless of algorithm quality — the algorithm requires a transcript to find sponsor language. Effective accuracy over the captioned subset is what matters.

Over the 39,701 videos with captions, **57.6% of ground-truth sponsor segments** were correctly covered at ≥50% overlap.

### Detection Algorithm (at time of run)

Key constants in `benchmark.py`:

| Constant | Value | Purpose |
|---|---|---|
| `MERGE_GAP_SEC` | 120 s | Bridge sponsor intro → ad body → outro clusters |
| `WINDOW_SEC` | 25 s | Sliding window for per-cue scoring |
| `MIN_SCORE` | 3 | Threshold to flag a cue (one STRONG match = 3) |
| `MIN_SEGMENT_DURATION` | 45 s | Floor on any detected segment length |
| `BOUNDARY_WALK_SEC` | 25 s | Max walk past cluster anchor for natural boundaries |

Notable patterns (STRONG, +3 each): `sponsor(?:ed)? by`, `brought to you by`, `use (?:my|our)? code`, `go to *.com`, `percent off`, `free trial`, `thanks? to * for sponsor`, `this video (?:is )?sponsor`, `our sponsors`.

### Interpretation

- **Precision 57.2%** — when we detect a sponsor segment, about 57% of the detected time overlaps ground truth. The remainder is over-extension into surrounding content, mainly from the `MERGE_GAP_SEC=120` bridging strategy needed to cover long ad reads.
- **Recall 61.3%** — we cover 61% of the ground-truth sponsor time across all captioned videos. The main miss cases are: (a) sponsors with non-verbal intro cards (no spoken keywords), (b) unusual brand-read language outside our pattern set, (c) sponsors in the no-captions population.
- **IoU 36.9%** — the harmonic balance of precision and recall. A segment we detect is roughly the right spot but wider than ideal.

### Baseline Comparison (100-video sample, before tuning)

| Metric | Before | After full run |
|---|---|---|
| Precision | 83.3% | 57.2% |
| Recall | 15.7% | 61.3% |
| IoU | 12.5% | 36.9% |

The shift from the 100-video sample reflects the intentional tradeoff: `MERGE_GAP_SEC` was increased from 15 s to 120 s to dramatically improve recall (covering the full sponsor read, not just the disclosure sentence), at the cost of wider detected segments and lower precision.
