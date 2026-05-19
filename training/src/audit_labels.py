"""audit_labels.py — SponsorBlock label quality audit.

Analyses the cached .npz files (always available) and optionally the raw
sponsorTimes.csv (for vote distribution) to diagnose label noise.

Outputs a human-readable report to stdout and a summary JSON.

Usage:
    # Cache-only audit (always works):
    python audit_labels.py --cache-dir training/cache/embeddings

    # Full audit including vote distribution:
    python audit_labels.py --cache-dir training/cache/embeddings \\
                           --csv sponsorTimes.csv

    # Simulate raising MIN_VOTES threshold (requires --csv):
    python audit_labels.py --cache-dir training/cache/embeddings \\
                           --csv sponsorTimes.csv \\
                           --vote-thresholds 3,5,8,10,15,20
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

WINDOW_SEC = 5.0  # must match data_pipeline.py


def load_cache(cache_dir: Path) -> list[dict]:
    """Load all .npz files; return a list of per-video dicts."""
    videos = []
    for npz_path in sorted(cache_dir.glob("*.npz")):
        try:
            d = np.load(npz_path, allow_pickle=True)
            labels   = d["labels"].astype(np.int8)          # [N]
            segments = d["segments"].astype(np.float32)     # [N, 2]  window start/end
            duration = float(d["video_duration"])
            videos.append({
                "video_id": npz_path.stem,
                "labels":   labels,
                "segments": segments,
                "duration": duration,
                "n_windows": len(labels),
            })
        except Exception as e:
            print(f"  Warning: could not load {npz_path.name}: {e}")
    return videos


def sponsor_runs(labels: np.ndarray) -> list[tuple[int, int]]:
    """Return (start_idx, end_idx) inclusive for each run of consecutive 1s."""
    runs = []
    in_run = False
    start = 0
    for i, v in enumerate(labels):
        if v == 1 and not in_run:
            in_run = True
            start = i
        elif v == 0 and in_run:
            runs.append((start, i - 1))
            in_run = False
    if in_run:
        runs.append((start, len(labels) - 1))
    return runs


def boundary_windows(labels: np.ndarray) -> int:
    """Count windows that are at the edge of a sponsor run (first or last window of a run)."""
    runs = sponsor_runs(labels)
    boundary = set()
    for s, e in runs:
        boundary.add(s)
        boundary.add(e)
    return len(boundary)


def ascii_hist(values: list[float], bins: int = 10,
               lo: float | None = None, hi: float | None = None,
               bar_width: int = 30) -> str:
    if not values:
        return "  (no data)"
    lo = lo if lo is not None else min(values)
    hi = hi if hi is not None else max(values)
    if lo == hi:
        return f"  All values = {lo:.3f}"
    width = (hi - lo) / bins
    counts = [0] * bins
    for v in values:
        idx = min(int((v - lo) / width), bins - 1)
        counts[idx] += 1
    max_c = max(counts) or 1
    lines = []
    for i, c in enumerate(counts):
        lo_b = lo + i * width
        bar  = "█" * int(c / max_c * bar_width)
        lines.append(f"  [{lo_b:7.2f} – {lo_b + width:7.2f})  {bar:<{bar_width}}  {c:>6,}")
    return "\n".join(lines)


def pct(n: int, total: int) -> str:
    return f"{n:,} ({100 * n / total:.1f}%)" if total else "0"


# ---------------------------------------------------------------------------
# Cache audit
# ---------------------------------------------------------------------------

def audit_cache(videos: list[dict]) -> dict:
    total_windows   = sum(v["n_windows"] for v in videos)
    total_positive  = sum(int(v["labels"].sum()) for v in videos)
    total_negative  = total_windows - total_positive
    base_pos_rate   = total_positive / total_windows if total_windows else 0

    # Per-video positive rates
    pos_rates = [int(v["labels"].sum()) / v["n_windows"] for v in videos
                 if v["n_windows"] > 0]

    # Sponsor run lengths (in windows → multiply by WINDOW_SEC for seconds)
    run_lengths_sec = []
    for v in videos:
        for s, e in sponsor_runs(v["labels"]):
            run_lengths_sec.append((e - s + 1) * WINDOW_SEC)

    # Boundary windows (edge of a sponsor run — most likely mislabeled)
    total_boundary = sum(boundary_windows(v["labels"]) for v in videos)
    boundary_pct   = total_boundary / total_positive if total_positive else 0

    # Videos with suspiciously high positive rate (>40%)
    high_pos_videos = [v for v in videos
                       if v["n_windows"] > 0 and v["labels"].mean() > 0.40]

    # Videos with zero positives
    zero_pos_videos = [v for v in videos if v["labels"].sum() == 0]

    # Short sponsor runs (1 window = 5 sec — often noise or boundary bleed)
    short_runs = sum(
        1 for v in videos
        for s, e in sponsor_runs(v["labels"])
        if (e - s + 1) == 1
    )

    # Videos with only 1–2 positive windows (very sparse signal)
    sparse_pos_videos = [v for v in videos
                         if 1 <= int(v["labels"].sum()) <= 2]

    return dict(
        n_videos          = len(videos),
        total_windows     = total_windows,
        total_positive    = total_positive,
        total_negative    = total_negative,
        base_pos_rate     = base_pos_rate,
        pos_rates         = pos_rates,
        run_lengths_sec   = run_lengths_sec,
        n_runs            = len(run_lengths_sec),
        total_boundary    = total_boundary,
        boundary_pct      = boundary_pct,
        n_high_pos_videos = len(high_pos_videos),
        n_zero_pos_videos = len(zero_pos_videos),
        n_short_runs      = short_runs,
        n_sparse_pos_vids = len(sparse_pos_videos),
        high_pos_examples = [(v["video_id"], float(v["labels"].mean()))
                             for v in sorted(high_pos_videos,
                                             key=lambda x: x["labels"].mean(),
                                             reverse=True)][:20],
    )


# ---------------------------------------------------------------------------
# CSV audit (optional — requires sponsorTimes.csv)
# ---------------------------------------------------------------------------

def audit_csv(csv_path: Path, cached_video_ids: set[str],
              vote_thresholds: list[int]) -> dict:
    """Parse sponsorTimes.csv and compute vote / duration distributions."""
    rows = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("category", "") != "sponsor":
                continue
            vid = row.get("videoID", "").strip()
            if not vid:
                continue
            try:
                votes = int(row.get("votes", "0"))
                start = float(row.get("startTime", "0"))
                end   = float(row.get("endTime", "0"))
            except ValueError:
                continue
            hidden = row.get("hidden", "0")
            if hidden not in ("0", ""):
                continue
            if end <= start:
                continue
            rows.append({
                "video_id": vid,
                "votes":    votes,
                "duration": end - start,
                "in_cache": vid in cached_video_ids,
            })

    cached_rows = [r for r in rows if r["in_cache"]]
    votes_cached = [r["votes"] for r in cached_rows]
    durations    = [r["duration"] for r in cached_rows]

    # Threshold simulation: how many segments/videos survive at each MIN_VOTES
    threshold_stats = {}
    for t in vote_thresholds:
        surviving = [r for r in cached_rows if r["votes"] >= t]
        surviving_vids = len({r["video_id"] for r in surviving})
        threshold_stats[t] = {
            "segments":  len(surviving),
            "videos":    surviving_vids,
            "seg_pct":   len(surviving) / len(cached_rows) * 100 if cached_rows else 0,
            "vid_pct":   surviving_vids / len(cached_video_ids) * 100 if cached_video_ids else 0,
        }

    # Low-vote segment breakdown
    vote_buckets = {
        "3-5":   sum(1 for v in votes_cached if 3 <= v <= 5),
        "6-10":  sum(1 for v in votes_cached if 6 <= v <= 10),
        "11-20": sum(1 for v in votes_cached if 11 <= v <= 20),
        "21-50": sum(1 for v in votes_cached if 21 <= v <= 50),
        "51+":   sum(1 for v in votes_cached if v > 50),
    }

    return dict(
        total_csv_segments = len(rows),
        cached_segments    = len(cached_rows),
        votes              = votes_cached,
        durations          = durations,
        vote_buckets       = vote_buckets,
        threshold_stats    = threshold_stats,
        median_votes       = float(np.median(votes_cached)) if votes_cached else 0,
        mean_votes         = float(np.mean(votes_cached)) if votes_cached else 0,
        pct_low_votes      = sum(1 for v in votes_cached if v < 8) / len(votes_cached) * 100
                             if votes_cached else 0,
    )


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_report(cache_stats: dict, csv_stats: dict | None,
                 vote_thresholds: list[int]) -> None:
    c = cache_stats

    lines = [
        "=" * 72,
        "  SPONSORBLOCK LABEL QUALITY AUDIT",
        "=" * 72,
        "",
        "── Dataset Overview ────────────────────────────────────────────────",
        f"  Videos in cache       : {c['n_videos']:,}",
        f"  Total windows         : {c['total_windows']:,}",
        f"  Positive windows      : {pct(c['total_positive'], c['total_windows'])}",
        f"  Negative windows      : {pct(c['total_negative'], c['total_windows'])}",
        f"  Class imbalance ratio : {c['total_negative'] / c['total_positive']:.1f}:1"
          if c['total_positive'] else "  Class imbalance ratio : N/A",
        "",
        "── Sponsor Run Analysis ────────────────────────────────────────────",
        f"  Total sponsor runs    : {c['n_runs']:,}",
        f"  Mean run duration     : {np.mean(c['run_lengths_sec']):.1f}s"
          if c['run_lengths_sec'] else "  Mean run duration     : N/A",
        f"  Median run duration   : {np.median(c['run_lengths_sec']):.1f}s"
          if c['run_lengths_sec'] else "  Median run duration   : N/A",
        f"  Short runs (≤5s)      : {pct(c['n_short_runs'], c['n_runs'])}"
          if c['n_runs'] else "  Short runs (≤5s)      : N/A",
        "",
        "  Run duration distribution (seconds):",
        ascii_hist(c['run_lengths_sec'], bins=12, lo=0,
                   hi=min(max(c['run_lengths_sec']), 300) if c['run_lengths_sec'] else 300),
        "",
        "── Boundary Window Analysis ────────────────────────────────────────",
        f"  Boundary windows      : {pct(c['total_boundary'], c['total_positive'])}",
        "  of all positive windows",
        f"  Interpretation        : {'HIGH noise risk — boundary imprecision may be significant'  if c['boundary_pct'] > 0.35 else 'Moderate' if c['boundary_pct'] > 0.20 else 'Low — boundaries look clean'}",
        "",
        "── Per-Video Positive Rate Distribution ────────────────────────────",
        "  (what fraction of each video's windows are labeled sponsor)",
        ascii_hist(c['pos_rates'], bins=10, lo=0, hi=1),
        "",
        f"  Videos with 0% positive (no sponsor label)   : {pct(c['n_zero_pos_videos'], c['n_videos'])}",
        f"  Videos with >40% positive (over-labeling?)   : {pct(c['n_high_pos_videos'], c['n_videos'])}",
        f"  Videos with 1–2 positive windows (sparse)    : {pct(c['n_sparse_pos_vids'], c['n_videos'])}",
    ]

    if c['n_high_pos_videos'] > 0:
        lines += [
            "",
            "  Top over-labeled videos (>40% positive rate):",
        ]
        for vid, rate in c['high_pos_examples'][:10]:
            lines.append(f"    https://youtube.com/watch?v={vid}  ({100*rate:.0f}% positive)")

    if csv_stats:
        s = csv_stats
        lines += [
            "",
            "── Vote Distribution (from sponsorTimes.csv) ───────────────────────",
            f"  Segments in cache     : {s['cached_segments']:,}",
            f"  Median votes          : {s['median_votes']:.0f}",
            f"  Mean votes            : {s['mean_votes']:.1f}",
            f"  Segments with <8 votes: {s['pct_low_votes']:.1f}%",
            "",
            "  Vote bucket breakdown:",
            f"    3–5   votes : {pct(s['vote_buckets']['3-5'],  s['cached_segments'])}",
            f"    6–10  votes : {pct(s['vote_buckets']['6-10'], s['cached_segments'])}",
            f"    11–20 votes : {pct(s['vote_buckets']['11-20'],s['cached_segments'])}",
            f"    21–50 votes : {pct(s['vote_buckets']['21-50'],s['cached_segments'])}",
            f"    51+   votes : {pct(s['vote_buckets']['51+'],  s['cached_segments'])}",
            "",
            "  Sponsor segment duration distribution (seconds):",
            ascii_hist(s['durations'], bins=12, lo=0,
                       hi=min(max(s['durations']), 600) if s['durations'] else 600),
            "",
            "── MIN_VOTES Threshold Simulation ──────────────────────────────────",
            "  How much data survives if you raise the vote threshold:",
            f"  {'Threshold':<12}  {'Segments':<18}  {'Videos':<18}  Recommendation",
        ]
        for t in vote_thresholds:
            ts = s['threshold_stats'][t]
            rec = ""
            if t == 3:
                rec = "← current"
            elif ts['seg_pct'] >= 70 and ts['vid_pct'] >= 70:
                rec = "✓ safe to use"
            elif ts['seg_pct'] >= 50:
                rec = "worth trying"
            else:
                rec = "too aggressive"
            lines.append(
                f"  {t:<12}  {ts['segments']:>6,} ({ts['seg_pct']:4.0f}%)   "
                f"{ts['videos']:>6,} ({ts['vid_pct']:4.0f}%)   {rec}"
            )

    lines += [
        "",
        "── Summary & Recommendations ───────────────────────────────────────",
    ]

    # Generate recommendations based on findings
    recs = []

    if c['boundary_pct'] > 0.35:
        recs.append(
            "• HIGH boundary noise: >35% of positive windows are at segment edges.\n"
            "  → Consider soft labeling: assign 0.3–0.4 to edge windows instead of 1.0."
        )
    if c['n_short_runs'] / c['n_runs'] > 0.15 if c['n_runs'] else False:
        recs.append(
            "• >15% of sponsor runs are ≤5s (single window).\n"
            "  → These are likely noise or boundary bleed. Consider filtering runs < 10s."
        )
    if c['n_high_pos_videos'] / c['n_videos'] > 0.05:
        recs.append(
            "• >5% of videos have >40% positive rate — likely over-labeled.\n"
            "  → Review the top offending videos listed above. Consider capping per-video\n"
            "    positive rate or excluding these videos from training."
        )
    if csv_stats and csv_stats['pct_low_votes'] > 40:
        recs.append(
            "• >40% of segments have <8 votes — significant label uncertainty.\n"
            "  → Raise MIN_VOTES to 8–10. Check threshold simulation above for data impact."
        )
    elif csv_stats and csv_stats['pct_low_votes'] > 20:
        recs.append(
            "• 20–40% of segments have <8 votes — moderate label uncertainty.\n"
            "  → Consider raising MIN_VOTES to 5–8 if data permits."
        )

    if not recs:
        recs.append("• No major label quality issues detected. Noise ceiling may be architectural.")

    lines += recs
    lines += ["", "=" * 72]

    print("\n".join(lines))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Audit SponsorBlock label quality.")
    p.add_argument("--cache-dir", required=True, type=Path,
                   help="Directory of per-video .npz cache files.")
    p.add_argument("--csv", type=Path, default=None,
                   help="Path to sponsorTimes.csv (optional, enables vote analysis).")
    p.add_argument("--vote-thresholds", type=str, default="3,5,8,10,15,20",
                   help="Comma-separated MIN_VOTES values to simulate (default: 3,5,8,10,15,20).")
    p.add_argument("--out", type=Path, default=None,
                   help="Optional path to write summary JSON.")
    args = p.parse_args()

    thresholds = [int(x) for x in args.vote_thresholds.split(",")]

    print(f"Loading cache from {args.cache_dir} …")
    videos = load_cache(args.cache_dir)
    print(f"Loaded {len(videos):,} videos.")

    cache_stats = audit_cache(videos)

    csv_stats = None
    if args.csv:
        print(f"Parsing CSV: {args.csv} …")
        cached_ids = {v["video_id"] for v in videos}
        csv_stats = audit_csv(args.csv, cached_ids, thresholds)

    print_report(cache_stats, csv_stats, thresholds)

    if args.out:
        summary = {
            "cache": {k: v for k, v in cache_stats.items()
                      if k not in ("pos_rates", "run_lengths_sec", "high_pos_examples")},
            "csv": csv_stats,
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=2))
        print(f"\nSummary JSON → {args.out}")


if __name__ == "__main__":
    main()
