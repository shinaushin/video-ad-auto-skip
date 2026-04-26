#!/usr/bin/env python3
"""
Sponsor Detection Benchmark

Compares transcript-based sponsor detection against SponsorBlock's
community-maintained ground truth.

Modes:

  DATABASE MODE (recommended for large-scale testing):
    python3 benchmark.py --db                   # sample 50 videos
    python3 benchmark.py --db --sample 200      # sample 200 videos
    python3 benchmark.py --db --workers 5       # parallel caption fetches

  API MODE (for testing specific videos):
    python3 benchmark.py VIDEO_ID [VIDEO_ID ...]

  Options:
    --db            Download SponsorBlock's full database (~2-4 GB, cached)
    --sample N      Number of videos to sample in DB mode (default: 50)
    --workers N     Parallel YouTube caption fetches (default: 3)
    --quiet         One-line-per-video output

Requirements: Python 3.7+. Zero pip dependencies (stdlib only).
"""

import argparse
import csv
import json
import os
import random
import re
import sys
import time
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from xml.etree import ElementTree

# ─── Detection logic (mirrored from content.js) ────────────────

STRONG_PATTERNS = [
    re.compile(r"this (?:video|segment|portion) is (?:brought to you|sponsored|made possible) by", re.I),
    re.compile(r"(?:sponsored|presented) by", re.I),
    re.compile(r"today'?s sponsor", re.I),
    re.compile(r"thanks to .{1,40} for sponsoring", re.I),
    re.compile(r"a (?:huge|big|special) thanks? to", re.I),
    re.compile(r"brought to you by", re.I),
    re.compile(r"use (?:my |our )?(?:code|link)", re.I),
    re.compile(r"use code .{1,20} (?:at|for) checkout", re.I),
    re.compile(r"go to .{1,40}\.com", re.I),
    re.compile(r"head (?:on )?over to .{1,40}\.com", re.I),
    re.compile(r"visit .{1,40}\.com", re.I),
    re.compile(r"check (?:them )?out at", re.I),
    re.compile(r"click (?:the|my) link", re.I),
    re.compile(r"link (?:is )?in (?:the )?description", re.I),
    re.compile(r"first \d+ (?:people|users|customers|subscribers)", re.I),
    re.compile(r"free trial", re.I),
    re.compile(r"percent off", re.I),
    re.compile(r"% off", re.I),
    re.compile(r"discount code", re.I),
    re.compile(r"promo code", re.I),
    re.compile(r"coupon code", re.I),
]

WEAK_PATTERNS = [
    re.compile(r"sign up", re.I),
    re.compile(r"download the app", re.I),
    re.compile(r"available (?:now )?at", re.I),
    re.compile(r"money back guarantee", re.I),
    re.compile(r"limited time", re.I),
    re.compile(r"exclusive (?:deal|offer)", re.I),
    re.compile(r"subscribe", re.I),
    re.compile(r"premium", re.I),
]

PADDING_BEFORE = 1.5
PADDING_AFTER = 2.0
MERGE_GAP_SEC = 8
MIN_KEYWORD_HITS = 2


def score_cue(text: str) -> int:
    """Score a caption cue for sponsor likelihood."""
    score = 0
    for pat in STRONG_PATTERNS:
        if pat.search(text):
            score += 3
    for pat in WEAK_PATTERNS:
        if pat.search(text):
            score += 1
    return score


def detect_sponsor_segments(cues: list[dict]) -> list[dict]:
    """
    Run sponsor detection on a list of caption cues.
    Each cue: {start: float, dur: float, text: str}.
    Returns: [{start: float, end: float}, ...].
    """
    if not cues:
        return []

    scored = []
    for c in cues:
        s = score_cue(c["text"])
        if s > 0:
            scored.append({**c, "end": c["start"] + c["dur"], "score": s})

    if not scored:
        return []

    # Cluster nearby hits
    clusters = [[scored[0]]]
    for i in range(1, len(scored)):
        prev = clusters[-1][-1]
        curr = scored[i]
        if curr["start"] - prev["end"] <= MERGE_GAP_SEC:
            clusters[-1].append(curr)
        else:
            clusters.append([curr])

    # Filter by minimum keyword density and build segments
    segments = []
    for cluster in clusters:
        total_score = sum(c["score"] for c in cluster)
        if total_score < MIN_KEYWORD_HITS * 3:
            continue
        start = max(0, cluster[0]["start"] - PADDING_BEFORE)
        end = cluster[-1]["end"] + PADDING_AFTER
        segments.append({"start": start, "end": end})

    return segments


# ─── HTTP helpers ───────────────────────────────────────────────

HEADERS = {"User-Agent": "SponsorSpeedBenchmark/1.0 (Python)"}


def http_get(url: str, timeout: int = 30) -> str:
    """Fetch a URL and return its body as a string."""
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def http_get_json(url: str, timeout: int = 30):
    """Fetch a URL and parse the response as JSON."""
    try:
        body = http_get(url, timeout)
        return json.loads(body)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise


# ─── YouTube caption fetching ──────────────────────────────────

def get_captions_for_video(video_id: str) -> list[dict] | None:
    """
    Fetch YouTube captions for a video.
    Returns a list of cues [{start, dur, text}] or None if unavailable.
    """
    html = http_get(f"https://www.youtube.com/watch?v={video_id}", timeout=15)

    # Extract captionTracks JSON from the player response
    match = re.search(r'"captionTracks":\s*(\[.*?\])', html)
    if not match:
        return None

    try:
        tracks = json.loads(match.group(1))
    except json.JSONDecodeError:
        return None

    if not tracks:
        return None

    # Prefer English
    english = next((t for t in tracks if t.get("languageCode", "").startswith("en")), None)
    track = english or tracks[0]
    base_url = track.get("baseUrl")
    if not base_url:
        return None

    # Fetch and parse timed-text XML
    xml_text = http_get(base_url, timeout=15)
    try:
        root = ElementTree.fromstring(xml_text)
    except ElementTree.ParseError:
        return None

    cues = []
    for elem in root.iter("text"):
        start = float(elem.get("start", "0"))
        dur = float(elem.get("dur", "2"))
        text = (elem.text or "").replace("\n", " ").strip()
        if text:
            cues.append({"start": start, "dur": dur, "text": text})

    return cues if cues else None


# ─── SponsorBlock API (single-video mode) ──────────────────────

def get_sponsorblock_segments(video_id: str) -> list[dict]:
    """Fetch sponsor segments from the SponsorBlock API for one video."""
    url = (
        f"https://sponsor.ajay.app/api/skipSegments"
        f"?videoID={video_id}"
        f"&categories=%5B%22sponsor%22%5D"
    )
    data = http_get_json(url)
    if not data or not isinstance(data, list):
        return []
    return [
        {"start": seg["segment"][0], "end": seg["segment"][1]}
        for seg in data
        if seg.get("category") == "sponsor" and seg.get("actionType") == "skip"
    ]


# ═══════════════════════════════════════════════════════════════
#  DATABASE MODE
# ═══════════════════════════════════════════════════════════════

CACHE_DIR = Path(__file__).parent / ".cache"
CACHE_FILE = CACHE_DIR / "sponsorTimes.csv"

DB_MIRRORS = [
    "https://sb.ltn.fi/database/sponsorTimes.csv",
    "https://mirror.sb.mchang.xyz/sponsorTimes.csv",
    "https://sponsor.ajay.app/database/sponsorTimes.csv",
]


def ensure_database() -> Path:
    """Download the SponsorBlock database if not cached. Returns CSV path."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if CACHE_FILE.exists() and CACHE_FILE.stat().st_size > 1_000_000:
        size_mb = CACHE_FILE.stat().st_size / 1e6
        age_hours = (time.time() - CACHE_FILE.stat().st_mtime) / 3600
        print(f"Using cached database ({size_mb:.0f} MB, {age_hours:.1f}h old)")
        if age_hours > 168:
            print("  Hint: database is over a week old. Delete .cache/ to re-download.\n")
        return CACHE_FILE

    print("Downloading SponsorBlock database...")
    print("(This is a large file, ~2-4 GB. It will be cached for future runs.)\n")

    for mirror_url in DB_MIRRORS:
        try:
            print(f"Trying {mirror_url}...")
            download_file(mirror_url, CACHE_FILE)
            size_mb = CACHE_FILE.stat().st_size / 1e6
            print(f"Download complete ({size_mb:.0f} MB)\n")
            return CACHE_FILE
        except Exception as e:
            print(f"  Failed: {e}")

    raise RuntimeError(
        "Could not download the SponsorBlock database from any mirror.\n"
        "You can download it manually from https://sb.ltn.fi/database/\n"
        f"and place it at: {CACHE_FILE}"
    )


def download_file(url: str, dest: Path):
    """Download a URL to a local file with progress display."""
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=60) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        last_pct = -1

        with open(dest, "wb") as f:
            while True:
                chunk = resp.read(1024 * 256)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if total > 0:
                    pct = int(downloaded / total * 100)
                    if pct != last_pct and pct % 5 == 0:
                        print(f"  {pct}% ({downloaded / 1e6:.0f} MB)", end="\r")
                        last_pct = pct

    print()


def parse_database(csv_path: Path, sample_size: int = 50) -> list[dict]:
    """
    Parse sponsorTimes.csv and extract high-confidence sponsor segments.

    Filters for: category=sponsor, votes>=1, not hidden, video>60s.
    Groups by videoID, keeps only videos with at least one segment
    with votes>=3, merges overlapping segments, and returns a random
    sample.

    Returns: [{videoId: str, segments: [{start, end}]}]
    """
    print("Parsing database for high-confidence sponsor segments...")

    video_segments: dict[str, list[dict]] = {}
    line_count = 0
    match_count = 0

    with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)

        for row in reader:
            line_count += 1

            if line_count % 500_000 == 0:
                print(
                    f"  Scanned {line_count / 1e6:.1f}M rows, "
                    f"found {len(video_segments)} qualifying videos...",
                    end="\r",
                )

            # Filter criteria
            if row.get("category") != "sponsor":
                continue
            if row.get("hidden") == "1" or row.get("shadowHidden") == "1":
                continue
            try:
                votes = float(row.get("votes", "0"))
                if votes < 1:
                    continue
                duration = float(row.get("videoDuration", "0"))
                if duration < 60:
                    continue
            except ValueError:
                continue

            service = row.get("service", "")
            if service and service != "YouTube":
                continue

            vid = row.get("videoID", "")
            if not vid or len(vid) != 11:
                continue

            try:
                start = float(row["startTime"])
                end = float(row["endTime"])
            except (ValueError, KeyError):
                continue

            video_segments.setdefault(vid, []).append(
                {"start": start, "end": end, "votes": int(votes)}
            )
            match_count += 1

    print()
    print(f"Scanned {line_count:,} rows total")
    print(
        f"Found {match_count:,} qualifying sponsor segments "
        f"across {len(video_segments):,} videos"
    )

    # Filter to videos with at least one segment with votes >= 3
    high_confidence = []
    for vid, segs in video_segments.items():
        if not any(s["votes"] >= 3 for s in segs):
            continue

        # Merge overlapping/adjacent segments
        sorted_segs = sorted(segs, key=lambda s: s["start"])
        merged = [dict(sorted_segs[0])]
        for seg in sorted_segs[1:]:
            prev = merged[-1]
            if seg["start"] <= prev["end"] + 2:
                prev["end"] = max(prev["end"], seg["end"])
                prev["votes"] = max(prev["votes"], seg["votes"])
            else:
                merged.append(dict(seg))

        high_confidence.append({
            "videoId": vid,
            "segments": [{"start": s["start"], "end": s["end"]} for s in merged],
        })

    print(f"High-confidence videos (votes >= 3): {len(high_confidence):,}")

    random.shuffle(high_confidence)
    sample = high_confidence[:sample_size]
    print(f"Sampled {len(sample)} videos for benchmarking\n")
    return sample


# ─── Metrics ───────────────────────────────────────────────────

def compute_metrics(detected: list[dict], ground_truth: list[dict]) -> dict:
    """
    Compute precision, recall, and IoU between detected and ground-truth
    segments using a sweep-line algorithm.
    """
    if not ground_truth and not detected:
        return {"precision": 1, "recall": 1, "iou": 1, "true_negative": True, "details": []}
    if not ground_truth and detected:
        return {
            "precision": 0, "recall": 1, "iou": 0,
            "details": [{"type": "false_positive", "detected": d} for d in detected],
        }
    if not detected and ground_truth:
        return {
            "precision": 1, "recall": 0, "iou": 0,
            "details": [{"type": "missed", "ground_truth": gt} for gt in ground_truth],
        }

    gt_total = sum(gt["end"] - gt["start"] for gt in ground_truth)
    det_total = sum(d["end"] - d["start"] for d in detected)

    # Sweep-line intersection
    events = []
    for gt in ground_truth:
        events.append((gt["start"], "gs"))
        events.append((gt["end"], "ge"))
    for d in detected:
        events.append((d["start"], "ds"))
        events.append((d["end"], "de"))
    events.sort()

    gt_active = 0
    det_active = 0
    prev_time = 0
    intersection = 0

    for t, kind in events:
        dt = t - prev_time
        if dt > 0 and gt_active > 0 and det_active > 0:
            intersection += dt
        prev_time = t
        if kind == "gs":
            gt_active += 1
        elif kind == "ge":
            gt_active -= 1
        elif kind == "ds":
            det_active += 1
        else:
            det_active -= 1

    union = det_total + gt_total - intersection
    precision = intersection / det_total if det_total > 0 else 0
    recall = intersection / gt_total if gt_total > 0 else 0
    iou = intersection / union if union > 0 else 0

    # Per-GT-segment coverage
    details = []
    for gt in ground_truth:
        overlap = 0
        for d in detected:
            s = max(gt["start"], d["start"])
            e = min(gt["end"], d["end"])
            if e > s:
                overlap += e - s
        gt_dur = gt["end"] - gt["start"]
        details.append({
            "ground_truth": gt,
            "coverage": overlap / gt_dur if gt_dur > 0 else 0,
        })

    return {"precision": precision, "recall": recall, "iou": iou, "details": details}


# ─── Formatting ────────────────────────────────────────────────

def fmt_time(sec: float) -> str:
    return f"{int(sec) // 60}:{int(sec) % 60:02d}"


def pct(n: float) -> str:
    return f"{n * 100:.1f}%"


def seg_str(seg: dict) -> str:
    dur = seg["end"] - seg["start"]
    return f"{fmt_time(seg['start'])} → {fmt_time(seg['end'])} ({dur:.0f}s)"


# ─── Benchmark a single video ──────────────────────────────────

def benchmark_video(
    video_id: str,
    ground_truth: list[dict],
    verbose: bool = True,
) -> dict:
    """Benchmark our detection against ground truth for one video."""
    if verbose:
        header = f"── Video: {video_id} "
        print(f"\n{header}{'─' * max(0, 60 - len(header))}")
        print(f"  Ground truth: {len(ground_truth)} segment(s)")
        if ground_truth:
            print("    " + "\n    ".join(seg_str(s) for s in ground_truth))

    # Fetch captions
    try:
        cues = get_captions_for_video(video_id)
    except Exception as e:
        if verbose:
            print(f"  Captions: error — {e}")
        return {"videoId": video_id, "ground_truth": ground_truth, "detected": [],
                "metrics": None, "error": str(e)}

    if not cues:
        if verbose:
            print("  Captions: not available")
        return {"videoId": video_id, "ground_truth": ground_truth, "detected": [],
                "metrics": None, "no_captions": True}

    if verbose:
        print(f"  Captions: {len(cues)} cues")

    detected = detect_sponsor_segments(cues)
    if verbose:
        print(f"  Detected: {len(detected)} segment(s)")
        if detected:
            print("    " + "\n    ".join(seg_str(s) for s in detected))

    metrics = compute_metrics(detected, ground_truth)
    if verbose:
        print(f"  Precision: {pct(metrics['precision'])}  "
              f"Recall: {pct(metrics['recall'])}  "
              f"IoU: {pct(metrics['iou'])}")
        for d in metrics.get("details", []):
            if d.get("type") == "false_positive":
                print(f"  ⚠ False positive: {seg_str(d['detected'])}")
            elif d.get("type") == "missed":
                print(f"  ✗ Missed: {seg_str(d['ground_truth'])}")
            elif "coverage" in d:
                icon = "✓" if d["coverage"] >= 0.5 else "✗"
                print(f"  {icon} GT {seg_str(d['ground_truth'])} — {pct(d['coverage'])} covered")

    return {"videoId": video_id, "ground_truth": ground_truth,
            "detected": detected, "metrics": metrics}


# ─── Main ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark sponsor detection against SponsorBlock ground truth"
    )
    parser.add_argument("video_ids", nargs="*", help="YouTube video IDs (API mode)")
    parser.add_argument("--db", action="store_true",
                        help="Use full SponsorBlock database (~2-4 GB, cached)")
    parser.add_argument("--sample", type=int, default=50,
                        help="Videos to sample in DB mode (default: 50)")
    parser.add_argument("--workers", type=int, default=3,
                        help="Parallel caption fetches (default: 3)")
    parser.add_argument("--quiet", action="store_true",
                        help="One-line-per-video output")
    args = parser.parse_args()

    if not args.db and not args.video_ids:
        parser.print_help()
        print("\nExamples:")
        print("  python3 benchmark.py --db                   # sample 50 from full database")
        print("  python3 benchmark.py --db --sample 500      # sample 500 videos")
        print("  python3 benchmark.py VIDEO_ID1 VIDEO_ID2    # test specific videos")
        sys.exit(0)

    print("YouTube Sponsor Speeder — Detection Benchmark")
    print("═" * 50)

    # Build test cases: [{videoId, segments}]
    test_cases = []

    if args.db:
        csv_path = ensure_database()
        test_cases = parse_database(csv_path, args.sample)
    else:
        for vid in args.video_ids:
            try:
                segments = get_sponsorblock_segments(vid)
                test_cases.append({"videoId": vid, "segments": segments})
            except Exception as e:
                print(f"  Warning: Could not fetch SponsorBlock data for {vid}: {e}")

    print(f"Testing {len(test_cases)} video(s) with {args.workers} worker(s)...\n")

    # Run benchmarks in parallel
    results = [None] * len(test_cases)
    completed = 0
    verbose = not args.quiet

    def run_one(idx_tc):
        idx, tc = idx_tc
        time.sleep(idx * 0.3)  # stagger requests to be polite
        return idx, benchmark_video(tc["videoId"], tc["segments"], verbose=verbose)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_one, (i, tc)): i for i, tc in enumerate(test_cases)}
        for future in as_completed(futures):
            idx, result = future.result()
            results[idx] = result
            completed += 1
            if not verbose:
                status = ("no-captions" if result.get("no_captions")
                          else f"error" if result.get("error")
                          else f"P:{pct(result['metrics']['precision'])} "
                               f"R:{pct(result['metrics']['recall'])}")
                print(f"  [{completed}/{len(test_cases)}] {result['videoId']} — {status}")

    # ── Aggregate summary ──────────────────────────────────────

    print("\n" + "═" * 50)
    print("AGGREGATE RESULTS")
    print("═" * 50)

    successful = [r for r in results if r and r.get("metrics")]
    with_segments = [r for r in successful if not r["metrics"].get("true_negative")]
    no_captions = [r for r in results if r and r.get("no_captions")]
    errors = [r for r in results if r and r.get("error")]
    true_neg = [r for r in successful if r["metrics"].get("true_negative")]

    print(f"Total videos tested:     {len(results)}")
    print(f"  Successful benchmarks: {len(successful)}")
    print(f"  With sponsor segments: {len(with_segments)}")
    print(f"  True negatives:        {len(true_neg)}")
    print(f"  No captions available: {len(no_captions)}")
    print(f"  Errors:                {len(errors)}")

    if with_segments:
        avg_p = sum(r["metrics"]["precision"] for r in with_segments) / len(with_segments)
        avg_r = sum(r["metrics"]["recall"] for r in with_segments) / len(with_segments)
        avg_iou = sum(r["metrics"]["iou"] for r in with_segments) / len(with_segments)

        total_gt = 0
        covered_gt = 0
        total_fp = 0
        for r in with_segments:
            for d in r["metrics"].get("details", []):
                if d.get("type") == "false_positive":
                    total_fp += 1
                elif d.get("type") == "missed":
                    total_gt += 1
                elif "coverage" in d:
                    total_gt += 1
                    if d["coverage"] >= 0.5:
                        covered_gt += 1

        print(f"\n── Detection Accuracy ──")
        print(f"Average Precision:  {pct(avg_p)}")
        print(f"Average Recall:     {pct(avg_r)}")
        print(f"Average IoU:        {pct(avg_iou)}")
        print(f"\nSegment-level:")
        print(f"  Ground truth segments:    {total_gt}")
        print(f"  Correctly covered (≥50%): {covered_gt} "
              f"({pct(covered_gt / total_gt) if total_gt else 'N/A'})")
        print(f"  False positives:          {total_fp}")

        print("\n── Interpretation ──")
        print("  Precision = of the time we flagged, how much was actually a sponsor")
        print("  Recall    = of the real sponsor time, how much did we catch")
        print("  IoU       = overall overlap quality (higher is better)")

        if avg_r < 0.3:
            print("\n⚠ Low recall: many sponsor segments being missed.")
            print("  → Add more keyword patterns or lower MIN_KEYWORD_HITS")
        elif avg_r < 0.5:
            print("\n⚠ Moderate recall: some sponsors being missed.")
        if avg_p < 0.3:
            print("\n⚠ Low precision: many false positives.")
            print("  → Raise MIN_KEYWORD_HITS or remove noisy WEAK_PATTERNS")
        elif avg_p < 0.5:
            print("\n⚠ Moderate precision: some false positives.")
        if avg_iou >= 0.4:
            print("\n✓ Decent IoU — detection is roughly aligned with ground truth.")

    # Write JSON report
    report_path = Path(__file__).parent / "benchmark-results.json"
    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mode": "database" if args.db else "api",
        "sample_size": len(test_cases),
        "summary": {
            "total": len(results),
            "successful": len(successful),
            "with_segments": len(with_segments),
            "no_captions": len(no_captions),
            "errors": len(errors),
        },
        "averages": {
            "precision": sum(r["metrics"]["precision"] for r in with_segments) / len(with_segments),
            "recall": sum(r["metrics"]["recall"] for r in with_segments) / len(with_segments),
            "iou": sum(r["metrics"]["iou"] for r in with_segments) / len(with_segments),
        } if with_segments else None,
        "videos": [
            {
                "videoId": r["videoId"],
                "ground_truth_count": len(r.get("ground_truth") or []),
                "detected_count": len(r.get("detected") or []),
                "precision": r["metrics"]["precision"] if r.get("metrics") else None,
                "recall": r["metrics"]["recall"] if r.get("metrics") else None,
                "iou": r["metrics"]["iou"] if r.get("metrics") else None,
                "no_captions": r.get("no_captions", False),
                "error": r.get("error"),
            }
            for r in results if r
        ],
    }
    report_path.write_text(json.dumps(report, indent=2))
    print(f"\nDetailed results written to: {report_path}")
    print()


if __name__ == "__main__":
    main()
