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
import http.cookiejar
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from xml.etree import ElementTree

# ─── Detection logic (mirrored from content.js) ────────────────

STRONG_PATTERNS = [
    # Explicit sponsorship disclosure
    re.compile(r"this (?:video|segment|portion|episode) is (?:brought to you|sponsored|made possible) by", re.I),
    re.compile(r"(?:sponsored|presented) by", re.I),
    re.compile(r"today'?s (?:video is )?sponsor", re.I),
    re.compile(r"this (?:week|episode)'?s sponsor", re.I),
    re.compile(r"our (?:sponsor|partner)", re.I),
    re.compile(r"(?:a )?(?:quick |brief )?(?:word|message) from (?:our |today'?s )?sponsor", re.I),
    re.compile(r"(?:video|episode) (?:is )?(?:sponsored|supported) by", re.I),
    re.compile(r"made (?:this video )?possible by", re.I),
    re.compile(r"thanks? to .{1,40} for (?:sponsoring|supporting|making this)", re.I),
    re.compile(r"a (?:huge|big|special) thanks? to", re.I),
    re.compile(r"brought to you by", re.I),
    re.compile(r"partnered with", re.I),
    re.compile(r"in partnership with", re.I),
    # Calls to action with codes / links
    re.compile(r"use (?:my |our )?(?:code|link|referral)", re.I),
    re.compile(r"use code .{1,20} (?:at|for) checkout", re.I),
    re.compile(r"(?:my|our|the) (?:affiliate |referral )?(?:code|link) (?:is |in )", re.I),
    re.compile(r"if you (?:use|click) (?:my|the) link", re.I),
    re.compile(r"link (?:is )?in (?:the )?description", re.I),
    re.compile(r"click (?:the|my) link", re.I),
    re.compile(r"check (?:them )?out at", re.I),
    re.compile(r"go to .{1,40}\.com", re.I),
    re.compile(r"head (?:on )?over to .{1,40}\.com", re.I),
    re.compile(r"visit .{1,40}\.com", re.I),
    re.compile(r"download .{1,30} (?:app|today|now|for free)", re.I),
    # Offers and discounts
    re.compile(r"first \d+ (?:people|users|customers|subscribers)", re.I),
    re.compile(r"(?:get |try ).{1,30} for free", re.I),
    re.compile(r"free trial", re.I),
    re.compile(r"\d+ (?:days?|months?) (?:free|trial)", re.I),
    re.compile(r"free (?:\d+-)?(?:day|month|week) trial", re.I),
    re.compile(r"percent off", re.I),
    re.compile(r"\d+% off", re.I),
    re.compile(r"(?:save|get) \d+%", re.I),
    re.compile(r"discount code", re.I),
    re.compile(r"promo code", re.I),
    re.compile(r"coupon code", re.I),
    re.compile(r"exclusive (?:discount|deal|offer|code) (?:for )?(?:my )?(?:viewers?|subscribers?|listeners?|audience)", re.I),
    re.compile(r"(?:my |our )?viewers? (?:get|receive|save|can get)", re.I),
]

WEAK_PATTERNS = [
    re.compile(r"sign up", re.I),
    re.compile(r"download the app", re.I),
    re.compile(r"available (?:now )?at", re.I),
    re.compile(r"money.?back guarantee", re.I),
    re.compile(r"limited time", re.I),
    re.compile(r"exclusive (?:deal|offer)", re.I),
    re.compile(r"start (?:your )?(?:free )?(?:trial|subscription)", re.I),
    re.compile(r"check (?:it|them) out", re.I),
    re.compile(r"learn more", re.I),
    re.compile(r"(?:scan|use) (?:the )?QR code", re.I),
    re.compile(r"premium (?:plan|subscription|membership|account)", re.I),
    re.compile(r"no ?(?:cost|charge|credit card)(?: required)?", re.I),
]

# Detection tuning constants
PADDING_BEFORE       = 3.0   # seconds before first keyword hit to include
PADDING_AFTER        = 5.0   # seconds after last keyword hit to include
MERGE_GAP_SEC        = 15    # merge clusters with gaps up to this many seconds
WINDOW_SEC           = 25    # sliding window width for context-aware scoring
MIN_SCORE            = 3     # minimum window score to flag (one strong hit = 3)
MIN_SEGMENT_DURATION = 45.0  # floor: any detected segment is at least this long
SILENCE_BOUNDARY_SEC = 1.5   # gap between consecutive cues treated as a break

# Phrases that signal "sponsor is starting" — used for backward boundary walk
SPONSOR_INTRO_PATTERNS = [
    re.compile(r"\bbefore (?:we|I) (?:get into|start|continue|dive)\b", re.I),
    re.compile(r"\breal quick\b", re.I),
    re.compile(r"\bquick(?:ly)? (?:want to|shout.?out)\b", re.I),
    re.compile(r"\bspeaking of\b", re.I),
    re.compile(r"\btake a (?:quick )?break\b", re.I),
    re.compile(r"\bfirst,? (?:a )?(?:quick )?word\b", re.I),
]

# Phrases that signal "sponsor is over, regular content resumes"
CONTENT_RETURN_PATTERNS = [
    re.compile(r"\banyway(?:s)?\b", re.I),
    re.compile(r"\bwith that (?:said|out of the way)\b", re.I),
    re.compile(r"\bback to (?:the|today|our|it)\b", re.I),
    re.compile(r"\blet'?s (?:get back|continue|dive back)\b", re.I),
    re.compile(r"\benough (?:about|of) that\b", re.I),
    re.compile(r"\bnow (?:where were we|let'?s get back)\b", re.I),
    re.compile(r"\bso(?: anyway)?,? (?:back|let'?s)\b", re.I),
]


def score_text(text: str) -> int:
    """Score a block of text for sponsor likelihood."""
    score = 0
    for pat in STRONG_PATTERNS:
        if pat.search(text):
            score += 3
    for pat in WEAK_PATTERNS:
        if pat.search(text):
            score += 1
    return score


# Keep the old name as an alias so tests and callers don't break
def score_cue(text: str) -> int:
    return score_text(text)


def _walk_boundary_back(cues: list[dict], from_idx: int) -> float:
    """
    Walk backward from cues[from_idx] to find a natural start boundary.

    Stops (and returns that cue's start) when either:
    - the gap between cues[i] and cues[i-1] exceeds SILENCE_BOUNDARY_SEC, OR
    - cues[i].text matches a SPONSOR_INTRO_PATTERNS phrase.

    Falls back to cues[from_idx]["start"] if no boundary is found within
    the look-back window (we never walk past the start of the cluster's
    cue or more than WINDOW_SEC seconds behind it).
    """
    anchor_start = cues[from_idx]["start"]
    best = anchor_start
    for i in range(from_idx, -1, -1):
        cue = cues[i]
        # Don't walk more than WINDOW_SEC back from the anchor
        if anchor_start - cue["start"] > WINDOW_SEC:
            break
        # Silence gap: gap between cues[i-1].end and cues[i].start
        if i > 0:
            prev_end = cues[i - 1]["start"] + cues[i - 1].get("dur", 0)
            if cue["start"] - prev_end > SILENCE_BOUNDARY_SEC:
                # Natural break just before cue[i] — start here
                best = cue["start"]
                break
        # Sponsor-intro phrase: start at the beginning of this cue
        if any(p.search(cue["text"]) for p in SPONSOR_INTRO_PATTERNS):
            best = cue["start"]
            break
    return best


def _walk_boundary_forward(cues: list[dict], from_idx: int) -> float:
    """
    Walk forward from cues[from_idx] to find a natural end boundary.

    Stops when either:
    - the gap between cues[i] and cues[i+1] exceeds SILENCE_BOUNDARY_SEC, OR
    - cues[i+1].text matches a CONTENT_RETURN_PATTERNS phrase.

    Returns the end timestamp (start + dur) of the stopping cue.
    """
    anchor_end = cues[from_idx]["start"] + cues[from_idx].get("dur", 0)
    best = anchor_end
    for i in range(from_idx, len(cues)):
        cue = cues[i]
        cue_end = cue["start"] + cue.get("dur", 0)
        # Don't walk more than WINDOW_SEC forward from the anchor
        if cue["start"] - anchor_end > WINDOW_SEC:
            break
        best = cue_end
        # Silence gap after this cue
        if i + 1 < len(cues):
            next_cue = cues[i + 1]
            if next_cue["start"] - cue_end > SILENCE_BOUNDARY_SEC:
                break
            # Content-return phrase in the *next* cue — end before it starts
            if any(p.search(next_cue["text"]) for p in CONTENT_RETURN_PATTERNS):
                break
    return best


def detect_sponsor_segments(cues: list[dict]) -> list[dict]:
    """
    Sliding-window sponsor detection with natural boundary walking.

    Step 1 — Score each cue using combined text of its ±WINDOW_SEC/2 window,
              catching sponsor language split across cues.
    Step 2 — Cluster nearby hits (gap ≤ MERGE_GAP_SEC).
    Step 3 — For each cluster, walk backward/forward through the cues list
              to find silence gaps or intro/outro boundary phrases instead of
              using fixed padding.  This expands detections to cover the full
              sponsor read rather than just the disclosure sentence.
    Step 4 — Apply MIN_SEGMENT_DURATION floor: if the resulting segment is
              shorter than 45 s, extend it symmetrically to that length.

    Each cue: {start: float, dur: float, text: str}.
    Returns: [{start: float, end: float}, ...].
    """
    if not cues:
        return []

    half = WINDOW_SEC / 2

    # ── Step 1: score each cue against its sliding window ──
    scored = []
    for i, c in enumerate(cues):
        t = c["start"]
        window_text = " ".join(
            x["text"] for x in cues
            if t - half <= x["start"] <= t + half
        )
        s = score_text(window_text)
        if s >= MIN_SCORE:
            scored.append({**c, "end": c["start"] + c["dur"], "score": s})

    if not scored:
        return []

    # ── Step 2: cluster nearby hits ──
    clusters = [[scored[0]]]
    for i in range(1, len(scored)):
        prev = clusters[-1][-1]
        curr = scored[i]
        if curr["start"] - prev["end"] <= MERGE_GAP_SEC:
            clusters[-1].append(curr)
        else:
            clusters.append([curr])

    # Build a fast index: cue start-time → position in cues list
    # (needed to locate the cluster anchors inside cues[])
    start_to_idx: dict[float, int] = {c["start"]: i for i, c in enumerate(cues)}

    # ── Step 3 + 4: walk boundaries, enforce minimum duration ──
    segments = []
    for cluster in clusters:
        anchor_start_cue = cluster[0]
        anchor_end_cue   = cluster[-1]

        # Find positions in the original cues list
        back_idx  = start_to_idx.get(anchor_start_cue["start"], 0)
        fwd_idx   = start_to_idx.get(anchor_end_cue["start"],
                                     len(cues) - 1)

        raw_start = _walk_boundary_back(cues, back_idx)
        raw_end   = _walk_boundary_forward(cues, fwd_idx)

        # Clamp start to ≥ 0
        raw_start = max(0.0, raw_start)

        # Step 4: enforce minimum segment duration
        duration = raw_end - raw_start
        if duration < MIN_SEGMENT_DURATION:
            deficit = MIN_SEGMENT_DURATION - duration
            raw_start = max(0.0, raw_start - deficit / 2)
            raw_end   = raw_end + deficit / 2

        segments.append({"start": raw_start, "end": raw_end})

    return segments


# ─── HTTP helpers ───────────────────────────────────────────────

BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)
HEADERS = {"User-Agent": BROWSER_UA}


def http_get(url: str, timeout: int = 30) -> str:
    """Fetch a URL and return its body as a string."""
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def http_post_json(url: str, body: dict, timeout: int = 30):
    """POST JSON to a URL and return the parsed JSON response."""
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={**HEADERS, "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8", errors="replace"))


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

# YouTube InnerTube API client configs, tried in order.
# TVHTML5 bypasses consent walls and age-gates most reliably.
# ANDROID is a robust fallback. WEB is last as it's most bot-detected.
# Note: the ?key= parameter was removed — it is deprecated and optional.
INNERTUBE_CLIENTS = [
    {
        "clientName": "TVHTML5",
        "clientVersion": "7.20240201.16.00",
    },
    {
        "clientName": "ANDROID",
        "clientVersion": "19.09.37",
        "androidSdkVersion": 30,
    },
    {
        "clientName": "WEB",
        "clientVersion": "2.20250401.00.00",
    },
]


def _caption_cache_path(video_id: str) -> Path:
    return CAPTION_CACHE_DIR / f"{video_id}.json"


def _load_caption_cache(video_id: str) -> list[dict] | None:
    p = _caption_cache_path(video_id)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text())
        # None sentinel means we previously confirmed no captions available
        if data is None:
            return []          # empty list → caller treats as no_captions
        return data or None
    except Exception:
        return None


def _save_caption_cache(video_id: str, cues: list[dict] | None):
    CAPTION_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    p = _caption_cache_path(video_id)
    # Store None as JSON null so we don't re-fetch videos with no captions
    p.write_text(json.dumps(cues))


def get_captions_for_video(
    video_id: str,
    debug: bool = False,
    cookies_from_browser: str | None = None,
    cookies_file: str | None = None,
) -> list[dict] | None:
    """
    Fetch YouTube captions using four strategies in order:

    1. InnerTube /player API — YouTube's internal JSON API, tries multiple
       client configs (TVHTML5 → ANDROID → WEB).
    2. HTML scrape (URL-only) — extracts a signed timedtext URL from the
       watch page, then fetches it in a separate request.
    3. HTML scrape with cookie session — like strategy 2 but shares
       a CookieJar (including SOCS consent) between the page fetch and
       the timedtext fetch.
    4. yt-dlp — used if the `yt-dlp` CLI is installed. Bypasses all
       bot detection via its own anti-fingerprinting layer. Most reliable.
       Install with: pip install yt-dlp

    Args:
        cookies_from_browser: Browser to extract cookies from for yt-dlp,
            e.g. "chrome", "firefox", "safari", "edge".  This is by far the
            most effective way to improve the caption fetch rate.
        cookies_file: Path to a Netscape cookies.txt file (alternative to
            cookies_from_browser).

    Returns a list of cues [{start, dur, text}] or None if unavailable.
    """
    def dbg(msg):
        if debug:
            print(f"    [captions] {msg}")

    # Check cache first (None = confirmed no captions; [] would mean empty but present)
    cached = _load_caption_cache(video_id)
    if cached is not None:
        dbg(f"cache hit — {len(cached)} cues" if cached else "cache hit — no captions")
        return cached if cached else None

    # Strategy 1: InnerTube API
    dbg("trying InnerTube API...")
    base_url = _get_caption_url_innertube(video_id)

    # Strategy 2: HTML scrape — URL extracted, fetched separately
    if not base_url:
        dbg("InnerTube failed, trying HTML scrape (no cookies)...")
        base_url = _get_caption_url_html(video_id)

    if base_url:
        dbg(f"got URL, fetching content...")
        raw = http_get(base_url, timeout=15)
        if raw:
            cues = _parse_caption_xml(raw) or _parse_caption_json(raw)
            if cues:
                dbg(f"success — {len(cues)} cues")
                _save_caption_cache(video_id, cues)
                return cues
            dbg(f"URL fetched {len(raw)} bytes but parsed 0 cues")
        else:
            dbg("URL fetch returned empty body")

    # Strategy 3: HTML scrape with shared cookie session
    dbg("trying HTML scrape with cookie session (SOCS)...")
    cues = _get_captions_via_html_session(video_id)
    if cues:
        dbg(f"cookie session success — {len(cues)} cues")
        _save_caption_cache(video_id, cues)
        return cues
    dbg("cookie session returned no cues")

    # Strategy 4: yt-dlp (optional, most reliable)
    import shutil
    if shutil.which("yt-dlp"):
        dbg("trying yt-dlp...")
        try:
            cues = _get_captions_via_ytdlp(
                video_id,
                cookies_from_browser=cookies_from_browser,
                cookies_file=cookies_file,
            )
        except YtdlpError as e:
            dbg(f"yt-dlp error — {e.reason}")
            # Bot-blocks are transient — don't negative-cache so they're retried
            if e.reason != "bot-blocked":
                _save_caption_cache(video_id, None)
            return None

        if cues:
            dbg(f"yt-dlp success — {len(cues)} cues")
            _save_caption_cache(video_id, cues)
            return cues
        dbg("yt-dlp returned no captions (video may have no English subtitles)")
    else:
        dbg("yt-dlp not found in PATH — skipping (install with: pip install yt-dlp)")

    # Cache the negative result so we don't retry this video on future runs.
    _save_caption_cache(video_id, None)
    return None


def _make_cookie(name: str, value: str, domain: str = ".youtube.com") -> http.cookiejar.Cookie:
    """Create a Cookie object suitable for insertion into a CookieJar."""
    return http.cookiejar.Cookie(
        version=0, name=name, value=value,
        port=None, port_specified=False,
        domain=domain, domain_specified=True, domain_initial_dot=True,
        path="/", path_specified=True,
        secure=False, expires=None, discard=True,
        comment=None, comment_url=None, rest={"HttpOnly": None},
    )


def _get_captions_via_html_session(video_id: str) -> list[dict] | None:
    """
    Fetch captions by replaying the watch page session with a CookieJar.

    YouTube's timedtext server returns 0 bytes (HTTP 200, text/html) for
    sessions that have not accepted cookie consent. The SOCS cookie
    ("CAI" = accept all) must be pre-seeded before the watch page is
    fetched, or the session is treated as unconsented and all timedtext
    requests are silently rejected.
    """
    jar = http.cookiejar.CookieJar()
    # Pre-accept cookie consent so YouTube serves full content.
    # SOCS=CAI is the minimal consent-accepted token recognised by YouTube.
    jar.set_cookie(_make_cookie("SOCS", "CAI"))
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))

    # Step 1: fetch the watch page (with consent cookie already set)
    watch_url = f"https://www.youtube.com/watch?v={video_id}"
    try:
        page_req = urllib.request.Request(watch_url, headers=HEADERS)
        with opener.open(page_req, timeout=15) as resp:
            html = resp.read().decode("utf-8", errors="replace")
    except Exception:
        return None

    # Step 2: extract a captionTracks URL from the page HTML
    match = re.search(r'"captionTracks":\s*(\[.*?\])', html)
    if not match:
        return None
    try:
        tracks = json.loads(match.group(1))
    except json.JSONDecodeError:
        return None
    if not tracks:
        return None

    english = next((t for t in tracks if t.get("languageCode", "").startswith("en")), None)
    base_url = (english or tracks[0]).get("baseUrl")
    if not base_url:
        return None

    # Step 3: fetch the caption content with the same opener (sends cookies)
    # Include Referer so the timedtext server treats this as a page-originated request.
    try:
        cap_req = urllib.request.Request(
            base_url,
            headers={**HEADERS, "Referer": watch_url},
        )
        with opener.open(cap_req, timeout=15) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except Exception:
        return None

    if not raw:
        return None

    cues = _parse_caption_xml(raw) or _parse_caption_json(raw)
    return cues if cues else None


class YtdlpError(Exception):
    """Raised when yt-dlp exits with a recognisable error we want to surface."""
    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


def _get_captions_via_ytdlp(
    video_id: str,
    cookies_from_browser: str | None = None,
    cookies_file: str | None = None,
) -> list[dict] | None:
    """
    Fetch captions using yt-dlp if it is installed (pip install yt-dlp).

    yt-dlp handles YouTube's bot detection, TLS fingerprinting, and
    consent requirements better than any pure-Python HTTP approach.
    This function is a no-op if yt-dlp is not found in PATH.

    Raises YtdlpError with a short reason string when yt-dlp exits with a
    recognisable error (bot block, private/deleted video, etc.) so callers
    can surface the reason rather than silently caching a negative result.

    Args:
        cookies_from_browser: Browser name to extract cookies from, e.g. "chrome",
            "firefox", "safari", "edge".  Passed as --cookies-from-browser to
            yt-dlp.  This is the single most effective way to bypass YouTube bot
            detection — yt-dlp will read your real browser session cookies.
        cookies_file: Path to a Netscape-format cookies.txt file.  Alternative to
            cookies_from_browser when you can't extract from the browser directly.
    """
    import shutil
    import subprocess
    import tempfile
    import os as _os

    if not shutil.which("yt-dlp"):
        return None

    # Retry up to 3 times to handle transient rate limits (HTTP 429)
    for attempt in range(3):
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                out_tmpl = _os.path.join(tmpdir, "%(id)s.%(ext)s")
                auth_args = []
                if cookies_from_browser:
                    auth_args = ["--cookies-from-browser", cookies_from_browser]
                elif cookies_file:
                    auth_args = ["--cookies", cookies_file]

                base_cmd = [
                    "yt-dlp",
                    "--write-subs",
                    "--write-auto-subs",
                    "--skip-download",
                    "--no-playlist",
                    "--no-warnings",
                    "--ignore-errors",   # continue past format errors (e.g. SABR/images-only)
                    "-o", out_tmpl,
                ] + auth_args

                url = f"https://www.youtube.com/watch?v={video_id}"
                cmd = base_cmd + ["--sub-langs", "en", url]

                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=60,
                    check=False,
                )

                stdout = getattr(proc, "stdout", "") or ""
                stderr = getattr(proc, "stderr", "") or ""
                combined = (stdout + stderr).lower()

                # Rate-limit → retry with back-off
                if "http error 429" in combined or "too many requests" in combined:
                    wait = 10 * (2 ** attempt)
                    time.sleep(wait)
                    continue

                # Permanent errors → raise immediately
                if "sign in to confirm" in combined or "bot" in combined:
                    raise YtdlpError("bot-blocked")
                if "private video" in combined:
                    raise YtdlpError("private")
                if "video unavailable" in combined or "not available" in combined:
                    raise YtdlpError("unavailable")
                if "has been removed" in combined:
                    raise YtdlpError("deleted")
                if proc.returncode != 0 and combined.strip():
                    first_err = next(
                        (ln.strip() for ln in (stdout + stderr).splitlines() if ln.strip()),
                        "unknown error",
                    )
                    raise YtdlpError(first_err[:80])

                files = sorted(_os.listdir(tmpdir))
                if not files:
                    # Nothing written — log yt-dlp output so we can diagnose
                    for line in (stdout + stderr).splitlines():
                        line = line.strip()
                        if line and not line.startswith("[debug]"):
                            print(f"    [yt-dlp] {line}", flush=True)

                for filename in files:
                    filepath = _os.path.join(tmpdir, filename)
                    try:
                        with open(filepath, "r", encoding="utf-8", errors="replace") as fh:
                            content = fh.read()
                    except Exception:
                        continue
                    if not content.strip():
                        continue
                    if filename.endswith(".vtt"):
                        cues = _parse_caption_vtt(content)
                    elif filename.endswith((".srv3", ".json3")):
                        cues = _parse_caption_json(content)
                    else:
                        cues = _parse_caption_xml(content) or _parse_caption_json(content)
                    if cues:
                        return cues

                return None  # yt-dlp ran cleanly but found no English captions

        except YtdlpError:
            raise  # propagate to caller
        except Exception:
            pass

    return None  # all retries exhausted (likely persistent rate limit)


def _parse_caption_vtt(text: str) -> list[dict] | None:
    """Parse WebVTT caption format into cues."""
    cues = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if "-->" in line:
            parts = line.split("-->")
            if len(parts) == 2:
                try:
                    start = _vtt_time_to_sec(parts[0].strip().split()[0])
                    end = _vtt_time_to_sec(parts[1].strip().split()[0])
                    dur = max(end - start, 0.001)
                except (ValueError, IndexError):
                    i += 1
                    continue
                i += 1
                text_parts = []
                while i < len(lines) and lines[i].strip():
                    # Strip VTT inline tags (<c>, <00:00:01.234>, etc.)
                    clean = re.sub(r"<[^>]+>", "", lines[i]).strip()
                    if clean:
                        text_parts.append(clean)
                    i += 1
                joined = " ".join(text_parts).strip()
                if joined:
                    cues.append({"start": start, "dur": dur, "text": joined})
                continue
        i += 1
    return cues or None


def _vtt_time_to_sec(t: str) -> float:
    """Convert a VTT timestamp (HH:MM:SS.mmm or MM:SS.mmm) to seconds."""
    parts = t.split(":")
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    if len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])
    return float(parts[0])


def _parse_caption_xml(text: str) -> list[dict] | None:
    """Parse YouTube's timed-text XML format into cues."""
    try:
        root = ElementTree.fromstring(text)
    except ElementTree.ParseError:
        return None
    cues = []
    for elem in root.iter("text"):
        start = float(elem.get("start", "0"))
        dur = float(elem.get("dur", "2"))
        raw_text = (elem.text or "").replace("\n", " ").strip()
        if raw_text:
            cues.append({"start": start, "dur": dur, "text": raw_text})
    return cues or None


def _parse_caption_json(text: str) -> list[dict] | None:
    """Parse YouTube's srv3/json3 caption format into cues."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    cues = []
    # srv3 / json3 format: {"events": [{"tStartMs": ..., "dDurationMs": ..., "segs": [...]}]}
    for event in data.get("events", []):
        start = event.get("tStartMs", 0) / 1000
        dur = event.get("dDurationMs", 2000) / 1000
        segs = event.get("segs", [])
        raw_text = "".join(s.get("utf8", "") for s in segs).replace("\n", " ").strip()
        if raw_text and raw_text != "\n":
            cues.append({"start": start, "dur": dur, "text": raw_text})
    return cues or None


def _get_caption_url_innertube(video_id: str) -> str | None:
    """
    Fetch caption track URL via YouTube's InnerTube /player API.
    Tries multiple client configs in order — TVHTML5 is most reliable
    for bypassing consent walls and bot detection.
    """
    api_url = "https://www.youtube.com/youtubei/v1/player"
    for client in INNERTUBE_CLIENTS:
        try:
            payload = {
                "context": {"client": client},
                "videoId": video_id,
            }
            data = http_post_json(api_url, payload, timeout=15)

            tracks = (
                data.get("captions", {})
                .get("playerCaptionsTracklistRenderer", {})
                .get("captionTracks", [])
            )
            if not tracks:
                continue  # try next client

            # Prefer English, fall back to first available
            english = next(
                (t for t in tracks if t.get("languageCode", "").startswith("en")),
                None,
            )
            track = english or tracks[0]
            base_url = track.get("baseUrl")
            if base_url:
                return base_url
        except Exception:
            continue  # try next client

    # Last-resort: direct timedtext API (no auth required, often still works)
    return _get_caption_url_timedtext(video_id)


def _get_caption_url_timedtext(video_id: str) -> str | None:
    """Fallback: YouTube's legacy timedtext API endpoint."""
    try:
        url = f"https://www.youtube.com/api/timedtext?v={video_id}&lang=en&fmt=srv3"
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=10) as resp:
            body = resp.read()
        # If we got a non-empty response, return the URL directly
        if body and len(body) > 50:
            return url
        return None
    except Exception:
        return None


def _get_caption_url_html(video_id: str) -> str | None:
    """Fallback: scrape caption track URL from the YouTube watch page HTML."""
    try:
        html = http_get(f"https://www.youtube.com/watch?v={video_id}", timeout=15)
        match = re.search(r'"captionTracks":\s*(\[.*?\])', html)
        if not match:
            return None
        tracks = json.loads(match.group(1))
        if not tracks:
            return None
        english = next(
            (t for t in tracks if t.get("languageCode", "").startswith("en")),
            None,
        )
        track = english or tracks[0]
        return track.get("baseUrl")
    except Exception:
        return None


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
CAPTION_CACHE_DIR = CACHE_DIR / "captions"

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


def parse_database(csv_path: Path, sample_size: int | None = 50) -> list[dict]:
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
    sample = high_confidence if sample_size is None else high_confidence[:sample_size]
    label = "all" if sample_size is None else str(len(sample))
    print(f"Sampled {label} videos for benchmarking\n")
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
    cookies_from_browser: str | None = None,
    cookies_file: str | None = None,
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
        cues = get_captions_for_video(
            video_id, debug=verbose,
            cookies_from_browser=cookies_from_browser,
            cookies_file=cookies_file,
        )
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
                        help="Videos to sample in DB mode (default: 50); "
                             "use 0 to process all qualifying videos")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel caption fetches (default: 1; use 3-5 only "
                             "when --cookies-from-browser is set to avoid rate limits)")
    parser.add_argument("--quiet", action="store_true",
                        help="One-line-per-video output")
    parser.add_argument("--prefetch", action="store_true",
                        help="Download and cache captions for all DB videos without "
                             "running detection. Re-run without --prefetch to benchmark "
                             "instantly from cache. Safe to interrupt and resume.")
    parser.add_argument("--cookies-from-browser", metavar="BROWSER",
                        dest="cookies_from_browser", default=None,
                        help="Extract cookies from this browser for yt-dlp "
                             "(e.g. chrome, firefox, safari, edge). "
                             "This is the single most effective way to bypass "
                             "YouTube bot detection and dramatically increases the "
                             "caption fetch rate. Example: --cookies-from-browser chrome")
    parser.add_argument("--cookies", metavar="FILE",
                        dest="cookies_file", default=None,
                        help="Path to a Netscape-format cookies.txt file for yt-dlp. "
                             "Alternative to --cookies-from-browser.")
    parser.add_argument("--retry-failed", action="store_true",
                        dest="retry_failed",
                        help="Delete negative-cache entries (videos where captions "
                             "previously returned nothing) so they are retried on the "
                             "next --prefetch run.  Useful after adding "
                             "--cookies-from-browser to a run that previously had none.")
    args = parser.parse_args()

    # ── --retry-failed: clear null cache entries ──────────────────
    if args.retry_failed:
        if not CAPTION_CACHE_DIR.exists():
            print("No caption cache found — nothing to clear.")
            return
        removed = 0
        import json as _json
        for p in CAPTION_CACHE_DIR.glob("*.json"):
            try:
                with open(p) as fh:
                    data = _json.load(fh)
                if data is None:
                    p.unlink()
                    removed += 1
            except Exception:
                pass
        print(f"Removed {removed:,} negative-cache entries from {CAPTION_CACHE_DIR}")
        print("Re-run with --prefetch (and --cookies-from-browser) to retry them.")
        return

    if not args.db and not args.video_ids:
        parser.print_help()
        print("\nExamples:")
        print("  python3 benchmark.py --db                                       # sample 50 from full database")
        print("  python3 benchmark.py --db --sample 500                          # sample 500 videos")
        print("  python3 benchmark.py --db --sample 0 --prefetch                 # cache ALL video captions")
        print("  python3 benchmark.py --db --sample 0 --prefetch \\")
        print("      --cookies-from-browser chrome                               # much higher fetch rate!")
        print("  python3 benchmark.py --db --sample 0                            # benchmark all (needs cache)")
        print("  python3 benchmark.py VIDEO_ID1 VIDEO_ID2                        # test specific videos")
        sys.exit(0)

    print("YouTube Sponsor Speeder — Detection Benchmark")
    print("═" * 50)

    # Build test cases: [{videoId, segments}]
    test_cases = []

    if args.db:
        csv_path = ensure_database()
        sample_size = args.sample if args.sample > 0 else None   # None = all
        test_cases = parse_database(csv_path, sample_size)
    else:
        for vid in args.video_ids:
            try:
                segments = get_sponsorblock_segments(vid)
                test_cases.append({"videoId": vid, "segments": segments})
            except Exception as e:
                print(f"  Warning: Could not fetch SponsorBlock data for {vid}: {e}")

    # ── Prefetch mode: download captions for all videos, skip detection ──
    if args.prefetch:
        cached_count = sum(1 for tc in test_cases
                           if _caption_cache_path(tc["videoId"]).exists())
        remaining = len(test_cases) - cached_count
        print(f"Prefetch mode: {len(test_cases):,} videos  "
              f"({cached_count:,} already cached, {remaining:,} to fetch)")
        if args.cookies_from_browser:
            print(f"Using browser cookies from: {args.cookies_from_browser}")
        elif args.cookies_file:
            print(f"Using cookies file: {args.cookies_file}")
        else:
            print("Tip: pass --cookies-from-browser chrome (or firefox/safari/edge) "
                  "for a much higher caption fetch rate.")
        print(f"Using {args.workers} worker(s). Safe to interrupt — resumes from cache.\n")

        completed = 0
        def fetch_one(tc):
            vid = tc["videoId"]
            if _caption_cache_path(vid).exists():
                return vid, "cached"
            try:
                cues = get_captions_for_video(
                    vid,
                    cookies_from_browser=args.cookies_from_browser,
                    cookies_file=args.cookies_file,
                )
                return vid, f"{len(cues)} cues" if cues else "no captions"
            except YtdlpError as e:
                return vid, f"yt-dlp error: {e.reason}"

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(fetch_one, tc): tc for tc in test_cases}
            for future in as_completed(futures):
                vid, status = future.result()
                completed += 1
                if completed % 100 == 0 or completed == len(test_cases):
                    pct_done = completed / len(test_cases) * 100
                    print(f"  [{completed:,}/{len(test_cases):,}  {pct_done:.1f}%]  {vid} — {status}")
                elif not args.quiet:
                    print(f"  {vid} — {status}")
        print(f"\nDone. Cached captions in: {CAPTION_CACHE_DIR}")
        return

    print(f"Testing {len(test_cases)} video(s) with {args.workers} worker(s)...\n")

    # Run benchmarks in parallel
    results = [None] * len(test_cases)
    completed = 0
    verbose = not args.quiet

    def run_one(idx_tc):
        idx, tc = idx_tc
        time.sleep(idx * 0.3)  # stagger requests to be polite
        return idx, benchmark_video(
            tc["videoId"], tc["segments"],
            verbose=verbose,
            cookies_from_browser=args.cookies_from_browser,
            cookies_file=args.cookies_file,
        )

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
