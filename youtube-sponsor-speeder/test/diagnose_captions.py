#!/usr/bin/env python3
"""
Diagnostic script: shows exactly what each caption-fetching strategy
returns for a given video ID. Run with:

    python3 diagnose_captions.py PPJ6NJkmDAo esQyYGezS7c bHIhgxav9LY
"""

import json
import sys
import urllib.request
import urllib.error

BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)
HEADERS = {"User-Agent": BROWSER_UA}

INNERTUBE_CLIENTS = [
    {"clientName": "TVHTML5",  "clientVersion": "7.20240201.16.00"},
    {"clientName": "ANDROID",  "clientVersion": "19.09.37", "androidSdkVersion": 30},
    {"clientName": "WEB",      "clientVersion": "2.20250401.00.00"},
]


def post_json(url, body):
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url, data=data,
        headers={**HEADERS, "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read().decode("utf-8", errors="replace"))


def check_innertube(video_id):
    api_url = "https://www.youtube.com/youtubei/v1/player"
    for client in INNERTUBE_CLIENTS:
        name = client["clientName"]
        try:
            payload = {"context": {"client": client}, "videoId": video_id}
            data = post_json(api_url, payload)

            status = data.get("playabilityStatus", {}).get("status", "?")
            reason = data.get("playabilityStatus", {}).get("reason", "")

            tracks = (
                data.get("captions", {})
                    .get("playerCaptionsTracklistRenderer", {})
                    .get("captionTracks", [])
            )
            langs = [t.get("languageCode") for t in tracks]
            base_url = tracks[0].get("baseUrl", "")[:80] if tracks else ""

            print(f"  [{name}] status={status} reason={reason!r} "
                  f"tracks={len(tracks)} langs={langs}")
            if base_url:
                print(f"           baseUrl={base_url}...")

        except Exception as e:
            print(f"  [{name}] ERROR: {e}")


def check_timedtext(video_id):
    for fmt in ("srv3", "vtt", "ttml"):
        url = f"https://www.youtube.com/api/timedtext?v={video_id}&lang=en&fmt={fmt}"
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=10) as resp:
                body = resp.read()
            print(f"  [timedtext/{fmt}] {len(body)} bytes — "
                  f"{'OK' if len(body) > 50 else 'EMPTY'}")
            if 0 < len(body) <= 300:
                print(f"    content: {body[:200]!r}")
        except urllib.error.HTTPError as e:
            print(f"  [timedtext/{fmt}] HTTP {e.code}")
        except Exception as e:
            print(f"  [timedtext/{fmt}] ERROR: {e}")


def check_html_scrape(video_id):
    import re, html as html_module
    url = f"https://www.youtube.com/watch?v={video_id}"
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=15) as resp:
            raw_html = resp.read().decode("utf-8", errors="replace")
    except Exception as e:
        print(f"  [HTML scrape] fetch ERROR: {e}")
        return

    size = len(raw_html)
    has_consent = "consent" in raw_html.lower() or "before you continue" in raw_html.lower()
    has_player = "ytInitialPlayerResponse" in raw_html

    # Extract captionTracks from the page
    match = re.search(r'"captionTracks":\s*(\[.*?\])', raw_html)
    if not match:
        print(f"  [HTML scrape] {size} bytes  consent_wall={has_consent}  "
              f"has_player={has_player}  captionTracks=NOT FOUND IN HTML")
        return

    try:
        tracks = json.loads(match.group(1))
    except json.JSONDecodeError as e:
        # Try unescaping HTML entities first (YouTube sometimes double-encodes)
        unescaped = html_module.unescape(match.group(1))
        try:
            tracks = json.loads(unescaped)
            print(f"  [HTML scrape] NOTE: needed html.unescape() to parse captionTracks")
        except json.JSONDecodeError:
            print(f"  [HTML scrape] JSON parse failed: {e}")
            print(f"    raw match: {match.group(1)[:200]!r}")
            return

    print(f"  [HTML scrape] {size} bytes  consent_wall={has_consent}  "
          f"has_player={has_player}  captionTracks={len(tracks)}")

    # Now try actually fetching each caption URL
    for i, track in enumerate(tracks):
        lang = track.get("languageCode", "?")
        base_url = track.get("baseUrl", "")
        if not base_url:
            print(f"    track[{i}] lang={lang}  NO baseUrl")
            continue

        # Check for &amp; encoding (common bug)
        if "&amp;" in base_url:
            print(f"    track[{i}] lang={lang}  WARNING: baseUrl contains &amp; — fixing")
            base_url = html_module.unescape(base_url)

        print(f"    track[{i}] lang={lang}  baseUrl={base_url[:80]}...")
        try:
            req2 = urllib.request.Request(base_url, headers=HEADERS)
            with urllib.request.urlopen(req2, timeout=10) as resp2:
                body = resp2.read()
            print(f"      → fetch: {len(body)} bytes  {'OK' if len(body) > 100 else 'TOO SMALL'}")
            if 0 < len(body) <= 500:
                print(f"        content: {body[:300]!r}")
        except urllib.error.HTTPError as e:
            print(f"      → fetch: HTTP {e.code} {e.reason}")
        except Exception as e:
            print(f"      → fetch: ERROR {e}")


if __name__ == "__main__":
    video_ids = sys.argv[1:] or ["PPJ6NJkmDAo", "esQyYGezS7c", "bHIhgxav9LY"]
    for vid in video_ids:
        print(f"\n{'═' * 55}")
        print(f"  Video: {vid}  https://youtu.be/{vid}")
        print(f"{'═' * 55}")
        print("── InnerTube /player API:")
        check_innertube(vid)
        print("── Timedtext API:")
        check_timedtext(vid)
        print("── HTML watch page:")
        check_html_scrape(vid)
    print()
