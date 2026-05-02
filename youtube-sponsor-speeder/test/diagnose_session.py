#!/usr/bin/env python3
"""
Traces _get_captions_via_html_session step by step for one video.
Run: python3 diagnose_session.py PPJ6NJkmDAo
"""
import http.cookiejar, json, re, sys, urllib.request, urllib.error

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

video_id = sys.argv[1] if len(sys.argv) > 1 else "PPJ6NJkmDAo"
print(f"Video: {video_id}\n")

jar = http.cookiejar.CookieJar()
# Pre-accept cookie consent — without SOCS, timedtext returns 0-byte HTML
jar.set_cookie(http.cookiejar.Cookie(
    version=0, name="SOCS", value="CAI",
    port=None, port_specified=False,
    domain=".youtube.com", domain_specified=True, domain_initial_dot=True,
    path="/", path_specified=True,
    secure=False, expires=None, discard=True,
    comment=None, comment_url=None, rest={"HttpOnly": None},
))
opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))

# Step 1: fetch watch page
print("── Step 1: fetch watch page")
req = urllib.request.Request(f"https://www.youtube.com/watch?v={video_id}", headers=HEADERS)
with opener.open(req, timeout=15) as resp:
    html = resp.read().decode("utf-8", errors="replace")

cookies = list(jar)
print(f"  HTML size: {len(html)} bytes")
print(f"  Cookies set: {[c.name for c in cookies]}")

# Step 2: extract captionTracks
print("\n── Step 2: extract captionTracks from HTML")
match = re.search(r'"captionTracks":\s*(\[.*?\])', html)
if not match:
    print("  NOT FOUND — exiting")
    sys.exit(1)

tracks = json.loads(match.group(1))
print(f"  Found {len(tracks)} track(s):")
for i, t in enumerate(tracks):
    print(f"    [{i}] lang={t.get('languageCode')}  kind={t.get('kind')}  "
          f"baseUrl={t.get('baseUrl','')[:100]}...")

english = next((t for t in tracks if t.get("languageCode","").startswith("en")), None)
track = english or tracks[0]
base_url = track.get("baseUrl", "")
print(f"\n  Selected: lang={track.get('languageCode')}  full URL:\n  {base_url}\n")

# Step 3: fetch caption content WITH cookies
print("── Step 3: fetch timedtext URL (with cookies)")
cap_req = urllib.request.Request(base_url, headers={
    **HEADERS,
    "Referer": f"https://www.youtube.com/watch?v={video_id}",
})
print(f"  Cookies in jar: {[c.name for c in jar]}")
try:
    with opener.open(cap_req, timeout=15) as resp:
        status = resp.status
        content_type = resp.headers.get("Content-Type", "")
        body = resp.read()
    print(f"  HTTP {status}  Content-Type: {content_type}")
    print(f"  Body: {len(body)} bytes")
    if body:
        print(f"  First 300 bytes: {body[:300]!r}")
    else:
        print("  EMPTY — cookie session didn't help")
except urllib.error.HTTPError as e:
    print(f"  HTTP ERROR {e.code}: {e.reason}")
except Exception as e:
    print(f"  ERROR: {e}")

# Step 4: also try WITHOUT cookies for comparison
print("\n── Step 4: same URL fetched WITHOUT cookies (baseline)")
try:
    bare_req = urllib.request.Request(base_url, headers={
        **HEADERS,
        "Referer": f"https://www.youtube.com/watch?v={video_id}",
    })
    with urllib.request.urlopen(bare_req, timeout=15) as resp:
        body2 = resp.read()
    print(f"  Body: {len(body2)} bytes {'OK' if len(body2) > 50 else 'EMPTY'}")
except Exception as e:
    print(f"  ERROR: {e}")
