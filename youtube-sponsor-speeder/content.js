/**
 * YouTube Sponsor Speeder — Content Script
 *
 * Detects built-in sponsor reads in YouTube videos and speeds through
 * them at 16×.
 *
 * Detection strategy (layered):
 *  PRIMARY — Transcript analysis:
 *   1. Extract the captions URL from YouTube's embedded player config.
 *   2. Fetch the timed-text XML and parse it into timestamped cues.
 *   3. Scan cues for clusters of sponsor-related keywords/phrases.
 *   4. Merge nearby sponsor cues into contiguous segments with padding.
 *
 *  FALLBACK — Audio/Visual analysis (when captions are unavailable):
 *   1. Sample video frames onto a canvas → colour histogram fingerprint.
 *   2. Monitor audio frequency bands via Web Audio API.
 *   3. Compare each time window to a running baseline.
 *   4. Flag windows with significant visual/audio deviation.
 *   5. Merge flagged windows into sponsor-like segments.
 *
 *  In both cases, playback is monitored and speed is toggled when
 *  the current time enters/exits a detected segment.
 */

(function () {
  "use strict";

  // ─── Configuration ────────────────────────────────────────────────

  const SPONSOR_SPEED = 16;
  const NORMAL_SPEED = 1;

  // Seconds of padding before/after detected sponsor cues to make
  // sure we catch the full read (captions often lag slightly).
  const PADDING_BEFORE = 1.5;
  const PADDING_AFTER = 2.0;

  // If two sponsor cues are within this many seconds of each other,
  // merge them into one segment (avoids rapid speed toggling).
  const MERGE_GAP_SEC = 8;

  // How often (ms) we check the current playback position.
  const POLL_MS = 250;

  // Minimum number of sponsor keyword hits in a cluster to count as
  // a real sponsor segment (reduces false positives).
  const MIN_KEYWORD_HITS = 2;

  // ─── Sponsor keyword / phrase patterns ────────────────────────────
  // These are phrases commonly used in English-language sponsor reads.
  // Each pattern is weighted: "strong" patterns are very likely sponsor
  // language; "weak" ones need corroboration from nearby cues.

  const STRONG_PATTERNS = [
    /this (?:video|segment|portion) is (?:brought to you|sponsored|made possible) by/i,
    /(?:sponsored|presented) by/i,
    /today'?s sponsor/i,
    /thanks to .{1,40} for sponsoring/i,
    /a (?:huge|big|special) thanks? to/i,
    /brought to you by/i,
    /use (?:my |our )?(?:code|link)/i,
    /use code .{1,20} (?:at|for) checkout/i,
    /go to .{1,40}\.com/i,
    /head (?:on )?over to .{1,40}\.com/i,
    /visit .{1,40}\.com/i,
    /check (?:them )?out at/i,
    /click (?:the|my) link/i,
    /link (?:is )?in (?:the )?description/i,
    /first \d+ (?:people|users|customers|subscribers)/i,
    /free trial/i,
    /percent off/i,
    /% off/i,
    /discount code/i,
    /promo code/i,
    /coupon code/i,
  ];

  const WEAK_PATTERNS = [
    /sign up/i,
    /download the app/i,
    /available (?:now )?at/i,
    /money back guarantee/i,
    /limited time/i,
    /exclusive (?:deal|offer)/i,
    /subscribe/i,
    /premium/i,
  ];

  // ─── State ────────────────────────────────────────────────────────

  let sponsorSegments = [];      // [{start, end}, ...]
  let insideSponsor = false;
  let savedSpeed = null;
  let savedVolume = null;
  let currentVideoId = null;
  let pollTimer = null;
  let indicatorEl = null;
  let detectionMode = "none";    // "transcript", "av", or "none"

  // AV analyzer instance (fallback when no captions available)
  let avAnalyzer = null;
  let avPollTimer = null;

  // ─── Caption fetching ─────────────────────────────────────────────

  /**
   * Extract the captions base URL from YouTube's player config.
   * YouTube embeds a JSON blob (ytInitialPlayerResponse) in the page
   * that contains the caption track URLs.
   */
  function getCaptionTrackUrl() {
    try {
      // Try the global player response first
      const playerResponse =
        window.ytInitialPlayerResponse ||
        window.ytplayer?.config?.args?.raw_player_response;

      if (playerResponse) {
        const tracks =
          playerResponse?.captions?.playerCaptionsTracklistRenderer?.captionTracks;
        if (tracks && tracks.length > 0) {
          // Prefer English, fall back to first available
          const english = tracks.find((t) =>
            t.languageCode?.startsWith("en")
          );
          const track = english || tracks[0];
          return track.baseUrl;
        }
      }

      // Fallback: scrape it from the page source
      const html = document.body.innerHTML;
      const match = html.match(/"captionTracks":\s*(\[.*?\])/);
      if (match) {
        const tracks = JSON.parse(match[1]);
        const english = tracks.find((t) =>
          t.languageCode?.startsWith("en")
        );
        const track = english || tracks[0];
        return track.baseUrl;
      }
    } catch (e) {
      console.warn("[Sponsor Speeder] Failed to extract caption URL:", e);
    }
    return null;
  }

  /**
   * Fetch and parse the timed-text XML into an array of cues:
   *   [{start: seconds, dur: seconds, text: string}, ...]
   */
  async function fetchCaptions(baseUrl) {
    try {
      const resp = await fetch(baseUrl);
      const xml = await resp.text();
      const parser = new DOMParser();
      const doc = parser.parseFromString(xml, "text/xml");
      const texts = doc.querySelectorAll("text");

      return Array.from(texts).map((node) => ({
        start: parseFloat(node.getAttribute("start")),
        dur: parseFloat(node.getAttribute("dur") || "2"),
        text: (node.textContent || "").replace(/\n/g, " ").trim(),
      }));
    } catch (e) {
      console.warn("[Sponsor Speeder] Failed to fetch captions:", e);
      return [];
    }
  }

  // ─── Sponsor detection ────────────────────────────────────────────

  /**
   * Score a single caption cue for sponsor likelihood.
   * Returns a numeric score (0 = not sponsor, higher = more likely).
   */
  function scoreCue(text) {
    let score = 0;
    for (const pat of STRONG_PATTERNS) {
      if (pat.test(text)) score += 3;
    }
    for (const pat of WEAK_PATTERNS) {
      if (pat.test(text)) score += 1;
    }
    return score;
  }

  /**
   * Scan all cues, score each one, and identify contiguous clusters
   * of sponsor-related language. Returns an array of segments:
   *   [{start: seconds, end: seconds}, ...]
   */
  function detectSponsorSegments(cues) {
    if (!cues.length) return [];

    // Score every cue
    const scored = cues.map((cue) => ({
      ...cue,
      end: cue.start + cue.dur,
      score: scoreCue(cue.text),
    }));

    // Find cues that scored > 0
    const hits = scored.filter((c) => c.score > 0);
    if (hits.length === 0) return [];

    // Cluster nearby hits into segments
    const rawSegments = [];
    let cluster = [hits[0]];

    for (let i = 1; i < hits.length; i++) {
      const prev = cluster[cluster.length - 1];
      const curr = hits[i];

      if (curr.start - prev.end <= MERGE_GAP_SEC) {
        cluster.push(curr);
      } else {
        rawSegments.push(cluster);
        cluster = [curr];
      }
    }
    rawSegments.push(cluster);

    // Filter out clusters that don't have enough keyword density
    // and convert to {start, end} segments
    const segments = [];
    for (const cluster of rawSegments) {
      const totalScore = cluster.reduce((sum, c) => sum + c.score, 0);
      if (totalScore < MIN_KEYWORD_HITS * 3) continue; // need at least MIN_KEYWORD_HITS strong matches equivalent

      const start = Math.max(0, cluster[0].start - PADDING_BEFORE);
      const end = cluster[cluster.length - 1].end + PADDING_AFTER;
      segments.push({ start, end });
    }

    return segments;
  }

  // ─── On-screen indicator ──────────────────────────────────────────

  function createIndicator() {
    if (indicatorEl) return;

    indicatorEl = document.createElement("div");
    indicatorEl.id = "sponsor-speeder-indicator";
    indicatorEl.textContent = "⏩ Sponsor · 16×";
    indicatorEl.style.cssText = `
      position: fixed;
      top: 16px;
      right: 16px;
      z-index: 999999;
      background: rgba(255, 140, 0, 0.92);
      color: #fff;
      font-family: "YouTube Sans", "Roboto", Arial, sans-serif;
      font-size: 13px;
      font-weight: 600;
      padding: 6px 14px;
      border-radius: 20px;
      pointer-events: none;
      opacity: 0;
      transition: opacity 0.25s ease;
      box-shadow: 0 2px 8px rgba(0,0,0,0.3);
    `;
    document.body.appendChild(indicatorEl);
  }

  function showIndicator() {
    createIndicator();
    const modeLabel = detectionMode === "av" ? "AV" : "Transcript";
    indicatorEl.textContent = `⏩ Sponsor (${modeLabel}) · 16×`;
    indicatorEl.style.opacity = "1";
  }

  function hideIndicator() {
    if (indicatorEl) indicatorEl.style.opacity = "0";
  }

  // ─── Playback control ─────────────────────────────────────────────

  function getVideo() {
    return (
      document.querySelector("video.html5-main-video") ||
      document.querySelector("#movie_player video") ||
      document.querySelector("video")
    );
  }

  function enterSponsor() {
    if (insideSponsor) return;
    const video = getVideo();
    if (!video) return;

    insideSponsor = true;
    savedSpeed = video.playbackRate;
    savedVolume = video.volume;

    video.playbackRate = SPONSOR_SPEED;
    video.muted = true;

    showIndicator();
    console.log(
      `[Sponsor Speeder] Sponsor segment started at ${video.currentTime.toFixed(1)}s — speeding to ${SPONSOR_SPEED}×`
    );
  }

  function exitSponsor() {
    if (!insideSponsor) return;
    const video = getVideo();
    if (!video) return;

    insideSponsor = false;

    video.playbackRate = savedSpeed || NORMAL_SPEED;
    video.volume = savedVolume ?? 1;
    video.muted = false;
    savedSpeed = null;
    savedVolume = null;

    hideIndicator();
    console.log(
      `[Sponsor Speeder] Sponsor segment ended at ${video.currentTime.toFixed(1)}s — restored normal speed`
    );
  }

  // ─── Playback monitor ─────────────────────────────────────────────

  function startMonitor() {
    stopMonitor();

    pollTimer = setInterval(() => {
      const video = getVideo();
      if (!video) return;

      const t = video.currentTime;
      const inSegment = sponsorSegments.some(
        (seg) => t >= seg.start && t <= seg.end
      );

      if (inSegment && !insideSponsor) {
        enterSponsor();
      } else if (!inSegment && insideSponsor) {
        exitSponsor();
      }
    }, POLL_MS);
  }

  function stopMonitor() {
    if (pollTimer) {
      clearInterval(pollTimer);
      pollTimer = null;
    }
    stopAVAnalyzer();
    exitSponsor();
  }

  // ─── AV Analyzer fallback ────────────────────────────────────────

  /**
   * Start the AV analyzer as a fallback when captions aren't available.
   * The analyzer collects visual/audio data as the video plays. We
   * periodically check its real-time score and also pull full segments
   * as enough data accumulates.
   */
  function startAVFallback() {
    const video = getVideo();
    if (!video) {
      console.log("[Sponsor Speeder] No video element found for AV analysis.");
      return;
    }

    detectionMode = "av";
    avAnalyzer = new AVAnalyzer();
    avAnalyzer.start(video);

    console.log("[Sponsor Speeder] AV fallback started — collecting data...");

    // Poll the analyzer for real-time anomaly detection.
    // Unlike transcript mode where we know segments upfront, AV mode
    // detects deviations as they happen.
    avPollTimer = setInterval(() => {
      if (!avAnalyzer) return;
      const video = getVideo();
      if (!video || video.paused || video.ended) return;

      // Real-time scoring: check if the current window looks anomalous
      const score = avAnalyzer.getCurrentScore();

      if (score >= AVAnalyzer.COMBINED_THRESHOLD && !insideSponsor) {
        enterSponsor();
      } else if (score < AVAnalyzer.COMBINED_THRESHOLD * 0.7 && insideSponsor) {
        // Use a lower threshold to exit (hysteresis) to avoid rapid
        // toggling at the boundary
        exitSponsor();
      }

      // Periodically refresh the full segment list from accumulated data
      // and merge with any real-time detections
      const avSegments = avAnalyzer.getSegments();
      if (avSegments.length > 0) {
        sponsorSegments = avSegments;
      }
    }, POLL_MS);
  }

  function stopAVAnalyzer() {
    if (avPollTimer) {
      clearInterval(avPollTimer);
      avPollTimer = null;
    }
    if (avAnalyzer) {
      avAnalyzer.stop();
      avAnalyzer = null;
    }
  }

  // ─── Video change detection ───────────────────────────────────────

  function getVideoId() {
    const params = new URLSearchParams(window.location.search);
    return params.get("v");
  }

  /**
   * Called when we detect a new video has loaded. Fetches captions,
   * runs sponsor detection, and starts monitoring playback.
   */
  async function onVideoChange() {
    const videoId = getVideoId();
    if (!videoId || videoId === currentVideoId) return;
    currentVideoId = videoId;

    stopMonitor();
    sponsorSegments = [];
    detectionMode = "none";

    console.log(`[Sponsor Speeder] New video detected: ${videoId}`);

    // YouTube's player config may not be ready immediately on SPA
    // navigation, so wait a moment before trying to read it.
    await delay(1500);

    // ── Try transcript-based detection first ──
    const captionUrl = getCaptionTrackUrl();
    if (captionUrl) {
      const cues = await fetchCaptions(captionUrl);
      if (cues.length > 0) {
        sponsorSegments = detectSponsorSegments(cues);

        if (sponsorSegments.length > 0) {
          detectionMode = "transcript";
          console.log(
            `[Sponsor Speeder] Transcript: detected ${sponsorSegments.length} sponsor segment(s):`,
            sponsorSegments.map(
              (s) =>
                `${formatTime(s.start)} → ${formatTime(s.end)} (${(s.end - s.start).toFixed(0)}s)`
            )
          );
          startMonitor();
          return;
        } else {
          console.log("[Sponsor Speeder] Transcript found but no sponsor segments detected.");
        }
      } else {
        console.log("[Sponsor Speeder] Caption track is empty.");
      }
    } else {
      console.log("[Sponsor Speeder] No captions available for this video.");
    }

    // ── Fall back to AV analysis ──
    console.log("[Sponsor Speeder] Falling back to audio/visual analysis...");
    startAVFallback();
  }

  // ─── Utilities ────────────────────────────────────────────────────

  function delay(ms) {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }

  function formatTime(sec) {
    const m = Math.floor(sec / 60);
    const s = Math.floor(sec % 60)
      .toString()
      .padStart(2, "0");
    return `${m}:${s}`;
  }

  // ─── Initialization ───────────────────────────────────────────────

  // Watch for YouTube SPA navigation (URL changes without page reload)
  let lastUrl = location.href;

  const navObserver = new MutationObserver(() => {
    if (location.href !== lastUrl) {
      lastUrl = location.href;
      onVideoChange();
    }
  });

  navObserver.observe(document.querySelector("head > title") || document.head, {
    childList: true,
    subtree: true,
    characterData: true,
  });

  // Also listen for the YouTube yt-navigate-finish event (more reliable
  // on some versions of the YouTube SPA)
  document.addEventListener("yt-navigate-finish", () => {
    onVideoChange();
  });

  // Initial run
  onVideoChange();

  console.log("[Sponsor Speeder] YouTube Sponsor Speeder v1.1 is active.");
})();
