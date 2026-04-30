/**
 * YouTube ML Sponsor Detector — Content Script
 *
 * Alternative sponsor detection method using a bimodal ML model
 * (keyword text features + MFCC audio features).
 *
 * ─── Detection pipeline ───────────────────────────────────────────────
 *
 *  STAGE 1 — Upfront text pre-detection (runs once per video load)
 *    1. Fetch the YouTube caption timed-text XML.
 *    2. Group cues into WINDOW_SEC-second windows.
 *    3. Extract a 64-dim keyword feature vector per window.
 *    4. Run MLSponsorDetector.scoreAsync() (uses ONNX if available).
 *    5. Windows scoring ≥ PRE_DETECT_THRESHOLD are "candidate segments."
 *    6. Adjacent candidate windows are merged → predetectedSegments[].
 *
 *  STAGE 2 — Real-time bimodal inference (runs every POLL_MS)
 *    1. Connect MFCC extractor to the video's audio stream.
 *    2. Each tick: capture one MFCC frame + extract text features for
 *       the caption cues currently on-screen.
 *    3. Run MLSponsorDetector.score() → combinedScore.
 *    4. Also read mfccExtractor.getAnomalyScore() as a supplementary
 *       audio-only signal.
 *    5. Blend: finalScore = mlScore×0.75 + audioAnomaly×0.25.
 *    6. Require MIN_CONSECUTIVE frames above ENTRY_THRESHOLD (0.85) before
 *       triggering. Lower threshold applies inside pre-detected regions.
 *
 *  PLAYBACK CONTROL — two modes depending on whether a segment end is known:
 *
 *    SEEK mode  (preferred — used when a pre-detected segment end is known)
 *      video.currentTime = segmentEnd + SEEK_BUFFER
 *      Instantaneous skip, no playback rate change. A brief flash indicator
 *      ("⏭ Skipped") confirms the action.
 *
 *    SPEED mode (fallback — used when real-time detection has no known end)
 *      video.playbackRate = SPEED_FALLBACK (16×), muted.
 *      Restored when the score drops below EXIT_THRESHOLD.
 *      Indicator shows "⏩ Speeding".
 *
 *    Shows a purple indicator (top-left) with the ML confidence %.
 *    The existing Sponsor Speeder uses an orange indicator (top-right),
 *    so the two extensions can run side-by-side for A/B comparison.
 *
 * ─── Files loaded before this script (manifest.json order) ───────────
 *   feature-extractor.js  → KeywordFeatureExtractor, MFCCExtractor
 *   ml-detector.js        → MLSponsorDetector
 *   content.js            → this file
 */

(function () {
  "use strict";

  // ─── Configuration ──────────────────────────────────────────────────────

  /**
   * Speed used in fallback SPEED mode (when no segment end time is known).
   * The preferred action is a direct seek; speed mode is a fallback only.
   */
  const SPEED_FALLBACK  = 16;
  const NORMAL_SPEED    = 1;

  /**
   * Seconds added after the detected segment end before seeking.
   * A small buffer ensures we land safely after any caption lag.
   */
  const SEEK_BUFFER     = 1.0;

  /** Poll interval for Stage 2 (ms). One MFCC frame + one inference call. */
  const POLL_MS         = 1000;

  /** Duration of each caption grouping window (seconds). */
  const WINDOW_SEC      = 5;

  /** Seconds of padding added around detected segments. */
  const PAD_BEFORE      = 1.5;
  const PAD_AFTER       = 2.0;

  /**
   * Max gap between adjacent flagged windows before they're split
   * into separate segments during the merge step.
   */
  const MERGE_GAP_SEC   = 8;

  /** Segments shorter than this are discarded as likely false positives. */
  const MIN_SEGMENT_SEC = 8;

  /**
   * Stage 1 uses a slightly lower threshold than Stage 2 — cast a
   * wider net upfront, then confirm in real time.
   */
  const PRE_DETECT_THRESHOLD = MLSponsorDetector.ENTRY_THRESHOLD * 0.80;

  /**
   * When Stage 2 detects we're inside a pre-detected segment, apply
   * this multiplier to lower the entry threshold.
   */
  const PRE_DETECT_BOOST = 0.85;

  /**
   * Weight of the ML score vs the audio anomaly in the blended
   * real-time score. ML score gets 0.75, raw audio anomaly gets 0.25.
   */
  const ML_WEIGHT    = 0.75;
  const AUDIO_WEIGHT = 0.25;

  // ─── State ──────────────────────────────────────────────────────────────

  let detector            = null;
  let mfccExtractor       = null;
  let captionCues         = [];    // All cues for the current video
  let predetectedSegments = [];    // Stage 1 output: [{start, end, score}]
  let insideSponsor       = false;
  let inSpeedMode         = false;  // true = speed fallback, false = seek (or idle)
  let savedSpeed          = null;
  let savedVolume         = null;
  let currentVideoId      = null;
  let pollTimer           = null;
  let indicatorEl         = null;
  let consecutiveFrames   = 0;     // consecutive frames above entry threshold
  let seekCooldownUntil   = 0;     // timestamp (ms) before which we skip re-entering
  let initialized         = false;

  // ─── Initialization ─────────────────────────────────────────────────────

  async function initialize() {
    if (initialized) return;
    initialized = true;

    detector      = new MLSponsorDetector();
    await detector.init();

    mfccExtractor = new MFCCExtractor();

    const backend = detector.isTrained ? "ONNX model" : "heuristic MLP";
    console.log(`[ML Detector] Active. Backend: ${backend}.`);
  }

  // ─── Caption helpers ────────────────────────────────────────────────────

  function getCaptionTrackUrl() {
    try {
      const pr =
        window.ytInitialPlayerResponse ||
        window.ytplayer?.config?.args?.raw_player_response;

      if (pr) {
        const tracks = pr?.captions?.playerCaptionsTracklistRenderer?.captionTracks;
        if (tracks?.length > 0) {
          const en = tracks.find(t => t.languageCode?.startsWith("en"));
          return (en || tracks[0]).baseUrl;
        }
      }

      // Fallback: scrape from page HTML
      const match = document.body.innerHTML.match(/"captionTracks":\s*(\[.*?\])/);
      if (match) {
        const tracks = JSON.parse(match[1]);
        const en = tracks.find(t => t.languageCode?.startsWith("en"));
        return (en || tracks[0])?.baseUrl;
      }
    } catch (e) {
      console.warn("[ML Detector] Caption URL extraction failed:", e.message);
    }
    return null;
  }

  async function fetchCaptions(baseUrl) {
    try {
      const resp = await fetch(baseUrl);
      const xml  = await resp.text();
      const doc  = new DOMParser().parseFromString(xml, "text/xml");
      return Array.from(doc.querySelectorAll("text")).map(node => ({
        start: parseFloat(node.getAttribute("start")),
        dur:   parseFloat(node.getAttribute("dur") || "2"),
        text:  (node.textContent || "").replace(/\n/g, " ").trim(),
      }));
    } catch (e) {
      console.warn("[ML Detector] Caption fetch failed:", e.message);
      return [];
    }
  }

  // ─── Stage 1: Upfront text-based pre-detection ──────────────────────────

  /**
   * Group all caption cues into WINDOW_SEC windows, score each window
   * using text features only (audio set to zeros), and return candidate
   * segments where the score exceeded PRE_DETECT_THRESHOLD.
   *
   * @param {Array} cues
   * @returns {Array<{start:number, end:number, score:number, source:string}>}
   */
  async function predetectFromCaptions(cues) {
    if (!cues.length) return [];

    const windows = groupCuesIntoWindows(cues);
    const candidates = [];

    for (const w of windows) {
      const textFeatures   = KeywordFeatureExtractor.extractFromCues(w.cues);
      // Stage 1 has no real-time audio — pass a zero frame buffer so the ONNX
      // model receives a correctly-shaped audio_input [1, N_FRAMES, 13].
      const zeroFrameBuffer = new Float32Array(MFCCExtractor.N_FRAMES * MFCCExtractor.NUM_COEFFICIENTS);

      // Use async path so the ONNX model is used if available
      const score = await detector.scoreAsync(textFeatures, zeroFrameBuffer);

      if (score >= PRE_DETECT_THRESHOLD) {
        candidates.push({ time: w.startTime, endTime: w.endTime, score });
      }
    }

    return mergeWindowsIntoSegments(candidates);
  }

  function groupCuesIntoWindows(cues) {
    const windows = [];
    let wStart = null;
    let wCues  = [];

    for (const cue of cues) {
      if (wStart === null) wStart = cue.start;

      if (cue.start - wStart >= WINDOW_SEC && wCues.length) {
        const last = wCues[wCues.length - 1];
        windows.push({
          startTime: wStart,
          endTime:   last.start + (last.dur || 2),
          cues:      wCues,
        });
        wStart = cue.start;
        wCues  = [];
      }
      wCues.push(cue);
    }

    if (wCues.length) {
      const last = wCues[wCues.length - 1];
      windows.push({
        startTime: wStart,
        endTime:   last.start + (last.dur || 2),
        cues:      wCues,
      });
    }
    return windows;
  }

  function mergeWindowsIntoSegments(scoredWindows) {
    if (!scoredWindows.length) return [];

    const sorted = [...scoredWindows].sort((a, b) => a.time - b.time);
    const groups = [];
    let group    = [sorted[0]];

    for (let i = 1; i < sorted.length; i++) {
      const prev = group[group.length - 1];
      const curr = sorted[i];
      if (curr.time - prev.endTime <= MERGE_GAP_SEC) {
        group.push(curr);
      } else {
        groups.push(group);
        group = [curr];
      }
    }
    groups.push(group);

    return groups
      .map(g => ({
        start:  Math.max(0, g[0].time - PAD_BEFORE),
        end:    g[g.length - 1].endTime + PAD_AFTER,
        score:  g.reduce((s, w) => s + w.score, 0) / g.length,
        source: "ml-text",
      }))
      .filter(s => (s.end - s.start) >= MIN_SEGMENT_SEC);
  }

  // ─── Stage 2: Real-time bimodal polling ─────────────────────────────────

  function startPolling() {
    stopPolling();

    const video = getVideo();
    if (video) mfccExtractor.connect(video);

    pollTimer = setInterval(() => {
      const video = getVideo();
      if (!video || video.paused || video.ended) return;

      const t = video.currentTime;

      // Capture MFCC frame, then get the full N_FRAMES frame buffer for
      // the CNN audio branch. Also capture the anomaly score separately
      // (still derived from the mean-vector path — unchanged behaviour).
      mfccExtractor.captureFrame();
      const frameBuffer  = mfccExtractor.getFrameBuffer(MFCCExtractor.N_FRAMES);
      const audioAnomaly = mfccExtractor.getAnomalyScore();

      // Caption cues active in the current time window
      const windowCues = captionCues.filter(
        c => c.start <= t + WINDOW_SEC && (c.start + c.dur) >= t
      );
      const textFeatures = windowCues.length
        ? KeywordFeatureExtractor.extractFromCues(windowCues)
        : new Float32Array(KeywordFeatureExtractor.FEATURE_DIM);

      // ML score from bimodal model (heuristic uses mean-pooled frame buffer)
      const mlScore = detector.score(textFeatures, frameBuffer);

      // Blend with raw audio anomaly
      const blendedScore = mlScore * ML_WEIGHT + audioAnomaly * AUDIO_WEIGHT;

      // Is the current position inside a pre-detected candidate segment?
      const inPredetected = predetectedSegments.some(
        s => t >= s.start && t <= s.end
      );

      // Apply pre-detection boost (lower threshold if we're in a known region)
      const entryThreshold = inPredetected
        ? MLSponsorDetector.ENTRY_THRESHOLD * PRE_DETECT_BOOST
        : MLSponsorDetector.ENTRY_THRESHOLD;

      // Consecutive-frame hysteresis: require MIN_CONSECUTIVE frames above
      // threshold before triggering, decrement on frames that fall below.
      if (blendedScore >= entryThreshold) {
        consecutiveFrames = Math.min(consecutiveFrames + 1, 10);
      } else {
        consecutiveFrames = Math.max(0, consecutiveFrames - 1);
      }

      const shouldEnter = consecutiveFrames >= MLSponsorDetector.MIN_CONSECUTIVE_FRAMES;
      const shouldExit  = blendedScore < MLSponsorDetector.EXIT_THRESHOLD
                          && consecutiveFrames === 0;

      const now = Date.now();
      if (shouldEnter && !insideSponsor && now >= seekCooldownUntil) {
        // ── Preferred: seek past the known segment end ─────────────────
        // Find the matching pre-detected segment (if any) so we know the end.
        const predetectedSeg = predetectedSegments.find(s => t >= s.start && t <= s.end);
        if (predetectedSeg) {
          seekPastSegment(predetectedSeg.end, blendedScore);
        } else {
          // ── Fallback: no known end time — speed through ───────────────
          enterSpeedMode(blendedScore);
        }
      } else if (shouldExit && insideSponsor) {
        exitSpeedMode();
      }

    }, POLL_MS);
  }

  function stopPolling() {
    if (pollTimer) {
      clearInterval(pollTimer);
      pollTimer = null;
    }
    resetPlayback();
  }

  // ─── Playback control ────────────────────────────────────────────────────

  function getVideo() {
    return (
      document.querySelector("video.html5-main-video") ||
      document.querySelector("#movie_player video") ||
      document.querySelector("video")
    );
  }

  // ─── Seek mode ────────────────────────────────────────────────────────────
  // Preferred path: we know the segment end, so jump past it immediately.
  // Does NOT change playbackRate — just moves currentTime.

  /**
   * Seek the video directly past the end of a detected segment.
   * Sets a cooldown so the polling loop doesn't immediately re-trigger on
   * the frames right after the seek.
   *
   * @param {number} segmentEnd - detected segment end time in seconds
   * @param {number} confidence - blended score that triggered this seek
   */
  function seekPastSegment(segmentEnd, confidence) {
    const video = getVideo();
    if (!video) return;

    const from   = video.currentTime.toFixed(1);
    const target = Math.min(segmentEnd + SEEK_BUFFER, (video.duration || Infinity) - 0.5);

    video.currentTime = target;

    // Reset consecutive-frame counter so we start fresh after the seek
    consecutiveFrames = 0;
    // Prevent re-triggering for 3 seconds (the seek may place us at the very
    // tail of a segment where the score is still briefly elevated)
    seekCooldownUntil = Date.now() + 3000;

    const backend = detector.isTrained ? "ONNX" : "heuristic";
    console.log(
      `[ML Detector] Seeked: ${from}s → ${target.toFixed(1)}s` +
      ` (segment end ${segmentEnd.toFixed(1)}s, confidence ${(confidence * 100).toFixed(0)}%, ${backend})`
    );

    // Flash the indicator briefly, then hide automatically
    showIndicator(confidence, "seek");
    setTimeout(hideIndicator, 2500);
  }

  // ─── Speed mode (fallback) ────────────────────────────────────────────────
  // Used when real-time detection fires but no pre-detected segment end is
  // available, so we can't know how far to seek.

  function enterSpeedMode(confidence) {
    if (insideSponsor) return;
    const video = getVideo();
    if (!video) return;

    insideSponsor = true;
    inSpeedMode   = true;
    savedSpeed    = video.playbackRate;
    savedVolume   = video.volume;

    video.playbackRate = SPEED_FALLBACK;
    video.muted        = true;

    showIndicator(confidence, "speed");

    const backend = detector.isTrained ? "ONNX" : "heuristic";
    console.log(
      `[ML Detector] Speed fallback at ${video.currentTime.toFixed(1)}s` +
      ` — confidence ${(confidence * 100).toFixed(0)}% (${backend}) — ${SPEED_FALLBACK}×` +
      ` (no segment end time known)`
    );
  }

  function exitSpeedMode() {
    if (!insideSponsor || !inSpeedMode) return;
    const video = getVideo();
    if (video) {
      video.playbackRate = savedSpeed  || NORMAL_SPEED;
      video.volume       = savedVolume ?? 1;
      video.muted        = false;
    }
    insideSponsor = false;
    inSpeedMode   = false;
    savedSpeed    = null;
    savedVolume   = null;
    hideIndicator();
    console.log(
      `[ML Detector] Speed mode ended at ${video?.currentTime?.toFixed(1) ?? "?"}s — playback restored.`
    );
  }

  /** Reset all playback state — safe to call at any time. */
  function resetPlayback() {
    if (insideSponsor && inSpeedMode) exitSpeedMode();
    insideSponsor     = false;
    inSpeedMode       = false;
    savedSpeed        = null;
    savedVolume       = null;
    consecutiveFrames = 0;
    seekCooldownUntil = 0;
  }

  // ─── On-screen indicator ─────────────────────────────────────────────────
  // Uses top-LEFT and purple colour to avoid clashing with the Sponsor
  // Speeder's orange top-RIGHT indicator when both extensions run together.

  function createIndicator() {
    if (indicatorEl) return;
    indicatorEl = document.createElement("div");
    indicatorEl.id = "ml-sponsor-detector-indicator";
    indicatorEl.style.cssText = `
      position: fixed;
      top: 16px;
      left: 16px;
      z-index: 999999;
      background: rgba(79, 70, 229, 0.93);
      color: #fff;
      font-family: "YouTube Sans", "Roboto", Arial, sans-serif;
      font-size: 13px;
      font-weight: 600;
      padding: 6px 14px;
      border-radius: 20px;
      pointer-events: none;
      opacity: 0;
      transition: opacity 0.25s ease;
      box-shadow: 0 2px 8px rgba(0,0,0,0.30);
    `;
    document.body.appendChild(indicatorEl);
  }

  /**
   * @param {number} confidence - blended score
   * @param {"seek"|"speed"} [mode="seek"] - which action was taken
   */
  function showIndicator(confidence, mode = "seek") {
    createIndicator();
    const pct     = Math.min(99, Math.round((confidence || 0) * 100));
    const backend = detector?.isTrained ? "ONNX" : "heuristic";
    if (mode === "seek") {
      indicatorEl.textContent = `⏭ Skipped (ML ${pct}% · ${backend})`;
    } else {
      indicatorEl.textContent = `⏩ Speeding (ML ${pct}% · ${backend}) · ${SPEED_FALLBACK}×`;
    }
    indicatorEl.style.opacity = "1";
  }

  function hideIndicator() {
    if (indicatorEl) indicatorEl.style.opacity = "0";
  }

  // ─── Video change detection ──────────────────────────────────────────────

  function getVideoId() {
    return new URLSearchParams(window.location.search).get("v");
  }

  async function onVideoChange() {
    const videoId = getVideoId();
    if (!videoId || videoId === currentVideoId) return;
    currentVideoId = videoId;

    stopPolling();
    captionCues         = [];
    predetectedSegments = [];

    console.log(`[ML Detector] New video: ${videoId}`);

    // Wait for YouTube's player config object to be available on SPA navigation
    await delay(1500);

    // ── Stage 1: Caption pre-detection ────────────────────────────────────
    const captionUrl = getCaptionTrackUrl();
    if (captionUrl) {
      captionCues = await fetchCaptions(captionUrl);

      if (captionCues.length > 0) {
        predetectedSegments = await predetectFromCaptions(captionCues);

        if (predetectedSegments.length > 0) {
          console.log(
            `[ML Detector] Stage 1: ${predetectedSegments.length} candidate segment(s) —`,
            predetectedSegments.map(s =>
              `${formatTime(s.start)}→${formatTime(s.end)} (score ${s.score.toFixed(2)})`
            ).join(", ")
          );
        } else {
          console.log("[ML Detector] Stage 1: no candidate segments detected from text.");
        }
      } else {
        console.log("[ML Detector] Stage 1: caption track empty.");
      }
    } else {
      console.log("[ML Detector] Stage 1: no captions available — running audio-only in Stage 2.");
    }

    // ── Stage 2: Real-time bimodal polling (always runs) ─────────────────
    startPolling();
  }

  // ─── Utilities ───────────────────────────────────────────────────────────

  function delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  function formatTime(sec) {
    const m = Math.floor(sec / 60);
    const s = Math.floor(sec % 60).toString().padStart(2, "0");
    return `${m}:${s}`;
  }

  // ─── Bootstrap ───────────────────────────────────────────────────────────

  // Watch for YouTube SPA navigation (URL changes without page reload)
  let lastUrl = location.href;
  new MutationObserver(() => {
    if (location.href !== lastUrl) {
      lastUrl = location.href;
      onVideoChange();
    }
  }).observe(document.querySelector("head > title") || document.head, {
    childList: true, subtree: true, characterData: true,
  });

  // yt-navigate-finish is more reliable on some YouTube versions
  document.addEventListener("yt-navigate-finish", () => onVideoChange());

  // Start everything
  initialize().then(() => onVideoChange());

})();
