/**
 * av-analyzer.js — Audio/Visual Sponsor Segment Detector
 *
 * Fallback detection layer for when captions are unavailable.
 *
 * VISUAL ANALYSIS:
 *   Periodically samples video frames onto a small canvas and computes
 *   a colour histogram fingerprint. The video is divided into fixed-
 *   length windows; each window gets an average fingerprint. When a
 *   window's fingerprint deviates sharply from the running baseline
 *   (median of recent windows), it's flagged as a potential sponsor
 *   segment — sponsor reads that cut to product demos, app screen-
 *   recordings, or branded graphics produce a clear visual shift.
 *
 * AUDIO ANALYSIS:
 *   Connects to the <video> element via the Web Audio API and monitors
 *   frequency-band energy levels. Sponsor transitions often come with
 *   background music changes, volume dips, or a shift in spectral
 *   balance. The analyser tracks the energy ratio between low and high
 *   frequency bands and flags sudden shifts.
 *
 * COMBINED SCORING:
 *   Visual and audio deviation scores are combined. A window is marked
 *   as "sponsor-like" only if the combined score exceeds a threshold.
 *   Adjacent flagged windows are merged into segments.
 *
 * This module exposes a single class: AVAnalyzer.
 */

// eslint-disable-next-line no-unused-vars
class AVAnalyzer {
  // ─── Configuration ──────────────────────────────────────────────

  /** How often (ms) to sample a frame for visual analysis. */
  static SAMPLE_INTERVAL_MS = 1000;

  /** Canvas size for frame sampling (small = fast). */
  static CANVAS_W = 64;
  static CANVAS_H = 36;

  /** Number of bins per colour channel in the histogram. */
  static HIST_BINS = 8;

  /** Length of each analysis window in seconds. */
  static WINDOW_SEC = 5;

  /**
   * How many recent windows form the "baseline." The median of this
   * sliding window is what we compare each new window against.
   */
  static BASELINE_WINDOW = 12;

  /** Minimum number of samples before we start flagging deviations. */
  static MIN_BASELINE_SAMPLES = 6;

  /**
   * Combined deviation threshold to flag a window as sponsor-like.
   * Higher = fewer false positives but may miss subtle segments.
   */
  static COMBINED_THRESHOLD = 0.45;

  /** Weight of visual vs audio signal in the combined score. */
  static VISUAL_WEIGHT = 0.6;
  static AUDIO_WEIGHT = 0.4;

  /** Merge gap: if flagged windows are within this many seconds,
   *  merge them into one segment. */
  static MERGE_GAP_SEC = 10;

  /** Minimum segment duration (seconds) to report. Short blips are
   *  likely false positives (jump cuts, B-roll, etc). */
  static MIN_SEGMENT_SEC = 12;

  /** Seconds of padding before/after detected segments. */
  static PAD_BEFORE = 1;
  static PAD_AFTER = 2;

  // FFT config
  static FFT_SIZE = 256;
  static LOW_BAND_END = 10;  // bin index dividing "low" from "high"

  // ─── Constructor ────────────────────────────────────────────────

  constructor() {
    this._canvas = null;
    this._ctx = null;
    this._audioCtx = null;
    this._analyserNode = null;
    this._sourceNode = null;
    this._freqData = null;

    // Per-window accumulated histograms
    this._currentWindowStart = 0;
    this._currentVisualSamples = [];
    this._currentAudioSamples = [];

    // Completed windows: [{time, visualHist, audioProfile}]
    this._windows = [];

    // Flagged window times
    this._flaggedWindows = [];

    this._video = null;
    this._timer = null;
    this._running = false;
    this._audioConnected = false;
  }

  // ─── Public API ─────────────────────────────────────────────────

  /**
   * Start analysing the given <video> element.
   * Call this once per video; call stop() before starting a new one.
   */
  start(videoEl) {
    this.stop();

    this._video = videoEl;
    this._running = true;
    this._windows = [];
    this._flaggedWindows = [];
    this._currentWindowStart = videoEl.currentTime;
    this._currentVisualSamples = [];
    this._currentAudioSamples = [];

    this._initCanvas();
    this._initAudio(videoEl);

    this._timer = setInterval(() => this._sample(), AVAnalyzer.SAMPLE_INTERVAL_MS);
    console.log("[AV Analyzer] Started.");
  }

  /** Stop analysis and clean up. */
  stop() {
    this._running = false;
    if (this._timer) {
      clearInterval(this._timer);
      this._timer = null;
    }
    // Don't close AudioContext — Chrome limits how many you can create
    // and reusing is fine.
    this._video = null;
    console.log("[AV Analyzer] Stopped.");
  }

  /**
   * Run a two-pass analysis on the video:
   *  Pass 1 (live): collect samples as the video plays (via start()).
   *  Pass 2 (on-demand): call getSegments() to retrieve detected
   *  sponsor segments based on data collected so far.
   *
   * Returns [{start, end}, ...] sorted by start time.
   */
  getSegments() {
    this._computeFlags();
    return this._mergeFlags();
  }

  /**
   * Check in real-time whether the current playback position is in
   * a region that looks anomalous (for live detection as data builds).
   * Returns a score 0..1 where >COMBINED_THRESHOLD is suspicious.
   */
  getCurrentScore() {
    if (this._windows.length < AVAnalyzer.MIN_BASELINE_SAMPLES) return 0;

    const latest = this._windows[this._windows.length - 1];
    if (!latest) return 0;

    const baseline = this._computeBaseline(this._windows.length - 1);
    const vScore = this._histogramDistance(latest.visualHist, baseline.visualHist);
    const aScore = this._audioDistance(latest.audioProfile, baseline.audioProfile);

    return (
      vScore * AVAnalyzer.VISUAL_WEIGHT +
      aScore * AVAnalyzer.AUDIO_WEIGHT
    );
  }

  // ─── Canvas setup ───────────────────────────────────────────────

  _initCanvas() {
    if (!this._canvas) {
      this._canvas = document.createElement("canvas");
      this._canvas.width = AVAnalyzer.CANVAS_W;
      this._canvas.height = AVAnalyzer.CANVAS_H;
      this._ctx = this._canvas.getContext("2d", { willReadFrequently: true });
    }
  }

  // ─── Audio setup ────────────────────────────────────────────────

  _initAudio(videoEl) {
    try {
      if (!this._audioCtx) {
        this._audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      }

      // Only create the source node once per <video> element.
      // Creating multiple MediaElementSourceNodes for the same element
      // throws an error.
      if (!this._audioConnected) {
        this._sourceNode = this._audioCtx.createMediaElementSource(videoEl);
        this._analyserNode = this._audioCtx.createAnalyser();
        this._analyserNode.fftSize = AVAnalyzer.FFT_SIZE;

        this._sourceNode.connect(this._analyserNode);
        this._analyserNode.connect(this._audioCtx.destination);

        this._freqData = new Uint8Array(this._analyserNode.frequencyBinCount);
        this._audioConnected = true;
      }
    } catch (e) {
      console.warn("[AV Analyzer] Audio setup failed (will use visual only):", e);
      this._analyserNode = null;
    }
  }

  // ─── Sampling ───────────────────────────────────────────────────

  _sample() {
    const video = this._video;
    if (!video || video.paused || video.ended) return;

    const t = video.currentTime;

    // Visual sample
    const hist = this._sampleFrame(video);
    if (hist) this._currentVisualSamples.push(hist);

    // Audio sample
    const audioProfile = this._sampleAudio();
    if (audioProfile) this._currentAudioSamples.push(audioProfile);

    // Check if we've crossed into a new window
    if (t - this._currentWindowStart >= AVAnalyzer.WINDOW_SEC) {
      this._closeWindow(this._currentWindowStart);
      this._currentWindowStart = t;
      this._currentVisualSamples = [];
      this._currentAudioSamples = [];
    }
  }

  /**
   * Draw the current video frame onto our tiny canvas and compute
   * a colour histogram (R, G, B channels, HIST_BINS bins each).
   */
  _sampleFrame(video) {
    try {
      this._ctx.drawImage(video, 0, 0, AVAnalyzer.CANVAS_W, AVAnalyzer.CANVAS_H);
      const imageData = this._ctx.getImageData(
        0, 0, AVAnalyzer.CANVAS_W, AVAnalyzer.CANVAS_H
      );
      return this._computeHistogram(imageData.data);
    } catch {
      // Cross-origin or other canvas tainting issues
      return null;
    }
  }

  /**
   * Compute a normalised colour histogram from raw pixel data.
   * Returns a Float32Array of length HIST_BINS * 3 (R, G, B).
   */
  _computeHistogram(pixels) {
    const bins = AVAnalyzer.HIST_BINS;
    const hist = new Float32Array(bins * 3);
    const totalPixels = pixels.length / 4;
    const binSize = 256 / bins;

    for (let i = 0; i < pixels.length; i += 4) {
      const r = pixels[i];
      const g = pixels[i + 1];
      const b = pixels[i + 2];

      hist[Math.min(Math.floor(r / binSize), bins - 1)] += 1;
      hist[bins + Math.min(Math.floor(g / binSize), bins - 1)] += 1;
      hist[bins * 2 + Math.min(Math.floor(b / binSize), bins - 1)] += 1;
    }

    // Normalise
    for (let i = 0; i < hist.length; i++) {
      hist[i] /= totalPixels;
    }

    return hist;
  }

  /**
   * Sample current audio frequency data and compute a profile:
   *   {lowEnergy, highEnergy, ratio}
   */
  _sampleAudio() {
    if (!this._analyserNode || !this._freqData) return null;

    this._analyserNode.getByteFrequencyData(this._freqData);

    let lowSum = 0;
    let highSum = 0;
    const mid = AVAnalyzer.LOW_BAND_END;

    for (let i = 0; i < this._freqData.length; i++) {
      if (i < mid) {
        lowSum += this._freqData[i];
      } else {
        highSum += this._freqData[i];
      }
    }

    const lowEnergy = lowSum / (mid || 1);
    const highEnergy = highSum / (this._freqData.length - mid || 1);
    const total = lowEnergy + highEnergy;
    const ratio = total > 0 ? lowEnergy / total : 0.5;

    return { lowEnergy, highEnergy, ratio };
  }

  // ─── Window management ─────────────────────────────────────────

  /**
   * Finalise the current window: average all samples into a single
   * histogram and audio profile, and push to the windows array.
   */
  _closeWindow(startTime) {
    const visualHist = this._averageHistograms(this._currentVisualSamples);
    const audioProfile = this._averageAudioProfiles(this._currentAudioSamples);

    if (visualHist || audioProfile) {
      this._windows.push({
        time: startTime,
        visualHist: visualHist || this._emptyHistogram(),
        audioProfile: audioProfile || { lowEnergy: 0, highEnergy: 0, ratio: 0.5 },
      });
    }
  }

  _averageHistograms(samples) {
    if (!samples.length) return null;
    const bins = AVAnalyzer.HIST_BINS * 3;
    const avg = new Float32Array(bins);
    for (const h of samples) {
      for (let i = 0; i < bins; i++) avg[i] += h[i];
    }
    for (let i = 0; i < bins; i++) avg[i] /= samples.length;
    return avg;
  }

  _averageAudioProfiles(samples) {
    if (!samples.length) return null;
    let low = 0, high = 0, ratio = 0;
    for (const s of samples) {
      low += s.lowEnergy;
      high += s.highEnergy;
      ratio += s.ratio;
    }
    const n = samples.length;
    return { lowEnergy: low / n, highEnergy: high / n, ratio: ratio / n };
  }

  _emptyHistogram() {
    return new Float32Array(AVAnalyzer.HIST_BINS * 3);
  }

  // ─── Baseline computation ──────────────────────────────────────

  /**
   * Compute the baseline (median) histogram and audio profile from
   * the surrounding windows (excluding the current one).
   */
  _computeBaseline(currentIdx) {
    const start = Math.max(0, currentIdx - AVAnalyzer.BASELINE_WINDOW);
    const end = currentIdx;
    const subset = this._windows.slice(start, end);

    if (!subset.length) {
      return {
        visualHist: this._emptyHistogram(),
        audioProfile: { lowEnergy: 0, highEnergy: 0, ratio: 0.5 },
      };
    }

    // Median histogram (per-bin median)
    const bins = AVAnalyzer.HIST_BINS * 3;
    const visualHist = new Float32Array(bins);
    for (let i = 0; i < bins; i++) {
      const vals = subset.map((w) => w.visualHist[i]).sort((a, b) => a - b);
      visualHist[i] = vals[Math.floor(vals.length / 2)];
    }

    // Median audio profile
    const ratios = subset.map((w) => w.audioProfile.ratio).sort((a, b) => a - b);
    const lows = subset.map((w) => w.audioProfile.lowEnergy).sort((a, b) => a - b);
    const highs = subset.map((w) => w.audioProfile.highEnergy).sort((a, b) => a - b);
    const mid = Math.floor(subset.length / 2);

    return {
      visualHist,
      audioProfile: {
        lowEnergy: lows[mid],
        highEnergy: highs[mid],
        ratio: ratios[mid],
      },
    };
  }

  // ─── Distance metrics ─────────────────────────────────────────

  /**
   * Chi-squared distance between two normalised histograms.
   * Returns a value in [0, 1] where 0 = identical.
   */
  _histogramDistance(a, b) {
    if (!a || !b) return 0;
    let dist = 0;
    for (let i = 0; i < a.length; i++) {
      const sum = a[i] + b[i];
      if (sum > 0) {
        dist += ((a[i] - b[i]) ** 2) / sum;
      }
    }
    // Normalise to [0, 1] — chi-squared max is 2 for normalised hists
    return Math.min(dist / 2, 1);
  }

  /**
   * Distance between two audio profiles based on the low/high
   * energy ratio. Returns a value in [0, 1].
   */
  _audioDistance(a, b) {
    if (!a || !b) return 0;
    // Ratio difference (both in [0, 1])
    const ratioDiff = Math.abs(a.ratio - b.ratio);

    // Also consider total energy change (normalised)
    const totalA = a.lowEnergy + a.highEnergy;
    const totalB = b.lowEnergy + b.highEnergy;
    const maxTotal = Math.max(totalA, totalB, 1);
    const energyDiff = Math.abs(totalA - totalB) / maxTotal;

    return Math.min((ratioDiff + energyDiff) / 2, 1);
  }

  // ─── Flagging & merging ────────────────────────────────────────

  /** Score all windows and flag those above threshold. */
  _computeFlags() {
    this._flaggedWindows = [];

    if (this._windows.length < AVAnalyzer.MIN_BASELINE_SAMPLES) return;

    for (let i = AVAnalyzer.MIN_BASELINE_SAMPLES; i < this._windows.length; i++) {
      const w = this._windows[i];
      const baseline = this._computeBaseline(i);

      const vScore = this._histogramDistance(w.visualHist, baseline.visualHist);
      const aScore = this._audioDistance(w.audioProfile, baseline.audioProfile);
      const combined =
        vScore * AVAnalyzer.VISUAL_WEIGHT +
        aScore * AVAnalyzer.AUDIO_WEIGHT;

      if (combined >= AVAnalyzer.COMBINED_THRESHOLD) {
        this._flaggedWindows.push({
          time: w.time,
          score: combined,
        });
      }
    }
  }

  /** Merge adjacent flagged windows into segments. */
  _mergeFlags() {
    if (!this._flaggedWindows.length) return [];

    const sorted = [...this._flaggedWindows].sort((a, b) => a.time - b.time);
    const raw = [];
    let group = [sorted[0]];

    for (let i = 1; i < sorted.length; i++) {
      const prev = group[group.length - 1];
      const curr = sorted[i];

      if (curr.time - prev.time <= AVAnalyzer.MERGE_GAP_SEC + AVAnalyzer.WINDOW_SEC) {
        group.push(curr);
      } else {
        raw.push(group);
        group = [curr];
      }
    }
    raw.push(group);

    // Convert groups to {start, end} and filter by minimum duration
    const segments = [];
    for (const group of raw) {
      const start = Math.max(0, group[0].time - AVAnalyzer.PAD_BEFORE);
      const end = group[group.length - 1].time + AVAnalyzer.WINDOW_SEC + AVAnalyzer.PAD_AFTER;
      const duration = end - start;

      if (duration >= AVAnalyzer.MIN_SEGMENT_SEC) {
        segments.push({ start, end, source: "av-analysis" });
      }
    }

    return segments;
  }
}
