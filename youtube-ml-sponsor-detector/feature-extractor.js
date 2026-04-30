/**
 * feature-extractor.js — Feature Extraction for ML Sponsor Detector
 *
 * Implements two feature extractors that feed the bimodal ML model:
 *
 *   KeywordFeatureExtractor
 *     Converts transcript text into a 64-dimensional binary indicator
 *     vector. Each dimension flags whether a specific keyword or phrase
 *     pattern was present in the text. Four groups of 16 features cover:
 *     sponsor intro language, call-to-action language, offer/discount
 *     language, and product/service endorsement language.
 *
 *     This is the "text modality" described in the project plan. It is
 *     a distilled version of the DistilBERT [CLS] embedding used in the
 *     teacher model — the student uses an explicit keyword vocabulary
 *     rather than a learned representation, keeping model size small.
 *
 *   MFCCExtractor
 *     Computes Mel-Frequency Cepstral Coefficients (MFCCs) from the
 *     video's audio stream in real time via the Web Audio API. Returns
 *     a 13-dimensional MFCC vector per time window. An anomaly score
 *     (cosine distance from a rolling baseline) is also provided.
 *
 *     This is the "audio modality." It captures tonal and spectral
 *     shifts accompanying sponsor reads: background music changes,
 *     rehearsed delivery tone, room acoustic differences, etc.
 */

"use strict";


// ─────────────────────────────────────────────────────────────────────────────
//  KeywordFeatureExtractor
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Converts text into a 64-dim binary keyword indicator vector.
 *
 * Feature groups:
 *   [0–15]   Sponsor intro phrases  (high precision, 3× weight in scoring)
 *   [16–31]  Call-to-action language (1.5× weight)
 *   [32–47]  Offer / discount language (1.5× weight)
 *   [48–63]  Product / service endorsement (0.5× weight — corroborating)
 */
class KeywordFeatureExtractor {

  static FEATURE_DIM = 64;

  // Each entry is a RegExp. Index in the array === feature dimension index.
  static VOCAB = [
    // ── Group 0: Sponsor intro phrases [0–15] ──────────────────────────────
    /\bsponsored by\b/i,
    /\bbrought to you by\b/i,
    /\btoday'?s sponsor\b/i,
    /\bthis (?:video|episode|segment|portion) (?:is )?(?:sponsored|presented|made possible)\b/i,
    /\bthanks? to .{1,40} for sponsoring\b/i,
    /\ba (?:huge|big|special) thanks?\b/i,
    /\bpartner(?:ed|ship)? with\b/i,
    /\bpaid (?:promotion|partnership|advertisement)\b/i,
    /\bofficial (?:sponsor|partner)\b/i,
    /\bsponsor of this\b/i,
    /\bpresented by\b/i,
    /\bmade possible by\b/i,
    /\bpowered by\b/i,
    /\bin (?:association|partnership) with\b/i,
    /\bsupported by\b/i,
    /\baffiliate\b/i,

    // ── Group 1: Call-to-action language [16–31] ───────────────────────────
    /\buse (?:my |our |the )?(?:code|link|coupon|promo)\b/i,
    /\buse code [A-Z0-9]{3,}/i,
    /\blink (?:is |in )?(?:the )?description\b/i,
    /\bgo to .{1,40}\.(?:com|io|co|net|org)\b/i,
    /\bhead (?:on )?over to\b/i,
    /\bvisit .{1,40}\.(?:com|io|co)\b/i,
    /\bcheck (?:them )?out\b/i,
    /\bclick (?:the |my |that )?link\b/i,
    /\bscan (?:the )?(?:qr|barcode)\b/i,
    /\bswipe up\b/i,
    /\btap (?:the )?link\b/i,
    /\bin (?:the )?bio\b/i,
    /\bdown(?:load|load the app)\b/i,
    /\bsign up\b/i,
    /\bget started\b/i,
    /\btry (?:it |them )?(?:free|for free|today)\b/i,

    // ── Group 2: Offer / discount language [32–47] ─────────────────────────
    /\b\d+\s*%\s*off\b/i,
    /\bpercent off\b/i,
    /\bfree trial\b/i,
    /\bdiscount code\b/i,
    /\bpromo code\b/i,
    /\bcoupon code\b/i,
    /\bspecial (?:offer|deal|discount)\b/i,
    /\bexclusive (?:deal|offer|discount)\b/i,
    /\blimited time\b/i,
    /\bmoney.?back guarantee\b/i,
    /\bfirst \d+ (?:people|users|customers|subscribers|listeners)\b/i,
    /\bno (?:cost|charge|commitment)\b/i,
    /\bcancel anytime\b/i,
    /\bno (?:risk|obligation)\b/i,
    /\bfirst (?:month|year|week) free\b/i,
    /\bsave \$?\d/i,

    // ── Group 3: Product / service endorsement [48–63] ────────────────────
    /\bvpn\b/i,
    /\bmeal (?:kit|prep|delivery)\b/i,
    /\bskin ?care\b/i,
    /\bsupplement\b/i,
    /\bapp\b/i,
    /\bsubscription\b/i,
    /\bpremium\b/i,
    /\bnootropic\b/i,
    /\bprotein\b/i,
    /\bshaker\b/i,
    /\bplatform\b/i,
    /\bsoftware\b/i,
    /\btools?\b/i,
    /\bservice\b/i,
    /\bproduct\b/i,
    /\bbrand\b/i,
  ];

  /**
   * Extract features from a single text string.
   * Returns a Float32Array(FEATURE_DIM) with 0/1 values — one per vocabulary entry.
   *
   * @param {string} text
   * @returns {Float32Array}
   */
  static extractFromText(text) {
    const features = new Float32Array(KeywordFeatureExtractor.FEATURE_DIM);
    for (let i = 0; i < KeywordFeatureExtractor.VOCAB.length; i++) {
      features[i] = KeywordFeatureExtractor.VOCAB[i].test(text) ? 1.0 : 0.0;
    }
    return features;
  }

  /**
   * Extract and aggregate features from a window of caption cues.
   * Each cue: {start: number, dur: number, text: string}
   * Returns Float32Array(FEATURE_DIM) — max-pooled across all cues in the window.
   * Max-pooling means a feature fires if *any* cue in the window matched it.
   *
   * @param {Array<{start:number, dur:number, text:string}>} cues
   * @returns {Float32Array}
   */
  static extractFromCues(cues) {
    const features = new Float32Array(KeywordFeatureExtractor.FEATURE_DIM);
    for (const cue of cues) {
      const cueFeatures = KeywordFeatureExtractor.extractFromText(cue.text || "");
      for (let i = 0; i < features.length; i++) {
        if (cueFeatures[i] > features[i]) features[i] = cueFeatures[i];
      }
    }
    return features;
  }

  /**
   * Compute a weighted keyword score from a feature vector.
   * Used for quick pre-filtering before running the full MLP.
   * Group 0 (intro phrases) is weighted 3×; group 1–2 (CTA, offers) 1.5×;
   * group 3 (product language) 0.5× (too common outside sponsors).
   *
   * @param {Float32Array} features
   * @returns {number}
   */
  static keywordScore(features) {
    let score = 0;
    for (let i = 0;  i < 16; i++) score += features[i] * 3.0;
    for (let i = 16; i < 48; i++) score += features[i] * 1.5;
    for (let i = 48; i < 64; i++) score += features[i] * 0.5;
    return score;
  }
}


// ─────────────────────────────────────────────────────────────────────────────
//  MFCCExtractor
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Real-time Mel-Frequency Cepstral Coefficient (MFCC) extractor.
 *
 * Connects to a <video> element's audio via the Web Audio API's AnalyserNode,
 * then at each call to captureFrame() computes a standard 13-coefficient MFCC
 * vector by:
 *
 *   1. Reading the current FFT magnitude spectrum (in dB) from the analyser.
 *   2. Converting dB to linear power spectrum.
 *   3. Applying a pre-computed mel filterbank (triangular filters, 26 bands).
 *   4. Taking the log of each band's energy.
 *   5. Applying the DCT-II to get the cepstral coefficients.
 *   6. Mean-subtracting against a rolling baseline (delta-MFCC).
 *
 * An anomaly score (cosine distance between current and baseline MFCC vectors)
 * is also tracked and exposed via getAnomalyScore().
 */
class MFCCExtractor {

  // ─── Configuration ──────────────────────────────────────────────────────

  /** Number of MFCC coefficients to return per frame. */
  static NUM_COEFFICIENTS = 13;

  /**
   * Number of MFCC frames buffered for the CNN audio branch of the student
   * model.  Must match N_FRAMES in models.py and ml-detector.js.
   * At 1 frame/second this covers the last 30 seconds of audio.
   */
  static N_FRAMES = 30;

  /** Number of triangular mel filters. */
  static NUM_MEL_FILTERS = 26;

  /** FFT size used by the Web Audio AnalyserNode. */
  static FFT_SIZE = 512;

  /** Mel filterbank frequency bounds (Hz). */
  static LOW_FREQ_HZ  = 80;
  static HIGH_FREQ_HZ = 8000;

  /**
   * Rolling history lengths.
   * SHORT_HISTORY: used for the "current" MFCC estimate (last ~5 seconds).
   * LONG_HISTORY:  used for the baseline (last ~60 seconds of non-recent data).
   */
  static SHORT_HISTORY = 5;
  static LONG_HISTORY  = 60;

  // ─── Constructor ────────────────────────────────────────────────────────

  constructor() {
    this._audioCtx      = null;
    this._analyserNode  = null;
    this._sourceNode    = null;
    this._freqData      = null;   // Float32Array for getFloatFrequencyData()
    this._melFilterbank = null;   // Array<Float32Array> — one filter per mel band
    this._connected     = false;

    // Ring buffers
    this._shortHistory = [];      // last SHORT_HISTORY MFCC frames
    this._longHistory  = [];      // last LONG_HISTORY  MFCC frames
  }

  // ─── Public API ─────────────────────────────────────────────────────────

  /**
   * Connect to a <video> element and start capturing audio.
   * Safe to call multiple times — only connects once.
   *
   * @param {HTMLVideoElement} videoEl
   */
  connect(videoEl) {
    try {
      if (!this._audioCtx) {
        this._audioCtx     = new (window.AudioContext || window.webkitAudioContext)();
        this._analyserNode = this._audioCtx.createAnalyser();
        this._analyserNode.fftSize = MFCCExtractor.FFT_SIZE;
        this._freqData     = new Float32Array(this._analyserNode.frequencyBinCount);

        // Pre-compute mel filterbank (depends only on FFT size + sample rate).
        this._melFilterbank = this._buildMelFilterbank(
          this._audioCtx.sampleRate,
          this._analyserNode.frequencyBinCount
        );
      }

      if (!this._connected) {
        this._sourceNode = this._audioCtx.createMediaElementSource(videoEl);
        this._sourceNode.connect(this._analyserNode);
        this._analyserNode.connect(this._audioCtx.destination);
        this._connected = true;
        console.log("[ML Detector] MFCC extractor connected to audio.");
      }
    } catch (e) {
      console.warn("[ML Detector] MFCC audio connect failed:", e.message);
    }
  }

  /** Reset history buffers (call when switching to a new video). */
  reset() {
    this._shortHistory = [];
    this._longHistory  = [];
  }

  /**
   * Capture one MFCC frame from the current audio state.
   * Updates internal history for anomaly scoring.
   *
   * @returns {Float32Array|null} 13-dim delta-MFCC vector, or null if not ready.
   */
  captureFrame() {
    if (!this._analyserNode || !this._freqData || !this._connected) return null;

    try {
      this._analyserNode.getFloatFrequencyData(this._freqData);
      const rawMfcc  = this._computeRawMFCC(this._freqData);
      const deltaMfcc = this._subtractBaseline(rawMfcc);

      // Update history
      this._shortHistory.push(deltaMfcc);
      if (this._shortHistory.length > MFCCExtractor.SHORT_HISTORY) {
        this._shortHistory.shift();
      }
      this._longHistory.push(deltaMfcc);
      if (this._longHistory.length > MFCCExtractor.LONG_HISTORY) {
        this._longHistory.shift();
      }

      return deltaMfcc;
    } catch {
      return null;
    }
  }

  /**
   * Return the mean MFCC vector over the last numFrames captured frames.
   * Useful as a lightweight single-vector audio summary (anomaly scoring,
   * heuristic fallback). For the ONNX CNN branch, use getFrameBuffer().
   *
   * @param {number} [numFrames=5]
   * @returns {Float32Array}
   */
  getRecentMeanMFCC(numFrames = 5) {
    if (!this._shortHistory.length) {
      return new Float32Array(MFCCExtractor.NUM_COEFFICIENTS);
    }
    return this._meanVector(this._shortHistory.slice(-numFrames));
  }

  /**
   * Return a flat frame buffer of the last nFrames MFCC frames, suitable
   * for the student model's CNN audio branch.
   *
   * Layout: row-major [frame0_c0, frame0_c1, …, frame0_c12, frame1_c0, …]
   * Total length: nFrames × NUM_COEFFICIENTS (30 × 13 = 390 by default).
   *
   * If fewer than nFrames frames have been captured since the last reset(),
   * the leading frames are zero-padded. This preserves the temporal
   * recency: the most recent frame is always at the end of the buffer.
   *
   * @param {number} [nFrames] - defaults to MFCCExtractor.N_FRAMES (30)
   * @returns {Float32Array} flat buffer of length nFrames × NUM_COEFFICIENTS
   */
  getFrameBuffer(nFrames = MFCCExtractor.N_FRAMES) {
    const C = MFCCExtractor.NUM_COEFFICIENTS;
    const buf = new Float32Array(nFrames * C);  // zero-initialised

    // Pull the most recent nFrames from the long history ring buffer.
    const available = this._longHistory;
    const nAvailable = available.length;

    if (nAvailable === 0) return buf;

    // Copy frames right-aligned: if we have fewer than nFrames, pad the front.
    const nCopy = Math.min(nAvailable, nFrames);
    const srcStart = nAvailable - nCopy;          // index into _longHistory
    const dstStart = nFrames   - nCopy;           // index into output buffer (frames)

    for (let i = 0; i < nCopy; i++) {
      const frame = available[srcStart + i];
      const dstOffset = (dstStart + i) * C;
      for (let c = 0; c < C; c++) {
        buf[dstOffset + c] = frame[c];
      }
    }

    return buf;
  }

  /**
   * Compute an audio anomaly score: cosine distance between the mean
   * of the last SHORT_HISTORY frames and the mean of the longer baseline.
   *
   * Returns a value in [0, 1] where higher = more acoustically anomalous.
   * A score above ~0.3 suggests the audio environment has shifted noticeably.
   *
   * @returns {number}
   */
  getAnomalyScore() {
    if (this._shortHistory.length < 2 || this._longHistory.length < 8) return 0;

    const recentFrames   = this._shortHistory.slice(-MFCCExtractor.SHORT_HISTORY);
    const baselineFrames = this._longHistory.slice(0, -MFCCExtractor.SHORT_HISTORY);

    if (!baselineFrames.length) return 0;

    const current  = this._meanVector(recentFrames);
    const baseline = this._meanVector(baselineFrames);

    return this._cosineDistance(current, baseline);
  }

  // ─── Mel filterbank construction ──────────────────────────────────────

  /**
   * Build a mel filterbank: NUM_MEL_FILTERS triangular filters,
   * equally spaced on the mel scale between LOW_FREQ_HZ and HIGH_FREQ_HZ.
   *
   * @param {number} sampleRate
   * @param {number} numBins - half of FFT size (frequencyBinCount)
   * @returns {Array<Float32Array>}
   */
  _buildMelFilterbank(sampleRate, numBins) {
    const nyquist  = sampleRate / 2;
    const lowMel   = this._hzToMel(MFCCExtractor.LOW_FREQ_HZ);
    const highMel  = this._hzToMel(Math.min(MFCCExtractor.HIGH_FREQ_HZ, nyquist));
    const nFilters = MFCCExtractor.NUM_MEL_FILTERS;

    // Linearly spaced mel points (nFilters + 2 = center + two boundary points)
    const melPoints = [];
    for (let i = 0; i < nFilters + 2; i++) {
      melPoints.push(lowMel + (i / (nFilters + 1)) * (highMel - lowMel));
    }

    // Convert to bin indices
    const bins = melPoints.map(mel => {
      const hz = this._melToHz(mel);
      return Math.min(numBins - 1, Math.floor((hz / nyquist) * numBins));
    });

    // Build triangular filters
    const filters = [];
    for (let m = 1; m <= nFilters; m++) {
      const filter = new Float32Array(numBins);
      const lo = bins[m - 1];
      const c  = bins[m];
      const hi = bins[m + 1];

      for (let k = lo; k < c;  k++) {
        if (c  > lo) filter[k] = (k - lo) / (c  - lo);
      }
      for (let k = c;  k < hi; k++) {
        if (hi > c)  filter[k] = (hi - k) / (hi - c);
      }
      filters.push(filter);
    }
    return filters;
  }

  _hzToMel(hz)  { return 2595 * Math.log10(1 + hz / 700); }
  _melToHz(mel) { return 700 * (Math.pow(10, mel / 2595) - 1); }

  // ─── MFCC computation ────────────────────────────────────────────────

  /**
   * Compute raw (non-normalized) MFCCs from FFT data in dB.
   *
   * Steps:
   *   dB spectrum → linear power → mel filterbank → log energies → DCT-II
   *
   * @param {Float32Array} freqDataDB - output of getFloatFrequencyData()
   * @returns {Float32Array} NUM_COEFFICIENTS raw MFCC values
   */
  _computeRawMFCC(freqDataDB) {
    const nFilters = MFCCExtractor.NUM_MEL_FILTERS;
    const nCoeff   = MFCCExtractor.NUM_COEFFICIENTS;

    // 1. dB → linear power
    const power = new Float32Array(freqDataDB.length);
    for (let i = 0; i < freqDataDB.length; i++) {
      power[i] = Math.pow(10, freqDataDB[i] / 10);
    }

    // 2. Mel filterbank → log energy per band
    const melEnergies = new Float32Array(nFilters);
    for (let m = 0; m < nFilters; m++) {
      let energy = 0;
      const filter = this._melFilterbank[m];
      for (let k = 0; k < filter.length; k++) {
        energy += filter[k] * power[k];
      }
      melEnergies[m] = Math.log(Math.max(energy, 1e-10));
    }

    // 3. DCT-II → cepstral coefficients
    // c[n] = sum_m( x[m] * cos(π * n * (2m + 1) / (2M)) )
    const mfcc = new Float32Array(nCoeff);
    for (let n = 0; n < nCoeff; n++) {
      let sum = 0;
      for (let m = 0; m < nFilters; m++) {
        sum += melEnergies[m] * Math.cos(Math.PI * n * (2 * m + 1) / (2 * nFilters));
      }
      mfcc[n] = sum;
    }
    return mfcc;
  }

  /**
   * Subtract the long-history baseline mean from a raw MFCC vector,
   * then scale so that typical deviations land near ±1.
   * This produces delta-MFCCs (how much has the spectrum shifted).
   *
   * @param {Float32Array} rawMfcc
   * @returns {Float32Array}
   */
  _subtractBaseline(rawMfcc) {
    if (this._longHistory.length < 5) return rawMfcc.slice();

    const baseline = this._meanVector(
      this._longHistory.slice(0, Math.max(1, this._longHistory.length - 2))
    );
    const delta = new Float32Array(rawMfcc.length);
    for (let i = 0; i < rawMfcc.length; i++) {
      // Soft normalisation: divide by |baseline| + small constant
      delta[i] = (rawMfcc[i] - baseline[i]) / (Math.abs(baseline[i]) + 1.0);
    }
    return delta;
  }

  // ─── Vector utilities ─────────────────────────────────────────────────

  _meanVector(frames) {
    const n = frames.length;
    const dim = MFCCExtractor.NUM_COEFFICIENTS;
    const mean = new Float32Array(dim);
    for (const f of frames) {
      for (let i = 0; i < dim; i++) mean[i] += f[i];
    }
    for (let i = 0; i < dim; i++) mean[i] /= n;
    return mean;
  }

  _cosineDistance(a, b) {
    let dot = 0, normA = 0, normB = 0;
    for (let i = 0; i < a.length; i++) {
      dot   += a[i] * b[i];
      normA += a[i] * a[i];
      normB += b[i] * b[i];
    }
    const denom = Math.sqrt(normA) * Math.sqrt(normB);
    if (denom === 0) return 0;
    const similarity = Math.max(-1, Math.min(1, dot / denom));
    // Map similarity ∈ [-1, 1] → distance ∈ [0, 1]
    return (1 - similarity) / 2;
  }
}
