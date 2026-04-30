/**
 * ml-detector.js — Bimodal MLP Inference Engine
 *
 * Combines a 64-dim keyword feature vector (text) and a 13-dim MFCC
 * feature vector (audio) through a small 3-layer MLP to produce a
 * sponsor probability score in [0, 1].
 *
 * This is the student model from the project plan — a lightweight
 * in-browser approximation of the bimodal teacher (DistilBERT + Whisper).
 * Architecture: [77 → 32 → 16 → 1] with ReLU + sigmoid output.
 *
 * ─── Two inference backends (tried in order) ─────────────────────────
 *
 *  1. ONNX Runtime Web
 *     If ort.js is available (listed in manifest.json's content_scripts
 *     or loaded separately) and model.onnx is present in the extension
 *     directory, the trained model is loaded and used for inference.
 *     This is the production path once the model is trained via Kaggle.
 *
 *     To set up: download ort.min.js from https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/
 *     and place it in this extension folder alongside model.onnx.
 *     Then add "ort.min.js" to content_scripts in manifest.json
 *     *before* this file.
 *
 *  2. Built-in Heuristic MLP (fallback)
 *     When no trained model is available, a parameter-free MLP runs with
 *     manually calibrated weights. The weights are designed using domain
 *     knowledge so that the extension gives useful results immediately:
 *
 *       • Strong sponsor intro phrase in text → score > 0.80
 *       • CTA + offer language together      → score > 0.70
 *       • Audio anomaly alone                → score ~0.50
 *       • Audio anomaly + any sponsor text   → score ~0.65
 *       • No signals                         → score < 0.15
 *
 *     The heuristic weights serve as a warm starting point; actual training
 *     on SponsorBlock data will produce much more accurate weights.
 *
 * ─── Usage ────────────────────────────────────────────────────────────
 *
 *   const detector = new MLSponsorDetector();
 *   await detector.init();
 *
 *   const textFeatures  = KeywordFeatureExtractor.extractFromCues(cues);
 *   const audioFeatures = mfccExtractor.getRecentMeanMFCC();
 *   const score = detector.score(textFeatures, audioFeatures);  // 0–1
 */

"use strict";

class MLSponsorDetector {

  // ─── Architecture ────────────────────────────────────────────────────────

  static TEXT_DIM  = 64;   // KeywordFeatureExtractor.FEATURE_DIM
  static MFCC_DIM  = 13;   // MFCCExtractor.NUM_COEFFICIENTS — one frame
  static N_FRAMES  = 30;   // MFCCExtractor.N_FRAMES — CNN input length
  static HIDDEN1   = 32;
  static HIDDEN2   = 16;
  // Legacy INPUT_DIM kept for heuristic backward compat (text + mean MFCC).
  static INPUT_DIM = 64 + 13;  // 77

  // ─── Decision thresholds ─────────────────────────────────────────────────

  /**
   * Score must exceed this to seek past (or speed through) a segment.
   * Set to 0.85 per the project plan — conservative to minimise false
   * positives on real content.
   */
  static ENTRY_THRESHOLD = 0.85;

  /**
   * Hysteresis exit threshold for the speed-mode fallback (used when
   * no segment end time is known). Lower than entry so we don't toggle
   * rapidly at boundaries.
   */
  static EXIT_THRESHOLD = 0.50;

  /**
   * Number of consecutive frames that must exceed ENTRY_THRESHOLD
   * before we actually trigger speedup (reduces false positives).
   */
  static MIN_CONSECUTIVE_FRAMES = 2;

  // ─── Constructor ─────────────────────────────────────────────────────────

  constructor() {
    this._onnxSession = null;
    this._useOnnx     = false;

    // MLP weight matrices (Float32Arrays, row-major)
    // Shapes: W1[HIDDEN1 × INPUT_DIM], W2[HIDDEN2 × HIDDEN1], W3[1 × HIDDEN2]
    this._W1 = null;
    this._b1 = null;
    this._W2 = null;
    this._b2 = null;
    this._W3 = null;
    this._b3 = null;

    this._initHeuristicWeights();
  }

  // ─── Initialization ───────────────────────────────────────────────────────

  /**
   * Attempt to load a trained ONNX model. Falls back to the heuristic MLP
   * silently if ort.js is unavailable or model.onnx is not present.
   * Always resolves — never throws.
   */
  async init() {
    try {
      if (typeof ort !== "undefined") {
        const modelUrl = chrome.runtime.getURL("model.onnx");
        this._onnxSession = await ort.InferenceSession.create(modelUrl);
        this._useOnnx = true;
        console.log("[ML Detector] Loaded ONNX model.");
        return;
      }
    } catch (e) {
      // Silently fall through — model.onnx may not exist yet (expected during dev)
    }
    console.log("[ML Detector] Using built-in heuristic MLP (no trained model found).");
  }

  /**
   * Whether the loaded backend is a trained ONNX model (true) or the
   * built-in heuristic weights (false). Useful for logging.
   */
  get isTrained() { return this._useOnnx; }

  // ─── Inference ────────────────────────────────────────────────────────────

  /**
   * Score using text features + an MFCC frame buffer.
   * Synchronous — uses the heuristic MLP (mean-pools the frame buffer
   * to a 13-dim vector before forwarding).  For the ONNX path use
   * scoreAsync() which supports the full 2D CNN input.
   *
   * @param {Float32Array} textFeatures   64-dim keyword indicator vector
   * @param {Float32Array} frameBuffer    [N_FRAMES × MFCC_DIM] flat buffer
   *                                      (or legacy 13-dim mean vector)
   * @returns {number} sponsor probability in [0, 1]
   */
  score(textFeatures, frameBuffer) {
    const text = textFeatures || new Float32Array(MLSponsorDetector.TEXT_DIM);
    const audioMean = this._meanPoolFrameBuffer(frameBuffer);
    return this._scoreHeuristic(text, audioMean);
  }

  /**
   * Async variant — uses the ONNX session with the two-input interface
   * when a trained model is available.
   *
   * ONNX inputs:
   *   text_input   float32 [1, 64]
   *   audio_input  float32 [1, N_FRAMES, 13]
   *
   * Falls back to the heuristic MLP silently on any error.
   *
   * @param {Float32Array} textFeatures  64-dim keyword indicator vector
   * @param {Float32Array} frameBuffer   [N_FRAMES × MFCC_DIM] flat buffer
   *                                     (zero buffer is fine for text-only pre-detection)
   * @returns {Promise<number>}
   */
  async scoreAsync(textFeatures, frameBuffer) {
    const text = textFeatures || new Float32Array(MLSponsorDetector.TEXT_DIM);
    const fb   = frameBuffer  || new Float32Array(MLSponsorDetector.N_FRAMES * MLSponsorDetector.MFCC_DIM);

    if (this._useOnnx && this._onnxSession) {
      try {
        // text_input: [1, 64]
        const textTensor  = new ort.Tensor("float32", text.slice(), [1, MLSponsorDetector.TEXT_DIM]);

        // audio_input: [1, N_FRAMES, MFCC_DIM]
        // fb is already flat [N_FRAMES * MFCC_DIM]; just reshape dims.
        const audioData   = fb.length === MLSponsorDetector.N_FRAMES * MLSponsorDetector.MFCC_DIM
          ? fb.slice()
          : this._padOrTrimFrameBuffer(fb);
        const audioTensor = new ort.Tensor(
          "float32",
          audioData,
          [1, MLSponsorDetector.N_FRAMES, MLSponsorDetector.MFCC_DIM]
        );

        const results = await this._onnxSession.run({
          text_input:  textTensor,
          audio_input: audioTensor,
        });
        // Model applies sigmoid internally → output is already a probability.
        return Math.max(0, Math.min(1, results.output.data[0]));
      } catch (e) {
        console.warn("[ML Detector] ONNX inference error, falling back:", e.message);
      }
    }
    const audioMean = this._meanPoolFrameBuffer(fb);
    return this._scoreHeuristic(text, audioMean);
  }

  // ─── Frame buffer helpers ─────────────────────────────────────────────────

  /**
   * Mean-pool a flat [N_FRAMES × MFCC_DIM] frame buffer → [MFCC_DIM] vector.
   * Handles legacy 13-dim mean-vector input for backward compatibility.
   *
   * @param {Float32Array} fb
   * @returns {Float32Array} MFCC_DIM-length mean vector
   */
  _meanPoolFrameBuffer(fb) {
    const C = MLSponsorDetector.MFCC_DIM;
    if (!fb || fb.length === 0) return new Float32Array(C);
    // Already a single mean vector (legacy call path from getRecentMeanMFCC)
    if (fb.length === C) return fb;

    const nFrames = Math.floor(fb.length / C);
    const mean = new Float32Array(C);
    for (let f = 0; f < nFrames; f++) {
      for (let c = 0; c < C; c++) {
        mean[c] += fb[f * C + c];
      }
    }
    for (let c = 0; c < C; c++) mean[c] /= nFrames;
    return mean;
  }

  /**
   * Pad or trim a frame buffer to exactly N_FRAMES × MFCC_DIM samples.
   * Short buffers are zero-padded at the front (preserving recency).
   *
   * @param {Float32Array} fb  input buffer (any length that is a multiple of MFCC_DIM)
   * @returns {Float32Array}   Float32Array of length N_FRAMES × MFCC_DIM
   */
  _padOrTrimFrameBuffer(fb) {
    const C = MLSponsorDetector.MFCC_DIM;
    const N = MLSponsorDetector.N_FRAMES;
    const target = N * C;
    const out = new Float32Array(target);  // zero-initialised

    if (!fb || fb.length === 0) return out;

    if (fb.length >= target) {
      // Trim: take the most recent N frames (tail of the buffer).
      out.set(fb.subarray(fb.length - target));
    } else {
      // Pad: place fb at the end (most recent frames last).
      out.set(fb, target - fb.length);
    }
    return out;
  }

  // ─── Built-in heuristic MLP ───────────────────────────────────────────────

  /**
   * Forward pass: [INPUT_DIM → HIDDEN1 → HIDDEN2 → 1] with ReLU + sigmoid.
   * Accepts the 13-dim mean MFCC vector (produced by _meanPoolFrameBuffer).
   */
  _scoreHeuristic(textFeatures, audioFeatures) {
    const input = new Float32Array(MLSponsorDetector.INPUT_DIM);
    input.set(textFeatures,  0);
    input.set(audioFeatures, MLSponsorDetector.TEXT_DIM);

    const h1 = this._relu(this._linear(input, this._W1, this._b1, MLSponsorDetector.HIDDEN1));
    const h2 = this._relu(this._linear(h1,    this._W2, this._b2, MLSponsorDetector.HIDDEN2));
    const out = this._linear(h2, this._W3, this._b3, 1);
    return this._sigmoid(out[0]);
  }

  /** y = W·x + b.  W stored row-major: W[i * inputDim + j] */
  _linear(x, W, b, outputDim) {
    const inputDim = x.length;
    const y = new Float32Array(outputDim);
    for (let i = 0; i < outputDim; i++) {
      let sum = b[i];
      for (let j = 0; j < inputDim; j++) {
        sum += W[i * inputDim + j] * x[j];
      }
      y[i] = sum;
    }
    return y;
  }

  _relu(v)       { for (let i = 0; i < v.length; i++) v[i] = Math.max(0, v[i]); return v; }
  _sigmoid(x)    { return 1 / (1 + Math.exp(-x)); }

  // ─── Heuristic weight initialisation ─────────────────────────────────────

  /**
   * Manually calibrated weights encoding domain knowledge about sponsor reads.
   *
   * Layer 1 (INPUT_DIM→HIDDEN1):
   *   Neurons 0–7   : respond to sponsor intro phrases (features 0–15)  → fire on any single intro phrase
   *   Neurons 8–15  : respond to CTA language (features 16–31)          → fire on 1+ CTA patterns
   *   Neurons 16–23 : respond to offer/discount language (features 32–47)
   *   Neurons 24–27 : respond to product/service language (features 48–63) → needs 2+ hits
   *   Neurons 28–31 : respond to audio MFCC anomaly (features 64–76)    → sensitive to lower coefficients
   *
   * Layer 2 (HIDDEN1→HIDDEN2):
   *   Neurons 0–3   : strong intro phrase alone        → very high score
   *   Neurons 4–7   : CTA + offer co-occurrence        → high score
   *   Neurons 8–11  : audio anomaly alone              → moderate score
   *   Neurons 12–15 : audio anomaly + product language → reinforced score
   *
   * Layer 3 (HIDDEN2→1):
   *   Each H2 neuron contributes a calibrated weight toward the final sigmoid.
   */
  _initHeuristicWeights() {
    const T  = MLSponsorDetector.TEXT_DIM;   // 64
    const A  = MLSponsorDetector.AUDIO_DIM;  // 13
    const H1 = MLSponsorDetector.HIDDEN1;    // 32
    const H2 = MLSponsorDetector.HIDDEN2;    // 16
    const IN = T + A;                        // 77

    // ── Layer 1: W1[H1, IN], b1[H1] ────────────────────────────────────────

    this._W1 = new Float32Array(H1 * IN).fill(0);
    this._b1 = new Float32Array(H1).fill(0);

    // Neurons 0–7: sponsor intro phrases (features 0–15)
    // Each intro phrase feature adds 0.8; one match enough to clear the bias.
    for (let n = 0; n < 8; n++) {
      for (let f = 0; f < 16; f++) {
        this._W1[n * IN + f] = 0.8;
      }
      this._b1[n] = -0.5;   // fires if ≥1 intro phrase present
    }

    // Neurons 8–15: call-to-action language (features 16–31)
    for (let n = 8; n < 16; n++) {
      for (let f = 16; f < 32; f++) {
        this._W1[n * IN + f] = 0.65;
      }
      this._b1[n] = -0.5;
    }

    // Neurons 16–23: offer / discount language (features 32–47)
    for (let n = 16; n < 24; n++) {
      for (let f = 32; f < 48; f++) {
        this._W1[n * IN + f] = 0.65;
      }
      this._b1[n] = -0.5;
    }

    // Neurons 24–27: product / service language (features 48–63)
    // Needs more hits to fire (these phrases occur in non-sponsor content too).
    for (let n = 24; n < 28; n++) {
      for (let f = 48; f < 64; f++) {
        this._W1[n * IN + f] = 0.45;
      }
      this._b1[n] = -0.8;   // needs ≥2 product-term hits
    }

    // Neurons 28–31: audio MFCC anomaly (features T..T+A-1)
    // Lower MFCC coefficients (spectral tilt) are more diagnostically useful.
    // Also weakly sensitive to intro phrases (text+audio agreement boosts confidence).
    for (let n = 28; n < 32; n++) {
      for (let f = 0; f < A; f++) {
        // Coefficients 0–5 carry more acoustic identity information
        this._W1[n * IN + T + f] = f < 6 ? 0.7 : 0.25;
      }
      // Weak text coupling
      for (let f = 0; f < 16; f++) {
        this._W1[n * IN + f] = 0.15;
      }
      this._b1[n] = -0.8;   // audio must be genuinely anomalous
    }

    // ── Layer 2: W2[H2, H1], b2[H2] ────────────────────────────────────────

    this._W2 = new Float32Array(H2 * H1).fill(0);
    this._b2 = new Float32Array(H2).fill(0);

    // Neurons 0–3: strong intro phrase → very high activation
    for (let n = 0; n < 4; n++) {
      for (let h = 0; h < 8; h++) {    // L1 intro neurons
        this._W2[n * H1 + h] = 1.2;
      }
      this._b2[n] = -0.3;
    }

    // Neurons 4–7: CTA + offer combination
    // Both groups must fire (modelled by requiring the sum to clear a higher bias)
    for (let n = 4; n < 8; n++) {
      for (let h = 8;  h < 16; h++) this._W2[n * H1 + h] = 0.8;  // CTA
      for (let h = 16; h < 24; h++) this._W2[n * H1 + h] = 0.8;  // offer
      this._b2[n] = -1.1;   // needs both CTA and offer active
    }

    // Neurons 8–11: audio anomaly alone → moderate activation
    for (let n = 8; n < 12; n++) {
      for (let h = 28; h < 32; h++) this._W2[n * H1 + h] = 1.0;
      this._b2[n] = -0.3;
    }

    // Neurons 12–15: audio anomaly + product language → reinforced
    // Bias raised to -1.5 so that product language ALONE does not fire
    // (common tech words like "app", "platform", "premium" appear in
    //  non-sponsor segments of tech videos). Audio must also be anomalous.
    for (let n = 12; n < 16; n++) {
      for (let h = 28; h < 32; h++) this._W2[n * H1 + h] = 0.7;  // audio
      for (let h = 24; h < 28; h++) this._W2[n * H1 + h] = 0.3;  // product (reduced weight)
      this._b2[n] = -1.5;   // requires genuine audio anomaly; product language alone won't clear this
    }

    // ── Layer 3: W3[1, H2], b3[1] ──────────────────────────────────────────

    this._W3 = new Float32Array(H2).fill(0);
    this._b3 = new Float32Array(1);

    // Calibrated to push sigmoid above ENTRY_THRESHOLD (0.65) when:
    //   - intro phrase present: W3[0–3] = 2.0 → logit ≈ 2.0 - 1.6 = +0.4 → sigmoid ≈ 0.60
    //     (+ CTA/offer combo adds more → easily exceeds 0.65)
    //   - CTA + offer:          W3[4–7] = 1.6 → logit ≈ 1.6*2 - 1.6 = +1.6 → sigmoid ≈ 0.83
    //   - audio alone:          W3[8–11] = 0.9 → logit ≈ 0.9 - 1.6 = -0.7 → sigmoid ≈ 0.33
    //   - audio + product:      W3[12–15] = 1.3 → logit ≈ 1.3*2 - 1.6 = +1.0 → sigmoid ≈ 0.73

    for (let h = 0;  h < 4;  h++) this._W3[h] = 2.0;   // intro → high
    for (let h = 4;  h < 8;  h++) this._W3[h] = 1.6;   // CTA+offer → high
    for (let h = 8;  h < 12; h++) this._W3[h] = 0.9;   // audio only → moderate
    for (let h = 12; h < 16; h++) this._W3[h] = 1.3;   // audio+product → reinforced

    // Global bias: below zero to keep baseline predictions low (not-sponsor default)
    this._b3[0] = -1.6;
  }
}
