/**
 * ort-env.js — Configure ONNX Runtime Web environment before any session is created.
 *
 * Must be injected AFTER ort.min.js and BEFORE ml-detector.js.
 * Sets:
 *   - numThreads = 1  (threading requires cross-origin isolation which extensions lack)
 *   - wasmPaths       (point ORT at the extension's own wasm binaries, not youtube.com)
 */
if (typeof ort !== "undefined") {
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.wasmPaths = {
    "ort-wasm-simd.wasm": chrome.runtime.getURL("ort-wasm-simd.wasm"),
    "ort-wasm.wasm":      chrome.runtime.getURL("ort-wasm.wasm"),
  };
}
