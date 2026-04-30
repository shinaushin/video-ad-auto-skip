"""export_onnx.py — Export the trained StudentModel to ONNX for browser inference.

The exported model has two named inputs to match the extension's two-branch architecture:
    text_input   float32 [batch, 64]          keyword indicator vector
    audio_input  float32 [batch, N_FRAMES, 13] MFCC frame buffer

And one named output:
    output       float32 [batch, 1]            sponsor confidence (post-sigmoid)

The ``audio_input`` N_FRAMES axis is exported as a **dynamic axis** so the runtime
can accept any sequence length; the extension always sends N_FRAMES=30 but the ONNX
graph remains flexible for evaluation scripts that vary the window.

Usage:
    python export_onnx.py --checkpoint outputs/student_best.pt --out outputs/model.onnx

    # Validate against onnxruntime after export:
    python export_onnx.py --checkpoint outputs/student_best.pt --out outputs/model.onnx --validate
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

from models import N_FRAMES, MFCC_DIM, TEXT_DIM, StudentModel, load_student

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def export(
    checkpoint_path: str | None,
    out_path: Path,
    n_frames: int = N_FRAMES,
    opset: int = 17,
) -> None:
    """Export the StudentModel to ONNX.

    Args:
        checkpoint_path: Path to a student_best.pt checkpoint, or None to use
                         random weights (useful for testing the export pipeline).
        out_path:        Destination .onnx file path.
        n_frames:        Default number of MFCC frames for the dummy input.
        opset:           ONNX opset version (17 is broadly supported by ort.js 1.17+).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    device = "cpu"

    model = load_student(checkpoint_path, device=device)
    model.eval()

    # Dummy inputs — batch size 1.
    dummy_text = torch.zeros(1, TEXT_DIM, dtype=torch.float32)    # [1, 64]
    dummy_audio = torch.zeros(1, n_frames, MFCC_DIM, dtype=torch.float32)  # [1, 30, 13]

    # Verify the model runs on the dummy input.
    with torch.no_grad():
        dummy_out = model(dummy_text, dummy_audio)
    assert dummy_out.shape == (1, 1), f"Expected [1,1] got {dummy_out.shape}"
    log.info("Model forward pass OK  (dummy output: %.4f)", float(dummy_out[0, 0]))

    torch.onnx.export(
        model,
        (dummy_text, dummy_audio),
        str(out_path),
        input_names=["text_input", "audio_input"],
        output_names=["output"],
        dynamic_axes={
            "text_input":  {0: "batch"},
            "audio_input": {0: "batch", 1: "n_frames"},
            "output":      {0: "batch"},
        },
        opset_version=opset,
        do_constant_folding=True,
    )
    size_kb = out_path.stat().st_size // 1024
    log.info("Exported → %s  (%d KB)", out_path, size_kb)


# ---------------------------------------------------------------------------
# Validation with onnxruntime
# ---------------------------------------------------------------------------


def validate(out_path: Path, n_frames: int = N_FRAMES) -> bool:
    """Run the exported model through onnxruntime and check shapes/values.

    Returns True if validation passes.
    """
    try:
        import onnx
        import onnxruntime as ort
    except ImportError:
        log.warning("onnx / onnxruntime not installed — skipping validation.")
        log.warning("Install with: pip install onnx onnxruntime")
        return True  # not a hard failure

    log.info("Validating with onnxruntime…")

    # Model check.
    model_proto = onnx.load(str(out_path))
    onnx.checker.check_model(model_proto)
    log.info("  ONNX checker: passed")

    # Runtime inference.
    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    inputs = sess.get_inputs()
    outputs = sess.get_outputs()
    log.info("  Inputs:  %s", [(i.name, i.shape, i.type) for i in inputs])
    log.info("  Outputs: %s", [(o.name, o.shape, o.type) for o in outputs])

    # Test 1: batch=1, n_frames=N_FRAMES (normal extension call).
    text_in = np.zeros((1, TEXT_DIM), dtype=np.float32)
    audio_in = np.zeros((1, n_frames, MFCC_DIM), dtype=np.float32)
    result = sess.run(None, {"text_input": text_in, "audio_input": audio_in})
    assert result[0].shape == (1, 1), f"Output shape mismatch: {result[0].shape}"
    score = float(result[0][0, 0])
    assert 0.0 <= score <= 1.0, f"Output out of [0,1]: {score}"
    log.info("  Test 1 (batch=1, n_frames=%d): output=%.4f  ✓", n_frames, score)

    # Test 2: batch=4 (bulk evaluation).
    text_in4 = np.zeros((4, TEXT_DIM), dtype=np.float32)
    audio_in4 = np.zeros((4, n_frames, MFCC_DIM), dtype=np.float32)
    result4 = sess.run(None, {"text_input": text_in4, "audio_input": audio_in4})
    assert result4[0].shape == (4, 1), f"Batch output shape mismatch: {result4[0].shape}"
    log.info("  Test 2 (batch=4): output shape=%s  ✓", result4[0].shape)

    # Test 3: dynamic n_frames (shorter buffer = 10 frames).
    audio_short = np.zeros((1, 10, MFCC_DIM), dtype=np.float32)
    result_short = sess.run(None, {"text_input": text_in, "audio_input": audio_short})
    assert result_short[0].shape == (1, 1)
    log.info("  Test 3 (n_frames=10): output=%.4f  ✓", float(result_short[0][0, 0]))

    # Test 4: strong sponsor signal — keyword "sponsored by" fires.
    text_sponsor = np.zeros((1, TEXT_DIM), dtype=np.float32)
    text_sponsor[0, 0] = 1.0   # pattern 0 = "sponsored by" (group 0)
    text_sponsor[0, 16] = 1.0  # pattern 16 = "use code" (group 1)
    result_sponsor = sess.run(None, {"text_input": text_sponsor, "audio_input": audio_in})
    score_sponsor = float(result_sponsor[0][0, 0])
    log.info("  Test 4 (sponsor keywords): output=%.4f", score_sponsor)

    log.info("Validation passed.")
    return True


# ---------------------------------------------------------------------------
# Manifest helper: write a companion JSON for the extension to read
# ---------------------------------------------------------------------------


def write_model_manifest(out_path: Path, n_frames: int = N_FRAMES) -> None:
    """Write model_manifest.json alongside model.onnx.

    The extension reads this to know how many MFCC frames to buffer and
    which input names to use for the ONNX session.
    """
    manifest = {
        "text_input_name": "text_input",
        "audio_input_name": "audio_input",
        "output_name": "output",
        "text_dim": TEXT_DIM,
        "mfcc_dim": MFCC_DIM,
        "n_frames": n_frames,
        "output_sigmoid": True,  # model already applies sigmoid internally
    }
    manifest_path = out_path.parent / "model_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    log.info("Manifest written → %s", manifest_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    p = argparse.ArgumentParser(
        description="Export StudentModel to ONNX for ONNX Runtime Web inference."
    )
    p.add_argument(
        "--checkpoint",
        default=None,
        type=str,
        help="Path to student_best.pt checkpoint.  Omit to export random weights (test mode).",
    )
    p.add_argument(
        "--out",
        default="outputs/model.onnx",
        type=Path,
        help="Destination .onnx file path.",
    )
    p.add_argument(
        "--n-frames",
        type=int,
        default=N_FRAMES,
        help=f"Number of MFCC frames in the dummy input (default: {N_FRAMES}).",
    )
    p.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version (default: 17).",
    )
    p.add_argument(
        "--validate",
        action="store_true",
        help="Validate the exported model with onnxruntime after export.",
    )
    p.add_argument(
        "--no-manifest",
        action="store_true",
        help="Skip writing model_manifest.json.",
    )
    args = p.parse_args()

    export(
        checkpoint_path=args.checkpoint,
        out_path=args.out,
        n_frames=args.n_frames,
        opset=args.opset,
    )

    if not args.no_manifest:
        write_model_manifest(args.out, n_frames=args.n_frames)

    if args.validate:
        ok = validate(args.out, n_frames=args.n_frames)
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
