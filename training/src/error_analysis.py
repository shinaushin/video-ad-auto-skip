"""error_analysis.py — Post-hoc error analysis for the teacher model.

Loads a trained teacher checkpoint, runs inference over the val (or test) split,
and produces a detailed breakdown of false positives and false negatives.

Outputs:
    error_report.txt     Human-readable summary
    errors.csv           One row per window: video_id, window_idx, label, prob,
                         pred, error_type, position, keyword_score, n_keywords
    fp_videos.csv        Aggregated false-positive rate per video
    fn_videos.csv        Aggregated false-negative rate per video

Usage:
    python error_analysis.py \\
        --checkpoint /tmp/outputs/teacher/teacher_best.pt \\
        --cache-dir  /tmp/embeddings_cache \\
        --out-dir    /tmp/outputs/error_analysis

    # Use the test split instead of val:
    python error_analysis.py --checkpoint ... --split test

    # Lower the decision threshold (default 0.5):
    python error_analysis.py --checkpoint ... --threshold 0.4
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data_pipeline import SponsorDataset, keyword_vector
from models import TeacherModel, WHISPER_DIM, DISTILBERT_DIM
from train import TeacherSequenceDataset, collate_teacher_sequences

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_inference(
    model: TeacherModel,
    dataset: TeacherSequenceDataset,
    device: str,
    batch_size: int = 8,
) -> list[dict]:
    """Run the teacher model over every window in `dataset`.

    Returns a list of dicts, one per window:
        video_id, window_idx, label, logit, prob, text (caption snippet)
    """
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_teacher_sequences,
        num_workers=0,
    )
    model.eval()

    results = []
    for batch in loader:
        audio_embs  = batch["audio_embs"].to(device)   # [B, seq, WHISPER_DIM]
        text_embs   = batch["text_embs"].to(device)    # [B, seq, DISTILBERT_DIM]
        labels      = batch["labels"]                  # [B, seq]
        lengths     = batch["lengths"]                 # [B]
        video_ids   = batch["video_ids"]               # list[str]
        window_idxs = batch["window_idxs"]             # list[list[int]]
        texts       = batch.get("texts", [[""] * int(l) for l in lengths])

        logits = model(audio_embs, text_embs, lengths).squeeze(-1)  # [B, seq]
        probs  = torch.sigmoid(logits).cpu().numpy()
        labels_np = labels.numpy()

        for b in range(len(video_ids)):
            vid = video_ids[b]
            seq_len = int(lengths[b])
            for t in range(seq_len):
                results.append({
                    "video_id":   vid,
                    "window_idx": window_idxs[b][t] if window_idxs else t,
                    "label":      int(labels_np[b, t]),
                    "logit":      float(logits[b, t].item()),
                    "prob":       float(probs[b, t]),
                    "text":       texts[b][t] if texts else "",
                })
    return results


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def classify_errors(results: list[dict], threshold: float) -> list[dict]:
    """Add 'pred' and 'error_type' fields to each result dict."""
    for r in results:
        r["pred"] = int(r["prob"] >= threshold)
        if r["label"] == 1 and r["pred"] == 0:
            r["error_type"] = "FN"
        elif r["label"] == 0 and r["pred"] == 1:
            r["error_type"] = "FP"
        elif r["label"] == 1:
            r["error_type"] = "TP"
        else:
            r["error_type"] = "TN"
    return results


def add_keyword_features(results: list[dict]) -> list[dict]:
    """Add keyword_score and n_keywords to each result using the cached text."""
    for r in results:
        text = r.get("text", "")
        if text:
            kw = keyword_vector(text)
            r["n_keywords"]    = int(kw.sum())
            r["keyword_score"] = float(
                sum(kw[i] * w for i, w in enumerate(
                    [3.0] * 16 + [1.5] * 32 + [0.5] * 16 +
                    [2.0] * 16 + [1.0] * 32 + [0.5] * 16
                ))
            )
        else:
            r["n_keywords"]    = 0
            r["keyword_score"] = 0.0
    return results


def add_position(results: list[dict]) -> list[dict]:
    """Add normalised window position [0, 1] within each video."""
    # Group by video to find total windows
    vid_windows: dict[str, list[int]] = defaultdict(list)
    for r in results:
        vid_windows[r["video_id"]].append(r["window_idx"])

    vid_total = {v: max(idxs) + 1 for v, idxs in vid_windows.items()}

    for r in results:
        total = vid_total[r["video_id"]]
        r["position"] = r["window_idx"] / max(total - 1, 1)
    return results


def compute_metrics(results: list[dict]) -> dict:
    tp = sum(1 for r in results if r["error_type"] == "TP")
    tn = sum(1 for r in results if r["error_type"] == "TN")
    fp = sum(1 for r in results if r["error_type"] == "FP")
    fn = sum(1 for r in results if r["error_type"] == "FN")

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy  = (tp + tn) / len(results) if results else 0.0

    return dict(tp=tp, tn=tn, fp=fp, fn=fn,
                precision=precision, recall=recall, f1=f1, accuracy=accuracy)


def score_distribution(values: list[float], bins: int = 10) -> str:
    """Return a compact ASCII histogram string."""
    if not values:
        return "(no data)"
    lo, hi = 0.0, 1.0
    width = (hi - lo) / bins
    counts = [0] * bins
    for v in values:
        idx = min(int((v - lo) / width), bins - 1)
        counts[idx] += 1
    max_c = max(counts) or 1
    bar_width = 20
    lines = []
    for i, c in enumerate(counts):
        lo_b = lo + i * width
        bar  = "█" * int(c / max_c * bar_width)
        lines.append(f"  [{lo_b:.1f}-{lo_b + width:.1f}) {bar:<{bar_width}} {c}")
    return "\n".join(lines)


def position_buckets(results: list[dict], error_type: str, n_buckets: int = 5) -> str:
    """Show where in a video errors cluster (early / middle / late)."""
    bucket_labels = ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]
    counts = [0] * n_buckets
    subset = [r for r in results if r["error_type"] == error_type]
    for r in subset:
        idx = min(int(r["position"] * n_buckets), n_buckets - 1)
        counts[idx] += 1
    total = sum(counts) or 1
    parts = [f"{lbl}: {c} ({100*c/total:.0f}%)" for lbl, c in zip(bucket_labels, counts)]
    return "  " + "  |  ".join(parts)


def top_error_videos(results: list[dict], error_type: str, n: int = 15) -> list[tuple]:
    """Return the n videos with the most errors of the given type."""
    counts: dict[str, int] = defaultdict(int)
    totals: dict[str, int] = defaultdict(int)
    for r in results:
        totals[r["video_id"]] += 1
        if r["error_type"] == error_type:
            counts[r["video_id"]] += 1
    ranked = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:n]
    return [(vid, cnt, totals[vid]) for vid, cnt in ranked]


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------

def write_report(results: list[dict], out_dir: Path, threshold: float) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    m = compute_metrics(results)

    fps = [r for r in results if r["error_type"] == "FP"]
    fns = [r for r in results if r["error_type"] == "FN"]
    tps = [r for r in results if r["error_type"] == "TP"]
    tns = [r for r in results if r["error_type"] == "TN"]

    report_lines = [
        "=" * 70,
        "  TEACHER MODEL ERROR ANALYSIS",
        "=" * 70,
        "",
        f"Decision threshold : {threshold:.2f}",
        f"Total windows      : {len(results):,}",
        f"  Positives (label=1) : {m['tp'] + m['fn']:,}",
        f"  Negatives (label=0) : {m['tn'] + m['fp']:,}",
        "",
        "── Overall Metrics ─────────────────────────────────────────────────",
        f"  Precision : {m['precision']:.4f}",
        f"  Recall    : {m['recall']:.4f}",
        f"  F1        : {m['f1']:.4f}",
        f"  Accuracy  : {m['accuracy']:.4f}",
        f"  TP={m['tp']:,}  TN={m['tn']:,}  FP={m['fp']:,}  FN={m['fn']:,}",
        "",
        "── Score Distributions ─────────────────────────────────────────────",
        "",
        "  True Positives (sponsor, predicted sponsor):",
        score_distribution([r["prob"] for r in tps]),
        "",
        "  False Negatives (sponsor, missed):",
        score_distribution([r["prob"] for r in fns]),
        "",
        "  False Positives (not sponsor, wrongly flagged):",
        score_distribution([r["prob"] for r in fps]),
        "",
        "  True Negatives (not sponsor, correctly ignored):",
        score_distribution([r["prob"] for r in tns]),
        "",
        "── Keyword Signal in Errors ────────────────────────────────────────",
        f"  FP avg keyword_score : {np.mean([r['keyword_score'] for r in fps]):.2f}  "
            f"avg n_keywords : {np.mean([r['n_keywords'] for r in fps]):.2f}"
            if fps else "  (no FPs)",
        f"  FN avg keyword_score : {np.mean([r['keyword_score'] for r in fns]):.2f}  "
            f"avg n_keywords : {np.mean([r['n_keywords'] for r in fns]):.2f}"
            if fns else "  (no FNs)",
        f"  TP avg keyword_score : {np.mean([r['keyword_score'] for r in tps]):.2f}  "
            f"avg n_keywords : {np.mean([r['n_keywords'] for r in tps]):.2f}"
            if tps else "  (no TPs)",
        "",
        "  FPs with zero keyword signal (pure audio FPs):",
        f"    {sum(1 for r in fps if r['n_keywords'] == 0)} / {len(fps)} "
            f"({100 * sum(1 for r in fps if r['n_keywords'] == 0) / max(len(fps), 1):.0f}%)",
        "  FNs with strong keyword signal (text present but still missed):",
        f"    {sum(1 for r in fns if r['keyword_score'] >= 3.0)} / {len(fns)} "
            f"({100 * sum(1 for r in fns if r['keyword_score'] >= 3.0) / max(len(fns), 1):.0f}%)",
        "",
        "── Position in Video ───────────────────────────────────────────────",
        "  False Positives by video position:",
        position_buckets(results, "FP"),
        "  False Negatives by video position:",
        position_buckets(results, "FN"),
        "  True Positives by video position:",
        position_buckets(results, "TP"),
        "",
        "── Top Videos by False Positive Count ──────────────────────────────",
        "  (videos most likely to be speed-skipped incorrectly)",
    ]
    for vid, err_cnt, total in top_error_videos(results, "FP"):
        report_lines.append(f"    {vid}  {err_cnt} FPs / {total} windows  "
                            f"({100*err_cnt/total:.0f}%)")

    report_lines += [
        "",
        "── Top Videos by False Negative Count ──────────────────────────────",
        "  (videos where sponsor segments were missed most often)",
    ]
    for vid, err_cnt, total in top_error_videos(results, "FN"):
        report_lines.append(f"    {vid}  {err_cnt} FNs / {total} windows  "
                            f"({100*err_cnt/total:.0f}%)")

    # Confidence analysis — how confident are we when we're wrong?
    report_lines += [
        "",
        "── Confidence of Wrong Predictions ─────────────────────────────────",
        f"  FP mean prob : {np.mean([r['prob'] for r in fps]):.3f}  "
            f"(higher = model very wrong)"
            if fps else "  (no FPs)",
        f"  FN mean prob : {np.mean([r['prob'] for r in fns]):.3f}  "
            f"(lower = model missed badly)"
            if fns else "  (no FNs)",
        f"  High-confidence FPs (prob > 0.8) : {sum(1 for r in fps if r['prob'] > 0.8)}",
        f"  Low-confidence FNs  (prob < 0.2) : {sum(1 for r in fns if r['prob'] < 0.2)}",
        "",
        "── Threshold Sensitivity ───────────────────────────────────────────",
    ]
    for t in [0.3, 0.4, 0.5, 0.6, 0.7]:
        classified = classify_errors(
            [{**r} for r in results], threshold=t
        )
        mt = compute_metrics(classified)
        report_lines.append(
            f"  thresh={t:.1f}  P={mt['precision']:.3f}  R={mt['recall']:.3f}  "
            f"F1={mt['f1']:.3f}  FP={mt['fp']:,}  FN={mt['fn']:,}"
        )

    report_lines += ["", "=" * 70]

    report_path = out_dir / "error_report.txt"
    report_path.write_text("\n".join(report_lines))
    log.info("Report written → %s", report_path)
    print("\n".join(report_lines))


def write_csvs(results: list[dict], out_dir: Path) -> None:
    """Write errors.csv, fp_videos.csv, fn_videos.csv."""
    # errors.csv — every window
    errors_path = out_dir / "errors.csv"
    fieldnames = ["video_id", "window_idx", "label", "prob", "pred",
                  "error_type", "position", "keyword_score", "n_keywords", "text"]
    with errors_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        # Only write errors + a sample of TNs (to keep file manageable)
        errors_only = [r for r in results if r["error_type"] in ("FP", "FN", "TP")]
        # Sample TNs (up to 5000)
        tns = [r for r in results if r["error_type"] == "TN"]
        import random; random.shuffle(tns)
        w.writerows(errors_only + tns[:5000])
    log.info("errors.csv → %s (%d rows)", errors_path, len(errors_only) + min(len(tns), 5000))

    # Per-video aggregates
    for error_type, fname in [("FP", "fp_videos.csv"), ("FN", "fn_videos.csv")]:
        agg: dict[str, dict] = defaultdict(lambda: {"errors": 0, "total": 0, "prob_sum": 0.0})
        for r in results:
            agg[r["video_id"]]["total"] += 1
            if r["error_type"] == error_type:
                agg[r["video_id"]]["errors"] += 1
                agg[r["video_id"]]["prob_sum"] += r["prob"]

        rows = [
            {
                "video_id": vid,
                "n_errors": d["errors"],
                "n_windows": d["total"],
                "error_rate": d["errors"] / d["total"],
                "mean_prob": d["prob_sum"] / d["errors"] if d["errors"] > 0 else 0.0,
            }
            for vid, d in agg.items() if d["errors"] > 0
        ]
        rows.sort(key=lambda x: x["n_errors"], reverse=True)
        csv_path = out_dir / fname
        with csv_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["video_id", "n_errors", "n_windows",
                                               "error_rate", "mean_prob"])
            w.writeheader()
            w.writerows(rows)
        log.info("%s → %s (%d videos)", fname, csv_path, len(rows))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    p = argparse.ArgumentParser(description="Error analysis for the teacher model.")
    p.add_argument("--checkpoint", required=True,
                   help="Path to teacher_best.pt checkpoint.")
    p.add_argument("--cache-dir", default="/tmp/embeddings_cache",
                   help="Directory of per-video .npz cache files.")
    p.add_argument("--out-dir", default="/tmp/outputs/error_analysis",
                   type=Path, help="Where to write reports.")
    p.add_argument("--split", choices=["val", "test", "train"], default="val",
                   help="Which data split to analyse (default: val).")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="Decision threshold for P/R/F1 (default: 0.5).")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Device: %s", device)

    # ── Load data ─────────────────────────────────────────────────────────
    log.info("Loading dataset from %s …", args.cache_dir)
    full_ds = SponsorDataset(cache_dir=args.cache_dir)

    # Reproduce the train/val/test split from train.py (80/10/10, seed 42).
    rng = np.random.default_rng(args.seed)
    n = len(full_ds)
    idx = rng.permutation(n)
    n_train = int(0.8 * n)
    n_val   = int(0.1 * n)

    if args.split == "train":
        chosen = idx[:n_train]
    elif args.split == "val":
        chosen = idx[n_train:n_train + n_val]
    else:
        chosen = idx[n_train + n_val:]

    subset = [full_ds[i] for i in chosen]
    seq_ds = TeacherSequenceDataset(subset)
    log.info("Split '%s': %d videos, %d sequences",
             args.split, len(subset), len(seq_ds))

    # ── Load model ────────────────────────────────────────────────────────
    log.info("Loading checkpoint: %s", args.checkpoint)
    ckpt  = torch.load(args.checkpoint, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)

    # Infer lstm_layers from checkpoint keys.
    lstm_layers = max(
        int(k.split(".")[1]) + 1
        for k in state if k.startswith("lstm.weight_ih_l")
    ) if any(k.startswith("lstm.weight_ih_l") for k in state) else 2

    model = TeacherModel(lstm_hidden=256, lstm_layers=lstm_layers, dropout=0.0).to(device)
    model.load_state_dict(state, strict=False)
    model.eval()
    log.info("Model loaded (lstm_layers=%d).", lstm_layers)

    # ── Inference ─────────────────────────────────────────────────────────
    log.info("Running inference …")
    results = run_inference(model, seq_ds, device, batch_size=args.batch_size)
    log.info("Inference done: %d windows.", len(results))

    # ── Enrich and analyse ────────────────────────────────────────────────
    results = classify_errors(results, threshold=args.threshold)
    results = add_keyword_features(results)
    results = add_position(results)

    write_report(results, args.out_dir, threshold=args.threshold)
    write_csvs(results, args.out_dir)

    log.info("Done. Outputs in %s", args.out_dir)


if __name__ == "__main__":
    main()
