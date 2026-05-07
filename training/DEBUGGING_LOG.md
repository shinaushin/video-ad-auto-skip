# Teacher Model Training — Debugging Log

A running record of training attempts, observed symptoms, root-cause analyses, and the fix applied each time.

---

## Attempt 1 — Baseline (CPU run, no GPU)

**Config:** `focal_alpha=0.5`, `focal_gamma=2.0`, `patience=7`, `epochs=40`, `seq_batch_size=8`, `lr=1e-3`, `dropout=0.3`

**Symptom:**
- F1 = 0.0 from epoch 2 onward; TP = 0, FP = 0
- Epoch 1 had TP=8, FP=99, then complete collapse
- Loss decreased steadily (0.042 → 0.033) despite zero detections
- Stopped at epoch 10 (best epoch 3 + patience 7)

**Root cause:**
Early stopping used `val_loss` as the monitored metric. BCE loss decreases when the model predicts *all non-sponsor* with increasing confidence — the model found a local minimum by saying "never" for everything. A ~93% accuracy base rate (class imbalance) makes this a valid loss-minimizing strategy. The `val_loss` signal was therefore misleading: it improved while the model learned nothing useful.

Secondary: only 84 training videos were used (see Dataset Issues below).

**Fix applied:**
- Changed early stopping from `val_loss → val_f1` (a model that predicts all-zero cannot game F1)
- Increased `patience` from 7 → 15 and `epochs` from 40 → 60
- Increased `focal_alpha` from 0.5 → 0.75 (more aggressive positive weighting)

---

## Attempt 2 — Early stopping on val_f1, alpha=0.75

**Config:** `focal_alpha=0.75`, `focal_gamma=2.0`, `patience=15`, `epochs=60`, `lr=1e-3`, `dropout=0.3`

**Symptom:**
- F1 = 0.0 for all 60 epochs; TP = 0 across the board
- Training and validation accuracy identical every epoch from epoch 2 onwards (0.9375 / 0.9455)
- Loss continued to slowly decrease toward ~0.026

**Root cause:**
`FocalLoss(alpha=0.75)` only gives a 3× relative weighting of the positive class (0.75 / 0.25 = 3). The actual class ratio is ~15:1 (16,447 negatives vs 1,095 positives across training videos). A 3× bonus is insufficient — the aggregate loss from 16,447 easy negatives still outweighs the signal from 1,095 positives, so the model converges to "predict nothing."

Additionally, `dropout=0.3` is too aggressive for a dataset of only 84 training videos, adding gradient noise that prevents the model from finding a stable positive-prediction policy.

The model architecture (BiLSTM + CrossAttention, ~2.5M params) is significantly over-parameterised relative to 84 training videos.

**Fix applied:**
- Replaced `FocalLoss` with `nn.BCEWithLogitsLoss(pos_weight=n_neg/n_pos)` — `pos_weight` is computed automatically from training data statistics and directly scales the positive-class gradient by the true imbalance ratio (~15). This is more reliable than tuning alpha manually.
- Reduced `dropout` 0.3 → 0.1 (less regularisation for small datasets)
- Reduced `lr` 1e-3 → 5e-4 (more stable gradient steps)
- Increased `epochs` 60 → 200 (room to converge once signal is found; early stopping limits waste)
- Added **probability distribution logging** per epoch: `mean_prob_pos` and `mean_prob_neg` are now written to `training_log.json` so future runs can diagnose partial learning even at F1=0 (e.g. if the model outputs 0.15 for sponsors and 0.05 for non-sponsors, it's learning but below the 0.5 threshold)

---

## Dataset Issues (parallel track)

### Problem 1 — Wrong source directory (860 files instead of 1,090)

**Symptom:** Upload command reported 860 `.npz` files; local master cache had 1,090.

**Root cause:** `--upload-cache-only` auto-search candidates did not include `_MASTER_CACHE_DIR` (`training/cache/embeddings/`), so it picked an older output directory with a subset of files.

**Fix:** Added `_MASTER_CACHE_DIR` as the first candidate in the auto-search list in `kaggle_bridge.py`.

---

### Problem 2 — Kaggle dataset only had 121 of 1,090 files

**Symptom:** Training used only 121 videos (84 train / 18 val / 19 test) despite 1,090 being uploaded. Kaggle polling showed "21/1090 files visible" for 10 minutes before giving up.

**Root cause:** Kaggle's per-version file-count limit (~100–200 files) silently truncates uploads when many small files are pushed individually. The CLI returns success but Kaggle only processes the first ~120 files.

**Fix:** Changed `_upload_dataset()` to bundle all `.npz` files into a single `data.zip` before uploading (one file always passes the limit). Updated the kernel mount cell to unzip `data.zip` on the Kaggle side, with a fallback to loose-file scan for backward compatibility.

---

### Problem 3 — Kaggle dataset not mounted (first teacher run)

**Symptom:** Kernel failed at cell 4: `RuntimeError: Dataset not mounted: /kaggle/input/yt-sponsor-embeddings-cache`.

**Root cause:** Kaggle changed the mount path layout — datasets are now mounted under `/kaggle/input/datasets/` rather than directly at `/kaggle/input/<slug>/`.

**Fix:** Rewrote the mount cell to scan all of `/kaggle/input` recursively (`rglob("*.npz")`) rather than assuming a fixed path. Also added a diagnostic that lists all entries under `/kaggle/input/` on failure.

---

---

## Dataset Issue — data.zip approach abandoned; switched to 100-file shards

**Symptom:** After switching to zip upload, kernel still showed "No data.zip found — falling back to loose file scan, Found 121 loose files."

**Root cause:** Kaggle never processed the uploaded `data.zip` — it remained invisible in dataset file listings. The zip approach assumed Kaggle would extract it; Kaggle instead expects loose files up to its per-version limit.

**Fix:** Abandoned zip; switched to `_SHARD_SIZE = 100` shards. Each shard is a separate Kaggle Dataset (`yt-sponsor-embeddings-cache`, `yt-sponsor-embeddings-cache-s1`, …`-s10`). The kernel mount cell scans all shards via `rglob("*.npz")`. All 11 shards confirmed visible at upload. Mount cell updated to fall back to loose file scan (skipping `data.zip` check which is no longer used).

---

## Attempt 3 — BCEWithLogitsLoss (intended), ran old FocalLoss code again

**Config:** same as attempt 2 (60 epochs, focal_alpha=0.75) — bridge bundled tarball before code changes were saved

**Symptom:**
- Results byte-for-byte identical to attempt 2 (train_loss epoch 1 = 0.03729, same TP/FP)
- No `mean_prob_pos` / `mean_prob_neg` keys in training_log.json (new logging code absent)
- Ran all 60 epochs despite patience=15 and val_f1=0 throughout

**Root cause (two bugs):**
1. User ran the bridge while code edits were still in progress — tarball bundled old `train.py` with FocalLoss, not the new BCEWithLogitsLoss version
2. Early stopping tiebreak bug: when `val_f1 == best_val_f1` (both 0), code fell back to `val_loss < best_val_loss` as the improvement condition. Since val_loss slowly decreases every epoch (model becomes more confident at predicting all-negative), `improved=True` every epoch, resetting patience forever. Model never stopped early.

**Fix applied:**
- Removed val_loss tiebreak from early stopping: `improved = val_f1 > best_val_f1` only
- Confirmed new `train.py` code (BCEWithLogitsLoss, prob logging) is in place
- Re-uploading dataset as zip before next run

---

---

## Attempt 4 — NameError: focal_alpha not defined (stale log.info line)

**Config:** BCEWithLogitsLoss intended, `epochs=200`, `patience=15`, all shard datasets mounted

**Symptom:**
- Kernel errored with `NameError: name 'focal_alpha' is not defined` before training started
- Bridge downloaded outputs — `training_log.json` was timestamped 04:18 (from attempt 3)
- Epoch 60/60 visible in downloaded log with old format (`acc=0.938`, no `mean_prob_pos` key)
- Bridge returned these stale files as if they were fresh (no freshness check for non-data phases)

**Root cause:**
After migrating from FocalLoss to BCEWithLogitsLoss, one `log.info` call still referenced the old variables:
```python
# Broken — focal_alpha and focal_gamma no longer exist:
log.info("Starting teacher training: %d epochs, %d videos/batch, focal(α=%.2f γ=%.1f)",
         epochs, seq_batch, focal_alpha, focal_gamma)
```
The kernel raised `NameError` at startup, before the training loop even began.

Secondary: phases 2-5 had no run_id freshness check. When a kernel fails, Kaggle's
`kaggle kernels output` returns the previous successful run's output files. The bridge
accepted these stale files without detecting the mismatch.

**Fix applied:**
- Updated the log.info to use `pos_weight_val`:
  ```python
  log.info("Starting teacher training: %d epochs, %d videos/batch, pos_weight=%.1f",
           epochs, seq_batch, pos_weight_val)
  ```
- Added run_id freshness verification to all phases in `kaggle_bridge.py`:
  - `run_id` generated for every phase at bridge startup
  - `_make_run_manifest_cell(run_id, phase)` appended as the LAST notebook cell (only runs on success)
  - After downloading outputs for phases 2-5: `_verify_fresh_outputs(files, run_id)` checks
    that `run_manifest.json` exists with matching run_id
  - Missing or mismatched → `RuntimeError` instead of silently accepting stale files

---

## Attempt 5 — BCEWithLogitsLoss, shard dataset, all fixes applied

**Config:** `BCEWithLogitsLoss(pos_weight=~15)`, `epochs=200`, `patience=15`, `dropout=0.1`, `lr=5e-4`, `weight_decay=1e-4`, `seq_batch_size=8`

**Symptom (partial — run in progress):**
- Epoch 14: val_f1=0.238, best_val_f1=0.257 (peaked around epoch 5)
- `mean_prob_pos=0.452` vs `mean_prob_neg=0.276` — clear signal separation ✓
- recall=0.517, precision=0.155 — over-predicts positives (expected with pos_weight=15)
- Patience counter: 9/15 at epoch 14

**Assessment:**
This is the first run where the model learned anything (F1 > 0). BCEWithLogitsLoss fix worked.
The high recall / low precision suggests pos_weight=15 may be slightly too aggressive.
A decision threshold of ~0.35 instead of 0.5 may improve F1 on this checkpoint.

**Next step:** Run Optuna tune phase to find better hyperparameters for a full re-run.

---

## What to watch in the next run

The new `training_log.json` includes `mean_prob_pos` and `mean_prob_neg` for every epoch. Interpret as follows:

| Observation | Conclusion |
|---|---|
| `mean_prob_pos ≈ mean_prob_neg` (both ~0.05) | Model is not learning; embeddings may not contain useful signal |
| `mean_prob_pos > mean_prob_neg` but both < 0.5 | Model IS learning but threshold 0.5 is too high; tune threshold downward |
| F1 > 0 and rising | Model is working; monitor for overfitting on val set |

If `mean_prob_pos ≈ mean_prob_neg` persists after this run, the next steps would be:
1. Collect more data (target 1,000+ videos on Kaggle)
2. Verify embedding quality with a simple logistic regression baseline
3. Consider a window-level (non-sequence) training approach to rule out BiLSTM issues
