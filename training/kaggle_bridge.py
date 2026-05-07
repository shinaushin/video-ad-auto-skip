r"""kaggle_bridge.py — Run sponsor-detection training phases on Kaggle's free GPU.

The bridge handles the full round-trip:
  1. Bundle training/src/ into a tarball (base64-embedded in the notebook).
  2. Embed the phase config into a generated Jupyter notebook.
  3. Push the kernel to Kaggle and wait for it to complete.
  4. Download outputs and route them to training/outputs/ locally.

Five phases — run them in order:

    Phase 1 — data
        Downloads the SponsorBlock CSV, runs data_pipeline.py to fetch yt-dlp
        audio + Whisper/DistilBERT embeddings for N videos, and uploads the
        resulting .npz cache as a Kaggle Dataset (reused by phases 2–5).

    Phase 2 — baseline
        Loads the embedding cache; evaluates the keyword heuristic against
        SponsorBlock ground truth and writes training_log.json.

    Phase 3 — teacher
        Trains TeacherModel (DistilBERT + Whisper + CrossAttention + BiLSTM).
        Outputs teacher_best.pt.

    Phase 4 — distill
        Loads the teacher checkpoint (uploaded as a Kaggle Dataset after phase 3),
        collects soft labels, trains StudentModel.  Outputs student_best.pt.

    Phase 5 — export
        Loads student_best.pt and exports model.onnx with two inputs
        (text_input [1,64] and audio_input [1,N_FRAMES,13]) for ONNX Runtime Web.

Authentication — Kaggle token system:
    a) Generate a token at https://www.kaggle.com/settings (API → Generate New Token).
    b) Export:
          export KAGGLE_API_TOKEN=<token>
          export KAGGLE_USERNAME=<your_username>
       or save the token string to ~/.kaggle/access_token and set KAGGLE_USERNAME.

Dependencies:
    pip install kaggle kagglehub nbformat

Usage:
    # Phase 1 — build embedding cache
    python training/kaggle_bridge.py --config training/configs/phase1_data.json --gpu

    # Phase 2 — baseline evaluation
    python training/kaggle_bridge.py --config training/configs/phase2_baseline.json --gpu

    # Phase 3 — teacher training
    python training/kaggle_bridge.py --config training/configs/phase3_teacher.json --gpu

    # Phase 4 — distillation (requires phase3 to have completed)
    python training/kaggle_bridge.py --config training/configs/phase4_distill.json --gpu

    # Phase 5 — ONNX export
    python training/kaggle_bridge.py --config training/configs/phase5_export.json

    # Dry-run any phase without consuming Kaggle quota
    python training/kaggle_bridge.py --config training/configs/phase3_teacher.json --dry-run
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Dependency checks
# ---------------------------------------------------------------------------

try:
    import nbformat
    from nbformat.v4 import new_code_cell, new_notebook
except ImportError:
    print("Missing: pip install nbformat")
    sys.exit(1)

try:
    import kagglehub  # noqa: F401
except ImportError:
    print("Missing: pip install kagglehub")
    sys.exit(1)

try:
    import kaggle  # noqa: F401
except ImportError:
    print("Missing: pip install kaggle")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_DEFAULT_USERNAME = "profileurlplz"

#: Kaggle Dataset slug for the per-video embedding cache (phases 1 → 2-5).
_EMBEDDINGS_DATASET_SLUG = "yt-sponsor-embeddings-cache"

#: Kaggle Dataset slug for teacher checkpoint (phase 3 → 4).
_TEACHER_CKPT_DATASET_SLUG = "yt-sponsor-teacher-checkpoint"

POLL_INTERVAL_SEC = 30
MAX_WAIT_SEC = 14400  # 4 hours — covers 500-video batches + accumulate seed copy

_TRAINING_ROOT = Path(__file__).resolve().parent          # training/
_PROJECT_ROOT = _TRAINING_ROOT.parent                     # repo root

#: Persistent local directory that accumulates all .npz files across every batch run.
_MASTER_CACHE_DIR = _TRAINING_ROOT / "cache" / "embeddings"

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _log(msg: str) -> None:
    print(f"[bridge {datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def _merge_into_master_cache(source_dir: Path) -> int:
    """Copy any new .npz files from source_dir into the persistent master cache.

    Returns the new master cache total.
    """
    _MASTER_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    added = 0
    for f in source_dir.glob("*.npz"):
        dst = _MASTER_CACHE_DIR / f.name
        if not dst.exists():
            import shutil as _shutil
            _shutil.copy2(f, dst)
            added += 1
    total = sum(1 for _ in _MASTER_CACHE_DIR.glob("*.npz"))
    if added:
        _log(f"Master cache: +{added} new files → {total} total")
    return total


def _print_cache_total(kaggle_total: int | None = None, label: str = "Cache total") -> None:
    """Print a banner showing the current cache totals.

    kaggle_total: count reported by the in-kernel summary JSON (reflects Kaggle dataset).
    Falls back to the master cache dir count if not provided.
    """
    local_count = sum(1 for _ in _MASTER_CACHE_DIR.glob("*.npz")) if _MASTER_CACHE_DIR.is_dir() else 0
    _log("=" * 50)
    if kaggle_total is not None:
        _log(f"  {label}")
        _log(f"    Kaggle dataset : {kaggle_total} videos")
        _log(f"    Local master   : {local_count} videos")
    else:
        _log(f"  {label}: {local_count} videos (local master cache)")
    _log("=" * 50)


def _whoami(token: str) -> str | None:
    """Ask the Kaggle API who owns this token. Returns username or None on failure."""
    import urllib.request as _req
    import base64 as _b64
    # The /whoami endpoint accepts Basic auth with any username + token as password,
    # or Bearer token — try Bearer first (works with new access_token format).
    for auth_header in (
        f"Bearer {token}",
        "Basic " + _b64.b64encode(f":{token}".encode()).decode(),
    ):
        try:
            req = _req.Request(
                "https://www.kaggle.com/api/v1/whoami",
                headers={"Authorization": auth_header, "Accept": "application/json"},
            )
            with _req.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())
                username = data.get("username") or data.get("name") or data.get("displayName")
                if username:
                    return username.strip()
        except Exception:
            pass
    return None


def _kaggle_username() -> str:
    username = os.environ.get("KAGGLE_USERNAME", "").strip()
    # Fallback 1: read from kaggle.json (has username even when access_token is used for key)
    if not username or username == "your_kaggle_username":
        p_json = Path.home() / ".kaggle" / "kaggle.json"
        if p_json.exists():
            try:
                username = json.loads(p_json.read_text()).get("username", "").strip()
            except (json.JSONDecodeError, KeyError):
                pass
    # Fallback 2: auto-fetch from Kaggle API using the token
    if not username or username == "your_kaggle_username":
        token = os.environ.get("KAGGLE_API_TOKEN") or _read_token_file_raw()
        if token:
            _log("Resolving username via Kaggle API…")
            fetched = _whoami(token)
            if fetched:
                _log(f"  resolved username: '{fetched}'")
                username = fetched
    # Fallback 3: module constant
    if not username or username == "your_kaggle_username":
        username = _DEFAULT_USERNAME
    if not username or username == "your_kaggle_username":
        print("\nKaggle username not configured and /whoami lookup failed.")
        print("  export KAGGLE_USERNAME=your_username")
        print(f"  or edit _DEFAULT_USERNAME in {__file__}")
        sys.exit(1)
    return username


def _read_token_file_raw() -> str | None:
    """Read just the raw token bytes without side-effects (no env var mutation)."""
    p_new = Path.home() / ".kaggle" / "access_token"
    if p_new.exists():
        return p_new.read_text().strip()
    p_json = Path.home() / ".kaggle" / "kaggle.json"
    if p_json.exists():
        try:
            return json.loads(p_json.read_text()).get("key")
        except (json.JSONDecodeError, KeyError):
            pass
    return None


def _read_token_file() -> str | None:
    """Read API key from ~/.kaggle/access_token (new format) or kaggle.json (classic)."""
    # New-style single-token file
    p_new = Path.home() / ".kaggle" / "access_token"
    if p_new.exists():
        return p_new.read_text().strip()
    # Classic kaggle.json: {"username": "...", "key": "..."}
    p_json = Path.home() / ".kaggle" / "kaggle.json"
    if p_json.exists():
        try:
            creds = json.loads(p_json.read_text())
            # Also set username from the JSON so _kaggle_username() finds it.
            if "username" in creds and not os.environ.get("KAGGLE_USERNAME"):
                os.environ["KAGGLE_USERNAME"] = creds["username"]
            return creds.get("key")
        except (json.JSONDecodeError, KeyError):
            pass
    return None


def _kaggle_cmd(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["kaggle", *args], capture_output=True, text=True)


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------


def authenticate() -> None:
    """Bridge the new Kaggle token format to KAGGLE_USERNAME + KAGGLE_KEY."""
    token = os.environ.get("KAGGLE_API_TOKEN") or _read_token_file()
    if not token:
        print("\nNo Kaggle token found.")
        print("Generate one at https://www.kaggle.com/settings (API → Generate New Token)")
        print("Then: export KAGGLE_API_TOKEN=your_token")
        sys.exit(1)
    username = _kaggle_username()
    os.environ["KAGGLE_USERNAME"] = username
    os.environ["KAGGLE_KEY"] = token
    _log(f"Authenticated as '{username}'")


# ---------------------------------------------------------------------------
# Source bundling
# ---------------------------------------------------------------------------


def _bundle_source(dest_tar: Path) -> None:
    """Bundle training/src/ into a gzipped tarball for embedding in the notebook."""
    _log(f"Bundling source → {dest_tar.name}")
    src_dir = _TRAINING_ROOT / "src"
    with tarfile.open(dest_tar, "w:gz") as tf:
        tf.add(src_dir, arcname="src")
    _log(f"  source tarball: {dest_tar.stat().st_size:,} bytes")


# ---------------------------------------------------------------------------
# Dataset upload helpers
# ---------------------------------------------------------------------------


_SHARD_SIZE = 100   # files per shard — well below Kaggle's per-version limit


def _upload_dataset(
    data_dir: Path,
    username: str,
    slug: str,
    title: str,
) -> str:
    """Upload a local directory as one or more sharded Kaggle Datasets.

    Kaggle silently truncates dataset versions that contain more than ~100-200
    files.  We work around this by splitting the files into shards of
    _SHARD_SIZE each and uploading each shard to a separate dataset slug:
        <slug>          (shard 0: files 0..99)
        <slug>-s1       (shard 1: files 100..199)
        <slug>-s2       (shard 2: files 200..299)
        …

    The mount cell in the kernel combines all shards transparently.

    Returns the primary dataset reference ``"<username>/<slug>"``.
    """
    all_files = sorted(f for f in data_dir.iterdir() if f.is_file())
    n_files    = len(all_files)
    n_shards   = max(1, (n_files + _SHARD_SIZE - 1) // _SHARD_SIZE)

    _log(f"Uploading {n_files} files in {n_shards} shard(s) of ≤{_SHARD_SIZE} each")

    primary_ref = f"{username}/{slug}"
    shard_refs: list[str] = []

    for shard_idx in range(n_shards):
        shard_files = all_files[shard_idx * _SHARD_SIZE : (shard_idx + 1) * _SHARD_SIZE]
        shard_slug  = slug if shard_idx == 0 else f"{slug}-s{shard_idx}"
        shard_ref   = f"{username}/{shard_slug}"
        shard_title = title if shard_idx == 0 else f"{title} (shard {shard_idx})"
        shard_refs.append(shard_ref)

        _log(f"  Shard {shard_idx}: {len(shard_files)} files → {shard_ref}")

        with tempfile.TemporaryDirectory(prefix="kaggle_dataset_") as tmp:
            tmp_path = Path(tmp)
            meta = {"title": shard_title, "id": shard_ref, "licenses": [{"name": "CC0-1.0"}]}
            (tmp_path / "dataset-metadata.json").write_text(json.dumps(meta, indent=2))

            total_bytes = 0
            for f in shard_files:
                shutil.copy2(f, tmp_path / f.name)
                total_bytes += f.stat().st_size
            _log(f"    {len(shard_files)} files  ({total_bytes:,} bytes)")

            result = _kaggle_cmd("datasets", "create", "-p", str(tmp_path))
            if result.returncode != 0 and "already exists" in (result.stderr + result.stdout).lower():
                _log("    Already exists — pushing new version…")
                result = _kaggle_cmd(
                    "datasets", "version", "-p", str(tmp_path), "-m", "Updated by kaggle_bridge.py"
                )
            if result.returncode != 0:
                raise RuntimeError(
                    f"Shard {shard_idx} upload failed:\n{result.stderr or result.stdout}"
                )

    ref = primary_ref
    n_files_uploaded = n_files  # used below for the total summary

    # Poll each shard until at least 1 file is visible.
    _log("  Waiting for Kaggle to process shards…")
    for shard_ref in shard_refs:
        for attempt in range(30):  # up to 5 minutes per shard
            check = _kaggle_cmd("datasets", "files", shard_ref)
            file_lines = [
                l.strip() for l in check.stdout.splitlines()
                if l.strip()
                and not l.strip().startswith("name")
                and not l.strip().startswith("---")
                and not l.strip().startswith("Next Page")
            ]
            if check.returncode == 0 and file_lines:
                _log(f"  {shard_ref}: {len(file_lines)}+ files visible ✓")
                break
            _log(f"  {shard_ref}: waiting (attempt {attempt+1}/30)…")
            time.sleep(10)
        else:
            _log(f"  WARNING: {shard_ref} may not be ready — proceeding anyway.")

    _log(f"  All shards ready → https://www.kaggle.com/datasets/{primary_ref}")
    return primary_ref


def upload_embeddings_cache(cache_dir: Path, username: str) -> str:
    """Upload per-video .npz embedding cache as a Kaggle Dataset."""
    return _upload_dataset(
        data_dir=cache_dir,
        username=username,
        slug=_EMBEDDINGS_DATASET_SLUG,
        title="YT Sponsor Embeddings Cache",
    )


def upload_teacher_checkpoint(ckpt_dir: Path, username: str) -> str:
    """Upload teacher checkpoint directory as a Kaggle Dataset."""
    return _upload_dataset(
        data_dir=ckpt_dir,
        username=username,
        slug=_TEACHER_CKPT_DATASET_SLUG,
        title="YT Sponsor Teacher Checkpoint",
    )


# ---------------------------------------------------------------------------
# Notebook cell templates
# ---------------------------------------------------------------------------

_CELL_SETUP = """\
# Auto-generated by kaggle_bridge.py — do not edit by hand.
import os, sys, torch
print(f"Python {sys.version}")
print(f"PyTorch {torch.__version__}  CUDA: {torch.cuda.is_available()}")
print(f"Working dir: {os.getcwd()}")
"""

def _make_install_deps_cell(phase: str = "") -> str:
    """Return the pip-install cell, adding optuna for the tune phase."""
    extra = '    "optuna>=3.0",\n' if phase == "tune" else ""
    return (
        "import subprocess, sys, urllib.request\n\n"
        "try:\n"
        "    urllib.request.urlopen('https://pypi.org', timeout=5)\n"
        "except Exception as _e:\n"
        "    raise RuntimeError(\n"
        "        'No internet access. Verify your phone at https://www.kaggle.com/settings/account'\n"
        "    ) from _e\n\n"
        "_pkgs = [\n"
        '    "transformers>=4.36",\n'
        '    "yt-dlp>=2024.1.1",\n'
        '    "soundfile>=0.12",\n'
        + extra +
        "]\n"
        "for _p in _pkgs:\n"
        "    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', _p], check=True)\n"
        'print("Extra deps installed:", _pkgs)\n'
    )


def _make_install_src_cell(src_tarball: Path) -> str:
    """Embed training/src/ as base64 and install it in the kernel."""
    b64 = base64.b64encode(src_tarball.read_bytes()).decode("ascii")
    return (
        "import base64, importlib, shutil, subprocess, sys, tarfile, warnings\n"
        "from pathlib import Path\n\n"
        f"_B64 = {b64!r}\n"
        "_tar = Path('/tmp/yt_sponsor_src.tar.gz')\n"
        "_tar.write_bytes(base64.b64decode(_B64))\n"
        "_dst = Path('/tmp/yt_sponsor_src')\n"
        "if _dst.exists(): shutil.rmtree(_dst)\n"
        "_dst.mkdir(parents=True)\n"
        "with warnings.catch_warnings():\n"
        "    warnings.simplefilter('ignore')\n"
        "    with tarfile.open(_tar, 'r:gz') as _tf:\n"
        "        _tf.extractall(_dst, filter='data')\n\n"
        "# Add src/ to sys.path so modules are importable directly.\n"
        "sys.path.insert(0, str(_dst / 'src'))\n"
        "importlib.invalidate_caches()\n"
        "print('src/ added to sys.path')\n"
        "print('  contents:', [f.name for f in (_dst / 'src').iterdir()])\n"
    )


def _make_config_cell(config: dict) -> str:
    """Write the patched config to /tmp/run_config.json inside the kernel."""
    config_json = json.dumps(config, indent=2)
    return (
        "import json\n"
        "from pathlib import Path\n\n"
        f"_CONFIG = json.loads({config_json!r})\n"
        "_CONFIG_PATH = Path('/tmp/run_config.json')\n"
        "_CONFIG_PATH.write_text(json.dumps(_CONFIG))\n"
        "print(f'Config → {_CONFIG_PATH}')\n"
        "print(f\"  phase:  {_CONFIG['phase']}\")\n"
        "print(f\"  name:   {_CONFIG['name']}\")\n"
    )


# ---------------------------------------------------------------------------
# Phase-specific notebook cell generators
# ---------------------------------------------------------------------------


def _make_phase1_data_cell(config: dict, already_processed: list[str] | None = None,
                            run_id: str = "") -> str:
    """Phase 1: download SponsorBlock CSV and run data_pipeline.py.

    Already-processed video IDs are embedded directly in the notebook as a
    skip-list file, so the kernel never needs to mount a large dataset.
    The kernel only downloads and processes NEW videos, which are then
    downloaded back by the bridge and merged into the local master cache.
    """
    n_videos = config.get("n_videos", 500)
    skip_audio = config.get("skip_audio", False)
    skip_flag = "--skip-audio" if skip_audio else ""

    # Embed the skip-list as a newline-separated string literal in the cell.
    if already_processed:
        ids_literal = "\\n".join(already_processed)
        skip_ids_block = (
            "# Write the skip-list of already-processed video IDs.\n"
            "_SKIP_IDS_PATH = Path('/kaggle/working/skip_ids.txt')\n"
            f"_SKIP_IDS_PATH.write_text('{ids_literal}')\n"
            f"print(f'Skipping {len(already_processed)} already-processed videos.')\n\n"
        )
        skip_ids_arg = "        '--skip-ids', str(_SKIP_IDS_PATH),\n"
    else:
        skip_ids_block = ""
        skip_ids_arg = ""

    return (
        "import json, subprocess, sys, urllib.request\n"
        "from pathlib import Path\n\n"
        + skip_ids_block +
        "_CSV_MIRRORS = [\n"
        "    'https://sb.ltn.fi/database/sponsorTimes.csv',\n"
        "    'https://mirror.sb.mchang.xyz/sponsorTimes.csv',\n"
        "]\n"
        "_CSV_PATH = Path('/kaggle/working/sponsorTimes.csv')\n"
        "_CACHE_DIR = Path('/kaggle/working/embeddings_cache')\n"
        "_CACHE_DIR.mkdir(parents=True, exist_ok=True)\n\n"
        "if not _CSV_PATH.exists():\n"
        "    print('Downloading SponsorBlock CSV (~2-4 GB)…')\n"
        "    _downloaded = False\n"
        "    for _url in _CSV_MIRRORS:\n"
        "        try:\n"
        "            print(f'  Trying {_url}')\n"
        "            urllib.request.urlretrieve(_url, _CSV_PATH)\n"
        "            print(f'  Downloaded: {_CSV_PATH.stat().st_size:,} bytes')\n"
        "            _downloaded = True\n"
        "            break\n"
        "        except Exception as _e:\n"
        "            print(f'  Failed: {_e}')\n"
        "    if not _downloaded:\n"
        "        raise RuntimeError('All SponsorBlock mirrors failed — check mirror availability.')\n"
        "else:\n"
        "    print(f'CSV already present: {_CSV_PATH.stat().st_size:,} bytes')\n\n"
        "# Run data pipeline — only processes videos not in the skip list.\n"
        "result = subprocess.run(\n"
        "    [\n"
        "        sys.executable, '/tmp/yt_sponsor_src/src/data_pipeline.py',\n"
        "        '--csv', str(_CSV_PATH),\n"
        "        '--out', str(_CACHE_DIR),\n"
        f"       '--videos', '{n_videos}',\n"
        "        '--device', 'cuda' if __import__('torch').cuda.is_available() else 'cpu',\n"
        + skip_ids_arg
        + (f"        '{skip_flag}',\n" if skip_flag else "") +
        "    ],\n"
        "    check=True,\n"
        ")\n"
        "_new_files = list(_CACHE_DIR.glob('*.npz'))\n"
        "print(f'New videos this batch: {len(_new_files)}')\n\n"
        "# Write a tiny summary JSON so the bridge can verify these outputs are fresh.\n"
        "_OUT_DIR = Path('/kaggle/working/outputs')\n"
        "_OUT_DIR.mkdir(parents=True, exist_ok=True)\n"
        "(_OUT_DIR / 'phase1_summary.json').write_text(json.dumps({\n"
        f"    'run_id': '{run_id}',\n"
        "    'new_videos_this_batch': len(_new_files),\n"
        "    'new_video_ids': [_f.stem for _f in _new_files],\n"
        "}))\n"
        f"print('run_id={run_id}')\n"
        "print('Summary written to outputs/phase1_summary.json')\n"
    )


def _make_phase_train_cell(config: dict) -> str:
    """Phases 2/3/4: run train.py with the patched config."""
    return (
        "import subprocess, sys\n"
        "result = subprocess.run(\n"
        "    [sys.executable, '/tmp/yt_sponsor_src/src/train.py', '--config', '/tmp/run_config.json'],\n"
        "    check=True,\n"
        ")\n"
        "print('train.py completed with exit code', result.returncode)\n"
    )


def _make_phase_tune_cell(config: dict) -> str:
    """Phase tune: run tune.py (Optuna hyperparameter search)."""
    return (
        "import subprocess, sys\n"
        "result = subprocess.run(\n"
        "    [sys.executable, '/tmp/yt_sponsor_src/src/tune.py', '--config', '/tmp/run_config.json'],\n"
        "    check=True,\n"
        ")\n"
        "print('tune.py completed with exit code', result.returncode)\n"
    )


def _make_phase5_export_cell(config: dict) -> str:
    """Phase 5: run export_onnx.py."""
    ckpt = config.get("student_ckpt", "/kaggle/working/outputs/student_best.pt")
    onnx_file = config.get("onnx_filename", "model.onnx")
    validate = "--validate" if config.get("validate", True) else ""
    return (
        "import subprocess, sys\n"
        "from pathlib import Path\n\n"
        "_OUT_DIR = Path('/kaggle/working/outputs')\n"
        "_OUT_DIR.mkdir(parents=True, exist_ok=True)\n\n"
        "result = subprocess.run(\n"
        "    [\n"
        "        sys.executable, '/tmp/yt_sponsor_src/src/export_onnx.py',\n"
        f"       '--checkpoint', '{ckpt}',\n"
        f"       '--out', str(_OUT_DIR / '{onnx_file}'),\n"
        + (f"       '{validate}',\n" if validate else "") +
        "    ],\n"
        "    check=True,\n"
        ")\n"
        "print('ONNX export completed.')\n"
        "_onnx = _OUT_DIR / 'model.onnx'\n"
        "if _onnx.exists():\n"
        "    print(f'  model.onnx: {_onnx.stat().st_size:,} bytes')\n"
    )


def _make_run_manifest_cell(run_id: str, phase: str) -> str:
    """Last notebook cell: stamp run_manifest.json so the bridge can verify freshness.

    Written as the FINAL cell so it only executes when all prior cells succeed.
    If the kernel errors out before reaching this cell, run_manifest.json will be
    absent from the downloaded outputs, which the bridge treats as a stale-output
    signal and raises RuntimeError rather than silently returning stale results.
    """
    return (
        "import json\n"
        "from pathlib import Path\n\n"
        "_OUT = Path('/kaggle/working/outputs')\n"
        "_OUT.mkdir(exist_ok=True)\n"
        f"_manifest = {{'run_id': '{run_id}', 'phase': '{phase}'}}\n"
        "(_OUT / 'run_manifest.json').write_text(json.dumps(_manifest, indent=2))\n"
        f"print(f'run_manifest written: run_id={run_id}  phase={phase}')\n"
    )


_CELL_COLLECT_OUTPUTS = """\
# Stage all outputs under /kaggle/working/outputs/ for download.
import shutil
from pathlib import Path

_OUT = Path('/kaggle/working/outputs')
_OUT.mkdir(exist_ok=True)

# Collect .pt checkpoints.
for _f in Path('/kaggle/working/outputs').glob('*.pt'):
    print(f'  checkpoint: {_f.name}  ({_f.stat().st_size:,} bytes)')

# Collect training log.
for _f in Path('/kaggle/working/outputs').glob('*.json'):
    print(f'  log: {_f.name}  ({_f.stat().st_size:,} bytes)')

# Collect ONNX model.
for _f in Path('/kaggle/working/outputs').glob('*.onnx'):
    print(f'  onnx: {_f.name}  ({_f.stat().st_size:,} bytes)')

print(f'\\nAll outputs in {_OUT}')
"""


def _make_phase1_upload_cell(username: str) -> str:
    """Upload the embeddings cache to a Kaggle Dataset from inside the kernel.

    The username is embedded at notebook-generation time so the kernel doesn't
    need to resolve it (access_token format has no username field).
    """
    slug = _EMBEDDINGS_DATASET_SLUG
    return (
        "import json, os, subprocess, sys\n"
        "from pathlib import Path\n\n"
        "_CACHE_DIR = Path('/kaggle/working/embeddings_cache')\n"
        "_UPLOAD_DIR = Path('/kaggle/working/dataset_upload')\n"
        "_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)\n\n"
        "# Copy .npz files to a clean upload staging dir.\n"
        "import shutil as _shutil\n"
        "_npz = sorted(_CACHE_DIR.glob('*.npz'))\n"
        "print(f'Staging {len(_npz)} .npz files for dataset upload…')\n"
        "for _f in _npz:\n"
        "    _shutil.copy2(_f, _UPLOAD_DIR / _f.name)\n\n"
        "# Username is embedded at notebook-generation time.\n"
        f"_username = '{username}'\n"
        f"_dataset_id = '{username}/{slug}'\n"
        "# Write dataset-metadata.json.\n"
        f"_meta = {{'title': 'YT Sponsor Embeddings Cache', 'id': _dataset_id, 'licenses': [{{'name': 'CC0-1.0'}}]}}\n"
        "(_UPLOAD_DIR / 'dataset-metadata.json').write_text(__import__('json').dumps(_meta, indent=2))\n\n"
        "# Try to create; if already exists, push a new version.\n"
        "_r = subprocess.run(\n"
        "    ['kaggle', 'datasets', 'create', '-p', str(_UPLOAD_DIR)],\n"
        "    capture_output=True, text=True,\n"
        ")\n"
        "if _r.returncode != 0 and 'already exists' in (_r.stderr + _r.stdout).lower():\n"
        "    print('Dataset exists — pushing new version…')\n"
        "    _r = subprocess.run(\n"
        "        ['kaggle', 'datasets', 'version', '-p', str(_UPLOAD_DIR),\n"
        "         '-m', f'Batch update: {len(_npz)} videos'],\n"
        "        capture_output=True, text=True,\n"
        "    )\n"
        "if _r.returncode != 0:\n"
        "    print('Dataset upload stderr:', _r.stderr)\n"
        "    print('Dataset upload stdout:', _r.stdout)\n"
        "    raise RuntimeError('kaggle dataset push failed — see output above.')\n"
        "print(f'Dataset uploaded: {_dataset_id}  ({len(_npz)} videos)')\n"
        "# Write a summary log for the bridge to download.\n"
        "_summary = {\n"
        "    'total_cached': len(_npz),\n"
        f"    'dataset': f'{{_username}}/{slug}',\n"
        "    'uploaded_from_kernel': True,\n"
        "}\n"
        "Path('/kaggle/working/outputs').mkdir(exist_ok=True)\n"
        "Path('/kaggle/working/outputs/phase1_summary.json').write_text(\n"
        "    __import__('json').dumps(_summary, indent=2)\n"
        ")\n"
        "print('Summary:', _summary)\n"
    )


# Phase 1: delete only the large SponsorBlock CSV before download.
# The new .npz files stay in /kaggle/working/embeddings_cache/ so the bridge
# can download them and merge them into the local master cache.
_CELL_COLLECT_OUTPUTS_PHASE1 = """\
import shutil
from pathlib import Path

# Delete only the large SponsorBlock CSV — the new .npz files will be
# downloaded by the bridge and merged into the local master cache.
_csv = Path('/kaggle/working/sponsorTimes.csv')
if _csv.exists():
    _csv.unlink()
    print(f'Removed {_csv.name}')

# Report what will be downloaded.
_CACHE_DIR = Path('/kaggle/working/embeddings_cache')
_npz = sorted(_CACHE_DIR.glob('*.npz')) if _CACHE_DIR.is_dir() else []
_total_bytes = sum(_f.stat().st_size for _f in _npz)
print(f'New .npz files to download: {len(_npz)}  ({_total_bytes / 1e6:.1f} MB total)')

_OUT = Path('/kaggle/working/outputs')
_OUT.mkdir(exist_ok=True)
for _f in _OUT.glob('phase1_summary.json'):
    print(f'  summary: {_f.name}  ({_f.stat().st_size:,} bytes)')
print(f'\\nReady for download.')
"""


# ---------------------------------------------------------------------------
# Dataset mount cell (copy from /kaggle/input/<slug>/ into working dir)
# ---------------------------------------------------------------------------


def _make_mount_dataset_cell(slug: str, dest_dir: str) -> str:
    """Copy mounted Kaggle Dataset files into the kernel working directory.

    Kaggle mounts datasets at /kaggle/input/<slug>/ in classic kernels but
    sometimes uses /kaggle/input/datasets/ or other paths in newer API versions.
    We search the entire /kaggle/input tree for the expected file extension so
    the cell works regardless of Kaggle's mount layout.
    """
    # Infer file extension from slug so we grab the right files.
    # Embeddings cache → .npz; teacher checkpoint → .pt
    if "embeddings" in slug:
        ext = ".npz"
    else:
        ext = ".pt"

    return (
        "import shutil, zipfile\n"
        "from pathlib import Path\n\n"
        f"_SLUG = '{slug}'\n"
        f"_EXT  = '{ext}'\n"
        f"_DEST = Path('{dest_dir}')\n"
        "_DEST.mkdir(parents=True, exist_ok=True)\n\n"
        "# --- locate data.zip anywhere under /kaggle/input ---\n"
        "_kaggle_input = Path('/kaggle/input')\n"
        "print(f'Scanning /kaggle/input for data.zip…')\n"
        "_zip_files = sorted(_kaggle_input.rglob('data.zip'))\n"
        "if _zip_files:\n"
        "    print(f'Found zip: {_zip_files[0]}  ({_zip_files[0].stat().st_size:,} bytes)')\n"
        "    with zipfile.ZipFile(_zip_files[0]) as _zf:\n"
        "        _members = [m for m in _zf.namelist() if m.endswith(_EXT)]\n"
        "        print(f'Extracting {len(_members)} {_EXT} files…')\n"
        "        for _m in _members:\n"
        "            _dst = _DEST / Path(_m).name\n"
        "            if not _dst.exists():\n"
        "                with _zf.open(_m) as _src, open(_dst, 'wb') as _out:\n"
        "                    _out.write(_src.read())\n"
        "else:\n"
        "    # Fallback: look for loose files (old-style dataset without zip).\n"
        "    print('No data.zip found — falling back to loose file scan')\n"
        "    _src_files = sorted(_kaggle_input.rglob(f'*{_EXT}'))\n"
        "    if not _src_files:\n"
        "        _available = sorted(str(p) for p in _kaggle_input.iterdir()) if _kaggle_input.exists() else []\n"
        "        raise RuntimeError(\n"
        f"            f'No {{_EXT}} files or data.zip found under /kaggle/input. '\n"
        "            f'Available: {_available}. Slug: {_SLUG}'\n"
        "        )\n"
        "    print(f'Found {len(_src_files)} loose files')\n"
        "    for _f in _src_files:\n"
        "        _dst = _DEST / _f.name\n"
        "        if not _dst.exists():\n"
        "            shutil.copy2(_f, _dst)\n\n"
        "_files = list(_DEST.glob(f'*{_EXT}'))\n"
        "print(f'Dataset ready: {len(_files)} {_EXT} files in {_DEST}')\n"
    )


# ---------------------------------------------------------------------------
# Notebook assembly
# ---------------------------------------------------------------------------


def build_notebook(
    config: dict,
    src_tarball: Path,
    phase: str,
    dataset_sources_slugs: list[str] | None = None,
    already_processed: list[str] | None = None,
    run_id: str = "",
    username: str = "",
) -> nbformat.NotebookNode:
    """Assemble the Kaggle training notebook for the given phase.

    Args:
        config:                 Phase config dict (will be embedded in the notebook).
        src_tarball:            Path to the bundled training/src/ tarball.
        phase:                  Phase name: "data" | "baseline" | "teacher" | "tune" |
                                "distill" | "export".
        dataset_sources_slugs:  List of Kaggle Dataset slugs mounted in the kernel.
        already_processed:      Phase 1 only. Video IDs already in the local master cache.
                                Embedded in the notebook as a skip-list so the kernel only
                                processes NEW videos.
        run_id:                 Unique ID stamped into run_manifest.json so the bridge
                                can detect stale (cached) outputs from a previous run.

    Returns:
        A nbformat.NotebookNode ready to be written to disk.
    """
    nb = new_notebook()
    cells = [
        new_code_cell(_CELL_SETUP),
        new_code_cell(_make_install_deps_cell(phase)),  # adds optuna for tune phase
        new_code_cell(_make_install_src_cell(src_tarball)),
    ]

    # Mount datasets if needed (phases 2-5 only — phase 1 uses embedded skip-list).
    if dataset_sources_slugs:
        for slug in dataset_sources_slugs:
            if slug == _EMBEDDINGS_DATASET_SLUG:
                cells.append(new_code_cell(
                    _make_mount_dataset_cell(slug, "/kaggle/working/embeddings_cache")
                ))
            elif slug == _TEACHER_CKPT_DATASET_SLUG:
                cells.append(new_code_cell(
                    _make_mount_dataset_cell(slug, "/kaggle/working/outputs")
                ))

    # Phase-specific execution cell.
    cells.append(new_code_cell(_make_config_cell(config)))

    if phase == "data":
        # Embed the already-processed IDs so the kernel skips them.
        # New .npz files are left in /kaggle/working/embeddings_cache/ for download.
        cells.append(new_code_cell(_make_phase1_data_cell(config, already_processed=already_processed, run_id=run_id)))
        cells.append(new_code_cell(_CELL_COLLECT_OUTPUTS_PHASE1))
        # Phase 1 uses _wait_phase1_fresh for staleness detection (run_id in phase1_summary.json).
        # No separate run_manifest needed.
    elif phase in ("baseline", "teacher", "distill"):
        cells.append(new_code_cell(_make_phase_train_cell(config)))
        cells.append(new_code_cell(_CELL_COLLECT_OUTPUTS))
        cells.append(new_code_cell(_make_run_manifest_cell(run_id, phase)))
    elif phase == "tune":
        cells.append(new_code_cell(_make_phase_tune_cell(config)))
        cells.append(new_code_cell(_CELL_COLLECT_OUTPUTS))
        cells.append(new_code_cell(_make_run_manifest_cell(run_id, phase)))
    elif phase == "export":
        cells.append(new_code_cell(_make_phase5_export_cell(config)))
        cells.append(new_code_cell(_CELL_COLLECT_OUTPUTS))
        cells.append(new_code_cell(_make_run_manifest_cell(run_id, phase)))
    else:
        raise ValueError(f"Unknown phase: {phase!r}")

    nb.cells = cells
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    return nb


# ---------------------------------------------------------------------------
# Kernel push / poll / fetch
# ---------------------------------------------------------------------------


def _write_kernel_metadata(
    folder: Path,
    kernel_slug: str,
    notebook_filename: str,
    enable_gpu: bool,
    username: str,
    dataset_sources: list[str] | None = None,
) -> None:
    metadata = {
        "id": f"{username}/{kernel_slug}",
        "title": kernel_slug.replace("-", " ").title(),
        "code_file": notebook_filename,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": enable_gpu,
        "enable_tpu": False,
        "enable_internet": True,
        "dataset_sources": dataset_sources or [],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    (folder / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2))


def push_kernel(folder: Path) -> str:
    _log("Pushing kernel to Kaggle…")
    result = _kaggle_cmd("kernels", "push", "-p", str(folder))
    push_out = (result.stdout + result.stderr).strip()
    if result.returncode != 0:
        raise RuntimeError(
            f"kaggle kernels push failed (exit {result.returncode}):\n{push_out}"
        )
    _log(f"  push output: {push_out}")
    # Kaggle returns exit 0 even for quota errors — detect them explicitly.
    if "quota" in push_out.lower() or "push error" in push_out.lower():
        raise RuntimeError(f"Kaggle kernel push rejected: {push_out}")
    meta = json.loads((folder / "kernel-metadata.json").read_text())
    ref = meta["id"]
    _log(f"Pushed → https://www.kaggle.com/code/{ref}")
    return ref


def _kaggle_api_get(path: str) -> dict | None:
    """GET request to the Kaggle REST API using credentials from env vars."""
    import urllib.request as _req, base64 as _b64
    username = os.environ.get("KAGGLE_USERNAME", "")
    key = os.environ.get("KAGGLE_KEY", "")
    if not (username and key):
        return None
    auth = _b64.b64encode(f"{username}:{key}".encode()).decode()
    url = f"https://www.kaggle.com/api/v1{path}"
    try:
        req = _req.Request(
            url,
            headers={"Authorization": f"Basic {auth}", "Accept": "application/json"},
        )
        with _req.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read())
    except Exception as exc:
        _log(f"  (API call failed for {path}: {exc})")
        return None


def wait_for_completion(kernel_ref: str) -> str:
    """Poll kernel status until it reaches a terminal state."""
    _log(f"Polling every {POLL_INTERVAL_SEC}s (timeout {MAX_WAIT_SEC}s)…")
    elapsed = 0
    _TERMINAL = {"complete": "complete", "error": "error", "cancelacknowledged": "cancelAcknowledged"}
    while elapsed < MAX_WAIT_SEC:
        result = _kaggle_cmd("kernels", "status", kernel_ref)
        output = (result.stdout + result.stderr).lower()
        matched = next((v for k, v in _TERMINAL.items() if k in output), None)
        status = matched or "running"
        _log(f"  status={status}  elapsed={elapsed}s")
        if matched:
            return matched
        time.sleep(POLL_INTERVAL_SEC)
        elapsed += POLL_INTERVAL_SEC
    return "timeout"


def _wait_phase1_fresh(kernel_ref: str, run_id: str, slug: str) -> tuple[list[Path], Path]:
    """Wait until phase-1 kernel outputs contain our run_id, then return them.

    Kaggle's ``kaggle kernels status`` and ``kaggle kernels output`` both return
    the PREVIOUS run's result while the newly-pushed version is still queued.
    We detect freshness by checking ``run_id`` in ``phase1_summary.json``.

    Returns ``(files, output_dir)`` once the fresh outputs are confirmed.
    """
    _log(f"Waiting for fresh phase-1 outputs (run_id={run_id})…")

    # Probe the Kaggle REST API to see what version is registered.
    kernel_info = _kaggle_api_get(f"/kernels/{kernel_ref}")
    if kernel_info:
        cur_ver = kernel_info.get("currentRunningVersion") or kernel_info.get("currentVersion")
        _log(f"  Kaggle API — kernel info: currentRunningVersion={cur_ver}  "
             f"keys={list(kernel_info.keys())[:10]}")
    else:
        _log("  (Kaggle REST API lookup failed — proceeding with CLI polling)")

    elapsed = 0
    attempt = 0

    while elapsed < MAX_WAIT_SEC:
        result = _kaggle_cmd("kernels", "status", kernel_ref)
        raw = (result.stdout + result.stderr).strip()
        lower = raw.lower()

        if "error" in lower and "complete" not in lower:
            raise RuntimeError(
                f"Kernel errored. Inspect at: https://www.kaggle.com/code/{kernel_ref}"
            )

        status = (
            "complete" if "complete" in lower
            else "running" if "running" in lower
            else "queued" if "queued" in lower
            else "pending"
        )
        _log(f"  status={status}  elapsed={elapsed}s  raw={raw!r}")

        # Try to download and verify outputs whenever something might have finished.
        if status in ("complete", "running", "queued"):
            attempt += 1
            ts = datetime.now().strftime("%H%M%S")
            out_dir = _TRAINING_ROOT / "kaggle_outputs" / f"{slug}_{run_id}_{attempt:02d}"
            try:
                files = fetch_outputs(kernel_ref, out_dir)
                for f in files:
                    if f.name == "phase1_summary.json":
                        try:
                            summary = json.loads(f.read_text())
                            if summary.get("run_id") == run_id:
                                _log(f"  Fresh outputs confirmed ✓  (attempt {attempt})")
                                return files, out_dir
                            else:
                                _log(
                                    f"  Stale output from a previous run "
                                    f"(expected run_id={run_id!r}, got {summary.get('run_id')!r})"
                                    " — still waiting…"
                                )
                        except Exception:
                            pass
                        break
            except RuntimeError as exc:
                _log(f"  Output download attempt {attempt} failed: {exc}")

        time.sleep(POLL_INTERVAL_SEC)
        elapsed += POLL_INTERVAL_SEC

    raise RuntimeError(
        f"Timed out ({MAX_WAIT_SEC}s) waiting for fresh phase-1 outputs. "
        f"Check https://www.kaggle.com/code/{kernel_ref}"
    )


def _verify_fresh_outputs(files: list[Path], run_id: str) -> bool:
    """Return True if downloaded outputs contain a run_manifest.json with matching run_id.

    The run_manifest.json is written as the LAST notebook cell, so it is absent
    whenever the kernel fails before reaching that point.  If it's missing or
    contains a different run_id, the outputs are stale (from a previous run) and
    should not be used.
    """
    for f in files:
        if f.name == "run_manifest.json":
            try:
                manifest = json.loads(f.read_text())
                got_id = manifest.get("run_id", "")
                if got_id == run_id:
                    _log(f"  run_manifest verified ✓ (run_id={run_id})")
                    return True
                _log(
                    f"  Stale output detected — expected run_id={run_id!r}, "
                    f"got {got_id!r}.  The kernel likely failed before completing."
                )
                return False
            except Exception as exc:
                _log(f"  WARNING: could not parse run_manifest.json: {exc}")
                return False
    _log(
        f"  run_manifest.json not found in downloaded outputs "
        f"(expected run_id={run_id!r}).  Kernel may have errored mid-run."
    )
    return False


def fetch_outputs(kernel_ref: str, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    _log(f"Fetching outputs → {output_dir}")
    result = _kaggle_cmd("kernels", "output", kernel_ref, "-p", str(output_dir), "--force")
    if result.returncode != 0:
        raise RuntimeError(
            f"kaggle kernels output failed:\n{result.stderr or result.stdout}"
        )
    files = [f for f in output_dir.rglob("*") if f.is_file()]
    for f in files:
        _log(f"  {f.name}  ({f.stat().st_size:,} bytes)")
    return files


# ---------------------------------------------------------------------------
# Output routing
# ---------------------------------------------------------------------------


def route_outputs(files: list[Path], output_dir: Path, no_overwrite: bool = False) -> None:
    """Copy fetched files to training/outputs/<phase>/ locally.

    *.pt    → model checkpoints
    *.onnx  → ONNX model (also copy to youtube-ml-sponsor-detector/ for the extension)
    *.json  → training logs
    *.npz   → embedding cache files (phase 1)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    extension_dir = _PROJECT_ROOT / "youtube-ml-sponsor-detector"

    for f in files:
        # Preserve embeddings_cache/ subdirectory structure for .npz files.
        if f.suffix == ".npz" and f.parent.name == "embeddings_cache":
            dest = output_dir / "embeddings_cache" / f.name
            dest.parent.mkdir(parents=True, exist_ok=True)
        else:
            dest = output_dir / f.name
        if no_overwrite and dest.exists():
            _log(f"  (skip — exists) {f.name}")
            continue

        shutil.copy2(f, dest)
        _log(f"  → {dest.relative_to(_PROJECT_ROOT)}")

        # Auto-install model.onnx into the extension directory.
        if f.suffix == ".onnx" and extension_dir.is_dir():
            ext_dest = extension_dir / f.name
            shutil.copy2(f, ext_dest)
            _log(f"  → {ext_dest.relative_to(_PROJECT_ROOT)}  (extension)")


# ---------------------------------------------------------------------------
# Kernel slug helpers
# ---------------------------------------------------------------------------


def _make_kernel_slug(config: dict, config_path: Path) -> str:
    name: str = config.get("name", config_path.stem)
    return name.lower().replace("_", "-")[:50]


# ---------------------------------------------------------------------------
# Phase-specific dataset source lists
# ---------------------------------------------------------------------------


def _embeddings_shard_slugs(n_total: int) -> list[str]:
    """Return all shard slugs for the embeddings dataset given total file count."""
    n_shards = max(1, (n_total + _SHARD_SIZE - 1) // _SHARD_SIZE)
    slugs = [_EMBEDDINGS_DATASET_SLUG]
    for i in range(1, n_shards):
        slugs.append(f"{_EMBEDDINGS_DATASET_SLUG}-s{i}")
    return slugs


def _get_dataset_sources(phase: str, username: str, n_embeddings: int = 0) -> list[str]:
    """Return the list of Kaggle Dataset references to mount for each phase.

    Phase 1 ("data") never mounts anything — already-processed IDs are embedded
    directly in the notebook as a skip-list, so no dataset mount is needed.
    """
    emb_slugs   = _embeddings_shard_slugs(n_embeddings)
    emb_refs    = [f"{username}/{s}" for s in emb_slugs]
    teacher_ref = f"{username}/{_TEACHER_CKPT_DATASET_SLUG}"

    return {
        "data":     [],
        "baseline": emb_refs,
        "teacher":  emb_refs,
        "tune":     emb_refs,
        "distill":  emb_refs + [teacher_ref],
        "export":   [teacher_ref],
    }.get(phase, [])


def _get_dataset_slugs(phase: str, n_embeddings: int = 0) -> list[str]:
    """Return only the slug parts (for notebook mount cells)."""
    emb_slugs = _embeddings_shard_slugs(n_embeddings)
    return {
        "data":     [],
        "baseline": emb_slugs,
        "teacher":  emb_slugs,
        "tune":     emb_slugs,
        "distill":  emb_slugs + [_TEACHER_CKPT_DATASET_SLUG],
        "export":   [_TEACHER_CKPT_DATASET_SLUG],
    }.get(phase, [])


# ---------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------


def dry_run(config_path: Path, enable_gpu: bool, kernel_slug: str | None = None) -> bool:
    config = json.loads(config_path.read_text())
    phase = config.get("phase", "teacher")
    slug = kernel_slug or _make_kernel_slug(config, config_path)
    username = _kaggle_username()

    _log("=== DRY RUN — nothing will be pushed to Kaggle ===")
    _log(f"Phase:  {phase}")
    _log(f"Config: {config_path}")
    _log(f"Slug:   {slug}")
    _log(f"GPU:    {enable_gpu}")

    out_dir = _TRAINING_ROOT / "kaggle_outputs" / "dry_run"
    out_dir.mkdir(parents=True, exist_ok=True)

    results: list[tuple[str, bool, str]] = []

    # Step 1 — credentials
    try:
        authenticate()
        results.append(("credentials", True, f"authenticated as '{username}'"))
    except SystemExit:
        results.append(("credentials", False, "auth failed — check token/username"))
        _print_dry_run_summary(results)
        return False

    # Step 2 — source bundle
    tar_path = out_dir / "yt_sponsor_src.tar.gz"
    try:
        _bundle_source(tar_path)
        size_kb = tar_path.stat().st_size // 1024
        results.append(("source bundle", True, f"{tar_path.name}  ({size_kb} KB)"))
    except Exception as exc:
        results.append(("source bundle", False, str(exc)))
        _print_dry_run_summary(results)
        return False

    # Step 3 — notebook generation
    nb_name = f"{slug}.ipynb"
    nb_path = out_dir / nb_name
    dataset_slugs = _get_dataset_slugs(phase)
    dry_run_id = "dryrun_00000000_000000"
    try:
        nb = build_notebook(config, tar_path, phase=phase, dataset_sources_slugs=dataset_slugs,
                            run_id=dry_run_id, username=username)
        nbformat.write(nb, nb_path)
        results.append(("notebook", True, f"{nb_name}  ({len(nb.cells)} cells)"))
    except Exception as exc:
        results.append(("notebook", False, str(exc)))
        _print_dry_run_summary(results)
        return False

    # Step 4 — kernel metadata
    try:
        dataset_sources = _get_dataset_sources(phase, username)
        _write_kernel_metadata(
            folder=out_dir,
            kernel_slug=slug,
            notebook_filename=nb_name,
            enable_gpu=enable_gpu,
            username=username,
            dataset_sources=dataset_sources,
        )
        meta = json.loads((out_dir / "kernel-metadata.json").read_text())
        results.append(("kernel metadata", True, f"id={meta['id']}"))
    except Exception as exc:
        results.append(("kernel metadata", False, str(exc)))
        _print_dry_run_summary(results)
        return False

    # Step 5 — notebook cell syntax check
    import ast
    bad_cells: list[int] = []
    for i, cell in enumerate(nb.cells):
        try:
            ast.parse(cell["source"])
        except SyntaxError as exc:
            bad_cells.append(i + 1)
            _log(f"  cell {i + 1} syntax error: {exc}")
    if bad_cells:
        results.append(("cell syntax", False, f"cells with errors: {bad_cells}"))
    else:
        results.append(("cell syntax", True, f"all {len(nb.cells)} cells parse cleanly"))

    _print_dry_run_summary(results)

    if all(ok for _, ok, _ in results):
        _log(f"Inspect the generated kernel at: {out_dir}")
        _log(f"  notebook:        {nb_name}")
        _log("  source tarball:  yt_sponsor_src.tar.gz")
        _log("  kernel metadata: kernel-metadata.json")
        _log("")
        _log("To run for real:")
        _log(f"  python {Path(__file__).name} --config {config_path} --gpu")
        return True

    return False


def _print_dry_run_summary(results: list[tuple[str, bool, str]]) -> None:
    print()
    print("  Dry-run checklist:")
    for step, ok, detail in results:
        mark = "✓" if ok else "✗"
        print(f"    [{mark}] {step:<22}  {detail}")
    print()


# ---------------------------------------------------------------------------
# Main bridge round-trip
# ---------------------------------------------------------------------------


def run_bridge(
    config_path: Path,
    enable_gpu: bool,
    kernel_slug: str | None = None,
    no_overwrite: bool = False,
    upload_cache: bool = False,
    upload_teacher_ckpt: bool = False,
    accumulate: bool = False,
) -> bool:
    """Execute the full bridge round-trip for one phase config.

    Args:
        config_path:          Phase config JSON.
        enable_gpu:           Whether to request a Kaggle GPU.
        kernel_slug:          Override kernel slug.
        no_overwrite:         Skip overwriting existing local outputs.
        upload_cache:         After phase 1 completes, upload the full master
                              embedding cache as a Kaggle Dataset so phases 2–5
                              can mount it.
        upload_teacher_ckpt:  After phase 3 completes, upload teacher_best.pt
                              as a Kaggle Dataset for phase 4 to consume.
        accumulate:           Phase 1 only: embed already-processed video IDs as a
                              skip-list so the kernel only processes NEW videos.
                              Use this on every run after the first.
    """
    config = json.loads(config_path.read_text())
    phase = config.get("phase", "teacher")
    slug = kernel_slug or _make_kernel_slug(config, config_path)

    _log(f"Phase:  {phase}")
    _log(f"Config: {config_path}")
    _log(f"Slug:   {slug}")
    _log(f"GPU:    {enable_gpu}")

    authenticate()
    username = _kaggle_username()

    # Generate a unique run_id for ALL phases.  It is embedded in the notebook
    # and written to run_manifest.json as the last cell.  The bridge checks this
    # after downloading outputs to detect stale results from a previous run.
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    _log(f"run_id={run_id}")

    # For phase 1, derive the already-processed video IDs from the master cache.
    already_processed: list[str] | None = None
    if phase == "data":
        master_ids = sorted(
            f.stem for f in _MASTER_CACHE_DIR.glob("*.npz")
        ) if _MASTER_CACHE_DIR.is_dir() else []
        if master_ids:
            already_processed = master_ids
            _log(f"Master cache: {len(master_ids)} videos already processed (will be skipped).")
        else:
            _log("Master cache is empty — processing fresh batch.")
        _print_cache_total(label="Before this batch")

    n_embeddings = sum(1 for _ in _MASTER_CACHE_DIR.glob("*.npz")) if _MASTER_CACHE_DIR.is_dir() else 0
    dataset_sources = _get_dataset_sources(phase, username, n_embeddings)
    dataset_slugs = _get_dataset_slugs(phase, n_embeddings)
    if dataset_slugs:
        _log(f"Dataset shards: {dataset_slugs}")

    tmp_dir = Path(tempfile.mkdtemp(prefix="yt_sponsor_kaggle_"))
    try:
        # 1. Bundle source.
        tar_path = tmp_dir / "yt_sponsor_src.tar.gz"
        _bundle_source(tar_path)

        # 2. Build notebook.
        nb = build_notebook(
            config, tar_path, phase=phase,
            dataset_sources_slugs=dataset_slugs,
            already_processed=already_processed,
            run_id=run_id,
            username=username,
        )
        nb_name = f"{slug}.ipynb"
        nb_path = tmp_dir / nb_name
        nbformat.write(nb, nb_path)
        _log(f"Notebook: {nb_name}  ({len(nb.cells)} cells)")

        # 3. Write kernel metadata.
        _write_kernel_metadata(
            folder=tmp_dir,
            kernel_slug=slug,
            notebook_filename=nb_name,
            enable_gpu=enable_gpu,
            username=username,
            dataset_sources=dataset_sources,
        )

        # 4. Push kernel.
        meta_path = tmp_dir / "kernel-metadata.json"
        _log(f"kernel-metadata.json:\n{meta_path.read_text()}")
        kernel_ref = push_kernel(tmp_dir)

        # 5. Wait for completion and fetch outputs.
        local_output_dir = _TRAINING_ROOT / "outputs" / phase
        if phase == "data":
            # Phase 1: use run_id to verify we got fresh (not cached) outputs.
            files, raw_output_dir = _wait_phase1_fresh(kernel_ref, run_id, slug)
        else:
            # Phases 2-5: poll until complete, then download and verify freshness.
            final_status = wait_for_completion(kernel_ref)
            _log(f"Kernel finished: {final_status}")
            if final_status != "complete":
                _log(f"Kernel did not complete — inspect at: https://www.kaggle.com/code/{kernel_ref}")
                return False
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            raw_output_dir = _TRAINING_ROOT / "kaggle_outputs" / f"{slug}_{timestamp}"
            files = fetch_outputs(kernel_ref, raw_output_dir)
            # Verify that we got fresh outputs for THIS run (not a cached previous run).
            # run_manifest.json is written as the final notebook cell; its absence or a
            # mismatched run_id means the kernel failed before completing.
            if not _verify_fresh_outputs(files, run_id):
                raise RuntimeError(
                    f"Downloaded outputs are stale (run_id mismatch or missing run_manifest.json). "
                    f"The kernel at https://www.kaggle.com/code/{kernel_ref} likely failed "
                    f"mid-run — Kaggle returned the previous run's output files. "
                    f"Check the kernel logs before re-running."
                )

        # 6. Route to training/outputs/<phase>/.
        route_outputs(files, local_output_dir, no_overwrite=no_overwrite)

        # 8. Phase-specific post-processing.
        if phase == "data":
            # Merge any newly downloaded .npz files into the persistent master cache.
            new_count = 0
            for src in [raw_output_dir / "embeddings_cache", raw_output_dir]:
                if src.is_dir() and any(src.glob("*.npz")):
                    new_count = _merge_into_master_cache(src)
                    break

            # Read summary JSON written by data_pipeline inside the kernel.
            summary_path = local_output_dir / "phase1_summary.json"
            batch_new = None
            if summary_path.exists():
                summary = json.loads(summary_path.read_text())
                batch_new = summary.get("new_videos_this_batch")
                _log(f"Kernel reported {batch_new} new videos this batch.")
            else:
                _log("WARNING: phase1_summary.json not found.")

            master_total = sum(1 for _ in _MASTER_CACHE_DIR.glob("*.npz")) if _MASTER_CACHE_DIR.is_dir() else 0
            _print_cache_total(label="Running total")

            # Optionally upload the full master cache to Kaggle for phases 2-5.
            if upload_cache:
                if master_total > 0:
                    _log(f"Uploading full master cache ({master_total} videos) to Kaggle Dataset…")
                    upload_embeddings_cache(_MASTER_CACHE_DIR, username)
                else:
                    _log("WARNING: master cache is empty — skipping Kaggle upload.")

        if upload_teacher_ckpt and phase == "teacher":
            ckpt_dir = local_output_dir
            if list(ckpt_dir.glob("teacher_best.pt")):
                upload_teacher_checkpoint(ckpt_dir, username)
            else:
                _log("WARNING: teacher_best.pt not found — skipping checkpoint upload.")

        _log(f"Done. Outputs in training/outputs/{phase}/")
        return True

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    p = argparse.ArgumentParser(
        prog="kaggle_bridge",
        description="Run YouTube sponsor-detection training phases on Kaggle GPU.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--config", required=True, type=Path,
        help="Path to phase config JSON (e.g. training/configs/phase3_teacher.json).",
    )
    p.add_argument(
        "--gpu", action="store_true", default=False,
        help="Enable Kaggle GPU accelerator (counts against 30 hr/week free quota).",
    )
    p.add_argument(
        "--slug", default=None,
        help="Kernel slug override. Default: derived from the config name field.",
    )
    p.add_argument(
        "--dry-run", action="store_true", default=False,
        help=(
            "Validate credentials, bundle source, generate notebook locally. "
            "Nothing pushed to Kaggle. Outputs to training/kaggle_outputs/dry_run/."
        ),
    )
    p.add_argument(
        "--no-overwrite", action="store_true", default=False,
        help="Skip output files that already exist locally.",
    )
    p.add_argument(
        "--upload-cache", action="store_true", default=False,
        help=(
            "Phase 1 only: after fetching outputs, upload the .npz embedding cache "
            "as a Kaggle Dataset so phases 2–5 can mount it without re-downloading."
        ),
    )
    p.add_argument(
        "--upload-teacher-ckpt", action="store_true", default=False,
        help=(
            "Phase 3 only: after fetching outputs, upload teacher_best.pt "
            "as a Kaggle Dataset so phase 4 can load it for distillation."
        ),
    )
    p.add_argument(
        "--accumulate", action="store_true", default=False,
        help=(
            "Phase 1 only: mount the existing embeddings cache dataset and seed the "
            "working cache from it before running data_pipeline.py.  New videos are "
            "added; already-processed ones are skipped.  Use together with "
            "--upload-cache to push the combined result back.  Not needed on the "
            "first run (when no dataset exists yet)."
        ),
    )
    p.add_argument(
        "--upload-cache-only", action="store_true", default=False,
        help=(
            "Skip the Kaggle kernel entirely and just upload the existing .npz cache "
            "to Kaggle Datasets.  Searches training/outputs/data/embeddings_cache/, "
            "training/outputs/data/, and training/kaggle_outputs/ for .npz files.  "
            "Use --cache-dir to point at a specific directory instead."
        ),
    )
    p.add_argument(
        "--cache-dir", type=Path, default=None,
        help=(
            "Explicit path to a directory of .npz files to upload with "
            "--upload-cache-only.  Overrides the automatic search."
        ),
    )
    args = p.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Config not found: {config_path}")
        sys.exit(1)

    if args.upload_cache_only:
        username = _kaggle_username()

        # Explicit dir takes priority.
        if args.cache_dir is not None:
            cache_dir = args.cache_dir.expanduser().resolve()
            if not cache_dir.is_dir():
                print(f"ERROR: --cache-dir does not exist: {cache_dir}")
                sys.exit(1)
        else:
            # Auto-search: pick the directory with the MOST .npz files.
            candidates = [
                _MASTER_CACHE_DIR,  # persistent master cache — always the best source
                _TRAINING_ROOT / "outputs" / "data" / "embeddings_cache",
                _TRAINING_ROOT / "outputs" / "data",
            ]
            kaggle_out = _TRAINING_ROOT / "kaggle_outputs"
            if kaggle_out.is_dir():
                for d in sorted(kaggle_out.iterdir(), reverse=True):
                    candidates.append(d / "embeddings_cache")
                    candidates.append(d)
            best, best_count = None, 0
            for candidate in candidates:
                if candidate.is_dir():
                    n = sum(1 for _ in candidate.glob("*.npz"))
                    if n > best_count:
                        best, best_count = candidate, n
            cache_dir = best

        if cache_dir is None or not any(cache_dir.glob("*.npz")):
            print("ERROR: No .npz files found. Use --cache-dir to specify the location.")
            sys.exit(1)

        count = sum(1 for _ in cache_dir.glob("*.npz"))
        _log(f"Uploading {count} .npz files from {cache_dir}")
        upload_embeddings_cache(cache_dir, username)
        _merge_into_master_cache(cache_dir)
        _print_cache_total(kaggle_total=count, label="Running total after upload")
        sys.exit(0)

    if args.dry_run:
        ok = dry_run(config_path=config_path, enable_gpu=args.gpu, kernel_slug=args.slug)
    else:
        ok = run_bridge(
            config_path=config_path,
            enable_gpu=args.gpu,
            kernel_slug=args.slug,
            no_overwrite=args.no_overwrite,
            upload_cache=args.upload_cache,
            upload_teacher_ckpt=args.upload_teacher_ckpt,
            accumulate=args.accumulate,
        )
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
