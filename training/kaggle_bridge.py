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

_DEFAULT_USERNAME = "your_kaggle_username"

#: Kaggle Dataset slug for the per-video embedding cache (phases 1 → 2-5).
_EMBEDDINGS_DATASET_SLUG = "yt-sponsor-embeddings-cache"

#: Kaggle Dataset slug for teacher checkpoint (phase 3 → 4).
_TEACHER_CKPT_DATASET_SLUG = "yt-sponsor-teacher-checkpoint"

POLL_INTERVAL_SEC = 30
MAX_WAIT_SEC = 7200  # 2 hours — ample for 300-video embedding + 30-epoch training

_TRAINING_ROOT = Path(__file__).resolve().parent          # training/
_PROJECT_ROOT = _TRAINING_ROOT.parent                     # repo root

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _log(msg: str) -> None:
    print(f"[bridge {datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def _kaggle_username() -> str:
    username = os.environ.get("KAGGLE_USERNAME", "").strip()
    if not username or username == "your_kaggle_username":
        username = _DEFAULT_USERNAME
    if not username or username == "your_kaggle_username":
        print("\nKaggle username not configured.")
        print("  export KAGGLE_USERNAME=your_username")
        print(f"  or edit _DEFAULT_USERNAME in {__file__}")
        sys.exit(1)
    return username


def _read_token_file() -> str | None:
    p = Path.home() / ".kaggle" / "access_token"
    return p.read_text().strip() if p.exists() else None


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


def _upload_dataset(
    data_dir: Path,
    username: str,
    slug: str,
    title: str,
) -> str:
    """Upload or update a local directory as a Kaggle Dataset.

    Returns the dataset reference ``"<username>/<slug>"``.
    """
    ref = f"{username}/{slug}"
    _log(f"Uploading dataset '{ref}'…")
    _log(f"  source: {data_dir}")

    with tempfile.TemporaryDirectory(prefix="kaggle_dataset_") as tmp:
        tmp_path = Path(tmp)
        meta = {"title": title, "id": ref, "licenses": [{"name": "CC0-1.0"}]}
        (tmp_path / "dataset-metadata.json").write_text(json.dumps(meta, indent=2))

        n_files = total_bytes = 0
        for f in sorted(data_dir.iterdir()):
            if f.is_file():
                shutil.copy2(f, tmp_path / f.name)
                n_files += 1
                total_bytes += f.stat().st_size
        _log(f"  {n_files} files  ({total_bytes:,} bytes)")

        result = _kaggle_cmd("datasets", "create", "-p", str(tmp_path))
        if result.returncode != 0 and "already exists" in (result.stderr + result.stdout).lower():
            _log("  Already exists — pushing new version…")
            result = _kaggle_cmd(
                "datasets", "version", "-p", str(tmp_path), "-m", "Updated by kaggle_bridge.py"
            )
        if result.returncode != 0:
            raise RuntimeError(
                f"Dataset upload failed:\n{result.stderr or result.stdout}"
            )

    # Poll until ready.
    _log("  Waiting for Kaggle to process the dataset…")
    for _ in range(30):
        check = _kaggle_cmd("datasets", "files", ref)
        if check.returncode == 0 and check.stdout.strip():
            break
        time.sleep(10)
    else:
        _log("  WARNING: dataset may not be ready yet — proceeding anyway.")

    _log(f"  Dataset ready → https://www.kaggle.com/datasets/{ref}")
    return ref


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

_CELL_INSTALL_DEPS = """\
import subprocess, sys, urllib.request

try:
    urllib.request.urlopen("https://pypi.org", timeout=5)
except Exception as _e:
    raise RuntimeError(
        "No internet access. Verify your phone at https://www.kaggle.com/settings/account"
    ) from _e

_pkgs = [
    "transformers>=4.36",
    "yt-dlp>=2024.1.1",
    "soundfile>=0.12",
]
for _p in _pkgs:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", _p], check=True)
print("Extra deps installed:", _pkgs)
"""


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


def _make_phase1_data_cell(config: dict) -> str:
    """Phase 1: download SponsorBlock CSV and run data_pipeline.py."""
    n_videos = config.get("n_videos", 300)
    skip_audio = config.get("skip_audio", False)
    skip_flag = "--skip-audio" if skip_audio else ""
    return (
        "import subprocess, sys, urllib.request\n"
        "from pathlib import Path\n\n"
        "# Download SponsorBlock CSV (cached across runs).\n"
        "_CSV_URL = 'https://sponsor.ajay.app/database/sponsorTimes.csv'\n"
        "_CSV_PATH = Path('/kaggle/working/sponsorTimes.csv')\n"
        "_CACHE_DIR = Path('/kaggle/working/embeddings_cache')\n"
        "_CACHE_DIR.mkdir(parents=True, exist_ok=True)\n\n"
        "if not _CSV_PATH.exists():\n"
        "    print('Downloading SponsorBlock CSV (~2-4 GB)…')\n"
        "    urllib.request.urlretrieve(_CSV_URL, _CSV_PATH)\n"
        "    print(f'Downloaded: {_CSV_PATH.stat().st_size:,} bytes')\n"
        "else:\n"
        "    print(f'CSV already present: {_CSV_PATH.stat().st_size:,} bytes')\n\n"
        "# Run data pipeline.\n"
        "result = subprocess.run(\n"
        "    [\n"
        "        sys.executable, '/tmp/yt_sponsor_src/src/data_pipeline.py',\n"
        "        '--csv', str(_CSV_PATH),\n"
        "        '--out', str(_CACHE_DIR),\n"
        f"       '--videos', '{n_videos}',\n"
        "        '--device', 'cuda' if __import__('torch').cuda.is_available() else 'cpu',\n"
        + (f"        '{skip_flag}',\n" if skip_flag else "") +
        "    ],\n"
        "    check=True,\n"
        ")\n"
        "_cached = list(_CACHE_DIR.glob('*.npz'))\n"
        "print(f'Cache: {len(_cached)} videos')\n"
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


# ---------------------------------------------------------------------------
# Dataset mount cell (copy from /kaggle/input/<slug>/ into working dir)
# ---------------------------------------------------------------------------


def _make_mount_dataset_cell(slug: str, dest_dir: str) -> str:
    """Copy mounted Kaggle Dataset files into the kernel working directory."""
    return (
        "import shutil\n"
        "from pathlib import Path\n\n"
        f"_DATASET_INPUT = Path('/kaggle/input/{slug}')\n"
        f"_DEST = Path('{dest_dir}')\n"
        "_DEST.mkdir(parents=True, exist_ok=True)\n\n"
        "if not _DATASET_INPUT.exists():\n"
        "    raise RuntimeError(\n"
        f"        f'Dataset not mounted: {{_DATASET_INPUT}}. '\n"
        "        'Check dataset_sources in kernel-metadata.json.'\n"
        "    )\n\n"
        "for _f in _DATASET_INPUT.iterdir():\n"
        "    if _f.is_file() and not (_DEST / _f.name).exists():\n"
        "        shutil.copy2(_f, _DEST / _f.name)\n"
        "        print(f'  copied: {_f.name}')\n\n"
        "_files = list(_DEST.iterdir())\n"
        "print(f'Dataset ready: {len(_files)} files in {_DEST}')\n"
    )


# ---------------------------------------------------------------------------
# Notebook assembly
# ---------------------------------------------------------------------------


def build_notebook(
    config: dict,
    src_tarball: Path,
    phase: str,
    dataset_sources_slugs: list[str] | None = None,
) -> nbformat.NotebookNode:
    """Assemble the Kaggle training notebook for the given phase.

    Args:
        config:                 Phase config dict (will be embedded in the notebook).
        src_tarball:            Path to the bundled training/src/ tarball.
        phase:                  Phase name: "data" | "baseline" | "teacher" | "distill" | "export".
        dataset_sources_slugs:  List of Kaggle Dataset slugs mounted in the kernel.

    Returns:
        A nbformat.NotebookNode ready to be written to disk.
    """
    nb = new_notebook()
    cells = [
        new_code_cell(_CELL_SETUP),
        new_code_cell(_CELL_INSTALL_DEPS),
        new_code_cell(_make_install_src_cell(src_tarball)),
    ]

    # Mount datasets if needed.
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
        cells.append(new_code_cell(_make_phase1_data_cell(config)))
    elif phase in ("baseline", "teacher", "distill"):
        cells.append(new_code_cell(_make_phase_train_cell(config)))
    elif phase == "export":
        cells.append(new_code_cell(_make_phase5_export_cell(config)))
    else:
        raise ValueError(f"Unknown phase: {phase!r}")

    cells.append(new_code_cell(_CELL_COLLECT_OUTPUTS))

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
    if result.returncode != 0:
        raise RuntimeError(
            f"kaggle kernels push failed (exit {result.returncode}):\n"
            f"{result.stderr or result.stdout}"
        )
    meta = json.loads((folder / "kernel-metadata.json").read_text())
    ref = meta["id"]
    _log(f"Pushed → https://www.kaggle.com/code/{ref}")
    return ref


def wait_for_completion(kernel_ref: str) -> str:
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
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    extension_dir = _PROJECT_ROOT / "youtube-ml-sponsor-detector"

    for f in files:
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


def _get_dataset_sources(phase: str, username: str) -> list[str]:
    """Return the list of Kaggle Dataset references to mount for each phase."""
    emb_ref = f"{username}/{_EMBEDDINGS_DATASET_SLUG}"
    teacher_ref = f"{username}/{_TEACHER_CKPT_DATASET_SLUG}"

    return {
        "data": [],                          # phase 1 builds the cache; nothing to mount
        "baseline": [emb_ref],               # phase 2 reads embeddings
        "teacher": [emb_ref],                # phase 3 reads embeddings
        "distill": [emb_ref, teacher_ref],   # phase 4 reads embeddings + teacher ckpt
        "export": [teacher_ref],             # phase 5 reads student ckpt (same dataset as teacher_ref after phase 4 updates it)
    }.get(phase, [])


def _get_dataset_slugs(phase: str) -> list[str]:
    """Return only the slug part (for notebook mount cells)."""
    return {
        "data": [],
        "baseline": [_EMBEDDINGS_DATASET_SLUG],
        "teacher": [_EMBEDDINGS_DATASET_SLUG],
        "distill": [_EMBEDDINGS_DATASET_SLUG, _TEACHER_CKPT_DATASET_SLUG],
        "export": [_TEACHER_CKPT_DATASET_SLUG],
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
    try:
        nb = build_notebook(config, tar_path, phase=phase, dataset_sources_slugs=dataset_slugs)
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
) -> bool:
    """Execute the full bridge round-trip for one phase config.

    Args:
        config_path:          Phase config JSON.
        enable_gpu:           Whether to request a Kaggle GPU.
        kernel_slug:          Override kernel slug.
        no_overwrite:         Skip overwriting existing local outputs.
        upload_cache:         After phase 1 completes, upload the embedding cache
                              as a Kaggle Dataset so subsequent phases can mount it.
        upload_teacher_ckpt:  After phase 3 completes, upload teacher_best.pt
                              as a Kaggle Dataset for phase 4 to consume.
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

    dataset_sources = _get_dataset_sources(phase, username)
    dataset_slugs = _get_dataset_slugs(phase)

    tmp_dir = Path(tempfile.mkdtemp(prefix="yt_sponsor_kaggle_"))
    try:
        # 1. Bundle source.
        tar_path = tmp_dir / "yt_sponsor_src.tar.gz"
        _bundle_source(tar_path)

        # 2. Build notebook.
        nb = build_notebook(config, tar_path, phase=phase, dataset_sources_slugs=dataset_slugs)
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
        kernel_ref = push_kernel(tmp_dir)

        # 5. Wait.
        final_status = wait_for_completion(kernel_ref)
        _log(f"Kernel finished: {final_status}")

        if final_status != "complete":
            _log(f"Kernel did not complete — inspect at: https://www.kaggle.com/code/{kernel_ref}")
            return False

        # 6. Fetch outputs.
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        raw_output_dir = _TRAINING_ROOT / "kaggle_outputs" / f"{slug}_{timestamp}"
        files = fetch_outputs(kernel_ref, raw_output_dir)

        # 7. Route to training/outputs/<phase>/.
        local_output_dir = _TRAINING_ROOT / "outputs" / phase
        route_outputs(files, local_output_dir, no_overwrite=no_overwrite)

        # 8. Phase-specific post-processing: upload artifacts for next phase.
        if upload_cache and phase == "data":
            cache_dir = local_output_dir / "embeddings_cache"
            if not cache_dir.is_dir():
                # Kaggle output was flat — use raw_output_dir.
                cache_dir = raw_output_dir
            if any(cache_dir.glob("*.npz")):
                upload_embeddings_cache(cache_dir, username)
            else:
                _log("WARNING: no .npz files found in output — skipping cache upload.")

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
    args = p.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Config not found: {config_path}")
        sys.exit(1)

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
        )
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
