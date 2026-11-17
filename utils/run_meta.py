"""Reproducibility metadata for a run (harness: .claude/rules/reproducibility.md).

Collects Git SHA/dirty flag, command, Slurm IDs, hardware and package versions,
guards against writing outputs inside the Git repo, and writes
`run_meta.json` + `resolved_config.yaml` into the run directory.
"""
from __future__ import annotations

import json
import os
import platform
import socket
import subprocess
import sys
import time
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Dict

_PACKAGES = ("torch", "torchvision", "numpy", "scipy", "wandb", "torch-fidelity", "einops", "diffusers", "transformers")
REPO_ROOT = Path(__file__).resolve().parents[1]


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(REPO_ROOT), *args], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


def _git_worktrees() -> list[Path]:
    out = _git("worktree", "list", "--porcelain")
    paths = [Path(line[len("worktree "):]).resolve() for line in out.splitlines() if line.startswith("worktree ")]
    return paths or [REPO_ROOT]


def assert_outside_repo(path: str | os.PathLike, what: str = "workdir") -> Path:
    p = Path(path).resolve()
    for wt in _git_worktrees():
        if p == wt or wt in p.parents:
            raise ValueError(f"{what}={p} is inside Git worktree {wt}; use $OUTPUT_ROOT (harness storage policy).")
    return p


def collect_run_meta(config_path: str = "") -> Dict[str, Any]:
    import torch

    versions = {}
    for name in _PACKAGES:
        try:
            versions[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            versions[name] = None
    gpus = []
    if torch.cuda.is_available():
        gpus = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    return {
        "git_sha": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
        "repo_root": str(REPO_ROOT),
        "command": " ".join([sys.executable, *sys.argv]),
        "config_path": config_path,
        "start_time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hostname": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_partition": os.environ.get("SLURM_JOB_PARTITION"),
        "slurm_nodelist": os.environ.get("SLURM_JOB_NODELIST"),
        "world_size": int(os.environ.get("WORLD_SIZE", "1")),
        "python": platform.python_version(),
        "cuda_build": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "gpus": gpus,
        "packages": versions,
        "env": {k: os.environ.get(k) for k in ("DATA_ROOT", "MODEL_ROOT", "OUTPUT_ROOT", "WANDB_PROJECT", "WANDB_RUN_GROUP", "WANDB_MODE", "DRIFT_COMPILE", "CONDA_DEFAULT_ENV")},
    }


def write_run_files(workdir: str | os.PathLike, config: Any, meta: Dict[str, Any]) -> None:
    import yaml

    wd = Path(workdir)
    wd.mkdir(parents=True, exist_ok=True)

    def _plain(x):
        if isinstance(x, dict):
            return {k: _plain(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [_plain(v) for v in x]
        return x

    (wd / "resolved_config.yaml").write_text(yaml.safe_dump(_plain(config), sort_keys=False), encoding="utf-8")
    meta_path = wd / "run_meta.json"
    # Keep a history across resumes instead of overwriting the first launch's metadata.
    history = []
    if meta_path.is_file():
        try:
            prev = json.loads(meta_path.read_text(encoding="utf-8"))
            history = prev.get("history", []) + [{k: v for k, v in prev.items() if k != "history"}]
        except Exception:
            pass
    meta_path.write_text(json.dumps({**meta, "history": history}, indent=2) + "\n", encoding="utf-8")


def update_run_meta(workdir: str | os.PathLike, **fields: Any) -> None:
    meta_path = Path(workdir) / "run_meta.json"
    data = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    data.update(fields)
    meta_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
