"""Storage paths, resolved from the environment (see `.env` / `scripts/common.sh`).

Harness policy: datasets, weights and outputs live outside the Git worktree.
Every path can be overridden by an env var of the same name; otherwise it is
derived from DATA_ROOT / MODEL_ROOT. If those roots are unset, derived paths
are empty strings and the code that needs them fails with a clear error.
"""
from __future__ import annotations

import os

DATA_ROOT = os.environ.get("DATA_ROOT", "")
MODEL_ROOT = os.environ.get("MODEL_ROOT", "")
OUTPUT_ROOT = os.environ.get("OUTPUT_ROOT", "")


def _path(name: str, root: str, *parts: str) -> str:
    override = os.environ.get(name)
    if override:
        return override
    return os.path.join(root, *parts) if root else ""


def require(path: str, name: str) -> str:
    if not path:
        raise RuntimeError(
            f"{name} is not configured. Set it (or DATA_ROOT/MODEL_ROOT) in .env and "
            "`source scripts/common.sh` before running."
        )
    return path


HF_REPO_ID = "Goodeat/drifting"
HF_ROOT = _path("HF_ROOT", MODEL_ROOT, "drifting_hf")
VAE_HF_PATH = _path("VAE_HF_PATH", MODEL_ROOT, "sdvae_hf")
TORCH_HUB_DIR = _path("TORCH_HUB_DIR", MODEL_ROOT, "torch_hub")

WFLOW_HF_REPO_ID = "jiaqihan99/W-Flow"
WFLOW_HF_ROOT = _path("WFLOW_HF_ROOT", MODEL_ROOT, "wflow_hf")

IMAGENET_PATH = _path("IMAGENET_PATH", DATA_ROOT, "imagenet-1k")
IMAGENET_CACHE_PATH = _path("IMAGENET_CACHE_PATH", DATA_ROOT, "imagenet256-latents-sdvae")
IMAGENET_FID_NPZ = _path("IMAGENET_FID_NPZ", DATA_ROOT, "fid_stats", "jit_in256_stats.npz")
IMAGENET_PR_NPZ = _path("IMAGENET_PR_NPZ", DATA_ROOT, "fid_stats", "imagenet_val_prc_arr0.npz")

CIFAR10_PATH = _path("CIFAR10_PATH", DATA_ROOT, "cifar10")
