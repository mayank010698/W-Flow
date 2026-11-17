# Research decisions (append-only)

Use the format:

## YYYY-MM-DD — Decision title
**Context:** ...
**Decision:** ...
**Alternatives:** ...
**Evidence:** ...
**Consequences / revisit trigger:** ...

## 2026-11-17 — CIFAR-10 feature encoder: pixel-space MAE (option a)
**Context:** Upstream W-Flow uses a latent MAE (`hf://mae_latent_256`) on 32x32x4 SD-VAE latents; it does not apply to 32x32 RGB CIFAR-10. Repo ships MAE configs but no MAE training script.
**Decision:** Use a pixel-space MAE trained (or obtained) for CIFAR-10 as the drift feature encoder for E001.
**Alternatives:** (b) ImageNet ConvNeXt-V2 features on upsampled images (no new code, different protocol); (c) raw pixels / `global` feature only.
**Evidence:** Code audit 2026-11-17 (`models/mae_model.py`, `configs/mae/*`, `train.py`).
**Consequences / revisit trigger:** Requires an MAE training entry point + CIFAR pixel config; encoder checkpoint must be versioned in `MODEL_ROOT` and recorded in every E001 run. Revisit if MAE pretraining cost dominates the budget.

## 2026-11-17 — Cluster and environment
**Context:** Harness requires explicit partition/environment choices.
**Decision:** Allowed Slurm partitions: `dali`, `scavenger`, `secondary`. Conda env: user named `w-flow`; the existing env on disk is `wflow` (`/projects/illinois/eng/cs/arindamb/mayanks4/envs/wflow`) — pending user confirmation.
**Alternatives:** Other partitions (eng-research-gpu, IllinoisComputes-GPU) not authorized.
**Evidence:** `sinfo -s`, `conda env list` on 2026-11-17.
**Consequences / revisit trigger:** `scavenger` (1-day limit) is likely preemptible -> runs there must resume from checkpoints; `secondary` has a 4h limit.
