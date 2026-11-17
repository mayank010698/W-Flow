# Experiment E001 — CIFAR-10 benchmark protocol for W-Flow

**Status:** proposed

**Research question:**
Can W-Flow (drift loss with debiased-OT projection) be trained end-to-end on class-conditional CIFAR-10 32x32 in pixel space, with a pixel-space MAE as the drift feature encoder, under a fixed, reproducible FID protocol that later experiments (E003, E004) can reuse unchanged?

**Hypothesis and falsification criterion:**
- H1 (trainability): With a CIFAR-10 pixel MAE encoder, generator training is numerically stable (finite loss and grad norm throughout) and FID-50k at the end is clearly below the FID of the step-1 sanity checkpoint.
  *Falsified if* loss/grad norm go non-finite, or final FID-50k is not lower than the step-1 FID (measured on the same protocol).
- H2 (protocol determinism): Re-evaluating the same EMA checkpoint with the same eval seed reproduces FID-50k to within 0.05.
  *Falsified if* two evaluations of one checkpoint differ by more than 0.05.
- No claim about matching any published CIFAR-10 FID is made by this experiment.

**Baseline and controls:**
- Lower anchor: step-1 (sanity) generator, same eval protocol.
- Real-data reference: FID of CIFAR-10 *test* set (10k) against the train-set reference stats, same Inception pipeline. This gives the protocol's noise floor with fewer samples (not a target).
- Literature CIFAR-10 FIDs are not used as baselines until a source and its exact protocol are cited here.

**Independent variables:** none (single protocol-establishing run, seed 0). Later sweeps vary from this config.

**Success metrics and evaluation protocol:**
- FID-50k: 50,000 class-conditional samples, labels taken from the CIFAR-10 *train* split in order of a fixed shuffled pass (sampler epoch 0, so exactly 5,000 per class). Reference statistics: mu/sigma of 2048-d pool features of all 50,000 CIFAR-10 train images.
- Feature extractor: torch-fidelity `inception-v3-compat` (TF-Inception port). Input is uint8 [0,255] at 32x32; torch-fidelity resizes internally to 299. Same code path (`utils/fid_util._extract_inception_features`) for reference stats and samples.
- Inception Score is logged as a secondary metric. CFG scales evaluated: [1.0, 1.5, 2.0]; report best CFG plus FID at CFG=1.0.
- Eval generator seed: `rng_eval=0`. The EMA model is evaluated.
- MAE encoder quality (secondary): masked-recon MSE on the test split, plus fc-head accuracy if the cls finetune phase is enabled.

**Dataset/version/split/preprocessing:**
- torchvision `CIFAR10` (python version, md5 verified by torchvision), root `$DATA_ROOT/cifar10`.
- Generator training: train split (50k), pixels scaled to [-1,1], channel-last, random horizontal flip.
- MAE training: train split with random crop (pad 4, reflect) and horizontal flip; MAE evaluation uses the test split, no augmentation.
- FID reference: train split, no augmentation, uint8.

**Feature encoder/checkpoint:**
- Pixel-space MAE-ResNet (`models/mae_model.MAEResNetJAX`, in_channels=3, input_patch_size=1), trained by `train_mae.py` on CIFAR-10 train with config `configs/mae/cifar10_pixel.yaml`. The trained EMA artifact is promoted to `$MODEL_ROOT/encoders/cifar10_pixel_mae_v1/` and referenced by the generator config. Its run dir, SHA and W&B ID are recorded below.
- Deviation from upstream: upstream used a latent-space MAE on SD-VAE latents of ImageNet-256. Here there is no VAE and no ImageNet pretraining.

**Model and resolved config:**
- Generator: `configs/gen/cifar10_pixel.yaml` (DiT-S-like: hidden 384, depth 12, heads 6, patch 2, 32x32x3). Debiased-OT drift loss, same OT kwargs as `ablation_ot_1node.yaml`.
- The resolved config is written to `<run_dir>/resolved_config.yaml` and to W&B config at runtime.

**Random seeds:** MAE train seed 0; generator train seed 0; eval rng 0.

**Git branch/worktree/SHA/dirty:** branch `exp/e001-cifar10-protocol`, worktree `/projects/illinois/eng/cs/arindamb/mayanks4/W-Flow-worktrees/e001-cifar10-protocol`. Pin SHA before full runs: TBD.

**Environment/hardware:** conda env `wflow` (`/projects/illinois/eng/cs/arindamb/mayanks4/envs/wflow`; Python 3.10.21, torch 2.5.1+cu124). Target GPU 1x A100 (`dali` or `secondary`, `--gres=gpu:A100:1`) so that step times are comparable. Per-run `pip freeze` goes to `<run_dir>/env_freeze.txt`.

**Budget (proposed, needs user approval):**
- Smoke Slurm job: 1x A100, ≤30 min, `secondary`.
- FID reference stats: 1x A100, ≤30 min (can share the smoke job).
- MAE pretraining: 1x A100, ≤4 h, `secondary`.
- Generator: 1x A100. Walltime to be set from the smoke-measured step time; `dali` (≤3 days) or `scavenger` with resume.

**W&B project/group/run IDs:** project `NWGF`, group `E001`. Run IDs TBD.

**Slurm job IDs and external log paths:** logs in `$SLURM_LOG_ROOT/%x-%j.{out,err}`. IDs TBD.

**Output/checkpoint paths:** `$OUTPUT_ROOT/E001-cifar10-protocol/<run_name>-<timestamp>-<jobid>/` containing `checkpoints/`, `params_ema/`, `resolved_config.yaml`, `run_meta.json`, `env_freeze.txt`, `wandb/`.

**Smoke test command and outcome:** TBD.

**Results (verified):** none yet.

**Interpretation (hypotheses):** none yet.

**Failed attempts / limitations:**
- No pretrained CIFAR encoder exists in the repo, so encoder quality is itself a variable here. FID depends on the encoder and is not comparable across encoder versions.
- `torch.compile` and DataLoader worker nondeterminism mean training is not bitwise reproducible, only reproducible given the config and seed.

**Next steps:** implement code (env-driven paths, CIFAR loader, MAE trainer, FID stats, run metadata, Slurm script), then run the smoke test.
