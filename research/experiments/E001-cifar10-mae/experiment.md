# Experiment E001 — Pixel-space MAE pretraining on CIFAR-10

**Status:** full MAE run completed 2026-11-17 (Slurm 12522363). H1 supported. H2 not yet tested. Encoder promoted to `$MODEL_ROOT/encoders/cifar10_pixel_mae_v1/` (2026-11-18).

**Scope:** train the CIFAR-10 pixel-space MAE only. Generator training and FID are out of scope (user decision, 2026-11-17). They are tracked separately in `research/PLAN.md`.

**Research question:**
Can the repo's MAE-ResNet (`models/mae_model.MAEResNetJAX`), trained with the upstream masked-reconstruction recipe adapted to 32x32 RGB pixels, be pretrained stably on CIFAR-10 to give a reusable, versioned encoder artifact?

**Hypothesis and falsification criterion:**
- H1 (learns): the final EMA model's masked-reconstruction MSE on the CIFAR-10 test split is clearly below the zero-predictor baseline on the identical masks. *Falsified if* final `eval/recon_loss` ≥ `eval/recon_loss_zero_baseline`, or training goes non-finite (the trainer aborts on a non-finite loss).
- H2 (eval determinism): eval masks come from a fixed RNG, so re-evaluating the same checkpoint gives the same `eval/recon_loss` (up to bf16 kernel nondeterminism). *Falsified if* two evaluations of one checkpoint differ by more than 1e-4.
- No threshold beyond "below baseline" is set a priori. A target can be added here before the full run if wanted.

**Baseline and controls:**
- Zero predictor: predict 0 (the pixel midpoint in [-1,1]) for every masked pixel. Computed on exactly the same masks inside each eval (`eval/recon_loss_zero_baseline`).
- Initial model: the eval at the first eval step serves as the early-training reference.

**Metrics and evaluation protocol:**
- Primary: `eval/recon_loss`, the masked-pixel MSE of the EMA model on all 10,000 test images. Mask ratio 0.5, 2x2-pixel patches, fixed mask RNG (seed 0 + rank), no augmentation. Evaluated every 2,000 steps and at the end.
- Logged: train loss / recon loss / grad norm / lr / step time. `eval/accuracy` is logged but meaningless while `lambda_cls=0`.

**Dataset/version/split/preprocessing:**
- Source: Hugging Face `uoft-cs/cifar10` @ `0b2714987fa478483af9968de7c934580d0bb9a2` (PNG, lossless). The UToronto tarball ran at ~150 KB/s from the cluster and was abandoned (user-approved switch to HF, 2026-11-17).
- Converted by `scripts/prepare_cifar10.py convert` to `$DATA_ROOT/cifar10/cifar10_{train,test}.npz` (uint8 NHWC). Verified: train 50,000 (5,000/class), test 10,000 (1,000/class). `SOURCE.json` holds parquet/npz sha256s (train npz `4a7ff56d…478e`, train pixel data `c5b8e1d0…6807`).
- Train: RandomCrop(32, pad 4, reflect) + RandomHorizontalFlip, scaled to [-1,1], channel-last. Test: no augmentation.

**Model and resolved config:** `configs/mae/cifar10_pixel.yaml`:
- Model: MAE-ResNet with base 64 and layers [2,2,2,2]; feature maps run from 32x32x64 down to 4x4x512. UNet decoder, mask patch 2.
- Training: AdamW at lr 1e-3, constant after 2k warmup steps, wd 0.01, betas (0.9, 0.95). Batch 512 for 40k steps (~410 epochs). EMA 0.9995, grad clip 1.0, bf16 autocast with fp32 params, no cls finetune.
- Deviations from upstream `latent_ablation_256.yaml`: pixels instead of SD-VAE latents, a smaller ResNet (base 64 vs 256, [2,2,2,2] vs [3,4,6,3]), batch 512 vs 8192, lr 1e-3 vs 4e-3, 40k vs 30k steps.
- The resolved config is written to `<run_dir>/resolved_config.yaml` and the W&B config.

**Random seeds:** train seed 0; eval mask seed 0.

**Git branch/worktree/SHA/dirty:** branch `exp/e001-cifar10-protocol`, worktree `/projects/illinois/eng/cs/arindamb/mayanks4/W-Flow-worktrees/e001-cifar10-protocol`. Full-run SHA: `f6e6dcef4344e028a2e63975841f0328caf36b55`, clean tree (`git_dirty: false` in `run_meta.json`).

**Environment/hardware:** conda env `wflow` (Python 3.10.21, torch 2.5.1+cu124). 1x A100 (`--gres=gpu:A100:1`). Per-run `pip freeze` goes to `<run_dir>/env_freeze.txt`.

**Budget (proposed, needs user approval):**
- GPU smoke job: 1x A100, ≤20 min, `secondary`.
- Full MAE run: 1x A100. Walltime set from the smoke-measured step time; expected to fit `secondary`'s 4 h limit.

**W&B project/group/run IDs:** project `NWGF` (entity `ml-climate`), group `E001`. Full run: `fe50c51bbec3d566`, https://wandb.ai/ml-climate/NWGF/runs/fe50c51bbec3d566 (synced).

**Slurm job IDs and external log paths:** `$SLURM_LOG_ROOT/%x-%j.{out,err}`.
- Full run: job `12522363`. Submitted 2026-11-17 15:15 with `sbatch ... -p secondary --time=04:00:00 slurm/e001.sbatch`. Accounting shows it ran on partition `dali`, node `ccc0284`, 1x A100 80GB PCIe, from 19:30:44 to 22:53:27 (3:22:43), `COMPLETED` with exit 0:0, MaxRSS ~3.7 GB. Logs: `$SLURM_LOG_ROOT/e001-12522363.{out,err}`.

**Output/checkpoint paths:** `$OUTPUT_ROOT/E001-cifar10-mae/<run_name>-<timestamp>-<jobid>/` containing `checkpoints/`, `params_ema/` (the encoder artifact), `resolved_config.yaml`, `run_meta.json`, `env_freeze.txt`, `wandb/`. After the run, the final `params_ema/` is copied to `$MODEL_ROOT/encoders/cifar10_pixel_mae_v1/` with its SHA and W&B ID recorded here.

**Smoke test command and outcome:**
- 2026-11-17, CPU on the login node, uncommitted tree on top of `fce8109`: `SMOKE_MODE=cpu bash scripts/e001_smoke.sh`, run dir `$OUTPUT_ROOT/E001-cifar10-mae/smoke-cpu-20261117-145844-local`.
  - Steps 1–4 passed: data loads; 20 tiny-model steps; evals at steps 10/20; checkpoints and EMA artifact saved.
  - Resume from step 10 worked. Step-20 eval was 1.12706 after resume vs 1.12702 without, so resume is not bitwise identical (data/augmentation RNG is not replayed).
  - Artifact reload gave 36 finite feature keys. The zero baseline was identical across all evals (0.763084), so eval masks are fixed.
  - Step 5 FAILED due to a bug in the check itself: the hygiene check flagged my uncommitted source edits. Fixed to diff the worktree state before and after the run. Tiny-model loss values are not results.

**Results (verified):** full run `mae-s0-20261117-193057-12522363`. Seed 0, SHA `f6e6dce`, config `configs/mae/cifar10_pixel.yaml` with no overrides.
- Run dir: `$OUTPUT_ROOT/E001-cifar10-mae/mae-s0-20261117-193057-12522363/`.
  - Checkpoints at steps 20k, 35k and 40k.
  - `params_ema/ema_params.pt`: step 40000, decay 0.9995.
  - Also `resolved_config.yaml`, `run_meta.json` and `env_freeze.txt`.
- All 40,000 of 40,000 steps completed at a steady ~3.33 it/s, and stdout ends with `finished task=mae`.
- EMA eval on all 10,000 test images (mask ratio 0.5, fixed masks):

  | step | eval/recon_loss | zero baseline |
  |---|---|---|
  | 2,000 | 0.09712 | 0.76442 |
  | 10,000 | 0.05188 | 0.76442 |
  | 20,000 | 0.04905 | 0.76442 |
  | 30,000 | 0.04807 | 0.76442 |
  | 40,000 | **0.04746** | 0.76442 |

- Final train summary: loss 0.04124, g_norm 0.046, lr 1e-3.
- `eval/accuracy` stayed between 0.069 and 0.091 (final 0.0773). That is at or below chance, as expected with `lambda_cls=0`, and is not a result.

**Interpretation (hypotheses):**
- H1 is supported: the final recon MSE of 0.0475 is about 16x below the zero baseline (0.764) on identical masks, and no non-finite loss occurred.
- H2 is untested; it needs two evaluations of the same checkpoint. The zero baseline was identical in all 20 evals, which suggests fixed masks but does not test model-eval determinism.
- The loss was still falling slowly at 40k steps (-0.0001 per 2k steps over the last 10k). It is near a plateau but not converged. More steps would probably give only marginal gains. This is a conjecture.
- Whether this encoder gives good drift features remains to be tested downstream (E006).

**Failed attempts / limitations:**
- Training is not bitwise reproducible (bf16 kernels, DataLoader workers); it is reproducible given the config and seed.
- No separate GPU smoke job ran; Slurm accounting shows only 12522363. The full run itself served as the first GPU run.
- Single seed (0).
- Reconstruction MSE alone does not show the encoder is good for drift features. That is evaluated downstream, out of scope here.

**Next steps:**
1. Copy the final `params_ema/` to `$MODEL_ROOT/encoders/cifar10_pixel_mae_v1/` and record its sha256 here.
2. Optional: re-evaluate the step-40k checkpoint twice to test H2.
3. Start E006 (generator benchmark + FID) using this encoder.

**Encoder promotion (2026-11-18 00:07):** `params_ema/` at step 40000 copied to `$MODEL_ROOT/encoders/cifar10_pixel_mae_v1/` together with `PROVENANCE.json` (source run, Slurm 12522363, W&B fe50c51bbec3d566, SHA f6e6dce), `SHA256SUMS`, `resolved_config.yaml`, `run_meta.json`. `sha256sum -c SHA256SUMS` verified OK on 2026-11-18. Used by E005 via `feature.mae_path`.
