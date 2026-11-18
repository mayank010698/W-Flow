# Current project status

Updated: 2026-11-17

## Active question
- E001 (encoder only): pixel-space MAE pretraining on CIFAR-10. The full run is done.
- Next up: E002, a full-pipeline GPU smoke test and Slurm workflow.

## Latest verified results
- **E001 full MAE run**
  - IDs: Slurm `12522363`, W&B `fe50c51bbec3d566` (`ml-climate/NWGF`, group `E001`).
  - Code: SHA `f6e6dce` (clean tree), seed 0.
  - Hardware: 1x A100 80GB on `dali`/`ccc0284`, 3:22:43, exit 0.
  - Ran 40k/40k steps. Final EMA `eval/recon_loss` = 0.04746 vs zero baseline 0.76442, on 10k test images with mask ratio 0.5. H1 is supported.
  - Artifacts: `$OUTPUT_ROOT/E001-cifar10-mae/mae-s0-20261117-193057-12522363/`.
  - Details are in `research/experiments/E001-cifar10-mae/experiment.md`.

## Open items / blockers
- The E001 encoder has not been copied to `$MODEL_ROOT/encoders/cifar10_pixel_mae_v1/` yet.
- E001 H2 (eval determinism) is untested.
- Not re-verified since E001: whether the generator path (`train.py`, `utils/fid_util.py`, `scripts/train_slurm/*`) handles CIFAR-10, CIFAR FID stats and env-driven paths/Slurm logs. This is in scope for E002/E006.

## Next actions
1. Plan E002: full-pipeline GPU smoke test and Slurm workflow.
2. Promote the E001 encoder to `$MODEL_ROOT` with a sha256.
3. E006: generator benchmark and FID using the E001 encoder.
