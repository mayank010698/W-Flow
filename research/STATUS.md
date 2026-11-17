# Current project status

Updated: 2026-11-17

## Active question
E001: establish a reproducible CIFAR-10 protocol for W-Flow (pixel-space MAE features, consistent FID).

## Latest verified results
None recorded yet.

## Current blockers
- No CIFAR-10 code path: loader, `num_classes` in memory bank (`train.py:509`), FID stats (`utils/fid_util.py:28`), pixel MAE encoder + trainer.
- `utils/env.py` and run scripts hard-code placeholder paths; default outputs (`runs/`, `results/`, `./log`, `./wandb`) land inside the repo.
- `scripts/train_slurm/*` hard-code `--output/--error` and placeholder partition/account.
- `.env` lacks `WANDB_PROJECT`; conda env name (`w-flow` vs existing `wflow`) to confirm.

## Next actions
1. Storage dirs created and validated (done 2026-11-17).
2. `/new-experiment` E001 on branch/worktree `exp/cifar10-protocol`: env-driven paths, CIFAR loader, pixel MAE, CIFAR FID stats, W&B SHA/Slurm metadata.
3. Smoke test, then short Slurm GPU job on `dali`/`secondary` after approval.
