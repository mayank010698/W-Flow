# Research plan

## Current questions

- [ ] E001: Pixel-space MAE pretraining on CIFAR-10 (encoder only; scope narrowed 2026-11-17).
- [ ] E006: CIFAR-10 generator benchmark protocol (epochs, hyperparameters, consistent FID evaluation) using the E001 encoder. Split out of E001 on 2026-11-17; not started.
- [ ] E002: Slurm job scheduling and quick full-pipeline GPU smoke test.
- [ ] E003: Number of inner regression steps in W-Flow; FID and projection error, including one-step analysis.
- [ ] E004: Compare likelihood-based projection and paired regression (NFM-R / NFM-D inspired baselines).
- [ ] E005: Evaluate ImageNet-64 scaling and Boltzmann-generator feasibility.

Each question should have a hypothesis and baseline in its experiment record before full runs.
