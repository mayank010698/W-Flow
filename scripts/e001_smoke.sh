#!/usr/bin/env bash
# E001 smoke test: pixel-space MAE pretraining on CIFAR-10 (tiny model).
# Checks: imports, data, train + eval + checkpoint/EMA artifact save, artifact reload as the
# drift feature encoder, mid-run resume, run metadata, and that nothing is written in the repo.
#   SMOKE_MODE=cpu (default): local CPU.
#   SMOKE_MODE=gpu: inside a Slurm GPU job; additionally runs the FULL MAE config for 300 steps
#                   to measure step time / memory for the walltime request.
set -euo pipefail
source "$(dirname "$0")/common.sh"
cd "$HARNESS_PROJECT_ROOT"
: "${OUTPUT_ROOT:?Set OUTPUT_ROOT}"
PY=${PYTHON:-python}
MODE=${SMOKE_MODE:-cpu}
D="$OUTPUT_ROOT/E001-cifar10-mae/smoke-${MODE}-$(date +%Y%m%d-%H%M%S)-${SLURM_JOB_ID:-local}"
mkdir -p "$D"
echo "smoke dir: $D"
export WANDB_MODE=${WANDB_MODE:-offline} WANDB_RUN_GROUP=E001-smoke
if [[ "$MODE" == cpu ]]; then export DRIFT_COMPILE=0; fi
fail() { echo "SMOKE FAIL: $*" >&2; exit 1; }
# Snapshot of worktree state (incl. ignored files) to detect anything this smoke writes into the repo.
repo_state() { git status --porcelain --ignored --untracked-files=all | grep -v '__pycache__' | sort; }
BEFORE=$(repo_state)
need() { [[ -e "$1" ]] || fail "missing $1"; }

echo "== [1] imports + environment + data"
$PY - <<'PY'
import os, torch, wandb, train_mae
from dataset.dataset import CIFAR10Arrays
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
for s in ("train", "test"):
    print(CIFAR10Arrays(os.environ["DATA_ROOT"] + "/cifar10", s))
PY

echo "== [2] MAE: 20 steps (cls-finetune phase in last 5), eval at 10/20, ckpt + EMA artifact at 10/20"
$PY train_mae.py --config configs/mae/cifar10_pixel_smoke.yaml --workdir "$D/mae" 2>&1 | tee "$D/mae.log"
for f in params_ema/ema_params.pt params_ema/metadata.json checkpoints/state_00000010.pt checkpoints/state_00000020.pt resolved_config.yaml run_meta.json; do need "$D/mae/$f"; done
grep -q "\[eval step 20\]" "$D/mae.log" || fail "no final eval"

echo "== [3] resume: drop step-20 ckpt, rerun -> must restore step 10 and re-create step 20"
rm "$D/mae/checkpoints/state_00000020.pt"
$PY train_mae.py --config configs/mae/cifar10_pixel_smoke.yaml --workdir "$D/mae" 2>&1 | tee "$D/mae_resume.log"
grep -q "Resumed MAE from step 10" "$D/mae_resume.log" || fail "resume path not taken"
need "$D/mae/checkpoints/state_00000020.pt"

echo "== [4] EMA artifact reloads via the drift feature-encoder loader; features finite"
$PY - "$D/mae" <<'PY'
import sys, torch
from models.mae_model import build_feature_model_and_params
m, _ = build_feature_model_and_params(path=sys.argv[1])
feats = m.get_activations(torch.randn(2, 32, 32, 3), patch_mean_size=[2, 4], patch_std_size=[2, 4], every_k_block=2)
bad = [k for k, v in feats.items() if not torch.isfinite(v).all()]
assert not bad, bad
print(f"{len(feats)} feature keys OK, layer4 {tuple(feats['layer4'].shape)}")
PY

echo "== [5] metadata + output hygiene"
$PY - "$D/mae" <<'PY'
import json, sys, pathlib
d = pathlib.Path(sys.argv[1])
m = json.loads((d / "run_meta.json").read_text())
assert m["git_sha"] and "git_dirty" in m and m.get("wandb_run_id"), m
assert any((d / "wandb").iterdir()), "wandb dir not inside run dir"
print("sha", m["git_sha"][:10], "dirty", m["git_dirty"], "wandb", m["wandb_run_id"], "resumes", len(m["history"]))
PY
stray=$(comm -13 <(echo "$BEFORE") <(repo_state) || true)
[[ -z "$stray" ]] || fail "files written inside the worktree during smoke: $stray"
echo "no files written inside the worktree"

if [[ "$MODE" == gpu ]]; then
  echo "== [6] timing: FULL MAE config, 300 steps (+1 test eval)"
  $PY train_mae.py --config configs/mae/cifar10_pixel.yaml --workdir "$D/timing_mae" \
    --override train.total_steps=300 train.eval_per_step=300 train.save_per_step=300
  nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
fi
echo "SMOKE OK ($MODE): $D"
