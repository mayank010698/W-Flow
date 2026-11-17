#!/usr/bin/env bash
# E001 CIFAR-10 pixel-MAE runner. Submit ONLY via scripts/submit-slurm.sh from the worktree root, e.g.
#   TASK=smoke EXPECTED_SHA=$(git rev-parse HEAD) \
#     bash scripts/submit-slurm.sh slurm/e001.sbatch -p secondary --time=00:45:00
#   TASK=mae CONFIG=configs/mae/cifar10_pixel.yaml RUN_NAME=mae-s0 EXPECTED_SHA=... \
#     bash scripts/submit-slurm.sh slurm/e001.sbatch -p secondary --time=04:00:00
# --output/--error are injected by submit-slurm.sh into $SLURM_LOG_ROOT. Partition and
# walltime are chosen at submission (approved by the user), not hard-coded here.
#SBATCH --job-name=e001
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?must run under Slurm}"
[[ -f scripts/common.sh ]] || { echo "Submit from the worktree root" >&2; exit 2; }
source scripts/common.sh
: "${TASK:?TASK=smoke|mae}" "${EXPECTED_SHA:?EXPECTED_SHA=\$(git rev-parse HEAD) at submission}"

# Code is read at runtime: refuse to run if HEAD moved or tracked files changed since submission.
sha=$(git rev-parse HEAD)
[[ "$sha" == "$EXPECTED_SHA" ]] || { echo "HEAD $sha != EXPECTED_SHA $EXPECTED_SHA" >&2; exit 3; }
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || { echo "Dirty worktree" >&2; exit 3; }

module load anaconda3/2024.06
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-wflow}"
export PYTHON=python
export WANDB_RUN_GROUP=${WANDB_RUN_GROUP:-E001}
export HF_HOME=${HF_HOME:-$MODEL_ROOT/hf_home}
export NCCL_DEBUG=WARN
if command -v gcc >/dev/null; then export CC=$(command -v gcc) CXX=$(command -v g++); fi

echo "job=${SLURM_JOB_ID} partition=${SLURM_JOB_PARTITION} node=$(hostname) task=$TASK sha=$sha env=$CONDA_DEFAULT_ENV"
nvidia-smi -L

case "$TASK" in
  smoke)     SMOKE_MODE=gpu bash scripts/e001_smoke.sh ;;
  mae)
    : "${CONFIG:?}" "${RUN_NAME:?}"
    entry=train_mae.py
    # New unique dir per launch; pass RUN_DIR=<existing dir> to resume (same W&B run id).
    RUN_DIR=${RUN_DIR:-$OUTPUT_ROOT/E001-cifar10-mae/${RUN_NAME}-$(date +%Y%m%d-%H%M%S)-${SLURM_JOB_ID}}
    mkdir -p "$RUN_DIR"
    pip freeze > "$RUN_DIR/env_freeze.txt"
    read -ra OV <<< "${OVERRIDES:-}"
    echo "run_dir=$RUN_DIR config=$CONFIG overrides=${OV[*]:-}"
    python "$entry" --config "$CONFIG" --workdir "$RUN_DIR" --override "${OV[@]}"
    ;;
  *) echo "Unknown TASK=$TASK" >&2; exit 2 ;;
esac
echo "finished task=$TASK"
