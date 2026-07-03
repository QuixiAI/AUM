#!/usr/bin/env bash
# Launch AUM-Ø training with Weights & Biases reporting enabled.
#
#   ./train/launch.sh                          # defaults: 1 epoch over train/data, project aum-ssm
#   RUN_NAME=my-run ./train/launch.sh          # name the run (default: aum-tiny-v6-<timestamp>)
#   ./train/launch.sh --batch-size 4 --grad-accum 4   # extra args pass through
#
# Any argument accepted by train/train.py can be appended (see python train/train.py --help);
# appended flags OVERRIDE the script defaults below (argparse keeps the last occurrence).
#
# CUDA nodes: execs `accelerate launch --num_processes <all GPUs>` with bf16 and the §13 recipe
# batch shape — 8 ranks x micro-batch 4 x grad-accum 4 x 4096 = 0.5M tokens/step. Micro-batch 4
# runs at ~21.5GiB/24GiB and NEEDS the expandable-segments allocator (exported below): with the
# default allocator, fragmentation that close to the ceiling collapses throughput 3x (measured
# 28.7k vs 8.6k tok/s on the 8x3090 node). Override with NUM_GPUS / BATCH_SIZE / GRAD_ACCUM env
# vars (e.g. NUM_GPUS=1 for a single-GPU run).
#
# MPS (Mac) default batch shape: micro-batch 8 x grad-accum 2 = 65,536 tokens/step at seq 4096 —
# sized for the 128GB machine (measured: batch 2 left system memory 84% free). If it OOMs or
# swaps, pass --batch-size 4 --grad-accum 4.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Prefer the repo venv; fall back to whatever python is active.
PY="$REPO_ROOT/.venv/bin/python"
[ -x "$PY" ] || PY="$(command -v python)"

RUN_NAME="${RUN_NAME:-aum-tiny-v6-$(date +%Y%m%d-%H%M)}"
WANDB_PROJECT="${WANDB_PROJECT:-aum-ssm}"
if command -v nvidia-smi >/dev/null 2>&1 && [ -e /dev/nvidia0 ]; then
    CUDA=1
    NUM_GPUS="${NUM_GPUS:-$(nvidia-smi -L | wc -l)}"
    BATCH_SIZE="${BATCH_SIZE:-4}"
    GRAD_ACCUM="${GRAD_ACCUM:-4}"
    export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
else
    CUDA=0
    BATCH_SIZE="${BATCH_SIZE:-8}"
    GRAD_ACCUM="${GRAD_ACCUM:-2}"
fi

# Fail fast on a missing wandb login instead of mid-startup (WANDB_API_KEY / prior
# `wandb login` / offline mode all count).
if [ "${WANDB_MODE:-}" != "offline" ] && [ -z "${WANDB_API_KEY:-}" ] \
        && ! grep -qs "api.wandb.ai" "$HOME/.netrc" 2>/dev/null; then
    echo "wandb is not logged in. Run 'wandb login' once (or export WANDB_API_KEY," >&2
    echo "or set WANDB_MODE=offline to sync later)." >&2
    exit 1
fi

if [ "$CUDA" = 1 ]; then
    ACCELERATE="$(dirname "$PY")/accelerate"
    [ -x "$ACCELERATE" ] || ACCELERATE="$(command -v accelerate)"
    echo "run: $RUN_NAME  project: $WANDB_PROJECT  gpus: $NUM_GPUS  batch: ${BATCH_SIZE}x${GRAD_ACCUM}"
    exec "$ACCELERATE" launch --num_processes "$NUM_GPUS" train/train.py \
        --wandb \
        --wandb-project "$WANDB_PROJECT" \
        --run-name "$RUN_NAME" \
        --batch-size "$BATCH_SIZE" \
        --grad-accum "$GRAD_ACCUM" \
        --mixed-precision bf16 \
        "$@"
fi

echo "run: $RUN_NAME  project: $WANDB_PROJECT  batch: ${BATCH_SIZE}x${GRAD_ACCUM}  python: $PY"
exec "$PY" train/train.py \
    --wandb \
    --wandb-project "$WANDB_PROJECT" \
    --run-name "$RUN_NAME" \
    --batch-size "$BATCH_SIZE" \
    --grad-accum "$GRAD_ACCUM" \
    "$@"
