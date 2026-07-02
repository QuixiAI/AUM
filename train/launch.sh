#!/usr/bin/env bash
# Launch AUM-Ø training with Weights & Biases reporting enabled.
#
#   ./train/launch.sh                          # defaults: 1 epoch over train/data, project aum-ssm
#   RUN_NAME=my-run ./train/launch.sh          # name the run (default: aum-tiny-v6-<timestamp>)
#   ./train/launch.sh --batch-size 4 --grad-accum 4 --mixed-precision bf16   # extra args pass through
#
# Any argument accepted by train/train.py can be appended (see python train/train.py --help);
# appended flags OVERRIDE the script defaults below (argparse keeps the last occurrence).
#
# Default batch shape: micro-batch 8 x grad-accum 2 = 65,536 tokens/step at seq 4096 — sized for
# this 128GB machine (measured: batch 2 left system memory 84% free). Same effective batch as
# train.py's own 2x8 defaults, so the schedule/LR math is unchanged; bigger micro-batches keep
# MPS busy with fewer kernel launches. If it OOMs or swaps, pass --batch-size 4 --grad-accum 4.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Prefer the repo venv; fall back to whatever python is active.
PY="$REPO_ROOT/.venv/bin/python"
[ -x "$PY" ] || PY="$(command -v python)"

RUN_NAME="${RUN_NAME:-aum-tiny-v6-$(date +%Y%m%d-%H%M)}"
WANDB_PROJECT="${WANDB_PROJECT:-aum-ssm}"
BATCH_SIZE="${BATCH_SIZE:-8}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"

# Fail fast on a missing wandb login instead of mid-startup (WANDB_API_KEY / prior
# `wandb login` / offline mode all count).
if [ "${WANDB_MODE:-}" != "offline" ] && [ -z "${WANDB_API_KEY:-}" ] \
        && ! grep -qs "api.wandb.ai" "$HOME/.netrc" 2>/dev/null; then
    echo "wandb is not logged in. Run 'wandb login' once (or export WANDB_API_KEY," >&2
    echo "or set WANDB_MODE=offline to sync later)." >&2
    exit 1
fi

echo "run: $RUN_NAME  project: $WANDB_PROJECT  batch: ${BATCH_SIZE}x${GRAD_ACCUM}  python: $PY"
exec "$PY" train/train.py \
    --wandb \
    --wandb-project "$WANDB_PROJECT" \
    --run-name "$RUN_NAME" \
    --batch-size "$BATCH_SIZE" \
    --grad-accum "$GRAD_ACCUM" \
    "$@"
