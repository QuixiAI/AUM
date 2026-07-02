#!/usr/bin/env bash
# Launch AUM-Ø training with Weights & Biases reporting enabled.
#
#   ./train/launch.sh                          # defaults: 1 epoch over train/data, project aum-ssm
#   RUN_NAME=my-run ./train/launch.sh          # name the run (default: aum-tiny-v6-<timestamp>)
#   ./train/launch.sh --batch-size 4 --grad-accum 4 --mixed-precision bf16   # extra args pass through
#
# Any argument accepted by train/train.py can be appended (see python train/train.py --help).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Prefer the repo venv; fall back to whatever python is active.
PY="$REPO_ROOT/.venv/bin/python"
[ -x "$PY" ] || PY="$(command -v python)"

RUN_NAME="${RUN_NAME:-aum-tiny-v6-$(date +%Y%m%d-%H%M)}"
WANDB_PROJECT="${WANDB_PROJECT:-aum-ssm}"

# Fail fast on a missing wandb login instead of mid-startup (WANDB_API_KEY / prior
# `wandb login` / offline mode all count).
if [ "${WANDB_MODE:-}" != "offline" ] && [ -z "${WANDB_API_KEY:-}" ] \
        && ! grep -qs "api.wandb.ai" "$HOME/.netrc" 2>/dev/null; then
    echo "wandb is not logged in. Run 'wandb login' once (or export WANDB_API_KEY," >&2
    echo "or set WANDB_MODE=offline to sync later)." >&2
    exit 1
fi

echo "run: $RUN_NAME  project: $WANDB_PROJECT  python: $PY"
exec "$PY" train/train.py \
    --wandb \
    --wandb-project "$WANDB_PROJECT" \
    --run-name "$RUN_NAME" \
    "$@"
