#!/usr/bin/env bash
# Ephemeral training entrypoint for Cloud Run Jobs (MODE=train).
#
# One-shot: it runs the train CLI and exits. The exit code is what tells Cloud
# Run the execution succeeded; a non-zero from the CLI propagates so max_retries
# can retry. Training status + metrics + artifacts are persisted to MLflow
# (Neon + GCS) via the Logger's env-gated sink — there is NO background sync loop
# and NO callback to the API. Auth to GCS/Neon is ADC via the runtime service
# account (metadata server); no bearer tokens, no JSON keys.
set -euo pipefail

MODE="${MODE:-train}"
MODEL="${MODEL:?MODEL env var is required}"
TRAINING_TIER="${TRAINING_TIER:-M}"
DEVICE="${DEVICE:-cpu}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/tmp/runs}"
RUN_NAME="${RUN_NAME:-${MODEL}-$(date -u +%Y%m%d-%H%M%S)}"
HPARAMS="${HPARAMS:-}"   # default empty; the JSON parse below treats empty as {}

echo "=== mini_networks train ==="
echo "model=${MODEL} tier=${TRAINING_TIER} device=${DEVICE} run=${RUN_NAME}"
echo "mlflow=${MN_MLFLOW_TRACKING_URI:-<unset>} artifacts=${MN_MLFLOW_ARTIFACT_ROOT:-<unset>}"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" || true

if [ "$MODE" != "train" ]; then
  echo "Unknown MODE='$MODE' (only 'train' is supported)" >&2
  exit 1
fi

# Build CLI flags. Pull recognized knobs out of the HPARAMS JSON; the rest are
# still captured as MLflow params by the sink (via the model config).
ARGS=(train --model "$MODEL" --training_tier "$TRAINING_TIER"
      --device "$DEVICE" --checkpoint_root "$CHECKPOINT_ROOT" --run_name "$RUN_NAME")

if [ -n "${EPOCHS:-}" ]; then
  ARGS+=(--epochs "$EPOCHS")
fi

BATCH_SIZE="$(python -c "import json,os; print(json.loads(os.environ.get('HPARAMS') or '{}').get('batch_size',''))")"
if [ -n "$BATCH_SIZE" ]; then
  ARGS+=(--batch_size "$BATCH_SIZE")
fi

EPOCHS_HP="$(python -c "import json,os; print(json.loads(os.environ.get('HPARAMS') or '{}').get('epochs',''))")"
if [ -z "${EPOCHS:-}" ] && [ -n "$EPOCHS_HP" ]; then
  ARGS+=(--epochs "$EPOCHS_HP")
fi

echo "+ python main.py ${ARGS[*]}"
exec python main.py "${ARGS[@]}"
