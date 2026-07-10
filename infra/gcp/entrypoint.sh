#!/usr/bin/env bash
# Ephemeral entrypoint for Cloud Run Jobs. Two modes:
#
#   MODE=train       one-shot single-model training (Pub/Sub-dispatched job)
#   MODE=sweep-task  ONE gate item per task — the parallel sweep job fans the
#                    catalog across tasks via CLOUD_RUN_TASK_INDEX; each task
#                    trains + gates its item and uploads a report shard to
#                    gs://$MN_SWEEP_BUCKET/$MN_SWEEP_PREFIX/sweeps/$SWEEP_ID/
#
# The exit code is what tells Cloud Run the task succeeded; a non-zero from
# the CLI propagates so max_retries can retry. Metrics/artifacts persist to
# the global MLflow tracker via the Logger's env-gated sink — no background
# sync loop, no callback to the API. Auth is ADC via the runtime service
# account; no bearer tokens, no JSON keys.
set -euo pipefail

MODE="${MODE:-train}"
TRAINING_TIER="${TRAINING_TIER:-M}"
DEVICE="${DEVICE:-cpu}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/tmp/runs}"

echo "=== mini_networks ${MODE} ==="
echo "tier=${TRAINING_TIER} device=${DEVICE}"
echo "mlflow=${MN_MLFLOW_TRACKING_URI:-<unset>} register=${MN_MLFLOW_REGISTER:-<unset>}"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" || true

case "$MODE" in
  train)
    MODEL="${MODEL:?MODEL env var is required for MODE=train}"
    RUN_NAME="${RUN_NAME:-${MODEL}-$(date -u +%Y%m%d-%H%M%S)}"
    HPARAMS="${HPARAMS:-}"   # default empty; the JSON parse below treats empty as {}
    echo "model=${MODEL} run=${RUN_NAME}"

    # Build CLI flags. Pull recognized knobs out of the HPARAMS JSON; the rest
    # are still captured as MLflow params by the sink (via the model config).
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
    ;;

  sweep-task)
    echo "sweep=${SWEEP_ID:-adhoc} task=${CLOUD_RUN_TASK_INDEX:-0} items=${ITEMS:-<full catalog>}"
    # TRAINING_TIER / DEVICE / EPOCHS / BATCH_SIZE / ITEMS / SWEEP_ID /
    # MN_SWEEP_BUCKET are read from env by the CLI itself.
    exec python main.py sweep-task \
      --data_root /tmp/data --checkpoint_root "$CHECKPOINT_ROOT"
    ;;

  *)
    echo "Unknown MODE='$MODE' (train | sweep-task)" >&2
    exit 1
    ;;
esac
