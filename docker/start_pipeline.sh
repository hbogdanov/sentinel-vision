#!/bin/sh
set -eu

CONFIG_PATH="${SENTINEL_CONFIG:-configs/default.yaml}"

set -- python -m src.main --config "$CONFIG_PATH"

if [ -n "${SENTINEL_PROFILE:-}" ]; then
  set -- "$@" --profile "$SENTINEL_PROFILE"
fi

if [ -n "${SENTINEL_SOURCE:-}" ]; then
  set -- "$@" --source "$SENTINEL_SOURCE"
fi

if [ -n "${SENTINEL_MODEL_PATH:-}" ]; then
  set -- "$@" --model "$SENTINEL_MODEL_PATH"
fi

if [ -n "${SENTINEL_DEVICE:-}" ]; then
  set -- "$@" --device "$SENTINEL_DEVICE"
fi

exec "$@"
