#!/usr/bin/env bash
set -euo pipefail
trap '' HUP

if [ "$#" -ne 3 ]; then
  echo "usage: $0 BUCKET CONCURRENCY MAX_TOKENS" >&2
  exit 2
fi

bucket="$1"
concurrency="$2"
max_tokens="$3"
case "$bucket" in
  short) finalizer_max_tokens=8192 ;;
  medium) finalizer_max_tokens=16384 ;;
  long) finalizer_max_tokens=32768 ;;
  *) echo "invalid bucket: $bucket" >&2; exit 2 ;;
esac
if ! [[ "$concurrency" =~ ^[0-9]+$ ]] || (( concurrency < 1 || concurrency > 128 )); then
  echo "concurrency must be an integer in [1, 128]" >&2
  exit 2
fi

credential_file="${INFERKNOCK_DSV4_API_KEY_FILE:-$HOME/.config/astrai/inferknock-dsv4.env}"
if [ -z "${INFERKNOCK_DSV4_API_KEY:-}" ] && [ -r "$credential_file" ]; then
  mode=$(stat -c '%a' "$credential_file")
  if [ "$mode" != 600 ] && [ "$mode" != 400 ]; then
    echo "credential file must have mode 0600 or 0400" >&2
    exit 2
  fi
  INFERKNOCK_DSV4_API_KEY=$(tr -d '\r\n' <"$credential_file")
  export INFERKNOCK_DSV4_API_KEY
fi
: "${INFERKNOCK_DSV4_API_KEY:?missing DSV4 API key and credential file}"

base=/mnt/nvme3/zbuser02/astrai-distill
state_base="${ASTRAI_DISTILL_STATE_DIR:-$base/state}"
run_id="${ASTRAI_DSV4_RUN_ID:-posttrain-dsv4-v1}"
cd /home/zbuser02/AstrAI-12b
mkdir -p "$base/logs" "$base/output" "$state_base"
log="$base/logs/${run_id}-${bucket}.log"
if [ -z "${JOURNAL_STREAM:-}" ]; then
  exec >>"$log" 2>&1
fi

echo "DISTILL_LAUNCH $(date --iso-8601=seconds) teacher=deepseek-v4-flash bucket=$bucket concurrency=$concurrency max_tokens=$max_tokens stream=true"
python3 scripts/data/recover_retryable_distill.py \
  "$state_base/${run_id}-${bucket}.sqlite3"
exec python3 scripts/data/distill_glm.py \
  --input "$base/seeds/${run_id}-${bucket}.jsonl" \
  --state "$state_base/${run_id}-${bucket}.sqlite3" \
  --output "$base/output/${run_id}-${bucket}.jsonl" \
  --failed-output "$base/output/${run_id}-${bucket}.failed.jsonl" \
  --base-url https://api.inferknock.ai/v1 \
  --model deepseek-v4-flash \
  --api-key-env INFERKNOCK_DSV4_API_KEY \
  --concurrency "$concurrency" \
  --max-tokens "$max_tokens" \
  --temperature 0.5 \
  --top-p 0.95 \
  --timeout 600 \
  --connect-timeout 20 \
  --max-attempts 3 \
  --retry-base 2 \
  --retry-max 300 \
  --outage-threshold 3 \
  --outage-cooldown 120 \
  --require-content \
  --finalize-reasoning-only \
  --finalizer-max-tokens "$finalizer_max_tokens" \
  --retry-finish-reason length \
  --stream
