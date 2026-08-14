#!/usr/bin/env bash
set -euo pipefail
# Detached workers must survive an SSH/control-session hangup. SIGINT/SIGTERM
# are still handled by the Python runner for checkpoint-safe shutdown.
trap '' HUP

credential_file="${INFERKNOCK_API_KEY_FILE:-$HOME/.config/astrai/inferknock.env}"
if [ -z "${INFERKNOCK_API_KEY:-}" ] && [ -r "$credential_file" ]; then
  # The credential file contains only the raw token, is mode 0600, and is
  # never committed. Strip line endings so PowerShell-created input is safe.
  INFERKNOCK_API_KEY=$(tr -d '\r\n' <"$credential_file")
  export INFERKNOCK_API_KEY
fi
: "${INFERKNOCK_API_KEY:?missing API key environment variable and credential file}"

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

base=/mnt/nvme3/zbuser02/astrai-distill
state_base="${ASTRAI_DISTILL_STATE_DIR:-$base/state}"
run_id="${ASTRAI_DISTILL_RUN_ID:-posttrain-v4}"
cd /home/zbuser02/AstrAI-12b
log="$base/logs/${run_id}-${bucket}.log"
# Under systemd, keep stdout/stderr on the journal. Redirecting a long-lived
# service to the old sparse NVMe log caused flush failures (including Python
# exit status 120) and hid the real traceback. Manual/tmux runs still append
# to the legacy file.
if [ -z "${JOURNAL_STREAM:-}" ]; then
  exec >>"$log" 2>&1
fi

echo "DISTILL_LAUNCH $(date --iso-8601=seconds) bucket=$bucket concurrency=$concurrency max_tokens=$max_tokens stream=true"
python3 scripts/data/recover_retryable_distill.py \
  "$state_base/${run_id}-${bucket}.sqlite3"
exec python3 scripts/data/distill_glm.py \
  --input "$base/seeds/${run_id}-${bucket}.jsonl" \
  --state "$state_base/${run_id}-${bucket}.sqlite3" \
  --output "$base/output/${run_id}-${bucket}.jsonl" \
  --failed-output "$base/output/${run_id}-${bucket}.failed.jsonl" \
  --base-url https://api.inferknock.ai/v1 \
  --model glm-5.2 \
  --concurrency "$concurrency" \
  --max-tokens "$max_tokens" \
  --temperature 0.5 \
  --top-p 0.95 \
  --timeout 600 \
  --connect-timeout 20 \
  --max-attempts 3 \
  --retry-base 2 \
  --retry-max 300 \
  --outage-threshold "${ASTRAI_OUTAGE_THRESHOLD:-3}" \
  --outage-cooldown 120 \
  --require-content \
  --finalize-reasoning-only \
  --finalizer-max-tokens "$finalizer_max_tokens" \
  --retry-finish-reason length \
  --stream
