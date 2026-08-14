#!/usr/bin/env bash
set -euo pipefail

repo=/home/zbuser02/AstrAI-12b
runner="$repo/scripts/data/run_glm_posttrain_bucket.sh"
lock=/tmp/astrai-glm-posttrain-supervisor.lock
supervisor_log=/mnt/nvme3/zbuser02/astrai-distill/logs/posttrain-supervisor.log

exec 9>"$lock"
flock -n 9 || exit 0

ensure_bucket() {
  local bucket="$1" concurrency="$2" max_tokens="$3"
  local session="astral-glm-v4-$bucket"
  local state="/mnt/nvme3/zbuser02/astrai-distill/state/posttrain-v4-$bucket.sqlite3"

  if pgrep -f "distill_glm.py.*$state" >/dev/null; then
    return
  fi

  # Do not relaunch a fully terminal bucket.
  if python3 - "$state" <<'PY'
import sqlite3, sys
con = sqlite3.connect(f"file:{sys.argv[1]}?mode=ro", uri=True)
unfinished = con.execute(
    "SELECT COUNT(*) FROM tasks WHERE status IN ('pending', 'running')"
).fetchone()[0]
raise SystemExit(0 if unfinished else 1)
PY
  then
    tmux kill-session -t "$session" 2>/dev/null || true
    # Do not leak the supervisor lock into tmux/the runner. A leaked fd keeps
    # the lock held after this script exits and disables future cron repairs.
    tmux new-session -d -s "$session" \
      "$runner $bucket $concurrency $max_tokens" 9>&-
    printf '%s restarted bucket=%s concurrency=%s max_tokens=%s\n' \
      "$(date --iso-8601=seconds)" "$bucket" "$concurrency" "$max_tokens" \
      >>"$supervisor_log"
  fi
}

ensure_bucket short 10 4096
ensure_bucket medium 6 8192
ensure_bucket long 4 32768
