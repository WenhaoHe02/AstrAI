#!/usr/bin/env bash
set -euo pipefail

for pid in "$@"; do
  while kill -0 "$pid" 2>/dev/null; do
    sleep 5
  done
done

systemctl --user start \
  astral-glm-v4@short.service \
  astral-glm-v4@medium.service \
  astral-glm-v4@long.service
