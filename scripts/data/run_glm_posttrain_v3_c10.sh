#!/usr/bin/env bash
set -euo pipefail

cd /home/zbuser02/AstrAI-12b
log=/mnt/nvme3/zbuser02/astrai-distill/logs/posttrain-v3-stream.log
mkdir -p "$(dirname "$log")"
exec >>"$log" 2>&1

echo "DISTILL_LAUNCH $(date --iso-8601=seconds) concurrency=10 stream=true"
exec python3 scripts/data/distill_glm.py \
  --input /mnt/nvme3/zbuser02/astrai-distill/seeds/posttrain-v3-stream.jsonl \
  --state /mnt/nvme3/zbuser02/astrai-distill/state/posttrain-v3-stream.sqlite3 \
  --output /mnt/nvme3/zbuser02/astrai-distill/output/posttrain-v3-stream.jsonl \
  --failed-output /mnt/nvme3/zbuser02/astrai-distill/output/posttrain-v3-stream.failed.jsonl \
  --base-url https://api.inferknock.ai/v1 \
  --model glm-5.2 \
  --concurrency 10 \
  --max-tokens 32768 \
  --temperature 0.6 \
  --top-p 0.95 \
  --timeout 600 \
  --connect-timeout 20 \
  --max-attempts 3 \
  --retry-base 2 \
  --retry-max 300 \
  --outage-threshold 5 \
  --outage-cooldown 120 \
  --require-content \
  --retry-finish-reason length \
  --stream
