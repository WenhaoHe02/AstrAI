#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "usage: $0 BUCKET" >&2
  exit 2
fi

case "$1" in
  # GLM-5.2 gateway allowance: 62 requests in total.  Split it in
  # proportion to the remaining short/medium/long v5 queues.
  short) concurrency=33; max_tokens=4096 ;;
  medium) concurrency=18; max_tokens=8192 ;;
  long) concurrency=11; max_tokens=32768 ;;
  *) echo "invalid bucket: $1" >&2; exit 2 ;;
esac

exec /home/zbuser02/AstrAI-12b/scripts/data/run_glm_posttrain_bucket.sh \
  "$1" "$concurrency" "$max_tokens"
