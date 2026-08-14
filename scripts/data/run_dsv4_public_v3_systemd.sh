#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "usage: $0 BUCKET" >&2
  exit 2
fi

# Keep the provider-wide DSV4 limit at 128 requests.
case "$1" in
  short) concurrency=72; max_tokens=4096 ;;
  medium) concurrency=40; max_tokens=8192 ;;
  long) concurrency=16; max_tokens=32768 ;;
  *) echo "invalid bucket: $1" >&2; exit 2 ;;
esac

export ASTRAI_DSV4_RUN_ID="posttrain-dsv4-public-v3"
exec /home/zbuser02/AstrAI-12b/scripts/data/run_dsv4_posttrain_bucket.sh \
  "$1" "$concurrency" "$max_tokens"

