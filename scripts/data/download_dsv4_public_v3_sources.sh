#!/usr/bin/env bash
set -euo pipefail

root="${1:-/mnt/nvme3/zbuser02/astrai-distill/sources/public-v3/nemotron-posttrain-v1}"
base="https://huggingface.co/datasets/nvidia/Nemotron-Post-Training-Dataset-v1/resolve/refs%2Fconvert%2Fparquet/default"
mkdir -p "$root"

for spec in chat/0004 code/0000 math/0000 stem/0000 tool_calling/0000; do
  split="${spec%/*}"
  shard="${spec#*/}"
  output="$root/${split}-${shard}.parquet"
  if [ -s "$output" ]; then
    echo "SOURCE_EXISTS split=$split path=$output"
    continue
  fi
  partial="$output.part"
  echo "SOURCE_DOWNLOAD split=$split"
  curl --fail --location --retry 5 --retry-all-errors \
    --connect-timeout 20 --continue-at - \
    --output "$partial" "$base/$split/$shard.parquet"
  mv "$partial" "$output"
done

sha256sum "$root"/*.parquet | tee "$root/SHA256SUMS"
