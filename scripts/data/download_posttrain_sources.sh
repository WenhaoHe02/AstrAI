#!/usr/bin/env bash
set -euo pipefail

base="${1:-/mnt/nvme3/zbuser02/astrai-distill/sources/additional}"
mkdir -p \
  "$base/mixture-of-thoughts" \
  "$base/chinese-instruct" \
  "$base/ultrafeedback"

download() {
  local output="$1"
  local url="$2"
  curl -L --fail --retry 10 --retry-delay 2 -C - -o "$output" "$url"
}

download_mixture_of_thoughts() {
  local index shard output url
  for index in $(seq 0 14); do
    shard=$(printf "%05d" "$index")
    output="$base/mixture-of-thoughts/train-${shard}-of-00015.parquet"
    url="https://huggingface.co/datasets/open-r1/Mixture-of-Thoughts/resolve/main/all/train-${shard}-of-00015.parquet"
    download "$output" "$url"
  done
}

download_mixture_of_thoughts &
mixture_pid=$!
download \
  "$base/chinese-instruct/stem_zh.jsonl" \
  "https://huggingface.co/datasets/lcccher/Chinese-Instruct/resolve/main/stem_zh/stem_zh.jsonl" &
stem_pid=$!
download \
  "$base/ultrafeedback/train_prefs.parquet" \
  "https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized/resolve/main/data/train_prefs-00000-of-00001.parquet" &
feedback_pid=$!

status=0
wait "$mixture_pid" || status=1
wait "$stem_pid" || status=1
wait "$feedback_pid" || status=1
exit "$status"
