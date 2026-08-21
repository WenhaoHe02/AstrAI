#!/usr/bin/env bash
set -euo pipefail

stop_file=${ASTRAI_STOP_FILE:-/root/data/v-boxiuli/state/STOP_PRETRAIN_7B_A1B}
mkdir -p "$(dirname "$stop_file")"
touch "$stop_file"
printf 'Graceful stop requested via %s\n' "$stop_file"
printf 'Training will stop after the current optimizer step and save a checkpoint.\n'
