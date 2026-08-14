#!/usr/bin/env bash
set -euo pipefail

repo="${REPO_ROOT:-/home/zbuser02/AstrAI-12b}"
probe="$repo/scripts/data/probe_distill_api.py"

probe_one() {
  local label="$1" model="$2" credential="$3" key_env="$4"
  if [ ! -r "$credential" ]; then
    echo "TEACHER_PROBE label=$label result=missing_credential"
    return
  fi
  local mode
  mode=$(stat -c '%a' "$credential")
  if [ "$mode" != 600 ] && [ "$mode" != 400 ]; then
    echo "TEACHER_PROBE label=$label result=unsafe_credential_mode mode=$mode"
    return
  fi
  local key
  key=$(tr -d '\r\n' <"$credential")
  echo "TEACHER_PROBE label=$label model=$model"
  env "$key_env=$key" python3 "$probe" \
    --model "$model" \
    --requests 1 \
    --concurrency 1 \
    --timeout 90 \
    --max-tokens 4 \
    --api-key-env "$key_env" \
    --show-first-error
}

probe_one glm glm-5.2 "$HOME/.config/astrai/inferknock.env" INFERKNOCK_API_KEY
probe_one dsv4 deepseek-v4-flash "$HOME/.config/astrai/inferknock-dsv4.env" INFERKNOCK_DSV4_API_KEY
