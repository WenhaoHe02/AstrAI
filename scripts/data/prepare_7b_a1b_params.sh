#!/usr/bin/env bash
set -euo pipefail

repo=${ASTRAI_REPO:-/root/data/v-boxiuli/AstrAI}
tokenizer_source=${ASTRAI_TOKENIZER_SOURCE:-/root/data/v-boxiuli/checkpoints/pretrain-12b-b200/epoch_0_step_60871}
output=${ASTRAI_BASE_PARAMS:-/root/data/v-boxiuli/params/astrai-7b-a1b-gqa-moe}
recipe=$repo/recipes/astrai-7b-a1b-gqa-moe/config.json

[[ -s "$recipe" ]] || { echo "Missing recipe: $recipe" >&2; exit 1; }
[[ -d "$tokenizer_source" ]] || {
    echo "Missing tokenizer source: $tokenizer_source" >&2
    exit 1
}

mkdir -p "$output"
cp "$recipe" "$output/config.json"
copied=0
for name in tokenizer.json tokenizer_config.json special_tokens_map.json; do
    if [[ -s "$tokenizer_source/$name" ]]; then
        cp "$tokenizer_source/$name" "$output/$name"
        copied=$((copied + 1))
    fi
done
if (( copied == 0 )); then
    echo "No tokenizer files found in $tokenizer_source" >&2
    exit 1
fi

python "$repo/scripts/tools/estimate_model_params.py" "$output/config.json"
echo "Prepared random-init 7B-A1B parameter directory: $output"
