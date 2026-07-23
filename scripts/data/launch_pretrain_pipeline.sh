#!/usr/bin/env bash
set -euo pipefail

repo=${ASTRAI_REPO:-/home/zbuser02/AstrAI-12b}
curation_python=${ASTRAI_CURATION_PYTHON:-/mnt/nvme2/astrai/envs/curation/bin/python}
train_python=${ASTRAI_TRAIN_PYTHON:-/mnt/nvme9/astrai/envs/train/bin/python}
log_root=${ASTRAI_LOG_ROOT:-/mnt/nvme9/astrai/logs}

cd "$repo"

launch() {
    local logfile=$1
    shift
    nohup setsid "$@" >"$logfile" 2>&1 </dev/null &
    launched_pid=$!
}

launch "$log_root/minhash-full-zh.log" \
    "$curation_python" scripts/data/curate_pretrain.py minhash \
    --input /mnt/nvme3/astrai/normalized/quality-zh/kept \
    --output /mnt/nvme4/astrai/dedup/zh \
    --work /mnt/nvme5/astrai/minhash/zh \
    --logs "$log_root/curation-full" \
    --source chinese-cosmopedia --language zh --tasks 64 --workers 48
zh_minhash_pid=$launched_pid

launch "$log_root/curation-full-en.log" \
    "$curation_python" scripts/data/curate_pretrain.py quality \
    --input /mnt/nvme2/astrai/raw/dolma-v1_7-30B \
    --output /mnt/nvme3/astrai/normalized/quality-en \
    --logs "$log_root/curation-full" \
    --source dolma-v1_7-30B --language en --glob '*.parquet' \
    --row-groups-per-chunk 16 --tasks 192 --workers 192
en_quality_pid=$launched_pid

launch "$log_root/minhash-full-en.log" \
    "$curation_python" scripts/data/wait_and_run.py --pid "$en_quality_pid" -- \
    "$curation_python" scripts/data/curate_pretrain.py minhash \
    --input /mnt/nvme3/astrai/normalized/quality-en/kept \
    --output /mnt/nvme4/astrai/dedup/en \
    --work /mnt/nvme5/astrai/minhash/en \
    --logs "$log_root/curation-full" \
    --source dolma-v1_7-30B --language en --tasks 192 --workers 192
en_minhash_pid=$launched_pid

launch "$log_root/balance-full.log" \
    "$curation_python" scripts/data/wait_and_run.py --pid "$zh_minhash_pid" -- \
    "$curation_python" scripts/data/wait_and_run.py --pid "$en_minhash_pid" -- \
    "$curation_python" scripts/data/balance_pretrain.py \
    --zh /mnt/nvme4/astrai/dedup/zh --en /mnt/nvme4/astrai/dedup/en \
    --tokenizer params/astrai-12b-mqa-moe/tokenizer.json \
    --output /mnt/nvme3/astrai/normalized/pretrain-balanced.jsonl \
    --tokens-per-language auto --batch-size 512
balance_pid=$launched_pid

launch "$log_root/preprocess-full.log" \
    "$curation_python" scripts/data/wait_and_run.py --pid "$balance_pid" -- \
    "$train_python" scripts/tools/preprocess.py \
    /mnt/nvme3/astrai/normalized/pretrain-balanced.jsonl \
    -o /mnt/nvme6/astrai/tokenized/pretrain-2048 \
    -c recipes/astrai-12b-mqa-moe/pretrain-2048.json \
    --tokenizer_path params/astrai-12b-mqa-moe
preprocess_pid=$launched_pid

launch "$log_root/train-pretrain-12b.log" \
    "$curation_python" scripts/data/wait_and_run.py --pid "$preprocess_pid" -- \
    /usr/bin/env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NCCL_DEBUG=WARN \
    "$train_python" scripts/tools/train.py \
    --nprocs=8 --parallel_mode=fsdp --train_type=seq \
    --data_root_path=/mnt/nvme6/astrai/tokenized/pretrain-2048 \
    --param_path=params/astrai-12b-mqa-moe \
    --batch_per_device=1 --grad_accum_steps=32 --gradient_checkpointing \
    --window_size=2048 --n_epoch=1 --num_workers=4 \
    --warmup_ratio=0.01 --max_lr=2e-4 --weight_decay=0.1 \
    --max_grad_norm=1.0 --schedule_type=wsd --ckpt_interval=5000 \
    --ckpt_dir=/mnt/nvme8/astrai/checkpoints/pretrain-12b \
    --log_dir="$log_root/train-pretrain-12b" \
    --metrics loss language_model_loss router_loss router_aux_loss \
    router_z_loss router_entropy expert_load_min expert_load_max \
    expert_load_cv lr grad_norm
train_watcher_pid=$launched_pid

pid_file="$log_root/pretrain-pipeline.pids"
printf '%s\n' \
    "zh_minhash=$zh_minhash_pid" \
    "en_quality=$en_quality_pid" \
    "en_minhash=$en_minhash_pid" \
    "balance=$balance_pid" \
    "preprocess=$preprocess_pid" \
    "train_watcher=$train_watcher_pid" >"$pid_file"
cat "$pid_file"
