#!/usr/bin/env bash
set -euo pipefail

repo=${ASTRAI_REPO:-/home/zbuser02/AstrAI-12b}
train_python=${ASTRAI_TRAIN_PYTHON:-/mnt/nvme9/astrai/envs/train/bin/python}
data_root=${ASTRAI_DATA_ROOT:-/mnt/nvme6/astrai/tokenized/pretrain-2048}
base_params=${ASTRAI_BASE_PARAMS:-$repo/params/astrai-12b-mqa-moe}
ckpt_root=${ASTRAI_CKPT_ROOT:-/mnt/nvme8/astrai/checkpoints/pretrain-12b}
log_root=${ASTRAI_LOG_ROOT:-/mnt/nvme9/astrai/logs}
stop_file=${ASTRAI_STOP_FILE:-/mnt/nvme9/astrai/STOP_PRETRAIN_12B}
pid_file=${ASTRAI_TRAIN_PID_FILE:-$log_root/train-pretrain-12b.pid}
log_file=${ASTRAI_TRAIN_LOG:-$log_root/train-pretrain-12b.log}

if pgrep -f 'scripts/tools/train.py.*pretrain-2048' >/dev/null; then
    echo 'AstrAI pretraining is already running.' >&2
    exit 1
fi

latest_ckpt=''
latest_step=-1
if [[ -d "$ckpt_root" ]]; then
    for candidate in "$ckpt_root"/epoch_*_step_*; do
        [[ -d "$candidate" ]] || continue
        step=${candidate##*_step_}
        [[ "$step" =~ ^[0-9]+$ ]] || continue
        [[ -s "$candidate/meta.json" ]] || continue
        [[ -s "$candidate/config.json" ]] || continue
        [[ -s "$candidate/model.safetensors" ]] || continue
        [[ -s "$candidate/scheduler.pt" ]] || continue

        complete=true
        for rank in {0..7}; do
            if [[ ! -s "$candidate/optimizer.rank${rank}.pt" ]]; then
                complete=false
                break
            fi
        done
        "$complete" || continue

        if (( step > latest_step )); then
            latest_step=$step
            latest_ckpt=$candidate
        fi
    done
fi

param_path=$base_params
resume_args=()
if [[ -n "$latest_ckpt" ]]; then
    param_path=$latest_ckpt
    resume_args=(--resume)
    echo "Resuming from complete checkpoint: $latest_ckpt"
else
    echo "No complete checkpoint found; starting from base parameters: $base_params"
fi

mkdir -p "$log_root" "$ckpt_root"
rm -f "$stop_file"
cd "$repo"

nohup setsid "$train_python" scripts/tools/train.py \
    --nprocs=8 --parallel_mode=fsdp --train_type=seq \
    --data_root_path="$data_root" --param_path="$param_path" "${resume_args[@]}" \
    --batch_per_device=1 --grad_accum_steps=32 --gradient_checkpointing \
    --window_size=2048 --n_epoch=1 --num_workers=4 \
    --warmup_ratio=0.01 --max_lr=2e-4 --weight_decay=0.1 \
    --max_grad_norm=1.0 --schedule_type=wsd --ckpt_interval=250 \
    --checkpoint_after_first_step --stop_file="$stop_file" \
    --ckpt_dir="$ckpt_root" --log_dir="$log_root/train-pretrain-12b" \
    --metrics loss language_model_loss router_loss router_aux_loss \
    router_z_loss router_entropy expert_load_min expert_load_max \
    expert_load_cv lr grad_norm \
    >>"$log_file" 2>&1 </dev/null &

train_pid=$!
printf '%s\n' "$train_pid" >"$pid_file"
echo "Started AstrAI pretraining as PID $train_pid"
echo "Log: $log_file"
echo "Graceful stop: $repo/scripts/data/stop_pretrain_12b.sh"
