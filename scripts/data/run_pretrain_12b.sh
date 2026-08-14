#!/usr/bin/env bash
set -euo pipefail

repo=${ASTRAI_REPO:-/home/zbuser02/AstrAI-12b}
train_python=${ASTRAI_TRAIN_PYTHON:-/mnt/nvme9/astrai/envs/train/bin/python}
data_root=${ASTRAI_DATA_ROOT:-/mnt/nvme6/astrai/tokenized/pretrain-2048-docmask}
val_data_root=${ASTRAI_VAL_DATA_ROOT:-}
base_params=${ASTRAI_BASE_PARAMS:-$repo/params/astrai-12b-gqa-moe}
ckpt_root=${ASTRAI_CKPT_ROOT:-/mnt/nvme8/astrai/checkpoints/pretrain-12b-gqa}
log_root=${ASTRAI_LOG_ROOT:-/mnt/nvme9/astrai/logs}
stop_file=${ASTRAI_STOP_FILE:-/mnt/nvme9/astrai/STOP_PRETRAIN_12B_GQA}
pid_file=${ASTRAI_TRAIN_PID_FILE:-$log_root/train-pretrain-12b-gqa.pid}
log_file=${ASTRAI_TRAIN_LOG:-$log_root/train-pretrain-12b-gqa.log}
# H200 has enough memory to replicate the ~1.96B non-expert parameters.
# EP-aware DDP removes FSDP parameter all-gathers while excluding rank-local
# routed experts from broadcasts and gradient reductions.
parallel_mode=${ASTRAI_PARALLEL_MODE:-ddp}
fsdp_sharding=${ASTRAI_FSDP_SHARDING:-no_shard}
loss_backend=${ASTRAI_LOSS_BACKEND:-liger}
swiglu_backend=${ASTRAI_SWIGLU_BACKEND:-liger}
residual_norm_backend=${ASTRAI_RESIDUAL_NORM_BACKEND:-liger}
router_score_dtype=${ASTRAI_ROUTER_SCORE_DTYPE:-fp32}
attention_backend=${ASTRAI_ATTENTION_BACKEND:-transformer_engine}
route_scale_before_down=${ASTRAI_ROUTE_SCALE_BEFORE_DOWN:-1}
deepep_reuse_nccl_comm=${ASTRAI_DEEPEP_REUSE_NCCL_COMM:-0}
deepep_cpu_sync=${ASTRAI_DEEPEP_CPU_SYNC:-0}
# Preserve the 524,288-token global optimizer batch while halving the number
# of forward/backward and EP dispatch rounds per optimizer step.
batch_per_device=${ASTRAI_BATCH_PER_DEVICE:-8}
grad_accum_steps=${ASTRAI_GRAD_ACCUM_STEPS:-4}
max_lr=${ASTRAI_MAX_LR:-2e-4}
warmup_ratio=${ASTRAI_WARMUP_RATIO:-0.01}
min_rate=${ASTRAI_MIN_RATE:-0.0}
stable_steps=${ASTRAI_STABLE_STEPS:-}
decay_steps=${ASTRAI_DECAY_STEPS:-}
val_step=${ASTRAI_VAL_STEP:-1000}
ckpt_keep_last=${ASTRAI_CKPT_KEEP_LAST:-3}
gradient_checkpoint_args=()
if [[ ${ASTRAI_GRADIENT_CHECKPOINTING:-0} == 1 ]]; then
    gradient_checkpoint_args=(--gradient_checkpointing)
fi
route_scale_args=(--moe_route_scale_before_down)
if [[ "$route_scale_before_down" == 0 ]]; then
    route_scale_args=(--no-moe_route_scale_before_down)
fi
deepep_sync_args=(--no-deepep_cpu_sync)
if [[ "$deepep_cpu_sync" == 1 ]]; then
    deepep_sync_args=(--deepep_cpu_sync)
fi
validation_args=()
if [[ -n "$val_data_root" ]]; then
    validation_args=(--val_data_root_path="$val_data_root" --val_step="$val_step")
fi
schedule_args=(--min_rate="$min_rate")
if [[ -n "$stable_steps" ]]; then
    schedule_args+=(--stable_steps="$stable_steps")
fi
if [[ -n "$decay_steps" ]]; then
    schedule_args+=(--decay_steps="$decay_steps")
fi

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
if [[ ! -w "$ckpt_root" ]]; then
    echo "Checkpoint directory is not writable: $ckpt_root" >&2
    echo "Fix its owner/group before training; refusing to waste steps without checkpoints." >&2
    exit 1
fi
rm -f "$stop_file"
cd "$repo"
export EP_REUSE_NCCL_COMM="$deepep_reuse_nccl_comm"
# Prefer NVIDIA's cuDNN Graph fused attention on Hopper.  TE can still be
# pointed at its external FlashAttention backend by overriding these values.
export NVTE_FUSED_ATTN=${NVTE_FUSED_ATTN:-1}
export NVTE_FLASH_ATTN=${NVTE_FLASH_ATTN:-0}

nohup setsid "$train_python" scripts/tools/train.py \
    --nprocs=8 --parallel_mode="$parallel_mode" \
    --fsdp_sharding_strategy="$fsdp_sharding" \
    --loss_backend="$loss_backend" --swiglu_backend="$swiglu_backend" \
    --residual_norm_backend="$residual_norm_backend" \
    --router_score_dtype="$router_score_dtype" \
    --attention_backend="$attention_backend" \
    "${deepep_sync_args[@]}" \
    --fused_qkv --fused_mlp_gate_up \
    "${route_scale_args[@]}" \
    --train_type=seq \
    --data_root_path="$data_root" --param_path="$param_path" "${resume_args[@]}" \
    "${validation_args[@]}" \
    --batch_per_device="$batch_per_device" \
    --grad_accum_steps="$grad_accum_steps" "${gradient_checkpoint_args[@]}" \
    --window_size=2048 --n_epoch=1 --num_workers=4 \
    --warmup_ratio="$warmup_ratio" --max_lr="$max_lr" --weight_decay=0.1 \
    --max_grad_norm=1.0 --schedule_type=wsd --ckpt_interval=250 \
    "${schedule_args[@]}" \
    --ckpt_keep_last="$ckpt_keep_last" \
    --checkpoint_after_first_step --stop_file="$stop_file" \
    --ckpt_dir="$ckpt_root" --log_dir="$log_root/train-pretrain-12b-gqa" \
    --metrics loss language_model_loss router_loss router_aux_loss \
    router_z_loss router_entropy expert_load_min expert_load_max \
    expert_load_cv step_time tokens_per_second peak_memory_gb lr grad_norm \
    val_loss global_batch_tokens seen_tokens effective_epochs \
    >>"$log_file" 2>&1 </dev/null &

train_pid=$!
printf '%s\n' "$train_pid" >"$pid_file"
echo "Started AstrAI pretraining as PID $train_pid"
echo "Log: $log_file"
echo "Graceful stop: $repo/scripts/data/stop_pretrain_12b.sh"
