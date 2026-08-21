#!/usr/bin/env bash
set -euo pipefail

repo=${ASTRAI_REPO:-/root/data/v-boxiuli/AstrAI}
train_python=${ASTRAI_TRAIN_PYTHON:-/usr/bin/python3}
data_root=${ASTRAI_DATA_ROOT:-/root/data/v-boxiuli/data/ultra-fineweb-2048}
val_data_root=${ASTRAI_VAL_DATA_ROOT:-}
base_params=${ASTRAI_BASE_PARAMS:-/root/data/v-boxiuli/params/astrai-7b-a1b-gqa-moe}
ckpt_root=${ASTRAI_CKPT_ROOT:-/root/data/v-boxiuli/checkpoints/pretrain-7b-a1b-b200}
log_root=${ASTRAI_LOG_ROOT:-/root/data/v-boxiuli/logs}
stop_file=${ASTRAI_STOP_FILE:-/root/data/v-boxiuli/state/STOP_PRETRAIN_7B_A1B}
pid_file=${ASTRAI_TRAIN_PID_FILE:-/root/data/v-boxiuli/state/pretrain-7b-a1b.pid}
log_file=${ASTRAI_TRAIN_LOG:-$log_root/train-7b-a1b-8gpu.log}

# EP1 keeps every expert local. DDP is the fast path: no expert all-to-all and
# no parameter all-gather. The full BF16 model plus AdamW state fits B200.
parallel_mode=${ASTRAI_PARALLEL_MODE:-ddp}
batch_per_device=${ASTRAI_BATCH_PER_DEVICE:-8}
grad_accum_steps=${ASTRAI_GRAD_ACCUM_STEPS:-4}
num_workers=${ASTRAI_NUM_WORKERS:-4}
max_lr=${ASTRAI_MAX_LR:-3e-4}
warmup_ratio=${ASTRAI_WARMUP_RATIO:-0.01}
min_rate=${ASTRAI_MIN_RATE:-0.0}
stable_steps=${ASTRAI_STABLE_STEPS:-}
decay_steps=${ASTRAI_DECAY_STEPS:-}
ckpt_interval=${ASTRAI_CKPT_INTERVAL:-5000}
ckpt_keep_last=${ASTRAI_CKPT_KEEP_LAST:-3}

[[ -s "$base_params/config.json" ]] || {
    echo "Missing 7B base params. Run scripts/data/prepare_7b_a1b_params.sh first." >&2
    exit 1
}
[[ -d "$data_root" ]] || { echo "Missing data root: $data_root" >&2; exit 1; }

python - "$base_params/config.json" <<'PY'
import json, sys
c = json.load(open(sys.argv[1], encoding="utf-8"))
assert c["expert_parallel_size"] == 1
assert c["expert_dispatch_backend"] == "torch"
assert c["n_routed_experts"] == 32 and c["n_activated_experts"] == 2
PY

if pgrep -f 'scripts/tools/train.py.*pretrain-7b-a1b' >/dev/null; then
    echo 'AstrAI 7B-A1B pretraining is already running.' >&2
    exit 1
fi

latest_ckpt=''
latest_step=-1
if [[ -d "$ckpt_root" ]]; then
    for candidate in "$ckpt_root"/epoch_*_step_*; do
        [[ -d "$candidate" ]] || continue
        step=${candidate##*_step_}
        [[ "$step" =~ ^[0-9]+$ ]] || continue
        complete=true
        for required in _SUCCESS meta.json config.json model.safetensors scheduler.pt; do
            [[ -s "$candidate/$required" ]] || complete=false
        done
        for rank in {0..7}; do
            [[ -s "$candidate/optimizer.rank${rank}.pt" ]] || complete=false
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
    echo "Starting 7B-A1B from random initialization. Tokenizer: $base_params"
fi

validation_args=()
if [[ -n "$val_data_root" ]]; then
    validation_args=(--val_data_root_path="$val_data_root" --val_step=1000)
fi
schedule_args=(--min_rate="$min_rate")
[[ -n "$stable_steps" ]] && schedule_args+=(--stable_steps="$stable_steps")
[[ -n "$decay_steps" ]] && schedule_args+=(--decay_steps="$decay_steps")

mkdir -p "$log_root" "$ckpt_root" "$(dirname "$stop_file")" "$(dirname "$pid_file")"
[[ -w "$ckpt_root" ]] || { echo "Checkpoint root is not writable: $ckpt_root" >&2; exit 1; }
rm -f "$stop_file"
cd "$repo"
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export NVTE_FUSED_ATTN=${NVTE_FUSED_ATTN:-1}
export NVTE_FLASH_ATTN=${NVTE_FLASH_ATTN:-0}

nohup setsid "$train_python" scripts/tools/train.py \
    --nprocs=8 --parallel_mode="$parallel_mode" \
    --loss_backend=liger --swiglu_backend=liger \
    --residual_norm_backend=liger --router_score_dtype=fp32 \
    --attention_backend=transformer_engine --fused_qkv --fused_mlp_gate_up \
    --moe_route_scale_before_down --train_type=seq \
    --data_root_path="$data_root" --param_path="$param_path" "${resume_args[@]}" \
    "${validation_args[@]}" \
    --batch_per_device="$batch_per_device" --grad_accum_steps="$grad_accum_steps" \
    --window_size=2048 --n_epoch=1 --num_workers="$num_workers" \
    --warmup_ratio="$warmup_ratio" --max_lr="$max_lr" --weight_decay=0.1 \
    --max_grad_norm=1.0 --schedule_type=wsd --ckpt_interval="$ckpt_interval" \
    "${schedule_args[@]}" --ckpt_keep_last="$ckpt_keep_last" \
    --checkpoint_after_first_step --stop_file="$stop_file" \
    --ckpt_dir="$ckpt_root" --log_dir="$log_root/train-7b-a1b" \
    --metrics loss language_model_loss router_loss router_aux_loss \
    router_z_loss router_entropy expert_load_min expert_load_max \
    expert_load_cv step_time tokens_per_second peak_memory_gb lr grad_norm \
    val_loss global_batch_tokens seen_tokens effective_epochs \
    >>"$log_file" 2>&1 </dev/null &

train_pid=$!
printf '%s\n' "$train_pid" >"$pid_file"
echo "Started AstrAI 7B-A1B DP8 pretraining as PID $train_pid"
echo "Log: $log_file"
echo "Graceful stop: $repo/scripts/data/stop_pretrain_7b_a1b.sh"
