#!/usr/bin/env bash
set -euo pipefail

# Checkpoints must remain writable when training is later resumed outside the
# root-running container on a single node.  The checkpoint root is setgid to
# the training user's group by the launcher/operator.
umask 0002

# Launch inside identical fresh containers on gpu002 and gpu004. Required:
# NODE_RANK=0|1 and a unique MASTER_PORT. Rank 1 should be started first.
: "${NODE_RANK:?set NODE_RANK to 0 or 1}"
: "${MASTER_PORT:?set a fresh MASTER_PORT for every run}"

export MASTER_ADDR="${MASTER_ADDR:-172.16.8.32}"
export NNODES=2
export NPROC_PER_NODE=8
export WORLD_SIZE=16
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

export EP_DISABLE_GIN="${EP_DISABLE_GIN:-0}"
export EP_NIC_NAME="${EP_NIC_NAME:-mlx5_0}"
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
export NCCL_GIN_CROSS_NIC="${NCCL_GIN_CROSS_NIC:-0}"
export NCCL_IB_HCA="${NCCL_IB_HCA:-^mlx5_3}"
export NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX:-3}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-ens99f0}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-ens99f0}"
export NCCL_NET_PLUGIN="${NCCL_NET_PLUGIN:-none}"
export NCCL_NET="${NCCL_NET:-IB}"
export NCCL_GIN_TYPE="${NCCL_GIN_TYPE:-3}"
export NCCL_GIN_GDAKI_NIC_HANDLER="${NCCL_GIN_GDAKI_NIC_HANDLER:-1}"

REPO_ROOT="${REPO_ROOT:-/home/zbuser02/AstrAI-12b}"
PYTHON_BIN="${PYTHON_BIN:-/mnt/nvme9/astrai/envs/train/bin/python}"
PARAM_PATH="${PARAM_PATH:?set PARAM_PATH to a complete 8-rank checkpoint}"
DATA_ROOT="${DATA_ROOT:-/mnt/nvme6/astrai/tokenized/pretrain-2048-docmask}"
CKPT_DIR="${CKPT_DIR:-/mnt/nvme8/astrai/checkpoints/pretrain-12b-gqa-2node}"
LOG_DIR="${LOG_DIR:-/mnt/nvme9/astrai/logs/train-pretrain-12b-gqa-2node}"
STOP_FILE="${STOP_FILE:-/mnt/nvme9/astrai/STOP_PRETRAIN_12B_GQA_2NODE}"

cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

exec "$PYTHON_BIN" -m torch.distributed.run \
  --nnodes="$NNODES" \
  --nproc-per-node="$NPROC_PER_NODE" \
  --node-rank="$NODE_RANK" \
  --master-addr="$MASTER_ADDR" \
  --master-port="$MASTER_PORT" \
  scripts/tools/train.py \
  --nprocs=16 \
  --parallel_mode=fsdp \
  --fsdp_sharding_strategy=no_shard \
  --loss_backend=liger \
  --swiglu_backend=liger \
  --residual_norm_backend=liger \
  --router_score_dtype=fp32 \
  --attention_backend=flash_sdpa \
  --expert_dispatch_backend=deepep \
  --deepep_cpu_sync \
  --fused_qkv \
  --fused_mlp_gate_up \
  --moe_route_scale_before_down \
  --train_type=seq \
  --data_root_path="$DATA_ROOT" \
  --param_path="$PARAM_PATH" \
  --resume \
  --batch_per_device=4 \
  --grad_accum_steps=4 \
  --window_size=2048 \
  --n_epoch=1 \
  --num_workers=4 \
  --warmup_ratio=0.01 \
  --max_lr=2e-4 \
  --weight_decay=0.1 \
  --max_grad_norm=1.0 \
  --schedule_type=wsd \
  --ckpt_interval=250 \
  --checkpoint_after_first_step \
  --min_rate=0.0 \
  --ckpt_keep_last=3 \
  --stop_file="$STOP_FILE" \
  --ckpt_dir="$CKPT_DIR" \
  --log_dir="$LOG_DIR" \
  --metrics loss language_model_loss router_loss router_aux_loss router_z_loss \
    router_entropy expert_load_min expert_load_max expert_load_cv step_time \
    tokens_per_second peak_memory_gb lr grad_norm val_loss global_batch_tokens \
    seen_tokens effective_epochs
