#!/usr/bin/env bash
set -euo pipefail

: "${NODE_RANK:?set NODE_RANK to 0 or 1}"
: "${MASTER_PORT:?set a fresh MASTER_PORT for every run}"
: "${PARAM_PATH:?set PARAM_PATH to the shared input checkpoint}"

IMAGE="${IMAGE:-crpi-w4le1oy1gd4wy3vu.cn-hangzhou.personal.cr.aliyuncs.com/ai-stack/ngc:26-03-deepepv2}"
RUN_KIND="${RUN_KIND:-train}"
CONTAINER_NAME="astrai-12b-2n-${RUN_KIND}-r${NODE_RANK}-p${MASTER_PORT}"
REPO_ROOT="${REPO_ROOT:-/home/zbuser02/AstrAI-12b}"
DATA_ROOT="${DATA_ROOT:-/mnt/nvme6/astrai/tokenized/pretrain-2048-docmask}"

test -x /mnt/nvme9/astrai/envs/train/bin/python
test -d "$REPO_ROOT"
test -d "$DATA_ROOT"
test -f "$PARAM_PATH/model.safetensors"

mkdir -p "/mnt/nvme9/astrai/jit/multinode-${MASTER_PORT}"

exec docker run --rm \
  --name "$CONTAINER_NAME" \
  --gpus all \
  --privileged \
  --network host \
  --ipc host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -e NODE_RANK="$NODE_RANK" \
  -e MASTER_ADDR="${MASTER_ADDR:-172.16.8.32}" \
  -e MASTER_PORT="$MASTER_PORT" \
  -e PARAM_PATH="$PARAM_PATH" \
  -e REPO_ROOT="$REPO_ROOT" \
  -e DATA_ROOT="$DATA_ROOT" \
  -e CUDA_CACHE_PATH="/mnt/nvme9/astrai/jit/multinode-${MASTER_PORT}" \
  -e TORCH_EXTENSIONS_DIR="/mnt/nvme9/astrai/jit/multinode-${MASTER_PORT}/torch" \
  -v /home/zbuser02:/home/zbuser02 \
  -v /mnt/nvme6:/mnt/nvme6 \
  -v /mnt/nvme8:/mnt/nvme8 \
  -v /mnt/nvme9:/mnt/nvme9 \
  -v /mnt/nfs:/mnt/nfs \
  "$IMAGE" \
  bash "$REPO_ROOT/scripts/cluster/run_pretrain_12b_multinode.sh"
