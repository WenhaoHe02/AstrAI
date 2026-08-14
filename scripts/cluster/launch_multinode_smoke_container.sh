#!/usr/bin/env bash
set -euo pipefail

: "${NODE_RANK:?set NODE_RANK to 0 or 1}"
: "${MASTER_PORT:?set a fresh MASTER_PORT for every run}"

IMAGE="${IMAGE:-crpi-w4le1oy1gd4wy3vu.cn-hangzhou.personal.cr.aliyuncs.com/ai-stack/ngc:26-03-deepepv2}"
REPO_ROOT="${REPO_ROOT:-/home/zbuser02/AstrAI-12b}"
CONTAINER_NAME="astrai-deepep-2n-smoke-r${NODE_RANK}-p${MASTER_PORT}"

exec docker run --rm \
  --name "$CONTAINER_NAME" \
  --gpus all \
  --privileged \
  --network host \
  --ipc host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -e NCCL_NET_PLUGIN=none \
  -e NCCL_NET=IB \
  -e NCCL_NVLS_ENABLE=0 \
  -e NCCL_GIN_CROSS_NIC=0 \
  -e 'NCCL_IB_HCA=^mlx5_3' \
  -e NCCL_IB_GID_INDEX=3 \
  -e NCCL_SOCKET_IFNAME=ens99f0 \
  -e GLOO_SOCKET_IFNAME=ens99f0 \
  -e EP_DISABLE_GIN=0 \
  -e EP_NIC_NAME=mlx5_0 \
  -e ASTRAI_SMOKE_DEEPEP=1 \
  -e ASTRAI_SMOKE_EP_SIZE=8 \
  -v /home/zbuser02:/home/zbuser02 \
  -v /mnt/nvme9:/mnt/nvme9 \
  "$IMAGE" \
  bash -lc "export PYTHONPATH='$REPO_ROOT'; exec /mnt/nvme9/astrai/envs/train/bin/python -m torch.distributed.run --nnodes=2 --nproc-per-node=8 --node-rank='$NODE_RANK' --master-addr=172.16.8.32 --master-port='$MASTER_PORT' '$REPO_ROOT/scripts/tools/smoke_multinode.py'"
