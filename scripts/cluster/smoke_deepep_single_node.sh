#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-crpi-w4le1oy1gd4wy3vu.cn-hangzhou.personal.cr.aliyuncs.com/ai-stack/ngc:26-03-deepepv2}"
REPO_ROOT="${REPO_ROOT:-/home/zbuser02/AstrAI-12b}"
CONTAINER_NAME="astrai-deepep-smoke-${RANDOM}-$$"

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
  -e 'NCCL_IB_HCA=^mlx5_3' \
  -e NCCL_IB_GID_INDEX=3 \
  -e NCCL_SOCKET_IFNAME=ens99f0 \
  -e GLOO_SOCKET_IFNAME=ens99f0 \
  -e EP_DISABLE_GIN=0 \
  -e EP_NIC_NAME=mlx5_0 \
  -e ASTRAI_SMOKE_DEEPEP=1 \
  -v /home/zbuser02:/home/zbuser02 \
  -v /mnt/nvme9:/mnt/nvme9 \
  "$IMAGE" \
  bash -lc "export PYTHONPATH='$REPO_ROOT'; exec /mnt/nvme9/astrai/envs/train/bin/python -m torch.distributed.run --standalone --nproc-per-node=8 '$REPO_ROOT/scripts/tools/smoke_multinode.py'"
