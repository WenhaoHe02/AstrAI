"""Inspect the parameter/optimizer layout of an FSDP strategy without training."""

import argparse
import json
import os
from functools import partial

import torch
import torch.distributed as dist
from torch.distributed.fsdp import ShardingStrategy

from astrai.config.model_config import ConfigFactory
from astrai.parallel.executor import FSDPExecutor
from scripts.tools.train import create_model, create_optimizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--optimizer-dir",
        help="Optional checkpoint directory whose per-rank optimizer state is loaded.",
    )
    parser.add_argument(
        "--sharding", choices=("full_shard", "shard_grad_op", "no_shard"), required=True
    )
    args = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))
    with open(args.config, encoding="utf-8") as handle:
        config = ConfigFactory.load(json.load(handle))

    strategy = {
        "full_shard": ShardingStrategy.FULL_SHARD,
        "shard_grad_op": ShardingStrategy.SHARD_GRAD_OP,
        "no_shard": ShardingStrategy.NO_SHARD,
    }[args.sharding]
    executor = FSDPExecutor(sharding_strategy=strategy)
    model, optimizer, _ = executor.prepare(
        partial(create_model, config),
        partial(create_optimizer, lr=2e-4, weight_decay=0.1),
        before_wrap=lambda module: module.cuda(local_rank),
    )

    inner = optimizer.optimizer
    if args.optimizer_dir:
        optimizer_path = os.path.join(
            args.optimizer_dir, f"optimizer.rank{dist.get_rank()}.pt"
        )
        optimizer.load_state_dict(
            torch.load(optimizer_path, map_location="cpu", weights_only=False)
        )
    if dist.get_rank() == 0:
        print(
            "FSDP_LAYOUT",
            f"sharding={args.sharding}",
            f"muon_params={sum(len(g['params']) for g in inner.muon.param_groups) if inner.muon else 0}",
            f"adamw_params={sum(len(g['params']) for g in inner.adamw.param_groups)}",
            f"allocated_gb={torch.cuda.memory_allocated() / 1e9:.3f}",
            f"optimizer_states={len(inner.adamw.state)}",
        )
        for index, parameter in enumerate(inner.adamw.param_groups[0]["params"][:5]):
            print("ADAMW_PARAM", index, tuple(parameter.shape), parameter.numel())
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
