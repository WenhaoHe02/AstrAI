"""Small NCCL/Gloo rendezvous and collective correctness smoke test."""

from __future__ import annotations

import os
import socket
import time

import torch
import torch.distributed as dist


def main() -> None:
    backend = os.environ.get("ASTRAI_SMOKE_BACKEND", "nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    if backend == "nccl":
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")

    dist.init_process_group(backend=backend, init_method="env://")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    value = torch.tensor(float(rank + 1), device=device)
    dist.all_reduce(value)
    expected = world_size * (world_size + 1) / 2
    if value.item() != expected:
        raise RuntimeError(f"all_reduce={value.item()}, expected={expected}")

    # DeepEP validates the NCCL runtime at import time.  Importing it before
    # init_process_group is insufficient because NCCL net plugins have not yet
    # been loaded into the process, which can hide duplicate-runtime failures.
    if os.environ.get("ASTRAI_SMOKE_DEEPEP", "0") == "1":
        from deep_ep import ElasticBuffer  # noqa: F401

        print(f"DEEPEP_IMPORT_OK rank={rank}", flush=True)

    dist.barrier()
    ep_size = int(os.environ.get("ASTRAI_SMOKE_EP_SIZE", "0"))
    if ep_size:
        from astrai.model.components.mlp import (
            _expert_data_parallel_group,
            _expert_parallel_group,
        )

        ep_group, ep_rank = _expert_parallel_group(ep_size)
        dp_group, dp_rank, dp_size = _expert_data_parallel_group(ep_size)
        ep_value = torch.ones((), device=device)
        dist.all_reduce(ep_value, group=ep_group)
        if ep_value.item() != ep_size:
            raise RuntimeError(f"EP all_reduce={ep_value.item()}, expected={ep_size}")
        dp_value = torch.tensor(float(rank), device=device)
        if dp_group is not None:
            dist.all_reduce(dp_value, group=dp_group)
        local_ep_rank = rank % ep_size
        replicas = world_size // ep_size
        expected_dp = sum(
            local_ep_rank + replica * ep_size for replica in range(replicas)
        )
        if dp_value.item() != expected_dp:
            raise RuntimeError(
                f"expert-DP all_reduce={dp_value.item()}, expected={expected_dp}"
            )
        print(
            f"TOPOLOGY_OK rank={rank} ep_rank={ep_rank} "
            f"dp_rank={dp_rank}/{dp_size}",
            flush=True,
        )

    dist.barrier()
    started = time.perf_counter()
    for _ in range(20):
        dist.all_reduce(value)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    print(
        f"SMOKE_OK host={socket.gethostname()} rank={rank}/{world_size} "
        f"local_rank={local_rank} backend={backend} collectives_s={elapsed:.4f}",
        flush=True,
    )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
