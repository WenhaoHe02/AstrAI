"""Benchmark the full routed-expert forward/backward with Torch or DeepEP."""

import argparse
import gc
import os
import statistics

import torch
import torch.distributed as dist

from astrai.model.components.mlp import DeepSeekMoE


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("torch", "deepep"), required=True)
    parser.add_argument("--tokens", type=int, default=2048)
    parser.add_argument("--hidden", type=int, default=3072)
    parser.add_argument("--ffn-hidden", type=int, default=2176)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--expert-alignment", type=int, default=1)
    parser.add_argument("--overlap-with-compute", action="store_true")
    parser.add_argument("--shared-experts", type=int, default=0)
    parser.add_argument("--shared-expert-overlap", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    if world_size != 8:
        raise RuntimeError(f"expected 8 ranks, got {world_size}")

    torch.manual_seed(1234)
    model = DeepSeekMoE(
        dim=args.hidden,
        dim_ffn=args.ffn_hidden,
        n_routed_experts=16,
        n_shared_experts=args.shared_experts,
        n_activated_experts=2,
        n_layers=32,
        expert_parallel_size=8,
        expert_dispatch_backend=args.backend,
        deepep_expert_alignment=args.expert_alignment,
        deepep_overlap_with_compute=args.overlap_with_compute,
        moe_shared_expert_overlap=args.shared_expert_overlap,
    ).to(device=local_rank, dtype=torch.bfloat16)
    model.apply(
        lambda module: (
            module.reset_parameters() if hasattr(module, "reset_parameters") else None
        )
    )

    torch.manual_seed(10_000 + rank)
    inputs = torch.randn(
        1,
        args.tokens,
        args.hidden,
        device=local_rank,
        dtype=torch.bfloat16,
    )

    elapsed_ms = []
    for iteration in range(args.warmup + args.iters):
        model.zero_grad(set_to_none=True)
        x = inputs.detach().requires_grad_(True)
        dist.barrier()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        output, aux_loss, z_loss, _, _ = model(x)
        loss = output.float().square().mean() + 0.01 * aux_loss + 0.001 * z_loss
        loss.backward()
        end.record()
        end.synchronize()

        local_ms = torch.tensor(start.elapsed_time(end), device=local_rank)
        dist.all_reduce(local_ms, op=dist.ReduceOp.MAX)
        if iteration >= args.warmup:
            elapsed_ms.append(local_ms.item())

    if rank == 0:
        median_ms = statistics.median(elapsed_ms)
        tokens_per_second = args.tokens * world_size / (median_ms / 1000)
        print(
            "EXPERT_DISPATCH_BENCH",
            f"backend={args.backend}",
            f"alignment={args.expert_alignment}",
            f"overlap={args.overlap_with_compute}",
            f"shared_experts={args.shared_experts}",
            f"shared_overlap={args.shared_expert_overlap}",
            f"median_ms={median_ms:.3f}",
            f"p90_ms={sorted(elapsed_ms)[int(0.9 * (len(elapsed_ms) - 1))]:.3f}",
            f"tokens_per_second={tokens_per_second:.1f}",
        )

    del model, inputs
    gc.collect()
    torch.cuda.empty_cache()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
