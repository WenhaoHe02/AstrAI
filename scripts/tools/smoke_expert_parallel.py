"""Eight-GPU forward/backward smoke test for expert-parallel grouped GEMM."""

import os

import torch
import torch.distributed as dist

from astrai.model.components.mlp import DeepSeekMoE


def main() -> None:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    if world_size != 8:
        raise RuntimeError(f"expected 8 ranks, got {world_size}")

    torch.manual_seed(1234)
    model = DeepSeekMoE(
        dim=256,
        dim_ffn=384,
        n_routed_experts=16,
        n_shared_experts=1,
        n_activated_experts=2,
        n_layers=2,
        expert_parallel_size=8,
    ).to(device=local_rank, dtype=torch.bfloat16)
    model.apply(
        lambda module: module.reset_parameters()
        if hasattr(module, "reset_parameters")
        else None
    )

    torch.manual_seed(10_000 + rank)
    inputs = torch.randn(
        2,
        32,
        256,
        device=local_rank,
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    output, aux_loss, z_loss, expert_load, _ = model(inputs)
    loss = output.float().square().mean() + 0.01 * aux_loss + 0.001 * z_loss
    loss.backward()

    tensors = [output, loss, inputs.grad]
    tensors.extend(param.grad for param in model.parameters() if param.grad is not None)
    local_ok = torch.tensor(
        int(all(torch.isfinite(tensor).all() for tensor in tensors)),
        device=local_rank,
    )
    dist.all_reduce(local_ok, op=dist.ReduceOp.MIN)
    global_load = expert_load.to(local_rank)
    dist.all_reduce(global_load, op=dist.ReduceOp.SUM)
    global_load /= world_size

    if not local_ok.item():
        raise RuntimeError("non-finite output, loss, or gradient")
    if rank == 0:
        print(
            "EP_GROUPED_GEMM_OK",
            f"loss={loss.item():.6f}",
            f"load_min={global_load.min().item():.6f}",
            f"load_max={global_load.max().item():.6f}",
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
