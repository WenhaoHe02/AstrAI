"""Eight-GPU forward/backward smoke test for expert-parallel grouped GEMM."""

import argparse
import os

import torch
import torch.distributed as dist

from astrai.model.components.mlp import DeepSeekMoE


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("torch", "deepep"), default="torch")
    parser.add_argument(
        "--compare-backends",
        action="store_true",
        help="Compare Torch and DeepEP outputs and gradients with identical weights.",
    )
    parser.add_argument("--shared-expert-overlap", action="store_true")
    return parser.parse_args()


def build_model(
    backend: str, device: int, shared_expert_overlap: bool = False
) -> DeepSeekMoE:
    model = DeepSeekMoE(
        dim=256,
        dim_ffn=384,
        n_routed_experts=16,
        n_shared_experts=1,
        n_activated_experts=2,
        n_layers=2,
        expert_parallel_size=8,
        expert_dispatch_backend=backend,
        moe_shared_expert_overlap=shared_expert_overlap,
    ).to(device=device, dtype=torch.bfloat16)
    model.apply(
        lambda module: (
            module.reset_parameters() if hasattr(module, "reset_parameters") else None
        )
    )
    return model


def run_model(model: DeepSeekMoE, inputs: torch.Tensor):
    model.zero_grad(set_to_none=True)
    x = inputs.detach().clone().requires_grad_(True)
    output, aux_loss, z_loss, expert_load, _ = model(x)
    loss = output.float().square().mean() + 0.01 * aux_loss + 0.001 * z_loss
    loss.backward()
    grads = {
        name: param.grad.detach().clone()
        for name, param in model.named_parameters()
        if param.grad is not None
    }
    return output.detach(), loss.detach(), x.grad.detach(), expert_load, grads


def relative_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    numerator = torch.linalg.vector_norm((actual.float() - expected.float()).flatten())
    denominator = torch.linalg.vector_norm(expected.float().flatten()).clamp_min(1e-12)
    return (numerator / denominator).item()


def compare_backends(
    inputs: torch.Tensor, device: int, rank: int, shared_expert_overlap: bool
) -> None:
    torch.manual_seed(1234)
    torch_model = build_model("torch", device)
    reference = run_model(torch_model, inputs)

    torch.manual_seed(1234)
    deepep_model = build_model(
        "deepep", device, shared_expert_overlap=shared_expert_overlap
    )
    deepep_model.load_state_dict(torch_model.state_dict())
    actual = run_model(deepep_model, inputs)

    output_error = relative_l2(actual[0], reference[0])
    input_grad_error = relative_l2(actual[2], reference[2])
    loss_error = abs(actual[1].item() - reference[1].item()) / max(
        abs(reference[1].item()), 1e-12
    )
    grad_errors = {
        name: relative_l2(actual[4][name], reference[4][name]) for name in reference[4]
    }
    max_param_grad_error = max(grad_errors.values(), default=0.0)

    errors = torch.tensor(
        [output_error, loss_error, input_grad_error, max_param_grad_error],
        dtype=torch.float64,
        device=device,
    )
    dist.all_reduce(errors, op=dist.ReduceOp.MAX)
    if not torch.all(errors < torch.tensor([0.02, 0.02, 0.05, 0.05], device=device)):
        worst_name = max(grad_errors, key=grad_errors.get, default="none")
        raise RuntimeError(
            "DeepEP numerical comparison failed: "
            f"output={errors[0].item():.3e}, loss={errors[1].item():.3e}, "
            f"input_grad={errors[2].item():.3e}, "
            f"param_grad={errors[3].item():.3e} ({worst_name})"
        )
    if rank == 0:
        print(
            "DEEPEP_NUMERICS_OK",
            f"output_rel_l2={errors[0].item():.3e}",
            f"loss_rel={errors[1].item():.3e}",
            f"input_grad_rel_l2={errors[2].item():.3e}",
            f"param_grad_rel_l2={errors[3].item():.3e}",
        )


def main() -> None:
    args = parse_args()
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    if world_size != 8:
        raise RuntimeError(f"expected 8 ranks, got {world_size}")

    torch.manual_seed(10_000 + rank)
    inputs = torch.randn(
        2,
        32,
        256,
        device=local_rank,
        dtype=torch.bfloat16,
    )

    if args.compare_backends:
        compare_backends(inputs, local_rank, rank, args.shared_expert_overlap)
        dist.destroy_process_group()
        return

    torch.manual_seed(1234)
    model = build_model(
        args.backend,
        local_rank,
        shared_expert_overlap=args.shared_expert_overlap,
    )
    output, loss, input_grad, expert_load, grads = run_model(model, inputs)

    tensors = [output, loss, input_grad, *grads.values()]
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
            f"backend={args.backend}",
            f"shared_overlap={args.shared_expert_overlap}",
            f"loss={loss.item():.6f}",
            f"load_min={global_load.min().item():.6f}",
            f"load_max={global_load.max().item():.6f}",
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
