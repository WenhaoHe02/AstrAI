"""Benchmark residual-add plus RMSNorm at the 12B training shape."""

import argparse
import statistics

import torch

from astrai.model.components.norm import RMSNorm


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("torch", "liger"), required=True)
    parser.add_argument("--rows", type=int, default=8192)
    parser.add_argument("--width", type=int, default=3072)
    parser.add_argument("--eps", type=float, default=1e-5)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def run(backend, norm, x_base, residual_base, grad_norm, grad_residual):
    norm.zero_grad(set_to_none=True)
    x = x_base.clone().requires_grad_()
    residual = residual_base.clone().requires_grad_()
    normalized, residual_sum = norm.forward_with_residual(x, residual, backend)
    torch.autograd.backward((normalized, residual_sum), (grad_norm, grad_residual))
    return (
        normalized.detach(),
        residual_sum.detach(),
        x.grad.detach(),
        residual.grad.detach(),
        norm.weight.grad.detach(),
    )


def relative_l2(actual, expected):
    numerator = torch.linalg.vector_norm((actual.float() - expected.float()).flatten())
    denominator = torch.linalg.vector_norm(expected.float().flatten()).clamp_min(1e-12)
    return (numerator / denominator).item()


def check_numerics(backend, device, eps):
    torch.manual_seed(1234)
    x = torch.randn(128, 256, device=device, dtype=torch.bfloat16)
    residual = torch.randn_like(x)
    grad_norm = torch.randn_like(x)
    grad_residual = torch.randn_like(x)
    reference_norm = RMSNorm(256, eps).to(device=device, dtype=torch.bfloat16)
    actual_norm = RMSNorm(256, eps).to(device=device, dtype=torch.bfloat16)
    actual_norm.load_state_dict(reference_norm.state_dict())
    reference = run("torch", reference_norm, x, residual, grad_norm, grad_residual)
    actual = run(backend, actual_norm, x, residual, grad_norm, grad_residual)
    errors = [relative_l2(value, target) for value, target in zip(actual, reference)]
    if max(errors) >= 0.02:
        raise RuntimeError(
            "residual RMSNorm numerical check failed: "
            + ", ".join(f"tensor_{i}={error:.3e}" for i, error in enumerate(errors))
        )
    print(
        "TRAINING_RESIDUAL_NORM_NUMERICS_OK",
        f"backend={backend}",
        "errors=" + ",".join(f"{error:.3e}" for error in errors),
    )


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    if args.check:
        check_numerics(args.backend, device, args.eps)
        torch.cuda.empty_cache()

    torch.manual_seed(1234)
    norm = RMSNorm(args.width, args.eps).to(device=device, dtype=torch.bfloat16)
    x = torch.randn(args.rows, args.width, device=device, dtype=torch.bfloat16)
    residual = torch.randn_like(x)
    grad_norm = torch.randn_like(x)
    grad_residual = torch.randn_like(x)

    elapsed_ms = []
    torch.cuda.reset_peak_memory_stats()
    for iteration in range(args.warmup + args.iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        run(args.backend, norm, x, residual, grad_norm, grad_residual)
        end.record()
        end.synchronize()
        if iteration >= args.warmup:
            elapsed_ms.append(start.elapsed_time(end))

    median_ms = statistics.median(elapsed_ms)
    p90_ms = sorted(elapsed_ms)[int(0.9 * (len(elapsed_ms) - 1))]
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    print(
        "TRAINING_RESIDUAL_NORM_BENCH",
        f"backend={args.backend}",
        f"shape=R{args.rows}-H{args.width}",
        f"median_ms={median_ms:.3f}",
        f"p90_ms={p90_ms:.3f}",
        f"peak_allocated_gb={peak_gb:.3f}",
    )


if __name__ == "__main__":
    main()
