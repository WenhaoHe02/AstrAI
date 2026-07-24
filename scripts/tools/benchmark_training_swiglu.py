"""Benchmark the SwiGLU activation at AstrAI's 12B expert shapes."""

import argparse
import statistics

import torch

from astrai.model.components.mlp import apply_swiglu


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("torch", "liger"), required=True)
    parser.add_argument("--rows", type=int, default=8192)
    parser.add_argument("--width", type=int, default=2176)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def run(backend, gate_base, up_base, grad_output):
    # Liger intentionally reuses the projection-output buffers for gradients,
    # matching a real MLP where these tensors are newly produced each forward.
    gate = gate_base.clone().requires_grad_()
    up = up_base.clone().requires_grad_()
    output = apply_swiglu(gate, up, backend)
    torch.autograd.backward(output, grad_output)
    return output.detach(), gate.grad.detach(), up.grad.detach()


def relative_l2(actual, expected):
    numerator = torch.linalg.vector_norm((actual.float() - expected.float()).flatten())
    denominator = torch.linalg.vector_norm(expected.float().flatten()).clamp_min(1e-12)
    return (numerator / denominator).item()


def check_numerics(backend, device):
    torch.manual_seed(1234)
    gate = torch.randn(128, 256, device=device, dtype=torch.bfloat16)
    up = torch.randn_like(gate)
    grad_output = torch.randn_like(gate)
    reference = run("torch", gate, up, grad_output)
    actual = run(backend, gate, up, grad_output)
    errors = [relative_l2(value, target) for value, target in zip(actual, reference)]
    if max(errors) >= 0.02:
        raise RuntimeError(
            "SwiGLU numerical check failed: "
            f"output={errors[0]:.3e}, gate_grad={errors[1]:.3e}, "
            f"up_grad={errors[2]:.3e}"
        )
    print(
        "TRAINING_SWIGLU_NUMERICS_OK",
        f"backend={backend}",
        f"output_rel_l2={errors[0]:.3e}",
        f"gate_grad_rel_l2={errors[1]:.3e}",
        f"up_grad_rel_l2={errors[2]:.3e}",
    )


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    if args.check:
        check_numerics(args.backend, device)
        torch.cuda.empty_cache()

    torch.manual_seed(1234)
    gate = torch.randn(args.rows, args.width, device=device, dtype=torch.bfloat16)
    up = torch.randn_like(gate)
    grad_output = torch.randn_like(gate)

    elapsed_ms = []
    torch.cuda.reset_peak_memory_stats()
    for iteration in range(args.warmup + args.iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        run(args.backend, gate, up, grad_output)
        end.record()
        end.synchronize()
        if iteration >= args.warmup:
            elapsed_ms.append(start.elapsed_time(end))

    median_ms = statistics.median(elapsed_ms)
    p90_ms = sorted(elapsed_ms)[int(0.9 * (len(elapsed_ms) - 1))]
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    print(
        "TRAINING_SWIGLU_BENCH",
        f"backend={args.backend}",
        f"shape=R{args.rows}-H{args.width}",
        f"median_ms={median_ms:.3f}",
        f"p90_ms={p90_ms:.3f}",
        f"peak_allocated_gb={peak_gb:.3f}",
    )


if __name__ == "__main__":
    main()
