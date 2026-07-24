"""Benchmark LM-head plus cross-entropy backends at the 12B recipe shape."""

import argparse
import statistics

import torch
import torch.nn.functional as F


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend",
        choices=("torch-fp32", "torch-native", "liger"),
        required=True,
    )
    parser.add_argument("--tokens", type=int, default=2048)
    parser.add_argument("--hidden-size", type=int, default=3072)
    parser.add_argument("--vocab-size", type=int, default=100000)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def create_loss_fn(backend: str):
    if backend != "liger":
        return None
    try:
        from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
    except ImportError as exc:
        raise RuntimeError("--backend liger requires liger-kernel") from exc
    return LigerFusedLinearCrossEntropyLoss()


def compute_loss(backend, loss_fn, weight, hidden_states, targets):
    if backend == "liger":
        return loss_fn(weight, hidden_states, targets)

    logits = F.linear(hidden_states, weight)
    if backend == "torch-fp32":
        logits = logits.float()
    return F.cross_entropy(logits, targets)


def run(backend, loss_fn, weight, hidden_states, targets):
    weight.grad = None
    hidden_states.grad = None
    loss = compute_loss(backend, loss_fn, weight, hidden_states, targets)
    loss.backward()
    return loss.detach(), hidden_states.grad.detach(), weight.grad.detach()


def relative_l2(actual, expected):
    numerator = torch.linalg.vector_norm((actual.float() - expected.float()).flatten())
    denominator = torch.linalg.vector_norm(expected.float().flatten()).clamp_min(1e-12)
    return (numerator / denominator).item()


def check_numerics(backend, loss_fn, device):
    torch.manual_seed(1234)
    hidden = torch.randn(
        64, 256, device=device, dtype=torch.bfloat16, requires_grad=True
    )
    weight = torch.randn(
        4096, 256, device=device, dtype=torch.bfloat16, requires_grad=True
    )
    targets = torch.randint(4096, (64,), device=device)

    reference = run("torch-fp32", None, weight, hidden, targets)
    actual = run(backend, loss_fn, weight, hidden, targets)
    loss_error = abs(actual[0].float() - reference[0].float()).item() / max(
        abs(reference[0].float()).item(), 1e-12
    )
    hidden_error = relative_l2(actual[1], reference[1])
    weight_error = relative_l2(actual[2], reference[2])
    if loss_error >= 0.02 or hidden_error >= 0.05 or weight_error >= 0.05:
        raise RuntimeError(
            "loss numerical check failed: "
            f"loss={loss_error:.3e}, hidden_grad={hidden_error:.3e}, "
            f"weight_grad={weight_error:.3e}"
        )
    print(
        "TRAINING_LOSS_NUMERICS_OK",
        f"backend={backend}",
        f"loss_rel={loss_error:.3e}",
        f"hidden_grad_rel_l2={hidden_error:.3e}",
        f"weight_grad_rel_l2={weight_error:.3e}",
    )


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    loss_fn = create_loss_fn(args.backend)

    if args.check:
        check_numerics(args.backend, loss_fn, device)
        torch.cuda.empty_cache()

    torch.manual_seed(1234)
    hidden = torch.randn(
        args.tokens,
        args.hidden_size,
        device=device,
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    weight = torch.randn(
        args.vocab_size,
        args.hidden_size,
        device=device,
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    targets = torch.randint(args.vocab_size, (args.tokens,), device=device)

    torch.cuda.reset_peak_memory_stats()
    elapsed_ms = []
    for iteration in range(args.warmup + args.iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        run(args.backend, loss_fn, weight, hidden, targets)
        end.record()
        end.synchronize()
        if iteration >= args.warmup:
            elapsed_ms.append(start.elapsed_time(end))

    median_ms = statistics.median(elapsed_ms)
    p90_ms = sorted(elapsed_ms)[int(0.9 * (len(elapsed_ms) - 1))]
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    print(
        "TRAINING_LOSS_BENCH",
        f"backend={args.backend}",
        f"shape=T{args.tokens}-H{args.hidden_size}-V{args.vocab_size}",
        f"median_ms={median_ms:.3f}",
        f"p90_ms={p90_ms:.3f}",
        f"tokens_per_second={args.tokens / (median_ms / 1000):.1f}",
        f"peak_allocated_gb={peak_gb:.3f}",
    )


if __name__ == "__main__":
    main()
