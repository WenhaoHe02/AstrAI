"""Benchmark mature GQA/MQA training attention kernels at AstrAI shapes."""

import argparse
import statistics

import torch
import torch.nn.functional as F


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend",
        choices=("torch-auto", "torch-flash", "flash-attn"),
        required=True,
    )
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--q-heads", type=int, default=24)
    parser.add_argument("--kv-heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def attention(backend: str, q, k, v):
    if backend == "flash-attn":
        try:
            from flash_attn import flash_attn_func
        except ImportError as exc:
            raise RuntimeError(
                "--backend flash-attn requires the flash-attn package"
            ) from exc
        return flash_attn_func(q, k, v, dropout_p=0.0, causal=True)

    q_bhld = q.transpose(1, 2)
    k_bhld = k.transpose(1, 2)
    v_bhld = v.transpose(1, 2)
    if backend == "torch-auto":
        out = F.scaled_dot_product_attention(
            q_bhld,
            k_bhld,
            v_bhld,
            is_causal=True,
            enable_gqa=q.shape[2] != k.shape[2],
        )
    else:
        from torch.nn.attention import SDPBackend, sdpa_kernel

        with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
            out = F.scaled_dot_product_attention(
                q_bhld,
                k_bhld,
                v_bhld,
                is_causal=True,
                enable_gqa=q.shape[2] != k.shape[2],
            )
    return out.transpose(1, 2)


def run(backend: str, tensors, backward: bool = True):
    q, k, v = (tensor.detach().requires_grad_(backward) for tensor in tensors)
    out = attention(backend, q, k, v)
    if backward:
        out.float().square().mean().backward()
        grads = (q.grad.detach(), k.grad.detach(), v.grad.detach())
    else:
        grads = ()
    return out.detach(), grads


def relative_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    diff = torch.linalg.vector_norm((actual.float() - expected.float()).flatten())
    denom = torch.linalg.vector_norm(expected.float().flatten()).clamp_min(1e-12)
    return (diff / denom).item()


def check_numerics(backend: str, tensors) -> None:
    reference_out, reference_grads = run("torch-auto", tensors)
    actual_out, actual_grads = run(backend, tensors)
    output_error = relative_l2(actual_out, reference_out)
    grad_error = max(
        relative_l2(actual, expected)
        for actual, expected in zip(actual_grads, reference_grads, strict=True)
    )
    if output_error >= 0.02 or grad_error >= 0.05:
        raise RuntimeError(
            f"attention numerical check failed: output={output_error:.3e}, "
            f"gradient={grad_error:.3e}"
        )
    print(
        "ATTENTION_NUMERICS_OK",
        f"backend={backend}",
        f"output_rel_l2={output_error:.3e}",
        f"grad_rel_l2={grad_error:.3e}",
    )


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.q_heads % args.kv_heads != 0:
        raise ValueError("q-heads must be divisible by kv-heads")

    torch.manual_seed(1234)
    device = torch.device("cuda")
    shapes = (
        (args.batch, args.seq_len, args.q_heads, args.head_dim),
        (args.batch, args.seq_len, args.kv_heads, args.head_dim),
        (args.batch, args.seq_len, args.kv_heads, args.head_dim),
    )
    tensors = tuple(
        torch.randn(shape, device=device, dtype=torch.bfloat16) for shape in shapes
    )

    if args.check:
        check_numerics(args.backend, tensors)
        torch.cuda.empty_cache()

    elapsed_ms = []
    for iteration in range(args.warmup + args.iters):
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        run(args.backend, tensors)
        end.record()
        end.synchronize()
        if iteration >= args.warmup:
            elapsed_ms.append(start.elapsed_time(end))

    median_ms = statistics.median(elapsed_ms)
    p90_ms = sorted(elapsed_ms)[int(0.9 * (len(elapsed_ms) - 1))]
    tokens_per_second = args.batch * args.seq_len / (median_ms / 1000)
    print(
        "ATTENTION_TRAINING_BENCH",
        f"backend={args.backend}",
        f"shape=B{args.batch}-S{args.seq_len}-Hq{args.q_heads}-Hkv{args.kv_heads}-D{args.head_dim}",
        f"median_ms={median_ms:.3f}",
        f"p90_ms={p90_ms:.3f}",
        f"tokens_per_second={tokens_per_second:.1f}",
    )


if __name__ == "__main__":
    main()
