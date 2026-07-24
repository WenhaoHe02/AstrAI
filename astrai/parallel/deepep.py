"""Optional DeepEP V2 autograd bridge for expert-parallel dispatch/combine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
from torch import Tensor


_BUFFER_CACHE: dict[tuple[int, int, int, int, bool], Any] = {}


def _load_deep_ep():
    try:
        import deep_ep
    except ImportError as exc:
        raise RuntimeError(
            "expert_dispatch_backend='deepep' requires DeepEP V2. "
            "Install deepseek-ai/DeepEP in the training environment."
        ) from exc

    if not hasattr(deep_ep, "ElasticBuffer"):
        raise RuntimeError("AstrAI's DeepEP backend requires the V2 ElasticBuffer API.")
    return deep_ep


def _get_buffer(
    group: dist.ProcessGroup,
    num_max_tokens_per_rank: int,
    hidden: int,
    num_topk: int,
    prefer_overlap_with_compute: bool,
):
    key = (
        id(group),
        num_max_tokens_per_rank,
        hidden,
        num_topk,
        prefer_overlap_with_compute,
    )
    buffer = _BUFFER_CACHE.get(key)
    if buffer is None:
        deep_ep = _load_deep_ep()
        buffer = deep_ep.ElasticBuffer(
            group,
            num_max_tokens_per_rank=num_max_tokens_per_rank,
            hidden=hidden,
            num_topk=num_topk,
            use_fp8_dispatch=False,
            allow_hybrid_mode=False,
            # Expand mode needs multiple reduction to carry top-k weight
            # gradients through dispatch-backward's combine operation.
            allow_multiple_reduction=True,
            prefer_overlap_with_compute=prefer_overlap_with_compute,
        )
        _BUFFER_CACHE[key] = buffer
    return buffer


@dataclass
class DeepEPDispatchState:
    """Per-forward routing state shared by dispatch and combine autograd ops."""

    buffer: Any
    num_experts: int
    num_max_tokens_per_rank: int
    expert_alignment: int = 1
    do_cpu_sync: bool = True
    num_sms: int = 0
    handle: Any = None
    counts: list[int] | Tensor | None = None


def _counts_from_gpu_prefix(psum: Tensor, alignment: int) -> Tensor:
    """Recover padded expert counts from DeepEP's expand-mode GPU prefix."""
    aligned_ends = (
        torch.div(
            psum + alignment - 1,
            alignment,
            rounding_mode="floor",
        )
        * alignment
    )
    starts = torch.cat((torch.zeros_like(aligned_ends[:1]), aligned_ends[:-1]))
    return aligned_ends - starts


class _Dispatch(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: Tensor,
        topk_idx: Tensor,
        topk_weights: Tensor,
        state: DeepEPDispatchState,
    ) -> tuple[Tensor, Tensor]:
        deep_ep = _load_deep_ep()
        topk_idx = topk_idx.to(dtype=deep_ep.topk_idx_t).contiguous()
        topk_weights = topk_weights.float().contiguous()

        recv_x, _, recv_topk_weights, handle, _ = state.buffer.dispatch(
            x.contiguous(),
            topk_idx=topk_idx,
            topk_weights=topk_weights,
            num_experts=state.num_experts,
            num_max_tokens_per_rank=state.num_max_tokens_per_rank,
            expert_alignment=state.expert_alignment,
            do_expand=True,
            do_zero_padding=state.expert_alignment > 1,
            do_cpu_sync=state.do_cpu_sync,
            num_sms=state.num_sms,
            async_with_compute_stream=False,
        )
        if recv_topk_weights is None:
            raise RuntimeError("DeepEP dispatch did not return routing weights")

        state.handle = handle
        if state.do_cpu_sync:
            state.counts = handle.num_recv_tokens_per_expert_list
        else:
            state.counts = _counts_from_gpu_prefix(
                handle.psum_num_recv_tokens_per_expert,
                state.expert_alignment,
            )
        ctx.state = state
        return recv_x, recv_topk_weights

    @staticmethod
    def backward(ctx, grad_recv_x: Tensor, grad_recv_topk_weights: Tensor):
        state = ctx.state
        grad_x, grad_topk_weights, _ = state.buffer.combine(
            grad_recv_x.contiguous(),
            handle=state.handle,
            topk_weights=grad_recv_topk_weights.float().contiguous(),
            num_sms=state.num_sms,
            async_with_compute_stream=False,
        )
        return grad_x, None, grad_topk_weights, None


class _Combine(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, state: DeepEPDispatchState) -> Tensor:
        combined_x, _, _ = state.buffer.combine(
            x.contiguous(),
            handle=state.handle,
            num_sms=state.num_sms,
            async_with_compute_stream=False,
        )
        ctx.state = state
        return combined_x

    @staticmethod
    def backward(ctx, grad_combined_x: Tensor):
        state = ctx.state
        grad_x, _, _, _, _ = state.buffer.dispatch(
            grad_combined_x.contiguous(),
            handle=state.handle,
            do_expand=True,
            do_zero_padding=state.expert_alignment > 1,
            num_sms=state.num_sms,
            async_with_compute_stream=False,
        )
        return grad_x, None


def dispatch(
    x: Tensor,
    topk_idx: Tensor,
    topk_weights: Tensor,
    group: dist.ProcessGroup,
    num_experts: int,
    expert_alignment: int = 1,
    prefer_overlap_with_compute: bool = False,
    do_cpu_sync: bool = True,
) -> tuple[Tensor, Tensor, Tensor, DeepEPDispatchState]:
    """Expand and dispatch tokens, returning expert-grouped local tensors."""
    if not x.is_cuda or x.dtype != torch.bfloat16:
        raise RuntimeError("DeepEP dispatch requires CUDA BF16 hidden states")
    if group is None:
        raise RuntimeError("DeepEP dispatch requires an expert process group")
    if expert_alignment < 1:
        raise ValueError("DeepEP expert alignment must be positive")

    num_tokens, hidden = x.shape
    num_topk = topk_idx.shape[1]
    buffer = _get_buffer(
        group,
        num_tokens,
        hidden,
        num_topk,
        prefer_overlap_with_compute,
    )
    state = DeepEPDispatchState(
        buffer=buffer,
        num_experts=num_experts,
        num_max_tokens_per_rank=num_tokens,
        expert_alignment=expert_alignment,
        do_cpu_sync=do_cpu_sync,
        num_sms=buffer.get_theoretical_num_sms(num_experts, num_topk),
    )
    recv_x, recv_weights = _Dispatch.apply(x, topk_idx, topk_weights, state)
    if state.counts is None:
        raise RuntimeError("DeepEP dispatch did not return expert token counts")
    counts = torch.as_tensor(state.counts, dtype=torch.int64, device=x.device)
    return recv_x, recv_weights, counts, state


def combine(x: Tensor, state: DeepEPDispatchState) -> Tensor:
    """Return expert outputs to their source ranks and reduce top-k routes."""
    return _Combine.apply(x, state)
