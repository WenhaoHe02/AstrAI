import torch
import torch.distributed as dist
import torch.distributed.nn.functional as dist_nn
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from astrai.factory import BaseFactory
from astrai.model.components.linear import Linear


class FFNFactory(BaseFactory[nn.Module]):
    pass


@FFNFactory.register("mlp")
class MLP(nn.Module):
    def __init__(self, dim: int, dim_ffn: int, down_init_std: float = 0.02):
        super().__init__()
        self.up = Linear(dim, dim_ffn)
        self.gate = Linear(dim, dim_ffn)
        self.down = Linear(dim_ffn, dim, init_std=down_init_std)

    def forward(self, x: Tensor) -> Tensor:
        gated = self.up(x) * F.silu(self.gate(x))
        out = self.down(gated)
        return out


_EP_GROUP_CACHE: dict[tuple[int, int], tuple[object, int]] = {}
_SHARED_EXPERT_STREAM_CACHE: dict[int, torch.cuda.Stream] = {}


def _shared_expert_stream(device: torch.device) -> torch.cuda.Stream:
    device_index = device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    stream = _SHARED_EXPERT_STREAM_CACHE.get(device_index)
    if stream is None:
        stream = torch.cuda.Stream(device=device_index)
        _SHARED_EXPERT_STREAM_CACHE[device_index] = stream
    return stream


def _expert_parallel_group(size: int):
    """Return this rank's contiguous expert-parallel group and local rank."""
    if size == 1:
        return None, 0
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError(
            "expert_parallel_size > 1 requires an initialized process group"
        )
    world_size = dist.get_world_size()
    if world_size % size != 0:
        raise ValueError(
            f"world_size ({world_size}) must be divisible by "
            f"expert_parallel_size ({size})"
        )

    key = (world_size, size)
    cached = _EP_GROUP_CACHE.get(key)
    if cached is not None:
        return cached

    global_rank = dist.get_rank()
    selected = None
    for start in range(0, world_size, size):
        ranks = list(range(start, start + size))
        group = dist.new_group(ranks=ranks)
        if global_rank in ranks:
            selected = (group, global_rank - start)
    assert selected is not None
    _EP_GROUP_CACHE[key] = selected
    return selected


class GroupedExperts(nn.Module):
    """Jagged grouped-GEMM experts, optionally sharded by expert rank."""

    def __init__(
        self,
        dim: int,
        dim_ffn: int,
        n_experts: int,
        down_init_std: float,
        expert_parallel_size: int = 1,
    ):
        super().__init__()
        if n_experts % expert_parallel_size != 0:
            raise ValueError(
                f"n_routed_experts ({n_experts}) must be divisible by "
                f"expert_parallel_size ({expert_parallel_size})"
            )
        self.dim = dim
        self.dim_ffn = dim_ffn
        self.n_experts = n_experts
        self.expert_parallel_size = expert_parallel_size
        self.process_group, self.expert_parallel_rank = _expert_parallel_group(
            expert_parallel_size
        )
        self.n_local_experts = n_experts // expert_parallel_size
        self.expert_start = self.expert_parallel_rank * self.n_local_experts
        self.expert_end = self.expert_start + self.n_local_experts
        self.down_init_std = down_init_std

        self.up_weight = nn.Parameter(torch.empty(self.n_local_experts, dim_ffn, dim))
        self.gate_weight = nn.Parameter(torch.empty(self.n_local_experts, dim_ffn, dim))
        self.down_weight = nn.Parameter(torch.empty(self.n_local_experts, dim, dim_ffn))

        # FSDP discovers this marker and leaves rank-local expert weights out
        # of data-parallel sharding.  Shared/router/attention parameters still
        # use full sharding.
        self._expert_parallel_local = expert_parallel_size > 1

    def reset_parameters(self):
        nn.init.normal_(self.up_weight, mean=0.0, std=0.02)
        nn.init.normal_(self.gate_weight, mean=0.0, std=0.02)
        nn.init.normal_(self.down_weight, mean=0.0, std=self.down_init_std)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        # Checkpoints store global expert tensors.  Each EP rank loads only
        # its contiguous expert slice before FSDP wrapping.
        for name in ("up_weight", "gate_weight", "down_weight"):
            key = prefix + name
            value = state_dict.get(key)
            if value is not None and value.size(0) == self.n_experts:
                state_dict[key] = value[self.expert_start : self.expert_end]
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    @staticmethod
    def _grouped_mm(x: Tensor, weight: Tensor, offsets: Tensor) -> Tensor:
        grouped_mm = getattr(F, "grouped_mm", None)
        if grouped_mm is not None and x.is_cuda and x.dtype == torch.bfloat16:
            return grouped_mm(x, weight.transpose(1, 2), offs=offsets)
        if hasattr(torch, "_grouped_mm") and x.is_cuda and x.dtype == torch.bfloat16:
            return torch._grouped_mm(x, weight.transpose(1, 2), offs=offsets)

        # CPU/unsupported-dtype correctness fallback used by unit tests and
        # tiny smoke checks. Production BF16 CUDA runs must take grouped_mm.
        chunks = []
        start = 0
        for expert_idx, end in enumerate(offsets.tolist()):
            chunks.append(F.linear(x[start:end], weight[expert_idx]))
            start = end
        if not chunks:
            return x.new_empty((0, weight.size(1)))
        return torch.cat(chunks, dim=0)

    def forward(self, x: Tensor, counts: Tensor) -> Tensor:
        offsets = counts.cumsum(0, dtype=torch.int32)
        up = self._grouped_mm(x, self.up_weight, offsets)
        gate = self._grouped_mm(x, self.gate_weight, offsets)
        hidden = up * F.silu(gate)
        return self._grouped_mm(hidden, self.down_weight, offsets)


@FFNFactory.register("moe")
class DeepSeekMoE(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_ffn: int,
        n_routed_experts: int,
        n_shared_experts: int = 1,
        n_activated_experts: int = 2,
        topk_method: str = "greedy",
        n_layers: int = 1,
        expert_parallel_size: int = 1,
        expert_dispatch_backend: str = "torch",
        deepep_expert_alignment: int = 1,
        deepep_overlap_with_compute: bool = False,
        moe_shared_expert_overlap: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.n_activated_experts = n_activated_experts
        self.topk_method = topk_method or "greedy"
        self.expert_parallel_size = expert_parallel_size
        self.expert_dispatch_backend = expert_dispatch_backend
        self.deepep_expert_alignment = deepep_expert_alignment
        self.deepep_overlap_with_compute = deepep_overlap_with_compute
        self.moe_shared_expert_overlap = moe_shared_expert_overlap

        if expert_dispatch_backend not in ("torch", "deepep"):
            raise ValueError(
                "expert_dispatch_backend must be 'torch' or 'deepep', got "
                f"{expert_dispatch_backend!r}"
            )
        if expert_dispatch_backend == "deepep" and expert_parallel_size == 1:
            raise ValueError("DeepEP dispatch requires expert_parallel_size > 1")
        if deepep_expert_alignment < 1:
            raise ValueError("deepep_expert_alignment must be positive")

        if self.topk_method != "greedy":
            raise ValueError(f"Unsupported MoE top-k method: {self.topk_method!r}")
        if not 0 < n_activated_experts <= n_routed_experts:
            raise ValueError("n_activated_experts must be in [1, n_routed_experts]")

        self.router = Linear(dim, n_routed_experts, bias=False)
        moe_scale = 1 / max(n_shared_experts, 1) + 1 / n_activated_experts
        down_init_std = 0.02 / (2 * n_layers * moe_scale) ** 0.5

        self.shared_experts = nn.ModuleList(
            [
                MLP(dim, dim_ffn, down_init_std=down_init_std)
                for _ in range(n_shared_experts)
            ]
        )
        self.routed_experts = GroupedExperts(
            dim,
            dim_ffn,
            n_routed_experts,
            down_init_std=down_init_std,
            expert_parallel_size=expert_parallel_size,
        )

    def forward(self, x: Tensor):
        bsz, seq_len, dim = x.shape
        x_flat = x.view(-1, dim)

        if (
            self.moe_shared_expert_overlap
            and self.n_shared_experts > 0
            and x_flat.is_cuda
        ):
            current_stream = torch.cuda.current_stream(x_flat.device)
            shared_stream = _shared_expert_stream(x_flat.device)
            shared_stream.wait_stream(current_stream)
            with torch.cuda.stream(shared_stream):
                shared_out = self._shared_forward(x_flat)
                x_flat.record_stream(shared_stream)

            routed_out, aux_loss, z_loss, expert_load, router_entropy = (
                self._routed_forward(x_flat)
            )
            current_stream.wait_stream(shared_stream)
            shared_out.record_stream(current_stream)
        else:
            shared_out = self._shared_forward(x_flat)
            routed_out, aux_loss, z_loss, expert_load, router_entropy = (
                self._routed_forward(x_flat)
            )

        out = (shared_out + routed_out).view(bsz, seq_len, dim)
        return out, aux_loss, z_loss, expert_load, router_entropy

    def _shared_forward(self, x: Tensor) -> Tensor:
        if self.n_shared_experts == 0:
            return torch.zeros_like(x)
        return sum(e(x) for e in self.shared_experts) / self.n_shared_experts

    def _routed_forward(self, x: Tensor):
        N, D = x.shape
        K = self.n_activated_experts

        router_logits = self.router(x)
        router_logits_fp32 = router_logits.float()
        router_probs_fp32 = torch.softmax(router_logits_fp32, dim=-1)
        router_probs = router_probs_fp32.to(x.dtype)

        topk_weights, topk_indices = torch.topk(router_probs, K, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        # Switch-style differentiable load-balancing loss.  Expert load is
        # measured over all top-k assignments, so a uniform router has loss 1.
        assignments = F.one_hot(topk_indices, num_classes=self.n_routed_experts).float()
        expert_load = assignments.mean(dim=(0, 1))
        mean_router_prob = router_probs_fp32.mean(dim=0)
        aux_loss = self.n_routed_experts * torch.sum(expert_load * mean_router_prob)
        z_loss = torch.logsumexp(router_logits_fp32, dim=-1).square().mean()
        router_entropy = -torch.sum(
            router_probs_fp32
            * torch.log(router_probs_fp32.clamp_min(torch.finfo(torch.float32).tiny)),
            dim=-1,
        ).mean()

        token_idx = torch.arange(N, device=x.device).repeat_interleave(K)
        expert_idx = topk_indices.reshape(-1)
        assignment_weights = topk_weights.reshape(-1)

        if self.expert_dispatch_backend == "deepep":
            output = self._deepep_forward(x, topk_indices, topk_weights.float())
        elif self.expert_parallel_size > 1:
            assignment_output = self._expert_parallel_forward(x[token_idx], expert_idx)
        else:
            assignment_output = self._local_grouped_forward(x[token_idx], expert_idx)

        if self.expert_dispatch_backend != "deepep":
            output = torch.zeros(N, D, device=x.device, dtype=x.dtype)
            output.index_add_(
                0,
                token_idx,
                assignment_output * assignment_weights.unsqueeze(-1),
            )

        return (
            output,
            aux_loss,
            z_loss,
            expert_load.detach(),
            router_entropy.detach(),
        )

    def _local_grouped_forward(self, assignment_x: Tensor, expert_idx: Tensor):
        order = torch.argsort(expert_idx, stable=True)
        grouped_x = assignment_x[order]
        counts = torch.bincount(expert_idx, minlength=self.n_routed_experts)
        grouped_out = self.routed_experts(grouped_x, counts)
        output = torch.empty_like(grouped_out)
        output[order] = grouped_out
        return output

    def _deepep_forward(
        self, x: Tensor, topk_indices: Tensor, topk_weights: Tensor
    ) -> Tensor:
        from astrai.parallel.deepep import combine, dispatch

        recv_x, recv_weights, counts, state = dispatch(
            x,
            topk_indices,
            topk_weights,
            self.routed_experts.process_group,
            self.n_routed_experts,
            expert_alignment=self.deepep_expert_alignment,
            prefer_overlap_with_compute=self.deepep_overlap_with_compute,
        )
        expert_out = self.routed_experts(recv_x, counts)
        weighted_out = expert_out * recv_weights.to(expert_out.dtype).unsqueeze(-1)
        return combine(weighted_out, state)

    def _expert_parallel_forward(self, assignment_x: Tensor, expert_idx: Tensor):
        ep_size = self.expert_parallel_size
        n_local = self.routed_experts.n_local_experts
        destination = torch.div(expert_idx, n_local, rounding_mode="floor")
        send_order = torch.argsort(destination, stable=True)
        send_x = assignment_x[send_order].contiguous()
        send_local_expert = (expert_idx[send_order] % n_local).contiguous()
        send_counts = torch.bincount(destination, minlength=ep_size).to(torch.int64)

        recv_counts = torch.empty_like(send_counts)
        dist.all_to_all_single(
            recv_counts,
            send_counts,
            group=self.routed_experts.process_group,
        )
        send_splits = send_counts.cpu().tolist()
        recv_splits = recv_counts.cpu().tolist()
        recv_total = sum(recv_splits)

        recv_x_buffer = send_x.new_empty((recv_total, send_x.size(1)))
        recv_x = dist_nn.all_to_all_single(
            recv_x_buffer,
            send_x,
            output_split_sizes=recv_splits,
            input_split_sizes=send_splits,
            group=self.routed_experts.process_group,
        )
        recv_expert = send_local_expert.new_empty(recv_total)
        dist.all_to_all_single(
            recv_expert,
            send_local_expert,
            output_split_sizes=recv_splits,
            input_split_sizes=send_splits,
            group=self.routed_experts.process_group,
        )

        recv_order = torch.argsort(recv_expert, stable=True)
        counts = torch.bincount(recv_expert, minlength=n_local)
        grouped_out = self.routed_experts(recv_x[recv_order], counts)
        recv_out = torch.empty_like(grouped_out)
        recv_out[recv_order] = grouped_out

        returned_buffer = recv_out.new_empty(send_x.shape)
        returned = dist_nn.all_to_all_single(
            returned_buffer,
            recv_out,
            output_split_sizes=send_splits,
            input_split_sizes=recv_splits,
            group=self.routed_experts.process_group,
        )
        assignment_output = torch.empty_like(returned)
        assignment_output[send_order] = returned
        return assignment_output
