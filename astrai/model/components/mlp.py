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


def apply_swiglu(gate: Tensor, up: Tensor, backend: str) -> Tensor:
    """Apply SwiGLU without changing the surrounding parameter layout."""
    if backend == "torch" or not gate.is_cuda:
        return F.silu(gate) * up
    if backend != "liger":
        raise ValueError(f"swiglu_backend must be 'torch' or 'liger', got {backend!r}")
    try:
        from liger_kernel.ops import LigerSiLUMulFunction
    except ImportError as exc:
        raise RuntimeError(
            "swiglu_backend='liger' requires the liger-kernel package"
        ) from exc
    return LigerSiLUMulFunction.apply(gate, up)


@FFNFactory.register("mlp")
class MLP(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_ffn: int,
        down_init_std: float = 0.02,
        swiglu_backend: str = "torch",
        fused_mlp_gate_up: bool = False,
    ):
        super().__init__()
        if swiglu_backend not in ("torch", "liger"):
            raise ValueError(
                f"swiglu_backend must be 'torch' or 'liger', got {swiglu_backend!r}"
            )
        self.swiglu_backend = swiglu_backend
        self.dim_ffn = dim_ffn
        self.fused_mlp_gate_up = fused_mlp_gate_up
        if fused_mlp_gate_up:
            # Preserve legacy flat order: up weights first, then gate weights.
            self.up_gate = Linear(dim, 2 * dim_ffn)
        else:
            self.up = Linear(dim, dim_ffn)
            self.gate = Linear(dim, dim_ffn)
        self.down = Linear(dim_ffn, dim, init_std=down_init_std)

    def forward(self, x: Tensor) -> Tensor:
        if self.fused_mlp_gate_up:
            up, gate = self.up_gate(x).split(self.dim_ffn, dim=-1)
        else:
            up, gate = self.up(x), self.gate(x)
        gated = apply_swiglu(gate, up, self.swiglu_backend)
        out = self.down(gated)
        return out

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
        fused_key = prefix + "up_gate.weight"
        up_key, gate_key = prefix + "up.weight", prefix + "gate.weight"
        if self.fused_mlp_gate_up and fused_key not in state_dict:
            if up_key in state_dict and gate_key in state_dict:
                state_dict[fused_key] = torch.cat(
                    [state_dict.pop(up_key), state_dict.pop(gate_key)], dim=0
                )
        elif not self.fused_mlp_gate_up and fused_key in state_dict:
            up, gate = state_dict.pop(fused_key).split(self.dim_ffn, dim=0)
            state_dict[up_key], state_dict[gate_key] = up, gate
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


_EP_GROUP_CACHE: dict[tuple[int, int], tuple[object, int]] = {}
_EXPERT_DP_GROUP_CACHE: dict[
    tuple[int, int], tuple[object | None, int, int]
] = {}
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


def _expert_data_parallel_group(size: int):
    """Return the replica group for this rank's expert shard.

    With ``world_size > expert_parallel_size`` the world is laid out as
    contiguous EP replicas.  Ranks at the same offset in each replica own the
    same expert slice and therefore form an expert data-parallel group.  Every
    rank creates every group in the same order, as required by ``new_group``.
    """
    if size == 1:
        return None, 0, 1
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
    cached = _EXPERT_DP_GROUP_CACHE.get(key)
    if cached is not None:
        return cached

    replicas = world_size // size
    if replicas == 1:
        selected = (None, 0, 1)
    else:
        global_rank = dist.get_rank()
        selected = None
        for expert_rank in range(size):
            ranks = [expert_rank + replica * size for replica in range(replicas)]
            group = dist.new_group(ranks=ranks)
            if global_rank in ranks:
                selected = (group, ranks.index(global_rank), replicas)
        assert selected is not None
    _EXPERT_DP_GROUP_CACHE[key] = selected
    return selected


class GroupedExperts(nn.Module):
    """Jagged grouped-GEMM experts, optionally sharded by expert rank."""

    # On H200, separate NVIDIA library GEMMs beat CUTLASS grouped GEMM for the
    # two-local-expert inference shapes below this size.  Training keeps the
    # grouped path because its backward is materially faster.
    _INDIVIDUAL_LINEAR_MAX_TOKENS = 1024

    def __init__(
        self,
        dim: int,
        dim_ffn: int,
        n_experts: int,
        down_init_std: float,
        expert_parallel_size: int = 1,
        expert_gemm_backend: str = "torch",
        swiglu_backend: str = "torch",
        fused_mlp_gate_up: bool = False,
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
        (
            self.expert_data_parallel_group,
            self.expert_data_parallel_rank,
            self.expert_data_parallel_size,
        ) = _expert_data_parallel_group(expert_parallel_size)
        self.n_local_experts = n_experts // expert_parallel_size
        self.expert_start = self.expert_parallel_rank * self.n_local_experts
        self.expert_end = self.expert_start + self.n_local_experts
        self.down_init_std = down_init_std
        if expert_gemm_backend not in ("torch", "transformer_engine"):
            raise ValueError(
                "expert_gemm_backend must be 'torch' or 'transformer_engine', got "
                f"{expert_gemm_backend!r}"
            )
        self.expert_gemm_backend = expert_gemm_backend
        if swiglu_backend not in ("torch", "liger"):
            raise ValueError(
                f"swiglu_backend must be 'torch' or 'liger', got {swiglu_backend!r}"
            )
        self.swiglu_backend = swiglu_backend
        self.fused_mlp_gate_up = fused_mlp_gate_up

        if expert_gemm_backend == "transformer_engine":
            try:
                from transformer_engine.pytorch import GroupedLinear
            except ImportError as exc:
                raise RuntimeError(
                    "expert_gemm_backend='transformer_engine' requires "
                    "Transformer Engine with GroupedLinear support"
                ) from exc

            def up_init(weight: Tensor) -> None:
                nn.init.normal_(weight, mean=0.0, std=0.02)

            def down_init(weight: Tensor) -> None:
                nn.init.normal_(weight, mean=0.0, std=down_init_std)

            if fused_mlp_gate_up:
                self.te_up_gate = GroupedLinear(
                    self.n_local_experts,
                    dim,
                    2 * dim_ffn,
                    bias=False,
                    init_method=up_init,
                    params_dtype=torch.float32,
                    device="cpu",
                )
            else:
                self.te_up = GroupedLinear(
                    self.n_local_experts,
                    dim,
                    dim_ffn,
                    bias=False,
                    init_method=up_init,
                    params_dtype=torch.float32,
                    device="cpu",
                )
                self.te_gate = GroupedLinear(
                    self.n_local_experts,
                    dim,
                    dim_ffn,
                    bias=False,
                    init_method=up_init,
                    params_dtype=torch.float32,
                    device="cpu",
                )
            self.te_down = GroupedLinear(
                self.n_local_experts,
                dim_ffn,
                dim,
                bias=False,
                init_method=down_init,
                params_dtype=torch.float32,
                device="cpu",
            )
        else:
            if fused_mlp_gate_up:
                self.up_gate_weight = nn.Parameter(
                    torch.empty(self.n_local_experts, 2 * dim_ffn, dim)
                )
            else:
                self.up_weight = nn.Parameter(
                    torch.empty(self.n_local_experts, dim_ffn, dim)
                )
                self.gate_weight = nn.Parameter(
                    torch.empty(self.n_local_experts, dim_ffn, dim)
                )
            self.down_weight = nn.Parameter(
                torch.empty(self.n_local_experts, dim, dim_ffn)
            )

        # FSDP discovers this marker and leaves rank-local expert weights out
        # of data-parallel sharding.  Shared/router/attention parameters still
        # use full sharding.
        self._expert_parallel_local = expert_parallel_size > 1

    def reset_parameters(self):
        # Transformer Engine owns and initializes its child GroupedLinear
        # parameters. Module.apply visits those children before this wrapper.
        if self.expert_gemm_backend == "transformer_engine":
            return
        if self.fused_mlp_gate_up:
            nn.init.normal_(self.up_gate_weight, mean=0.0, std=0.02)
        else:
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
        fused_key = prefix + "up_gate_weight"
        up_key, gate_key = prefix + "up_weight", prefix + "gate_weight"
        down_key = prefix + "down_weight"

        if self.expert_gemm_backend == "transformer_engine":
            # Import legacy packed expert tensors into TE's per-GEMM parameter
            # layout. This permits model-weight continuation across the B200
            # backend switch; optimizer state is intentionally restarted.
            if self.fused_mlp_gate_up and fused_key not in state_dict:
                if up_key in state_dict and gate_key in state_dict:
                    state_dict[fused_key] = torch.cat(
                        [state_dict.pop(up_key), state_dict.pop(gate_key)], dim=1
                    )
            legacy = [fused_key if self.fused_mlp_gate_up else None, up_key, gate_key]
            for key in [key for key in legacy if key is not None] + [down_key]:
                value = state_dict.get(key)
                if value is not None and value.size(0) == self.n_experts:
                    state_dict[key] = value[self.expert_start : self.expert_end]
            if self.fused_mlp_gate_up and fused_key in state_dict:
                value = state_dict.pop(fused_key)
                for expert_idx, weight in enumerate(value):
                    state_dict[prefix + f"te_up_gate.weight{expert_idx}"] = weight
            elif not self.fused_mlp_gate_up:
                for source_key, module_name in (
                    (up_key, "te_up"),
                    (gate_key, "te_gate"),
                ):
                    if source_key in state_dict:
                        value = state_dict.pop(source_key)
                        for expert_idx, weight in enumerate(value):
                            state_dict[prefix + f"{module_name}.weight{expert_idx}"] = weight
            if down_key in state_dict:
                value = state_dict.pop(down_key)
                for expert_idx, weight in enumerate(value):
                    state_dict[prefix + f"te_down.weight{expert_idx}"] = weight
            return super()._load_from_state_dict(
                state_dict,
                prefix,
                local_metadata,
                strict,
                missing_keys,
                unexpected_keys,
                error_msgs,
            )

        if self.fused_mlp_gate_up and fused_key not in state_dict:
            if up_key in state_dict and gate_key in state_dict:
                state_dict[fused_key] = torch.cat(
                    [state_dict.pop(up_key), state_dict.pop(gate_key)], dim=1
                )
        elif not self.fused_mlp_gate_up and fused_key in state_dict:
            up, gate = state_dict.pop(fused_key).split(self.dim_ffn, dim=1)
            state_dict[up_key], state_dict[gate_key] = up, gate
        parameter_names = (
            ("up_gate_weight", "down_weight")
            if self.fused_mlp_gate_up
            else ("up_weight", "gate_weight", "down_weight")
        )
        for name in parameter_names:
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
        # PyTorch 2.8's grouped GEMM kernels are Hopper-only (CC 9.0).  Calling
        # them on Blackwell fails at runtime, so B200 uses the cuBLAS-backed
        # per-expert F.linear fallback below.  Keep the grouped path on H100/H200.
        grouped_mm_supported = (
            x.is_cuda
            and x.dtype == torch.bfloat16
            and torch.cuda.get_device_capability(x.device) == (9, 0)
        )
        if grouped_mm_supported:
            grouped_mm = getattr(F, "grouped_mm", None)
            if grouped_mm is not None:
                return grouped_mm(x, weight.transpose(1, 2), offs=offsets)
            if hasattr(torch, "_grouped_mm"):
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

    @staticmethod
    def _linear_per_expert(
        x: Tensor, weight: Tensor, host_counts: list[int]
    ) -> Tensor:
        """Run one library-backed linear per contiguous expert segment."""
        chunks = []
        start = 0
        for expert_idx, count in enumerate(host_counts):
            end = start + int(count)
            chunks.append(F.linear(x[start:end], weight[expert_idx]))
            start = end
        if not chunks:
            return x.new_empty((0, weight.size(1)))
        return torch.cat(chunks, dim=0)

    def _use_individual_linear(
        self, x: Tensor, host_counts: list[int] | None
    ) -> bool:
        if (
            torch.is_grad_enabled()
            or not x.is_cuda
            or x.dtype != torch.bfloat16
            or host_counts is None
            or len(host_counts) != self.n_local_experts
        ):
            return False
        counts = [int(count) for count in host_counts]
        return (
            all(count >= 0 for count in counts)
            and sum(counts) == x.size(0)
            and max(counts, default=0) <= self._INDIVIDUAL_LINEAR_MAX_TOKENS
        )

    def forward(
        self,
        x: Tensor,
        counts: Tensor,
        row_scale: Tensor | None = None,
        offsets: Tensor | None = None,
        host_counts: list[int] | None = None,
    ) -> Tensor:
        if self.expert_gemm_backend == "transformer_engine":
            if host_counts is None:
                host_counts = [int(value) for value in counts.tolist()]
            if len(host_counts) != self.n_local_experts:
                raise ValueError("host_counts must have one entry per local expert")
            if self.fused_mlp_gate_up:
                up, gate = self.te_up_gate(x, host_counts).split(
                    self.dim_ffn, dim=-1
                )
            else:
                up = self.te_up(x, host_counts)
                gate = self.te_gate(x, host_counts)
            hidden = apply_swiglu(gate, up, self.swiglu_backend)
            if row_scale is not None:
                if row_scale.ndim != 1 or row_scale.size(0) != hidden.size(0):
                    raise ValueError("row_scale must have one value per expert input row")
                hidden = hidden * row_scale.to(hidden.dtype).unsqueeze(-1)
            return self.te_down(hidden, host_counts)

        if offsets is None:
            offsets = counts.cumsum(0, dtype=torch.int32)
        elif offsets.dtype != torch.int32:
            offsets = offsets.to(dtype=torch.int32)
        use_individual_linear = self._use_individual_linear(x, host_counts)

        def project(weight: Tensor) -> Tensor:
            if use_individual_linear:
                assert host_counts is not None
                return self._linear_per_expert(x, weight, host_counts)
            return self._grouped_mm(x, weight, offsets)

        if self.fused_mlp_gate_up:
            up, gate = project(self.up_gate_weight).split(self.dim_ffn, dim=-1)
        else:
            up = project(self.up_weight)
            gate = project(self.gate_weight)
        hidden = apply_swiglu(gate, up, self.swiglu_backend)
        if row_scale is not None:
            if row_scale.ndim != 1 or row_scale.size(0) != hidden.size(0):
                raise ValueError("row_scale must have one value per expert input row")
            # Scaling commutes with the bias-free down projection.  Applying
            # routing weights at the smaller FFN width reduces the row-scale
            # kernel's memory traffic versus scaling the model-width output.
            hidden = hidden * row_scale.to(hidden.dtype).unsqueeze(-1)
        if use_individual_linear:
            assert host_counts is not None
            return self._linear_per_expert(hidden, self.down_weight, host_counts)
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
        expert_gemm_backend: str = "torch",
        deepep_expert_alignment: int = 1,
        deepep_overlap_with_compute: bool = False,
        deepep_cpu_sync: bool = True,
        moe_shared_expert_overlap: bool = False,
        moe_route_scale_before_down: bool = False,
        swiglu_backend: str = "torch",
        fused_mlp_gate_up: bool = False,
        router_score_dtype: str = "model",
    ):
        super().__init__()
        self.dim = dim
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.n_activated_experts = n_activated_experts
        self.topk_method = topk_method or "greedy"
        self.expert_parallel_size = expert_parallel_size
        self.expert_dispatch_backend = expert_dispatch_backend
        self.expert_gemm_backend = expert_gemm_backend
        self.deepep_expert_alignment = deepep_expert_alignment
        self.deepep_overlap_with_compute = deepep_overlap_with_compute
        self.deepep_cpu_sync = deepep_cpu_sync
        self.moe_shared_expert_overlap = moe_shared_expert_overlap
        self.moe_route_scale_before_down = moe_route_scale_before_down
        if router_score_dtype not in ("model", "fp32"):
            raise ValueError(
                "router_score_dtype must be 'model' or 'fp32', got "
                f"{router_score_dtype!r}"
            )
        self.router_score_dtype = router_score_dtype

        if expert_dispatch_backend not in ("torch", "deepep"):
            raise ValueError(
                "expert_dispatch_backend must be 'torch' or 'deepep', got "
                f"{expert_dispatch_backend!r}"
            )
        if expert_dispatch_backend == "deepep" and expert_parallel_size == 1:
            raise ValueError("DeepEP dispatch requires expert_parallel_size > 1")
        if expert_gemm_backend not in ("torch", "transformer_engine"):
            raise ValueError(
                "expert_gemm_backend must be 'torch' or 'transformer_engine', got "
                f"{expert_gemm_backend!r}"
            )
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
                MLP(
                    dim,
                    dim_ffn,
                    down_init_std=down_init_std,
                    swiglu_backend=swiglu_backend,
                    fused_mlp_gate_up=fused_mlp_gate_up,
                )
                for _ in range(n_shared_experts)
            ]
        )
        self.routed_experts = GroupedExperts(
            dim,
            dim_ffn,
            n_routed_experts,
            down_init_std=down_init_std,
            expert_parallel_size=expert_parallel_size,
            expert_gemm_backend=expert_gemm_backend,
            swiglu_backend=swiglu_backend,
            fused_mlp_gate_up=fused_mlp_gate_up,
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
            shared_out = (
                self._shared_forward(x_flat) if self.n_shared_experts > 0 else None
            )
            routed_out, aux_loss, z_loss, expert_load, router_entropy = (
                self._routed_forward(x_flat)
            )

        if self.n_shared_experts == 0:
            out = routed_out.view(bsz, seq_len, dim)
        else:
            assert shared_out is not None
            out = (shared_out + routed_out).view(bsz, seq_len, dim)
        return out, aux_loss, z_loss, expert_load, router_entropy

    def _shared_forward(self, x: Tensor) -> Tensor:
        if self.n_shared_experts == 0:
            return torch.zeros_like(x)
        if self.n_shared_experts == 1:
            return self.shared_experts[0](x)
        return sum(e(x) for e in self.shared_experts) / self.n_shared_experts

    def _routed_forward(self, x: Tensor):
        N, D = x.shape
        K = self.n_activated_experts

        router_logits = self.router(x)
        router_logits_fp32 = router_logits.float()
        router_probs_fp32 = torch.softmax(router_logits_fp32, dim=-1)
        router_probs = (
            router_probs_fp32
            if self.router_score_dtype == "fp32"
            else router_probs_fp32.to(x.dtype)
        )

        topk_weights, topk_indices = torch.topk(router_probs, K, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        # Switch-style differentiable load-balancing loss. Expert load is a
        # non-differentiable routing statistic; bincount avoids materializing
        # the [tokens, top-k, experts] one-hot tensor on every layer.
        expert_idx = topk_indices.reshape(-1)
        expert_load = torch.bincount(expert_idx, minlength=self.n_routed_experts).to(
            router_probs_fp32.dtype
        )
        expert_load = expert_load / (N * K)
        mean_router_prob = router_probs_fp32.mean(dim=0)
        aux_loss = self.n_routed_experts * torch.sum(expert_load * mean_router_prob)
        router_logsumexp = torch.logsumexp(router_logits_fp32, dim=-1)
        z_loss = router_logsumexp.square().mean()
        router_entropy = (
            router_logsumexp - torch.sum(router_probs_fp32 * router_logits_fp32, dim=-1)
        ).mean()

        if self.expert_dispatch_backend == "deepep":
            output = self._deepep_forward(x, topk_indices, topk_weights.float())
        else:
            token_idx = torch.arange(N, device=x.device).repeat_interleave(K)
            assignment_weights = topk_weights.to(x.dtype).reshape(-1)
            if self.expert_parallel_size > 1:
                assignment_output = self._expert_parallel_forward(
                    x[token_idx], expert_idx
                )
            else:
                assignment_output = self._local_grouped_forward(
                    x[token_idx], expert_idx
                )
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
        host_counts = None
        if self.expert_gemm_backend == "transformer_engine":
            host_counts = [int(value) for value in counts.tolist()]
        grouped_out = self.routed_experts(
            grouped_x,
            counts,
            host_counts=host_counts,
        )
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
            do_cpu_sync=self.deepep_cpu_sync,
        )
        recv_weights = recv_weights[: recv_x.size(0)]
        host_counts = state.counts if isinstance(state.counts, list) else None
        if self.moe_route_scale_before_down:
            expert_out = self.routed_experts(
                recv_x,
                counts,
                row_scale=recv_weights,
                offsets=state.expert_offsets,
                host_counts=host_counts,
            )
            weighted_out = expert_out
        else:
            expert_out = self.routed_experts(
                recv_x,
                counts,
                offsets=state.expert_offsets,
                host_counts=host_counts,
            )
            weighted_out = expert_out * recv_weights.to(expert_out.dtype).unsqueeze(-1)
        if not state.do_cpu_sync:
            valid_rows = (
                torch.arange(expert_out.size(0), device=expert_out.device)
                < counts.sum()
            )
            weighted_out = torch.where(
                valid_rows.unsqueeze(-1),
                weighted_out,
                torch.zeros((), dtype=weighted_out.dtype, device=weighted_out.device),
            )
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
