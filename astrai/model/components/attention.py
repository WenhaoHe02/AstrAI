import inspect
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.attention import SDPBackend, sdpa_kernel

try:
    from torch.nn.attention.varlen import varlen_attn
except ImportError:  # pragma: no cover - exercised only on older PyTorch
    varlen_attn = None

_VARLEN_SUPPORTS_ENABLE_GQA = (
    varlen_attn is not None
    and "enable_gqa" in inspect.signature(varlen_attn).parameters
)

from astrai.factory import BaseFactory
from astrai.inference.core.cache import CacheView
from astrai.model.components.linear import Linear
from astrai.model.components.norm import RMSNorm
from astrai.model.components.rope import apply_rotary_emb


def repeat_kv(x: Tensor, n_rep: int) -> Tensor:
    bs, slen, n_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_heads, n_rep, head_dim)
        .reshape(bs, slen, n_heads * n_rep, head_dim)
    )


def _run_varlen_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    cu_seqlens: Tensor,
    max_seqlen: int,
) -> Tensor:
    if varlen_attn is None:
        raise RuntimeError(
            "document-aware Flash attention requires "
            "torch.nn.attention.varlen.varlen_attn"
        )
    kwargs = {"window_size": (-1, 0)}
    if _VARLEN_SUPPORTS_ENABLE_GQA:
        kwargs["enable_gqa"] = q.size(1) != k.size(1)
    return varlen_attn(
        q,
        k,
        v,
        cu_seqlens,
        cu_seqlens,
        max_seqlen,
        max_seqlen,
        **kwargs,
    )


class AttnFactory(BaseFactory[nn.Module]):
    pass


@AttnFactory.register("gqa")
class GQA(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        use_qk_norm: bool,
        norm_eps: float,
        use_gated_attention: bool,
        layer_id: int,
        n_layers: int = 1,
        attention_backend: str = "auto",
        fused_qkv: bool = False,
    ):
        super().__init__()
        assert dim % n_heads == 0
        assert n_heads % n_kv_heads == 0

        self.head_dim = dim // n_heads
        self.layer_id = layer_id
        self.dim = dim
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep = n_heads // n_kv_heads
        self.use_qk_norm = use_qk_norm
        self.use_gated_attention = use_gated_attention
        self.attention_backend = attention_backend
        self.fused_qkv = fused_qkv
        if attention_backend not in ("auto", "flash_sdpa", "transformer_engine"):
            raise ValueError(
                "attention_backend must be 'auto', 'flash_sdpa', or "
                "'transformer_engine', got "
                f"{attention_backend!r}"
            )

        self.core_attention = None
        if attention_backend == "transformer_engine":
            try:
                from transformer_engine.pytorch import DotProductAttention
            except ImportError as exc:
                raise RuntimeError(
                    "attention_backend='transformer_engine' requires "
                    "Transformer Engine with PyTorch support"
                ) from exc
            # BF16 mode only: TE selects its Hopper FlashAttention/cuDNN fused
            # attention implementation without changing model parameters.
            self.core_attention = DotProductAttention(
                num_attention_heads=n_heads,
                kv_channels=self.head_dim,
                num_gqa_groups=n_kv_heads,
                attention_dropout=0.0,
                attn_mask_type="causal",
                qkv_format="bshd",
                layer_number=layer_id + 1,
            )

        self.q_size = n_heads * self.head_dim
        self.kv_size = n_kv_heads * self.head_dim
        if fused_qkv:
            self.qkv_proj = Linear(dim, self.q_size + 2 * self.kv_size)
        else:
            self.q_proj = Linear(dim, self.q_size)
            self.k_proj = Linear(dim, self.kv_size)
            self.v_proj = Linear(dim, self.kv_size)
        self.o_proj = Linear(dim, dim, init_std=0.02 / (2 * n_layers) ** 0.5)

        if self.use_qk_norm:
            self.q_norm = RMSNorm(self.head_dim, norm_eps)
            self.k_norm = RMSNorm(self.head_dim, norm_eps)

        if self.use_gated_attention:
            self.gate = Linear(dim, dim)

    def _split_heads(self, x: Tensor, n_heads) -> Tensor:
        batch_size, seq_len, _ = x.shape
        x = x.reshape(batch_size, seq_len, n_heads, self.head_dim)
        return x

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
        fused_key = prefix + "qkv_proj.weight"
        legacy_keys = [
            prefix + "q_proj.weight",
            prefix + "k_proj.weight",
            prefix + "v_proj.weight",
        ]
        if self.fused_qkv and fused_key not in state_dict:
            if all(key in state_dict for key in legacy_keys):
                state_dict[fused_key] = torch.cat(
                    [state_dict.pop(key) for key in legacy_keys], dim=0
                )
        elif not self.fused_qkv and fused_key in state_dict:
            q, k, v = state_dict.pop(fused_key).split(
                [self.q_size, self.kv_size, self.kv_size], dim=0
            )
            for key, value in zip(legacy_keys, (q, k, v)):
                state_dict[key] = value
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(
        self,
        x: Tensor,
        rotary_emb: Tensor,
        attn_mask: Tensor = None,
        paged_cache: Optional[CacheView] = None,
        is_causal: bool = False,
        document_cu_seqlens: Optional[Tensor] = None,
        document_max_seqlen: Optional[int] = None,
    ) -> Tensor:
        if self.fused_qkv:
            q, k, v = self.qkv_proj(x).split(
                [self.q_size, self.kv_size, self.kv_size], dim=-1
            )
        else:
            q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        q = self._split_heads(q, self.n_heads)
        k = self._split_heads(k, self.n_kv_heads)
        v = self._split_heads(v, self.n_kv_heads)
        q, k = apply_rotary_emb(q, rotary_emb), apply_rotary_emb(k, rotary_emb)

        if self.use_qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)

        if paged_cache is not None:
            paged_cache.write(self.layer_id, k, v)

            # Decode kernels use split-KV parallelism.  A PageCacheView also
            # reads the page table directly, avoiding a full contiguous KV
            # gather.  These forward-only kernels are inference-only; cache
            # views return None while autograd is enabled.
            cache_out = paged_cache.attend(
                self.layer_id,
                q,
                mask=attn_mask,
                causal_offset=-1,
            )
            if cache_out is not None:
                sdqa_out = cache_out.contiguous().flatten(2)
                if self.use_gated_attention:
                    sdqa_out = sdqa_out * F.sigmoid(self.gate(x))
                return self.o_proj(sdqa_out)

            k, v = paged_cache.gather(self.layer_id)

        if document_cu_seqlens is not None:
            if paged_cache is not None:
                raise ValueError("varlen training attention cannot use a paged cache")
            if document_max_seqlen is None:
                raise ValueError("document_max_seqlen is required for varlen attention")
            flat_q, flat_k, flat_v = (
                q.flatten(0, 1),
                k.flatten(0, 1),
                v.flatten(0, 1),
            )
            if self.core_attention is not None and q.is_cuda:
                attention_output = self.core_attention(
                    flat_q,
                    flat_k,
                    flat_v,
                    qkv_format="thd",
                    cu_seqlens_q=document_cu_seqlens,
                    cu_seqlens_kv=document_cu_seqlens,
                    max_seqlen_q=document_max_seqlen,
                    max_seqlen_kv=document_max_seqlen,
                    attn_mask_type="causal",
                )
                if attention_output.ndim == 3:
                    attention_output = attention_output.flatten(1)
                sdqa_out = attention_output.view(x.size(0), x.size(1), self.dim)
            else:
                attention_output = _run_varlen_attention(
                    flat_q,
                    flat_k,
                    flat_v,
                    document_cu_seqlens,
                    document_max_seqlen,
                ).view(q.shape)
                sdqa_out = attention_output.contiguous().flatten(2)
            if self.use_gated_attention:
                sdqa_out = sdqa_out * F.sigmoid(self.gate(x))
            return self.o_proj(sdqa_out)

        if self.core_attention is not None and q.is_cuda:
            mask_type = "causal" if is_causal else (
                "no_mask" if attn_mask is None else "arbitrary"
            )
            attention_output = self.core_attention(
                q,
                k,
                v,
                attention_mask=attn_mask,
                qkv_format="bshd",
                attn_mask_type=mask_type,
            )
            if attention_output.ndim == 4:
                attention_output = attention_output.flatten(2)
            sdqa_out = attention_output.contiguous().view(
                x.size(0), x.size(1), self.dim
            )
            if self.use_gated_attention:
                sdqa_out = sdqa_out * F.sigmoid(self.gate(x))
            return self.o_proj(sdqa_out)

        q, k, v = q.permute(0, 2, 1, 3), k.permute(0, 2, 1, 3), v.permute(0, 2, 1, 3)

        def run_sdpa():
            return F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask,
                is_causal=is_causal,
                enable_gqa=self.n_rep > 1,
            )

        if (
            self.attention_backend == "flash_sdpa"
            and q.is_cuda
            and attn_mask is None
        ):
            # Fail loudly instead of silently selecting the math backend. The
            # 12B Hopper recipe is shaped for fused Flash-SDPA training.
            # Arbitrary block-diagonal document masks are not accepted by
            # every Flash-SDPA build, so masked batches use PyTorch's normal
            # backend selection (which can choose a compatible fused kernel
            # or the math fallback) instead of failing with no kernel.
            with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
                attention_output = run_sdpa()
        else:
            attention_output = run_sdpa()

        sdqa_out = attention_output.permute(0, 2, 1, 3).contiguous().flatten(2)

        if self.use_gated_attention:
            sdqa_out = sdqa_out * F.sigmoid(self.gate(x))

        out = self.o_proj(sdqa_out)
        return out


@AttnFactory.register("mla")
class MLA(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        kv_lora_rank: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        norm_eps: float,
        use_qk_norm: bool,
        use_gated_attention: bool,
        layer_id: int,
        n_layers: int = 1,
    ):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.layer_id = layer_id
        self.n_rep = n_heads // n_kv_heads
        self.use_qk_norm = use_qk_norm
        self.use_gated_attention = use_gated_attention

        self.q_proj = Linear(dim, n_heads * self.head_dim, bias=False)

        if self.use_qk_norm:
            self.q_norm = RMSNorm(self.head_dim, norm_eps)
            self.k_norm = RMSNorm(self.head_dim, norm_eps)
        self.kv_a_proj = Linear(dim, kv_lora_rank, bias=False)
        self.kv_norm = RMSNorm(kv_lora_rank, norm_eps)

        self.kv_b_proj = Linear(
            kv_lora_rank,
            n_kv_heads * (2 * self.head_dim),
        )

        self.o_proj = Linear(
            dim, dim, bias=False, init_std=0.02 / (2 * n_layers) ** 0.5
        )

        if use_gated_attention:
            self.gate = Linear(dim, dim, bias=False)

    def forward(
        self,
        x: Tensor,
        rotary_emb: Tensor,
        attn_mask: Tensor = None,
        paged_cache: Optional[CacheView] = None,
        is_causal: bool = False,
        document_cu_seqlens: Optional[Tensor] = None,
        document_max_seqlen: Optional[int] = None,
    ) -> Tensor:
        bsz, seq_len, _ = x.size()

        q = self.q_proj(x)
        q = q.view(bsz, seq_len, self.n_heads, self.head_dim)

        kv_compressed = self.kv_a_proj(x)
        kv_compressed = self.kv_norm(kv_compressed)

        kv = self.kv_b_proj(kv_compressed)
        kv = kv.view(bsz, seq_len, self.n_kv_heads, -1)

        k_nope, k_rope, v = torch.split(
            kv, [self.qk_nope_head_dim, self.qk_rope_head_dim, self.head_dim], dim=-1
        )

        q_nope, q_rope = (
            q[..., : self.qk_nope_head_dim],
            q[..., self.qk_nope_head_dim :],
        )
        q_rope = apply_rotary_emb(q_rope, rotary_emb)
        k_rope = apply_rotary_emb(k_rope, rotary_emb)

        q = torch.cat([q_nope, q_rope], dim=-1)
        k = torch.cat([k_nope, k_rope], dim=-1)

        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        if paged_cache is not None:
            paged_cache.write(self.layer_id, k, v)
            k, v = paged_cache.gather(self.layer_id)

        if document_cu_seqlens is not None:
            if paged_cache is not None:
                raise ValueError("varlen training attention cannot use a paged cache")
            if document_max_seqlen is None:
                raise ValueError("document_max_seqlen is required for varlen attention")
            attn_out = _run_varlen_attention(
                q.flatten(0, 1),
                k.flatten(0, 1),
                v.flatten(0, 1),
                document_cu_seqlens,
                document_max_seqlen,
            ).view(q.shape)
            attn_out = attn_out.contiguous().flatten(2)
            if self.use_gated_attention:
                attn_out = attn_out * F.sigmoid(self.gate(x))
            return self.o_proj(attn_out)

        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        attn_out = F.scaled_dot_product_attention(
            q, k, v, attn_mask, is_causal=is_causal
        )
        attn_out = attn_out.permute(0, 2, 1, 3).contiguous().flatten(2)

        if self.use_gated_attention:
            attn_out = attn_out * F.sigmoid(self.gate(x))

        out = self.o_proj(attn_out)
        return out
