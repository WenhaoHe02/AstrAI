from dataclasses import asdict
from typing import Optional

import torch.nn as nn
from torch import Tensor

from astrai.inference.core.cache import CacheView
from astrai.model.components.attention import AttnFactory
from astrai.model.components.mlp import FFNFactory
from astrai.model.components.norm import RMSNorm


class DecoderBlock(nn.Module):
    def __init__(self, config, layer_id: int):
        super().__init__()
        cfg = asdict(config)
        cfg.update(
            dim=config.hidden_size,
            dim_ffn=config.intermediate_size,
            n_layers=config.num_hidden_layers,
            n_heads=config.num_attention_heads,
            n_kv_heads=config.num_key_value_heads,
            norm_eps=config.rms_norm_eps,
            down_init_std=0.02 / (2 * config.num_hidden_layers) ** 0.5,
        )
        self.attention = AttnFactory.create(config.attn_type, **cfg, layer_id=layer_id)
        self.input_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        if config.residual_norm_backend not in ("torch", "liger"):
            raise ValueError(
                "residual_norm_backend must be 'torch' or 'liger', got "
                f"{config.residual_norm_backend!r}"
            )
        self.residual_norm_backend = config.residual_norm_backend
        self.mlp = FFNFactory.create(config.ffn_type, **cfg)

    def forward(
        self,
        x: Tensor,
        rotary_emb: Tensor,
        attention_mask: Optional[Tensor] = None,
        paged_cache: Optional[CacheView] = None,
        is_causal: bool = False,
        document_cu_seqlens: Optional[Tensor] = None,
        document_max_seqlen: Optional[int] = None,
        return_router_losses: bool = False,
    ):
        mlp_output, residual, router_outputs = self._forward_from_normalized(
            x,
            self.input_norm(x),
            rotary_emb,
            attention_mask,
            paged_cache,
            is_causal,
            document_cu_seqlens,
            document_max_seqlen,
        )
        x = mlp_output + residual

        if return_router_losses:
            return x, router_outputs
        return x

    def _forward_from_normalized(
        self,
        x: Tensor,
        normalized_x: Tensor,
        rotary_emb: Tensor,
        attention_mask: Optional[Tensor] = None,
        paged_cache: Optional[CacheView] = None,
        is_causal: bool = False,
        document_cu_seqlens: Optional[Tensor] = None,
        document_max_seqlen: Optional[int] = None,
    ):
        """Run a block while deferring the final MLP residual addition."""
        attn_output = self.attention(
            normalized_x,
            rotary_emb,
            attention_mask,
            paged_cache,
            is_causal,
            document_cu_seqlens,
            document_max_seqlen,
        )
        normalized_x, residual = self.post_attention_norm.forward_with_residual(
            attn_output,
            x,
            self.residual_norm_backend,
        )
        mlp_output = self.mlp(normalized_x)
        router_outputs = None
        if isinstance(mlp_output, tuple):
            mlp_output, *router_outputs = mlp_output
        return mlp_output, residual, router_outputs
