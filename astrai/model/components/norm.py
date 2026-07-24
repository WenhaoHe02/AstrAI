import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class RMSNorm(nn.Module):
    def __init__(self, dim, norm_eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.normalized_shape = (dim,)
        self.norm_eps = norm_eps

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, self.normalized_shape, self.weight, self.norm_eps)

    def forward_with_residual(
        self,
        x: Tensor,
        residual: Tensor,
        backend: str = "torch",
    ) -> tuple[Tensor, Tensor]:
        """Add a residual and normalize while preserving this module's state."""
        if backend == "torch" or not x.is_cuda:
            residual = x + residual
            return self(residual), residual
        if backend != "liger":
            raise ValueError(
                f"residual_norm_backend must be 'torch' or 'liger', got {backend!r}"
            )
        try:
            from liger_kernel.ops import LigerFusedAddRMSNormFunction
        except ImportError as exc:
            raise RuntimeError(
                "residual_norm_backend='liger' requires the liger-kernel package"
            ) from exc
        return LigerFusedAddRMSNormFunction.apply(
            x,
            residual,
            self.weight,
            self.norm_eps,
            0.0,
            "llama",
            False,
        )
