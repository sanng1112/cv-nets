"""RMSNorm — Root Mean Square Layer Normalization (Zhang & Sennrich, 2019)."""

import torch
from torch import Tensor, nn


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Normalises inputs by their root-mean-square statistic along the last
    dimension instead of mean+std as in LayerNorm.  Popular in modern
    transformers (LLaMA, ViT-22B, etc.) for its computational efficiency.

    Parameters
    ----------
    normalized_shape : int or tuple
        Shape of the normalisation dimension(s).
    eps : float
        Small constant for numerical stability (default ``1e-6``).
    """

    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.normalized_shape = (
            (normalized_shape,)
            if isinstance(normalized_shape, int)
            else normalized_shape
        )
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(self.normalized_shape))

    def forward(self, x: Tensor) -> Tensor:
        dtype = x.dtype
        x_f32 = x.float()
        rms = torch.sqrt(
            torch.mean(x_f32 ** 2, dim=-1, keepdim=True) + self.eps
        )
        return (x_f32 / rms).to(dtype) * self.weight

    def extra_repr(self) -> str:
        return f"normalized_shape={self.normalized_shape}, eps={self.eps}"
