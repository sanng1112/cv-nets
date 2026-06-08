"""LayerScale — per-channel learnable scaling for transformer stability (Touvron et al., 2021)."""

import torch
from torch import Tensor, nn


class LayerScale(nn.Module):
    """Learnable per-channel multiplicative scaling.

    Initialised near zero so the residual branch starts as an approximate
    identity.  Commonly used after the FFN and attention sub-layers in
    Vision Transformers.

    Parameters
    ----------
    dim : int
        Number of channels (features) to scale.
    init_value : float
        Initial value for the learnable scale parameter (default ``1e-5``).
    """

    def __init__(self, dim: int, init_value: float = 1e-5) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.full((dim,), init_value))

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() == 3:
            return x * self.scale[None, None, :]
        return x * self.scale[None, :, None, None]

    def extra_repr(self) -> str:
        return f"dim={len(self.scale)}, init_value={self.scale[0].item():.6f}"
