import torch
from torch import Tensor, nn

from . import register_act_fn


r"""
\[
\mathrm{Sparsemax}(x)=\arg\min_{p\in \Delta^K}\|p-x\|^2
\]
"""

@register_act_fn(name="sparsemax")
class Sparsemax(nn.Module):
    def __init__(self, dim: int = -1) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        x = x - x.max(dim=self.dim, keepdim=True).values
        z_sorted, _ = torch.sort(x, dim=self.dim, descending=True)
        z_cumsum = z_sorted.cumsum(dim=self.dim)
        k = torch.arange(1, x.size(self.dim) + 1, device=x.device, dtype=x.dtype)
        view_shape = [1] * x.dim()
        view_shape[self.dim] = -1
        k = k.view(view_shape)

        support = 1 + k * z_sorted > z_cumsum
        k_z = support.sum(dim=self.dim, keepdim=True).clamp(min=1)
        idx = k_z - 1
        tau = (z_cumsum.gather(self.dim, idx) - 1) / k_z.to(x.dtype)
        return torch.clamp(x - tau, min=0.0)

    def __repr__(self) -> str:
        return f"Sparsemax(dim={self.dim})"
