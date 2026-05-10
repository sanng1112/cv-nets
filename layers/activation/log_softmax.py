from torch import Tensor, nn
from torch.nn import functional as F

from . import register_act_fn


r"""
\[
\mathrm{LogSoftmax}(x)=\log(\mathrm{Softmax}(x))
\]
"""

@register_act_fn(name="log_softmax")
class LogSoftmax(nn.Module):
    def __init__(self, dim: int = -1) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        return F.log_softmax(x, dim=self.dim)

    def __repr__(self) -> str:
        return f"LogSoftmax(dim={self.dim})"
