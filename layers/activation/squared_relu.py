from torch import Tensor, nn
from torch.nn import functional as F

from . import register_act_fn


r"""
\[
\mathrm{SquaredReLU}(x)=\mathrm{ReLU}(x)^2
\]
"""

@register_act_fn(name="squared_relu")
class SquaredReLU(nn.Module):
    def __init__(self, inplace: bool = False) -> None:
        super().__init__()
        self.inplace = inplace

    def forward(self, x: Tensor) -> Tensor:
        y = F.relu(x, inplace=self.inplace)
        return y * y

    def __repr__(self) -> str:
        return f"SquaredReLU(inplace={self.inplace})"
