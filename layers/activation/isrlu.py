import torch
from torch import Tensor, nn

from . import register_act_fn


r"""
\[
\mathrm{ISRLU}(x)=
\begin{cases}
x, & x\ge 0 \\
\frac{x}{\sqrt{1+\alpha x^2}}, & x<0
\end{cases}
\]
"""

@register_act_fn(name="isrlu")
class ISRLU(nn.Module):
    def __init__(self, alpha: float = 1.0, inplace: bool = False) -> None:
        super().__init__()
        self.alpha = alpha
        self.inplace = inplace

    def forward(self, x: Tensor) -> Tensor:
        pos = torch.relu(x)
        neg = torch.where(x < 0, x / torch.sqrt(1.0 + self.alpha * x * x), x)
        return torch.where(x >= 0, pos, neg)

    def __repr__(self) -> str:
        return f"ISRLU(alpha={self.alpha}, inplace={self.inplace})"
