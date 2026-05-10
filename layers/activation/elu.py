from . import register_act_fn
from torch import nn


r"""
\[
\mathrm{ELU}(x)=
\begin{cases}
x, & x>0 \\
\alpha(e^x-1), & x\le 0
\end{cases}
\]
"""

@register_act_fn(name="elu")
class ELU(nn.ELU):
    def __init__(self, alpha: float = 1.0, inplace: bool = False) -> None:
        super().__init__(alpha=alpha, inplace=inplace)

    def __repr__(self) -> str:
        return f"ELU(alpha={self.alpha}, inplace={self.inplace})"
