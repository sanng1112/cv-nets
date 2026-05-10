from . import register_act_fn
from torch import nn


r"""
\[
\mathrm{CELU}(x)=
\begin{cases}
x, & x>0 \\
\alpha(e^{x/\alpha}-1), & x\le 0
\end{cases}
\]
"""

@register_act_fn(name="celu")
class CELU(nn.CELU):
    def __init__(self, alpha: float = 1.0, inplace: bool = False) -> None:
        super().__init__(alpha=alpha, inplace=inplace)

    def __repr__(self) -> str:
        return f"CELU(alpha={self.alpha}, inplace={self.inplace})"
