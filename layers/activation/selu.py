from . import register_act_fn
from torch import nn


r"""
\[
\mathrm{SELU}(x)=\lambda
\begin{cases}
x, & x>0 \\
\alpha(e^x-1), & x\le 0
\end{cases}
\]
"""

@register_act_fn(name="selu")
class SELU(nn.SELU):
    def __init__(self, inplace: bool = False) -> None:
        super().__init__(inplace=inplace)

    def __repr__(self) -> str:
        return f"SELU(inplace={self.inplace})"
