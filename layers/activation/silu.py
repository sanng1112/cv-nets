from . import register_act_fn
from torch import nn


r"""
\[
\mathrm{SiLU}(x)=x\cdot \sigma(x)
\]
"""

@register_act_fn(name="silu")
class SiLU(nn.SiLU):
    def __init__(self, inplace: bool = False) -> None:
        super().__init__(inplace=inplace)

    def __repr__(self) -> str:
        return f"SiLU(inplace={self.inplace})"
