from . import register_act_fn
from torch import nn


r"""
\[
\mathrm{HardSigmoid}(x)\approx \sigma(x)
\]
"""

@register_act_fn(name="hard_sigmoid")
class HardSigmoid(nn.Hardsigmoid):
    def __init__(self, inplace: bool = False) -> None:
        super().__init__()
        self.inplace = inplace

    def __repr__(self) -> str:
        return f"HardSigmoid(inplace={self.inplace})"
