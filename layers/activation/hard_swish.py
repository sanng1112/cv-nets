from . import register_act_fn
from torch import nn


r"""
\[
\mathrm{HardSwish}(x)=x\cdot\frac{\mathrm{ReLU6}(x+3)}{6}
\]
"""

@register_act_fn(name="hard_swish")
class HardSwish(nn.Hardswish):
    def __init__(self, inplace: bool = False) -> None:
        super().__init__(inplace=inplace)

    def __repr__(self) -> str:
        return f"HardSwish(inplace={self.inplace})"
